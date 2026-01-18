import asyncio
import io
import logging
import os
import subprocess
import threading
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks
from pydantic import BaseModel, Field
from queue import Queue, Full
from typing import Literal, Optional, AsyncIterator
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
QUEUE_SIZE = 32
QUEUE_TIMEOUT = 2.0
EOF_TIMEOUT = 1.0
CHUNK_SIZE = 64 * 1024
DEFAULT_SAMPLE_RATE = 24000

# map OpenAI voice names to pocket_tts voice names
VOICE_MAPPING = {
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

FFMPEG_FORMATS = {
    "mp3": ("mp3", "libmp3lame"),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


class SpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd"] = Field("tts-1", description="TTS model to use")
    input: str = Field(
        ..., min_length=1, max_length=4096, description="Text to generate"
    )
    voice: str = Field("alloy", description="Voice identifier (predefined or custom)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("wav")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)


class FileLikeQueueWriter:
    """File-like adapter that writes bytes to a queue with backpressure."""

    def __init__(self, queue: Queue, timeout: float = QUEUE_TIMEOUT):
        self.queue = queue
        self.timeout = timeout

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        try:
            self.queue.put(data, timeout=self.timeout)
            return len(data)
        except Full:
            logger.warning("Queue full, dropping chunk")
            return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=EOF_TIMEOUT)
        except (Full, Exception):
            try:
                self.queue.put_nowait(None)
            except (Full, Exception):
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            logger.exception("Error closing queue writer")
        return False


# Global model state
tts_model: Optional[TTSModel] = None
device: Optional[str] = None
sample_rate: Optional[int] = None


@asynccontextmanager
async def lifespan(app):
    """Load the TTS model on startup."""
    logger.info("🚀 Starting TTS API server...")
    load_tts_model()
    yield


def load_tts_model() -> None:
    """Load TTS model once and keep in memory."""
    global tts_model, device, sample_rate

    tts_model = TTSModel.load_model()
    device = tts_model.device
    sample_rate = getattr(tts_model, "sample_rate", DEFAULT_SAMPLE_RATE)

    logger.info(f"Pocket TTS loaded | Device: {device} | Sample Rate: {sample_rate}")


def _start_audio_producer(queue: Queue, voice_name: str, text: str) -> threading.Thread:
    """Start background thread that generates audio and writes to queue."""

    def producer():
        logger.info(f"Starting audio generation for voice: {voice_name}")
        try:
            model_state = tts_model.get_state_for_audio_prompt(voice_name)
            audio_chunks = tts_model.generate_audio_stream(
                model_state=model_state, text_to_generate=text
            )
            with FileLikeQueueWriter(queue) as writer:
                stream_audio_chunks(
                    writer, audio_chunks, sample_rate or DEFAULT_SAMPLE_RATE
                )
        except Exception:
            logger.exception("Audio generation failed")
        finally:
            try:
                queue.put(None, timeout=EOF_TIMEOUT)
            except (Full, Exception):
                pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return thread


async def _stream_queue_chunks(queue: Queue) -> AsyncIterator[bytes]:
    """Async generator that yields bytes from queue until EOF."""
    while True:
        chunk = await asyncio.to_thread(queue.get)
        if chunk is None:
            logger.info("Received EOF")
            break
        yield chunk


def _start_ffmpeg_process(format: str) -> tuple[subprocess.Popen, int, int]:
    """Start ffmpeg process with OS pipe for stdin."""
    out_fmt, codec = FFMPEG_FORMATS.get(format, ("mp3", "libmp3lame"))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-f",
        out_fmt,
        "-codec:a",
        codec,
        "pipe:1",
    ]
    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")
    proc = subprocess.Popen(cmd, stdin=r_file, stdout=subprocess.PIPE)
    return proc, w_fd, r_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
    """Start thread that writes queue chunks to OS pipe."""

    def pipe_writer():
        try:
            with os.fdopen(write_fd, "wb") as pipe:
                while True:
                    data = queue.get()
                    if data is None:
                        break
                    try:
                        pipe.write(data)
                    except (BrokenPipeError, OSError):
                        break
                pipe.flush()
        except Exception:
            try:
                os.close(write_fd)
            except (OSError, Exception):
                pass

    thread = threading.Thread(target=pipe_writer, daemon=True)
    thread.start()
    return thread


async def generate_audio(
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    format: str = "wav",
    chunk_size: int = CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Generate and stream audio in requested format.

    Args:
        text: Text to synthesize
        voice: Voice identifier (OpenAI format)
        speed: Playback speed multiplier (unused currently)
        format: Output audio format
        chunk_size: Size of chunks to read from ffmpeg stdout

    Yields:
        Audio bytes in requested format
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # If voice is a known alias, map it; otherwise pass-through custom voice names
    voice_name = VOICE_MAPPING.get(voice, voice)
    queue = Queue(maxsize=QUEUE_SIZE)
    producer_thread = _start_audio_producer(queue, voice_name, text)

    try:
        if format in ("wav", "pcm"):
            async for chunk in _stream_queue_chunks(queue):
                yield chunk
            producer_thread.join()
            return

        if format in FFMPEG_FORMATS:
            proc, write_fd, _ = _start_ffmpeg_process(format)
            writer_thread = _start_pipe_writer(queue, write_fd)

            try:
                while True:
                    chunk = await asyncio.to_thread(proc.stdout.read, chunk_size)
                    if not chunk:
                        logger.info(f"FFmpeg output complete for {format}")
                        break
                    yield chunk
            finally:
                proc.wait()
                producer_thread.join()
                writer_thread.join()
            return

        # Fallback for unsupported formats
        async for chunk in _stream_queue_chunks(queue):
            yield chunk
        producer_thread.join()

    except Exception:
        logger.exception(f"Error streaming audio format: {format}")
        raise


app = FastAPI(
    title="OpenAI-Compatible TTS API (Cached)",
    description="OpenAI Audio Speech API compatible endpoint using Kyutai TTS with model caching",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest) -> StreamingResponse:
    """Generate speech audio from text with streaming response."""
    return StreamingResponse(
        generate_audio(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            format=request.response_format,
        ),
        media_type=MEDIA_TYPES.get(request.response_format, "audio/wav"),
    )


if __name__ == "__main__":

    # Configure uvicorn logging for HTTP debugging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

    logger.info("Starting server with HTTP debug logging enabled")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config, access_log=True)
