"""
╔══════════════════════════════════════════════════════════════════╗
║  CALLEX GPU STT MICROSERVICE — Standalone Deployment            ║
║  Runs Faster Whisper large-v3 on GPU via WebSocket.             ║
║  Your PBX server connects here for real-time transcription.     ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python stt_server.py

API:
    WebSocket ws://<GPU_IP>:8123/ws
    Send: raw PCM16 audio bytes (16kHz mono)
    Receive: JSON {"event": "transcript", "text": "...", "latency": 0.12}
"""

import asyncio
import json
import time
import os
import signal
import subprocess
import numpy as np
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import logging

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Callex GPU STT")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Callex GPU STT API")
executor = ThreadPoolExecutor(max_workers=4)

# ── Global model reference (loaded on startup, NOT at import time) ──
global_model = None
MODEL_LOADED = False


def _kill_port(port: int):
    """Kill any process occupying the given port so we can bind cleanly."""
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"🔪 Killed stale process on port {port}")
            time.sleep(0.5)          # give OS time to release the socket
    except FileNotFoundError:
        # fuser not available, try lsof fallback
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"], timeout=5
            ).decode().strip()
            for pid in out.split("\n"):
                if pid and pid.isdigit():
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"🔪 Killed PID {pid} on port {port}")
            time.sleep(0.5)
        except Exception:
            pass
    except Exception:
        pass


@app.on_event("startup")
async def load_model():
    """Load Whisper model once at startup — avoids double-load on re-import."""
    global global_model, MODEL_LOADED

    logger.info("Loading Faster Whisper large-v3 model...")
    try:
        from faster_whisper import WhisperModel
        global_model = WhisperModel("large-v3", device="auto", compute_type="default")
        MODEL_LOADED = True
        logger.info("✅ Whisper large-v3 loaded successfully!")
    except Exception as e:
        logger.warning(f"⚠️ Fallback to small/cpu: {e}")
        try:
            from faster_whisper import WhisperModel
            global_model = WhisperModel("small", device="cpu", compute_type="int8")
            MODEL_LOADED = True
        except Exception as e2:
            logger.error(f"❌ Failed to load any model: {e2}")
            global_model = None
            MODEL_LOADED = False


@app.get("/health")
async def health_check():
    """Health check — PBX can ping this to verify STT server is alive."""
    import torch
    return JSONResponse({
        "status": "online",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "model_loaded": MODEL_LOADED
    })


@app.websocket("/ws")
async def stt_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    vad = webrtcvad.Vad(3)
    audio_buffer = bytearray()
    is_speaking = False
    silence_frames = 0
    language = "hi"

    def process_inference(audio_bytes):
        """Blocking Whisper inference — runs in ThreadPool."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        start = time.time()
        segments, _ = global_model.transcribe(
            audio_np, beam_size=1, language=language, 
            condition_on_previous_text=False
        )
        text = " ".join([seg.text for seg in segments]).strip()
        latency = time.time() - start
        return text, latency

    try:
        while True:
            data = await websocket.receive()
            
            # Handle text commands (FLUSH)
            if "text" in data:
                payload = data["text"]
                if payload == "FLUSH":
                    if len(audio_buffer) > 6400:
                        buffer_copy = bytearray(audio_buffer)
                        audio_buffer.clear()
                        is_speaking = False
                        
                        loop = asyncio.get_running_loop()
                        text, latency = await loop.run_in_executor(
                            executor, process_inference, buffer_copy
                        )
                        if text:
                            await websocket.send_json({
                                "event": "transcript", 
                                "text": text, 
                                "latency": latency
                            })
                continue

            # Handle binary audio
            if "bytes" in data:
                payload = data["bytes"]
                audio_buffer.extend(payload)
                
                CHUNK_SIZE = 640
                if len(payload) >= CHUNK_SIZE:
                    chunk = payload[:CHUNK_SIZE]
                    try:
                        is_speech = vad.is_speech(chunk, 16000)
                    except:
                        is_speech = False

                    if is_speech:
                        silence_frames = 0
                        if not is_speaking:
                            is_speaking = True
                            await websocket.send_json({"event": "speech_started"})
                    else:
                        silence_frames += 1
                        if is_speaking and silence_frames > 40:
                            is_speaking = False
                            await websocket.send_json({"event": "speech_ended"})
                            
                            buffer_copy = bytearray(audio_buffer)
                            audio_buffer.clear()
                            
                            loop = asyncio.get_running_loop()
                            text, latency = await loop.run_in_executor(
                                executor, process_inference, buffer_copy
                            )
                            if text:
                                await websocket.send_json({
                                    "event": "transcript",
                                    "text": text,
                                    "latency": latency
                                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    PORT = int(os.getenv("STT_PORT", "8123"))

    # Kill any zombie process holding the port from a previous crash
    _kill_port(PORT)

    logger.info(f"🚀 Starting Callex GPU STT on 0.0.0.0:{PORT}")
    # Pass app object directly — do NOT use the string "stt_server:app"
    # which causes uvicorn to re-import the module and double-load everything.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="error")
