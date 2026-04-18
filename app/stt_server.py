import asyncio
import json
import time
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging

# Mute heavy FastAPI system logs to keep terminal clean
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

print("[Microservice STT] Starting dedicated PyTorch inference environment...")
print("[Microservice STT] Loading 'faster-whisper' model exclusively to GPU/CPU memory...")

try:
    global_model = WhisperModel("large-v3", device="auto", compute_type="default")
    print(f"[Microservice STT] ✅ Model loaded successfully.")
except Exception as e:
    print(f"[Microservice STT] ⚠️ Fallback to 'int8/cpu' due to load error: {e}")
    global_model = WhisperModel("small", device="cpu", compute_type="int8")

@app.websocket("/ws")
async def stt_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    vad = webrtcvad.Vad(3)
    audio_buffer = bytearray()
    is_speaking = False
    silence_frames = 0
    language = "hi"
    
    def process_inference(audio_bytes):
        # Convert PCM to Float32 array (-1.0 to 1.0)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        start = time.time()
        segments, _ = global_model.transcribe(audio_np, beam_size=1, language=language, condition_on_previous_text=False)
        text = " ".join([seg.text for seg in segments]).strip()
        latency = time.time() - start
        return text, latency

    try:
        while True:
            # Receive binary packet from the Callex Voice Engine
            data = await websocket.receive()
            
            # Handles text commands (like FLUSH) vs binary audio
            if "text" in data:
                payload = data["text"]
                if payload == "FLUSH":
                    if len(audio_buffer) > 6400: # Ensure minimum length
                        buffer_copy = bytearray(audio_buffer)
                        audio_buffer.clear()
                        is_speaking = False
                        
                        loop = asyncio.get_running_loop()
                        text, latency = await loop.run_in_executor(executor, process_inference, buffer_copy)
                        if text:
                            await websocket.send_json({"event": "transcript", "text": text, "latency": latency})
                continue
                
            if "bytes" in data:
                payload = data["bytes"]
                audio_buffer.extend(payload)
                
                # Check VAD logic securely using latest incoming chunk
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
                        if is_speaking and silence_frames > 40: # ~0.8s internal silence barrier
                            is_speaking = False
                            await websocket.send_json({"event": "speech_ended"})
                            
                            buffer_copy = bytearray(audio_buffer)
                            audio_buffer.clear()
                            
                            loop = asyncio.get_running_loop()
                            text, latency = await loop.run_in_executor(executor, process_inference, buffer_copy)
                            if text:
                                await websocket.send_json({"event": "transcript", "text": text, "latency": latency})

    except WebSocketDisconnect:
        pass  # Expected disconnect when client ends call
    except Exception as e:
        print(f"[STT Server] Error Exception: {e}")

if __name__ == "__main__":
    uvicorn.run("stt_server:app", host="127.0.0.1", port=8123, log_level="info")
