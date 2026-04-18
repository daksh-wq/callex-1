import asyncio
import json
import os
import websockets

class CallexSTT:
    """
    Microservice Client mapping exactly to local Callex Internal Acoustic model logic.
    Dumps all heavy PyTorch computation off to the GPU `stt_server.py` microservice
    to guarantee zero GIL locking and perfect audio responsiveness.
    """
    def __init__(
        self,
        api_key: str = None, 
        on_transcript=None,
        on_speech_started=None,
        on_speech_ended=None,
        model: str = "large-v3", 
        language: str = "hi", 
        **kwargs
    ):
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._language = language[:2] if language else "hi"
        
        self._is_connected = False
        self._ws = None
        self._ws_queue = asyncio.Queue()
        self._listen_task = None
        self._send_task = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected
        
    async def connect(self):
        try:
            # Route to GPU server via environment variable (same as TTS)
            gpu_ip = os.getenv("CALLEX_GPU_URL", "127.0.0.1")
            stt_url = f"ws://{gpu_ip}:8123/ws"
            print(f"[Callex AI STT] 🔗 Connecting to GPU STT: {stt_url}")
            self._ws = await websockets.connect(stt_url)
            self._is_connected = True
            
            # Start background send/receive loops so we don't block VoIP
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            
            print(f"[Callex AI STT] ✅ Connected (Microservice Linked natively)")
        except Exception as e:
            print(f"[Callex AI STT] ❌ Connection to Local Microservice Failed: {e}. Is stt_server running?")
            self._is_connected = False
            
    def send_audio(self, pcm16_bytes: bytes):
        """Append incoming audio bytes cleanly to the non-blocking queue."""
        if not self._is_connected or not pcm16_bytes:
            return
        # Put synchronously, sending is handled async
        self._ws_queue.put_nowait({"type": "audio", "data": pcm16_bytes})

    def send_flush(self):
        """Force flush buffer dynamically to the server."""
        if self._is_connected:
            self._ws_queue.put_nowait({"type": "flush"})

    async def _send_loop(self):
        try:
            while self._is_connected and self._ws:
                msg = await self._ws_queue.get()
                if msg["type"] == "audio":
                    await self._ws.send(msg["data"])
                elif msg["type"] == "flush":
                    await self._ws.send("FLUSH")
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except asyncio.CancelledError:
            pass

    async def _listen_loop(self):
        try:
            while self._is_connected and self._ws:
                response = await self._ws.recv()
                data = json.loads(response)
                event = data.get("event")
                
                if event == "speech_started" and self._on_speech_started:
                    asyncio.create_task(self._on_speech_started())
                    
                elif event == "speech_ended" and self._on_speech_ended:
                    asyncio.create_task(self._on_speech_ended())
                    
                elif event == "transcript" and self._on_transcript:
                    text = data.get("text")
                    lat = data.get('latency', 0)
                    print(f"[Microservice STT] 📝 Transcript: '{text[:80]}' (latency={lat:.2f}s)")
                    await self._on_transcript(text)
                    
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
            print("[Callex AI STT] 🔌 WebSocket Closed by Server")
        except asyncio.CancelledError:
            pass

    async def disconnect(self):
        self._is_connected = False
        if self._listen_task: self._listen_task.cancel()
        if self._send_task: self._send_task.cancel()
        if self._ws:
            await self._ws.close()
        print("[Callex AI STT] 🔌 Client Disconnected completely from Backend")
