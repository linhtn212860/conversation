"""
WebSocket Server for Speech-to-Speech Pipeline
Provides streaming API for real-time voice conversation
"""
import asyncio
import json
import numpy as np
import base64
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .pipeline import SpeechToSpeechPipeline, PipelineState
from .config import (
    WS_HOST,
    WS_PORT,
    SAMPLE_RATE,
    TTS_OUTPUT_SAMPLE_RATE,
    CHUNK_SIZE
)


app = FastAPI(title="Speech-to-Speech API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (shared across connections)
# In production, may want per-connection pipelines
pipeline: Optional[SpeechToSpeechPipeline] = None


def set_pipeline(p: SpeechToSpeechPipeline):
    """Set the global pipeline (for preloading)."""
    global pipeline
    pipeline = p


def get_pipeline() -> SpeechToSpeechPipeline:
    """Get or create pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = SpeechToSpeechPipeline(device="cuda")
        pipeline.load_all_models()
    return pipeline


class WebSocketConnection:
    """Manages a single WebSocket connection."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.pipeline = get_pipeline()
        self.is_active = True
        self.is_connected = True
        self.audio_buffer = []
        self._processing = False
    
    async def send_json(self, data: dict):
        """Send JSON message to client."""
        if not self.is_connected:
            return
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            self.is_connected = False
            # Suppress log if client already disconnected
            error_str = str(e).lower()
            if "not connected" not in error_str and "accept" not in error_str and "closed" not in error_str:
                print(f"Error sending message: {e}")
    
    async def send_audio(self, audio: np.ndarray):
        """Send audio data to client as base64."""
        if not self.is_connected:
            return
        # Convert to int16 for transmission
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        await self.send_json({
            "type": "audio",
            "data": audio_b64,
            "sample_rate": TTS_OUTPUT_SAMPLE_RATE
        })
    
    async def handle_audio_chunk(self, audio_b64: str):
        """Handle incoming audio chunk from client."""
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(audio_b64)
            
            # Convert to numpy (assuming int16 input)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Feed to pipeline VAD
            result = self.pipeline.process_audio_chunk(audio_float)
            
            if result is not None:
                # Speech segment detected and processed
                await self.send_json({
                    "type": "transcription",
                    "text": result.transcription
                })
                
                await self.send_json({
                    "type": "response",
                    "text": result.response_text
                })
                
                if result.audio_chunk is not None and len(result.audio_chunk) > 0:
                    await self.send_audio(result.audio_chunk)
                
                await self.send_json({
                    "type": "processing_complete",
                    "processing_time_ms": result.processing_time_ms
                })
        
        except Exception as e:
            await self.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def handle_text(self, text: str):
        """Handle text input (alternative to speech)."""
        try:
            await self.send_json({
                "type": "state",
                "state": "processing"
            })
            
            # Generate response
            response = self.pipeline.llm.generate(text)
            
            await self.send_json({
                "type": "response",
                "text": response
            })
            
            # Generate speech
            audio = self.pipeline.tts.synthesize(response)
            
            if len(audio) > 0:
                await self.send_audio(audio)
            
            await self.send_json({
                "type": "state",
                "state": "idle"
            })
        
        except Exception as e:
            await self.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def handle_full_audio(self, audio_b64: str):
        """Handle complete audio segment (non-streaming)."""
        try:
            # Decode
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Process with streaming output
            async for result in self._process_streaming(audio_float):
                if result.transcription:
                    await self.send_json({
                        "type": "transcription",
                        "text": result.transcription
                    })
                
                if result.response_text:
                    await self.send_json({
                        "type": "response_chunk",
                        "text": result.response_text
                    })
                
                if result.audio_chunk is not None and len(result.audio_chunk) > 0:
                    await self.send_audio(result.audio_chunk)
            
            await self.send_json({
                "type": "processing_complete"
            })
        
        except Exception as e:
            await self.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def _process_streaming(self, audio: np.ndarray):
        """Async wrapper for streaming processing."""
        # Run the generator in a thread pool to avoid blocking the event loop
        import concurrent.futures
        loop = asyncio.get_event_loop()

        # Create a queue to pass results from thread to async
        queue = asyncio.Queue()
        done = asyncio.Event()

        def run_generator():
            try:
                for result in self.pipeline.process_speech_streaming(audio):
                    # Check if still connected before putting in queue
                    if not self.is_connected:
                        break
                    # Put result in queue from thread
                    asyncio.run_coroutine_threadsafe(queue.put(result), loop)
            except Exception as e:
                print(f"Generator error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Signal completion
                asyncio.run_coroutine_threadsafe(done.set(), loop)

        # Start generator in thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(run_generator)

        # Yield results as they come
        while not done.is_set() and self.is_connected:
            try:
                result = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield result
            except asyncio.TimeoutError:
                continue

        # Mark as disconnected if we broke out early
        if not self.is_connected:
            self._processing = False

        # Drain remaining items only if still connected
        while not queue.empty() and self.is_connected:
            try:
                result = await asyncio.wait_for(queue.get(), timeout=0.01)
                yield result
            except asyncio.TimeoutError:
                break

        executor.shutdown(wait=False)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint."""
    await websocket.accept()
    
    conn = WebSocketConnection(websocket)
    
    await conn.send_json({
        "type": "connected",
        "message": "Speech-to-Speech API connected",
        "sample_rate": SAMPLE_RATE
    })
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            msg_type = data.get("type", "")
            
            if msg_type == "audio_chunk":
                # Streaming audio input
                await conn.handle_audio_chunk(data.get("data", ""))
            
            elif msg_type == "audio":
                # Complete audio segment
                await conn.handle_full_audio(data.get("data", ""))
            
            elif msg_type == "text":
                # Text input
                await conn.handle_text(data.get("text", ""))
            
            elif msg_type == "reset":
                # Reset conversation
                conn.pipeline.reset()
                await conn.send_json({
                    "type": "reset_complete"
                })
            
            elif msg_type == "ping":
                await conn.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print("Client disconnected")
        conn.is_connected = False
    except Exception as e:
        print(f"WebSocket error: {e}")
        conn.is_connected = False
    finally:
        conn.is_connected = False


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Speech-to-Speech API",
        "websocket_url": f"ws://{WS_HOST}:{WS_PORT}/ws"
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


def run_server(host: str = WS_HOST, port: int = WS_PORT):
    """Run the WebSocket server."""
    print("=" * 60)
    print("🎤 Speech-to-Speech WebSocket Server")
    print("=" * 60)
    print(f"📡 WebSocket URL: ws://{host}:{port}/ws")
    print(f"🌐 HTTP URL: http://{host}:{port}")
    print("=" * 60)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
