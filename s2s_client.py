#!/usr/bin/env python3
"""Speech-to-Speech Client.

Client kết nối đến Speech-to-Speech WebSocket server (English-only pipeline).

Server mặc định của project (FastAPI) chạy tại endpoint `/ws`:
    python run_server.py

Cách dùng:
    python s2s_client.py
    python s2s_client.py --url ws://localhost:8765/ws
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import threading
import queue
import time
import tempfile
import numpy as np

# Audio
try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: pip install soundfile")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def log(self, msg): print(msg.replace('[', '').replace(']', ''))
    console = Console()


# =============================================================================
# CONFIG
# =============================================================================

SERVER_URL = "ws://localhost:8765/ws"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SPEECH_THRESHOLD = 0.5


# =============================================================================
# Silero VAD
# =============================================================================

class SileroVAD:
    """
    Silero VAD wrapper.
    Silero VAD yêu cầu audio chunk đúng 512 samples cho 16kHz hoặc 256 cho 8kHz.
    """
    
    def __init__(self, threshold: float = SPEECH_THRESHOLD):
        console.log("[cyan]Loading Silero VAD...[/cyan]")
        self.model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad',
            force_reload=False, trust_repo=True
        )
        self.threshold = threshold
        console.log("[green]✓ VAD loaded[/green]")
    
    def reset(self):
        """Reset VAD states."""
        self.model.reset_states()
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        Audio chunk MUST be exactly 512 samples for 16kHz.
        
        Args:
            audio_chunk: Audio samples (float32, 16kHz, exactly 512 samples)
            
        Returns:
            True if speech detected
        """
        if len(audio_chunk) == 0:
            return False
        
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
        
        return speech_prob > self.threshold


# =============================================================================
# Audio Player
# =============================================================================

class AudioPlayer:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = False
        self._is_playing = False
        self._play_end = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                data = self.queue.get(timeout=0.1)
                if data is None:
                    break
                
                self._is_playing = True
                try:
                    # data is (audio_array, sample_rate) or file path
                    if isinstance(data, str):
                        audio, sr = sf.read(data)
                    else:
                        audio, sr = data
                    
                    sd.play(audio, sr)
                    sd.wait()
                except Exception as e:
                    console.log(f"[red]Play error: {e}[/red]")
                finally:
                    self._is_playing = False
                    self._play_end = time.time()
            except queue.Empty:
                continue

    @property
    def is_playing(self) -> bool:
        return self._is_playing
    
    @property
    def is_suppressing(self) -> bool:
        return self._is_playing or (time.time() - self._play_end < 0.5)

    def play_pcm_int16(self, audio_bytes: bytes, sample_rate: int):
        """Play raw PCM int16 mono from bytes."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return
        audio = audio_int16.astype(np.float32) / 32768.0
        self.queue.put((audio, sample_rate))

    def stop(self):
        self.running = False
        self.queue.put(None)


# =============================================================================
# S2S Client
# =============================================================================

class S2SClient:
    def __init__(self, url: str = SERVER_URL):
        self.url = url
        self.ws = None
        
        console.log("")
        console.log("[bold]╔══════════════════════════════════════╗[/bold]")
        console.log("[bold]║   Speech-to-Speech Client            ║[/bold]")
        console.log("[bold]╚══════════════════════════════════════╝[/bold]")
        console.log("")
        
        self.vad = SileroVAD()
        self.player = AudioPlayer()
        self.player.start()
        
        console.log(f"[dim]Server: {self.url}[/dim]")
        console.log("")

    async def connect(self) -> bool:
        try:
            console.log(f"[cyan]Connecting to {self.url}...[/cyan]")
            self.ws = await websockets.connect(self.url, max_size=10*1024*1024)  # 10MB limit
            
            # Wait for welcome
            msg = await self.ws.recv()
            data = json.loads(msg)
            if data.get("type") == "connected":
                console.log(f"[green]✓ Connected! Session: {data.get('session_id')}[/green]")
                return True
            return False
        except Exception as e:
            console.log(f"[red]Connection failed: {e}[/red]")
            return False

    async def send_audio(self, audio: np.ndarray):
        """Send audio chunk to server."""
        if self.ws is None:
            return
        
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
        
        await self.ws.send(json.dumps({
            "type": "audio",
            "data": audio_b64,
            "sample_rate": SAMPLE_RATE
        }))

    async def send_text(self, text: str):
        """Send text to server."""
        if self.ws is None:
            return
        
        console.log(f"[cyan]🎤 Bạn: {text}[/cyan]")
        await self.ws.send(json.dumps({
            "type": "text",
            "text": text
        }))

    async def receive_loop(self):
        """Background loop to receive server messages."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "transcription":
                    user_text = data.get("text", "")
                    if user_text:
                        console.log(f"[cyan]🎤 Bạn: {user_text}[/cyan]")

                elif msg_type in ("response", "response_chunk"):
                    ai_text = data.get("text", "")
                    if ai_text:
                        console.log(f"[green]💬 AI: {ai_text}[/green]")

                elif msg_type == "audio":
                    audio_b64 = data.get("data", "")
                    sr = int(data.get("sample_rate", 24000))
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        self.player.play_pcm_int16(audio_bytes, sr)

                elif msg_type == "error":
                    console.log(f"[red]Server error: {data.get('message', '')}[/red]")

                elif msg_type == "processing_complete":
                    pass

                elif msg_type in ("connected", "state", "reset_complete", "pong"):
                    pass
        
        except websockets.exceptions.ConnectionClosed:
            console.log("[yellow]Connection closed[/yellow]")

    async def run(self):
        """Main client loop."""
        if not await self.connect():
            return

        # Start receive loop
        receive_task = asyncio.create_task(self.receive_loop())

        # Audio capture state
        audio_buffer = []
        is_speaking = False
        silence_samples = 0
        min_speech = int(SAMPLE_RATE * 0.5)  # 500ms
        max_silence = int(SAMPLE_RATE * 0.8)  # 800ms

        # Audio queue for async processing
        audio_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def audio_callback(indata, _frames, _time_info, status):
            """Callback for sounddevice - runs in audio thread."""
            if status:
                console.log(f"[red]Audio status: {status}[/red]")
            # Put audio data into async queue
            audio_chunk = indata.copy().flatten()
            asyncio.run_coroutine_threadsafe(
                audio_queue.put(audio_chunk),
                loop
            )

        console.log("")
        console.log("[bold green]════════════════════════════════════[/bold green]")
        console.log("[bold green]   SẴN SÀNG NÓI CHUYỆN!            [/bold green]")
        console.log("[bold green]════════════════════════════════════[/bold green]")
        console.log("")
        console.log("[dim]- Nói tự nhiên, không cần nhấn nút[/dim]")
        console.log("[dim]- Có thể nói chen khi AI đang nói[/dim]")
        console.log("[dim]- Ctrl+C để thoát[/dim]")
        console.log("")
        console.log("[cyan]🎤 Listening...[/cyan]")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):

                while True:
                    # Get audio from queue asynchronously
                    audio_chunk = await audio_queue.get()

                    # Suppress when playing
                    if self.player.is_suppressing:
                        audio_chunk = audio_chunk * 0.1

                    # VAD
                    speech_detected = self.vad.is_speech(audio_chunk)

                    if speech_detected:
                        if not is_speaking:
                            is_speaking = True
                            audio_buffer = []
                            silence_samples = 0
                            console.log("[yellow]>>> Đang nghe...[/yellow]")

                        audio_buffer.append(audio_chunk)
                        silence_samples = 0

                    elif is_speaking:
                        audio_buffer.append(audio_chunk)
                        silence_samples += len(audio_chunk)

                        if silence_samples >= max_silence:
                            is_speaking = False
                            console.log("[yellow]<<< Đang xử lý...[/yellow]")

                            if sum(len(c) for c in audio_buffer) >= min_speech:
                                full_audio = np.concatenate(audio_buffer)
                                await self.send_audio(full_audio)

                            audio_buffer = []
                            silence_samples = 0
                            self.vad.reset()
                            console.log("[cyan]🎤 Listening...[/cyan]")

        except KeyboardInterrupt:
            console.log("\n[yellow]Đang thoát...[/yellow]")
        finally:
            receive_task.cancel()
            if self.ws:
                await self.ws.close()
            self.player.stop()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="S2S Client")
    parser.add_argument("--url", default=SERVER_URL, help="Server WebSocket URL")
    
    args = parser.parse_args()
    
    client = S2SClient(url=args.url)
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
