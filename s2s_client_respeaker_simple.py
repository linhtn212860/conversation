#!/usr/bin/env python3
"""
Speech-to-Speech Client with ReSpeaker (Simple Version)

Uses parec (PulseAudio/PipeWire) to capture audio from ReSpeaker.
Configured with 5.1 surround profile to access all 6 channels.

Echo cancellation:
- ReSpeaker hardware AEC (channel 0 = front-left)
- Playback suppression

Usage:
    python s2s_client_respeaker_simple.py [--url ws://localhost:8765/ws]
"""

import asyncio
import websockets
import numpy as np
import json
import base64
import subprocess
import struct
import threading
import queue
from rich.console import Console

console = Console()

# =============================================================================
# CONFIG
# =============================================================================
SAMPLE_RATE = 16000
DEVICE = "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-51"
CHANNELS = 6
PROCESSED_CHANNEL = 0  # Channel with hardware AEC (front-left in 5.1 setup)
CHUNK_SIZE = 1024

# Gain
OUTPUT_GAIN = 2.0
MIC_GAIN = 1.5
SUPPRESSION_GAIN = 0.1

# =============================================================================
# AUDIO PLAYER
# =============================================================================

class SimplePlayer:
    """Simple audio player using aplay."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.is_playing = False
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        console.log(f"[green]✓ Audio player started[/green]")

    def _playback_loop(self):
        """Playback loop using aplay subprocess."""
        proc = None

        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
                if data is None:
                    break

                with self._lock:
                    self.is_playing = True

                # Apply gain
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples = samples * OUTPUT_GAIN
                samples = np.clip(samples, -32768, 32767)
                amplified = samples.astype(np.int16).tobytes()

                # Play using aplay
                proc = subprocess.Popen(
                    ['aplay', '-f', 'S16_LE', '-r', str(SAMPLE_RATE), '-c', '1', '-q'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc.stdin.write(amplified)
                proc.stdin.close()
                proc.wait()

            except queue.Empty:
                with self._lock:
                    self.is_playing = False
                continue
            except Exception as e:
                console.log(f"[red]Playback error: {e}[/red]")
                with self._lock:
                    self.is_playing = False

    @property
    def is_suppressing(self) -> bool:
        with self._lock:
            return self.is_playing

    def play(self, data: bytes):
        self.audio_queue.put(data)

    def clear(self):
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

    def stop(self):
        self.running = False
        self.audio_queue.put(None)
        if self.thread:
            self.thread.join(timeout=1)


# =============================================================================
# RESPEAKER CAPTURE
# =============================================================================

class ReSpeakerCapture:
    """Capture audio from ReSpeaker using parec (PipeWire)."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.proc = None

    def start(self):
        """Start parec subprocess to capture from PipeWire."""
        # Use parec to capture from PipeWire/PulseAudio
        cmd = [
            'parec',
            '--device', self.device,
            '--rate', str(SAMPLE_RATE),
            '--channels', str(CHANNELS),
            '--format', 's16le',
            '--raw',
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=CHUNK_SIZE * CHANNELS * 2
        )

        console.log("[green]✓ ReSpeaker capture started via PipeWire[/green]")

    def read_chunk(self) -> np.ndarray:
        """Read one chunk and extract channel 0."""
        if not self.proc:
            return None

        try:
            # Read interleaved data
            bytes_to_read = CHUNK_SIZE * CHANNELS * 2  # 2 bytes per sample
            data = self.proc.stdout.read(bytes_to_read)

            if not data or len(data) < bytes_to_read:
                console.log(f"[red]Incomplete read: got {len(data) if data else 0}/{bytes_to_read} bytes[/red]")
                return None

            # Convert to numpy
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples.reshape(-1, CHANNELS)

            # Extract channel 0 (hardware AEC)
            mono = samples[:, PROCESSED_CHANNEL]

            return mono
        except Exception as e:
            console.log(f"[red]Read error: {e}[/red]")
            return None

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


# =============================================================================
# CLIENT
# =============================================================================

class ReSpeakerClient:
    """Simple S2S client for ReSpeaker."""

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.is_connected = False
        self.player = SimplePlayer()
        self.capture = ReSpeakerCapture()

    async def connect(self) -> bool:
        try:
            console.log(f"[cyan]Connecting to {self.url}...[/cyan]")
            # Use longer timeout for initial connection
            self.ws = await asyncio.wait_for(
                websockets.connect(self.url, max_size=10*1024*1024),
                timeout=30.0
            )
            self.is_connected = True

            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(msg)
            console.log(f"[green]✓ Connected: {data.get('message', '')}[/green]")
            return True

        except asyncio.TimeoutError:
            console.log(f"[red]Connection timeout. Is server running?[/red]")
            return False
        except Exception as e:
            console.log(f"[red]Connection failed: {e}[/red]")
            return False

    async def send_audio(self, audio: np.ndarray):
        if not self.is_connected or not self.ws:
            return

        try:
            audio_bytes = audio.astype(np.int16).tobytes()
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')

            await self.ws.send(json.dumps({
                "type": "audio",
                "data": b64_audio
            }))
        except Exception as e:
            console.log(f"[red]Send error: {e}[/red]")

    async def receive_loop(self):
        try:
            while self.is_connected:
                msg = await self.ws.recv()
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "transcription":
                    text = data.get("text", "")
                    console.log(f"[yellow]🎤 You: {text}[/yellow]")

                elif msg_type == "response":
                    text = data.get("text", "")
                    console.log(f"[cyan]💬 AI: {text}[/cyan]")

                elif msg_type == "audio":
                    self.player.clear()
                    audio_b64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    self.player.play(audio_bytes)

                elif msg_type == "error":
                    console.log(f"[red]Error: {data.get('message')}[/red]")

        except websockets.exceptions.ConnectionClosed:
            console.log("[yellow]Connection closed[/yellow]")
            self.is_connected = False
        except Exception as e:
            console.log(f"[red]Receive error: {e}[/red]")
            self.is_connected = False

    async def run(self):
        # Connect
        if not await self.connect():
            return

        # Start components
        self.player.start()
        self.capture.start()

        # Start receive loop
        receive_task = asyncio.create_task(self.receive_loop())

        # VAD state
        audio_buffer = []
        is_speaking = False
        silence_samples = 0
        min_speech = int(SAMPLE_RATE * 0.5)
        max_silence = int(SAMPLE_RATE * 0.8)

        console.log("")
        console.log("[bold green]════════════════════════════════════[/bold green]")
        console.log("[bold green]   READY TO TALK!                  [/bold green]")
        console.log("[bold green]════════════════════════════════════[/bold green]")
        console.log("")
        console.log("[dim]- Speak naturally into ReSpeaker[/dim]")
        console.log("[dim]- Hardware AEC enabled[/dim]")
        console.log("[dim]- Playback suppression active[/dim]")
        console.log("[dim]- Ctrl+C to exit[/dim]")
        console.log("")
        console.log("[cyan]🎤 Listening...[/cyan]")

        try:
            chunk_count = 0
            while self.is_connected:
                # Read audio chunk
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.capture.read_chunk
                )

                if chunk is None:
                    console.log("[red]Got None chunk, stopping...[/red]")
                    break

                chunk_count += 1
                if chunk_count % 100 == 0:
                    console.log(f"[dim]Processed {chunk_count} chunks[/dim]")

                # Convert to float
                audio_float = chunk.astype(np.float32) / 32768.0

                # Apply suppression or gain
                if self.player.is_suppressing:
                    audio_float = audio_float * SUPPRESSION_GAIN
                else:
                    audio_float = audio_float * MIC_GAIN
                    audio_float = np.clip(audio_float, -1.0, 1.0)

                # Simple VAD
                energy = np.abs(audio_float).mean()
                threshold = 0.05 if self.player.is_suppressing else 0.02
                speech_detected = energy > threshold

                if speech_detected:
                    if not is_speaking:
                        is_speaking = True
                        audio_buffer = []
                        silence_samples = 0
                        if not self.player.is_suppressing:
                            console.log("[yellow]>>> Listening...[/yellow]")

                    audio_buffer.append(audio_float)
                    silence_samples = 0

                elif is_speaking:
                    audio_buffer.append(audio_float)
                    silence_samples += len(audio_float)

                    if silence_samples >= max_silence:
                        is_speaking = False

                        if sum(len(c) for c in audio_buffer) >= min_speech:
                            if not self.player.is_suppressing:
                                console.log("[yellow]<<< Processing...[/yellow]")
                                full_audio = np.concatenate(audio_buffer)
                                audio_int16 = (full_audio * 32767).astype(np.int16)
                                await self.send_audio(audio_int16)
                            else:
                                console.log("[dim]<<< Ignored (playback)[/dim]")

                        audio_buffer = []
                        silence_samples = 0
                        console.log("[cyan]🎤 Listening...[/cyan]")

                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            console.log("\n[yellow]Exiting...[/yellow]")
        finally:
            receive_task.cancel()
            self.capture.stop()
            if self.ws:
                await self.ws.close()
            self.player.stop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    console.log("")
    console.log("[bold]╔══════════════════════════════════════════╗[/bold]")
    console.log("[bold]║  ReSpeaker S2S Client (Simple)          ║[/bold]")
    console.log("[bold]╚══════════════════════════════════════════╝[/bold]")
    console.log("")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8765/ws")
    args = parser.parse_args()

    console.log(f"[cyan]Device: {DEVICE}[/cyan]")
    console.log(f"[cyan]Channels: {CHANNELS}[/cyan]")
    console.log(f"[cyan]Using channel: {PROCESSED_CHANNEL} (hardware AEC)[/cyan]")
    console.log("")

    client = ReSpeakerClient(url=args.url)
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
