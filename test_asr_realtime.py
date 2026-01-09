 #!/usr/bin/env python3
"""
Real-time ASR Testing Tool with Microphone
Test and compare different ASR models with your real voice
"""
import sys
import time
import queue
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    class Console:
        def log(self, msg): print(msg.replace('[', '').replace(']', ''))
        def print(self, msg): print(str(msg))
    console = Console()


# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SPEECH_THRESHOLD = 0.5


# =============================================================================
# Silero VAD
# =============================================================================

class SileroVAD:
    """Silero VAD for speech detection."""

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
        """Check if audio chunk contains speech."""
        if len(audio_chunk) == 0:
            return False

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        audio_tensor = torch.from_numpy(audio_chunk)

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        return speech_prob > self.threshold


# =============================================================================
# ASR Test Runner
# =============================================================================

class ASRTestRunner:
    """Real-time ASR testing tool."""

    def __init__(self, model_name: str = "whisper", device: str = "cuda", streaming: bool = True):
        """
        Initialize ASR test runner.

        Args:
            model_name: "whisper", "speechbrain", or "compare"
            device: "cuda" or "cpu"
            streaming: Enable streaming output (print words as they appear)
        """
        self.model_name = model_name
        self.device = device
        self.streaming = streaming
        self.asr_models = {}
        self.results = []

        console.log("")
        console.print(Panel.fit(
            "[bold cyan]ASR Real-time Testing Tool[/bold cyan]\n"
            f"Model: {model_name} | Device: {device}\n"
            f"Streaming: {'✅ Enabled' if streaming else '❌ Disabled'}",
            border_style="cyan"
        ))
        console.log("")

        # Load VAD
        self.vad = SileroVAD()

        # Load ASR models
        self._load_models()

    def _load_models(self):
        """Load selected ASR models."""
        if self.model_name in ["whisper", "compare"]:
            console.log("[cyan]Loading Whisper Large V3 Turbo...[/cyan]")
            from speech_pipeline.asr import WhisperASR
            self.asr_models["whisper"] = WhisperASR(device=self.device)
            console.log("[green]✓ Whisper loaded[/green]")

        if self.model_name in ["speechbrain", "compare"]:
            console.log("[cyan]Loading SpeechBrain Conformer...[/cyan]")
            try:
                # Apply torchaudio patch
                import torchaudio
                if not hasattr(torchaudio, 'list_audio_backends'):
                    torchaudio.list_audio_backends = lambda: ["soundfile"]

                from speechbrain.inference.ASR import EncoderDecoderASR

                model = EncoderDecoderASR.from_hparams(
                    source="speechbrain/asr-conformer-transformerlm-librispeech",
                    savedir="pretrained_models/speechbrain_asr-conformer-transformerlm-librispeech",
                    run_opts={"device": self.device}
                )
                self.asr_models["speechbrain"] = model
                console.log("[green]✓ SpeechBrain loaded[/green]")
            except Exception as e:
                console.log(f"[yellow]⚠ SpeechBrain load failed: {e}[/yellow]")

    def transcribe_whisper(self, audio: np.ndarray, streaming: bool = True) -> tuple:
        """Transcribe with Whisper (streaming)."""
        import sys
        start = time.time()

        if streaming:
            # Stream segments as they're generated
            full_text_parts = []

            # Print header with rich
            console.print("[green]Whisper:[/green] ", end="")

            try:
                segments, info = self.asr_models["whisper"].model.transcribe(
                    audio,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=100
                    )
                )

                # Stream each segment using sys.stdout for flush support
                for segment in segments:
                    text = segment.text
                    # Use sys.stdout.write for true streaming
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    full_text_parts.append(text)

                sys.stdout.write("\n")  # New line
                full_text = " ".join(full_text_parts).strip()
            except Exception as e:
                console.log(f"[red]Error: {e}[/red]")
                full_text = ""
        else:
            # Non-streaming (original)
            full_text = self.asr_models["whisper"].transcribe(audio)

        elapsed = time.time() - start
        return full_text, elapsed

    def transcribe_speechbrain(self, audio: np.ndarray) -> tuple:
        """Transcribe with SpeechBrain."""
        if "speechbrain" not in self.asr_models:
            return "", 0

        start = time.time()
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        predictions, _ = self.asr_models["speechbrain"].transcribe_batch(
            audio_tensor,
            torch.tensor([1.0])
        )
        text = predictions[0] if predictions else ""
        elapsed = time.time() - start
        return text, elapsed

    def process_audio(self, audio: np.ndarray):
        """Process audio with all loaded models."""
        audio_duration = len(audio) / SAMPLE_RATE

        console.log(f"\n[yellow]📊 Audio: {audio_duration:.2f}s[/yellow]")

        result = {
            "audio_duration": audio_duration,
            "timestamp": time.time()
        }

        # Test each model
        for model_name, model in self.asr_models.items():
            if model_name == "whisper":
                text, elapsed = self.transcribe_whisper(audio, streaming=self.streaming)
            elif model_name == "speechbrain":
                console.log("[green]SpeechBrain:[/green] ", end="")
                text, elapsed = self.transcribe_speechbrain(audio)
                console.log(f"[green]{text}[/green]")
            else:
                continue

            rtf = elapsed / audio_duration if audio_duration > 0 else 0

            result[model_name] = {
                "text": text,
                "time": elapsed,
                "rtf": rtf
            }

            # Display metrics
            console.log(f"[dim]⏱️  {elapsed:.3f}s | RTF: {rtf:.3f}x | {1/rtf if rtf > 0 else 0:.1f}x real-time[/dim]")

        self.results.append(result)
        console.log("[cyan]🎤 Ready for next...[/cyan]")

    def show_summary(self):
        """Show test summary."""
        if len(self.results) == 0:
            console.log("[yellow]No results to show[/yellow]")
            return

        console.log("\n" + "="*80)
        console.log("[bold cyan]TEST SUMMARY[/bold cyan]")
        console.log("="*80 + "\n")

        # Create table
        table = Table(title="ASR Performance Comparison")
        table.add_column("Test #", style="cyan")
        table.add_column("Duration", justify="right")

        for model_name in self.asr_models.keys():
            table.add_column(f"{model_name}\nText", style="green")
            table.add_column(f"{model_name}\nTime", justify="right")
            table.add_column(f"{model_name}\nRTF", justify="right")

        # Add rows
        for i, result in enumerate(self.results, 1):
            row = [
                str(i),
                f"{result['audio_duration']:.2f}s"
            ]

            for model_name in self.asr_models.keys():
                if model_name in result:
                    r = result[model_name]
                    text = r['text'][:30] + "..." if len(r['text']) > 30 else r['text']
                    row.extend([
                        text,
                        f"{r['time']:.3f}s",
                        f"{r['rtf']:.3f}x"
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A"])

            table.add_row(*row)

        console.print(table)

        # Calculate averages
        if len(self.results) > 0:
            console.log("\n[bold cyan]AVERAGES:[/bold cyan]")
            for model_name in self.asr_models.keys():
                times = [r[model_name]['time'] for r in self.results if model_name in r]
                rtfs = [r[model_name]['rtf'] for r in self.results if model_name in r]

                if times:
                    avg_time = sum(times) / len(times)
                    avg_rtf = sum(rtfs) / len(rtfs)
                    console.log(f"  {model_name}: {avg_time:.3f}s | RTF: {avg_rtf:.3f}x")

    def run(self):
        """Run real-time microphone test."""
        console.log("")
        console.log("[bold green]" + "="*60 + "[/bold green]")
        console.log("[bold green]   READY TO TEST![/bold green]")
        console.log("[bold green]" + "="*60 + "[/bold green]")
        console.log("")
        console.log("[dim]- Speak naturally (VAD will detect speech)[/dim]")
        console.log("[dim]- System will process when you stop speaking[/dim]")
        console.log("[dim]- Press Ctrl+C to see summary and exit[/dim]")
        console.log("")
        console.log("[cyan]🎤 Listening...[/cyan]")

        # Audio capture state
        audio_buffer = []
        is_speaking = False
        silence_samples = 0
        min_speech = int(SAMPLE_RATE * 0.5)  # 500ms minimum
        max_silence = int(SAMPLE_RATE * 0.8)  # 800ms silence to stop

        # Queue for audio chunks
        audio_queue = queue.Queue()

        def audio_callback(indata, _frames, _time_info, status):
            """Audio input callback."""
            if status:
                console.log(f"[red]Audio status: {status}[/red]")
            audio_chunk = indata.copy().flatten()
            audio_queue.put(audio_chunk)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):
                while True:
                    # Get audio chunk
                    audio_chunk = audio_queue.get()

                    # VAD check
                    speech_detected = self.vad.is_speech(audio_chunk)

                    if speech_detected:
                        if not is_speaking:
                            is_speaking = True
                            audio_buffer = []
                            silence_samples = 0
                            console.log("[yellow]>>> Speaking detected...[/yellow]")

                        audio_buffer.append(audio_chunk)
                        silence_samples = 0

                    elif is_speaking:
                        audio_buffer.append(audio_chunk)
                        silence_samples += len(audio_chunk)

                        if silence_samples >= max_silence:
                            is_speaking = False

                            if sum(len(c) for c in audio_buffer) >= min_speech:
                                full_audio = np.concatenate(audio_buffer)
                                self.process_audio(full_audio)
                            else:
                                console.log("[dim]Speech too short, ignored[/dim]")

                            audio_buffer = []
                            silence_samples = 0
                            self.vad.reset()
                            console.log("[cyan]🎤 Listening...[/cyan]")

        except KeyboardInterrupt:
            console.log("\n[yellow]Stopping...[/yellow]")
        finally:
            self.show_summary()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASR Testing Tool")
    parser.add_argument(
        "--model",
        type=str,
        default="whisper",
        choices=["whisper", "speechbrain", "compare"],
        help="ASR model to test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Enable streaming output (print words as they appear)"
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming output (print full result at once)"
    )

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        console.log("[yellow]⚠ CUDA not available, falling back to CPU[/yellow]")
        args.device = "cpu"

    runner = ASRTestRunner(model_name=args.model, device=args.device, streaming=args.streaming)
    runner.run()


if __name__ == "__main__":
    main()
