"""
TTS Wrapper - Kokoro-82M for English speech synthesis
Using Kokoro lightweight TTS model for fast and high-quality speech
"""
import os
import numpy as np
import tempfile
import re
from pathlib import Path
from typing import Optional
import torch

# Rich console
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def log(self, msg): print(msg.replace('[', '').replace(']', ''))
    console = Console()

from .config import TTS_MODEL_NAME, TTS_OUTPUT_SAMPLE_RATE, MODELS_DIR


class EnglishTTS:
    """
    English TTS using Kokoro-82M.
    Fast and lightweight neural speech synthesis.
    """

    def __init__(self, model_name: str = TTS_MODEL_NAME, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.sample_rate = 24000  # Kokoro uses 24kHz
        self.available = False

        self._load_model()

    def _load_model(self):
        """Load Kokoro ONNX model."""
        try:
            console.log(f"[cyan]Loading Kokoro TTS ONNX ({self.model_name})...[/cyan]")

            from kokoro_onnx import Kokoro

            # Use downloaded model files
            kokoro_dir = MODELS_DIR / "kokoro"
            model_path = str(kokoro_dir / "kokoro-v1.0.onnx")
            voices_path = str(kokoro_dir / "voices-v1.0.bin")

            self.pipeline = Kokoro(model_path, voices_path)
            self.available = True

            console.log(f"[green]Kokoro TTS ONNX loaded[/green]")

        except ImportError:
            console.log("[red]Kokoro ONNX not available. Install with: pip install kokoro-onnx[/red]")
            self.available = False
        except Exception as e:
            console.log(f"[red]Failed to load Kokoro: {e}[/red]")
            self.available = False

    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_path: Path to save WAV file

        Returns:
            True if successful
        """
        if not self.available:
            console.log("[red]Kokoro TTS not available[/red]")
            return False

        try:
            # Clean text - remove special characters
            cleaned = text.strip()

            # Skip empty input
            if not cleaned:
                return False

            cleaned = re.sub(r'[\u4e00-\u9fff]+', '', cleaned)  # Remove Chinese
            cleaned = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]+', '', cleaned)  # Remove non-Latin
            cleaned = cleaned.strip()

            if not cleaned:
                # Text was only special characters, skip silently
                return False

            console.log(f"[dim]Kokoro: Synthesizing '{cleaned[:50]}...'[/dim]")

            # Generate speech using Kokoro ONNX
            # voice='af_heart' is default American female voice
            audio, sample_rate = self.pipeline.create(
                text=cleaned,
                voice='af_heart',
                speed=1.0,
                lang='en-us'
            )

            if audio is None or len(audio) == 0:
                return False

            # Normalize audio to [-1, 1]
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            # Save as WAV
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)

            console.log(f"[dim]Kokoro: Saved to {output_path}[/dim]")
            return True

        except Exception as e:
            console.log(f"[red]Kokoro error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    def synthesize_to_array(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text directly to numpy array.

        Args:
            text: Text to synthesize

        Returns:
            Audio as numpy array (float32), or None if failed
        """
        if not self.available:
            return None

        try:
            # Clean text
            cleaned = text.strip()
            cleaned = re.sub(r'[\u4e00-\u9fff]+', '', cleaned)
            cleaned = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]+', '', cleaned)

            if not cleaned:
                return None

            # Generate speech using Kokoro ONNX
            audio, _ = self.pipeline.create(
                text=cleaned,
                voice='af_heart',
                speed=1.0,
                lang='en-us'
            )

            if audio is None or len(audio) == 0:
                return None

            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return audio

        except Exception as e:
            console.log(f"[red]Kokoro synthesis error: {e}[/red]")
            return None


# =============================================================================
# Test
# =============================================================================

def test_tts():
    """Test Kokoro TTS."""
    console.log("[bold]TEST KOKORO TTS[/bold]")

    tts = EnglishTTS()

    test_text = "Hello, I am an intelligent voice assistant. How can I help you today?"
    output_path = "/tmp/test_tts_kokoro.wav"

    success = tts.synthesize(test_text, output_path)

    if success:
        console.log(f"[green]TTS success! Output: {output_path}[/green]")
    else:
        console.log("[red]TTS failed[/red]")


if __name__ == "__main__":
    test_tts()
