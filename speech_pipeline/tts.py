"""
TTS Wrapper - VITS for English speech synthesis
Using VITS-based models for high-quality neural TTS
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

from .config import VITS_MODEL_NAME, TTS_OUTPUT_SAMPLE_RATE


class VITSTTS:
    """VITS-based TTS for English speech synthesis."""

    def __init__(self, model_name: str = VITS_MODEL_NAME, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.sample_rate = TTS_OUTPUT_SAMPLE_RATE
        self.available = False

        self._load_model()

    def _load_model(self):
        """Load VITS model."""
        try:
            console.log(f"[cyan]Loading VITS TTS ({self.model_name})...[/cyan]")

            from transformers import VitsModel, VitsTokenizer

            # Load model and tokenizer
            self.model = VitsModel.from_pretrained(self.model_name)
            self.processor = VitsTokenizer.from_pretrained(self.model_name)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
                self.device = "cpu"

            self.model.eval()
            self.available = True

            console.log(f"[green]✓ VITS TTS loaded: {self.model_name}[/green]")

        except ImportError:
            console.log("[red]✗ Transformers not available. Install with: pip install transformers[/red]")
            self.available = False
        except Exception as e:
            console.log(f"[red]✗ Failed to load VITS: {e}[/red]")
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
            console.log("[red]VITS TTS not available[/red]")
            return False

        try:
            # Clean text - remove special characters
            cleaned = text.strip()
            cleaned = re.sub(r'[\u4e00-\u9fff]+', '', cleaned)  # Remove Chinese
            cleaned = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]+', '', cleaned)  # Remove non-Latin

            if not cleaned:
                console.log("[yellow]VITS: Empty text after cleaning[/yellow]")
                return False

            console.log(f"[dim]VITS: Synthesizing '{cleaned[:50]}...'[/dim]")

            # Tokenize text
            inputs = self.processor(text=cleaned, return_tensors="pt")

            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                outputs = self.model(**inputs)
                audio = outputs.waveform.squeeze().cpu().numpy()

            # Normalize audio to [-1, 1]
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            # Save as WAV
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)

            console.log(f"[dim]VITS: Saved to {output_path}[/dim]")
            return True

        except Exception as e:
            console.log(f"[red]VITS error: {e}[/red]")
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

            # Tokenize
            inputs = self.processor(text=cleaned, return_tensors="pt")

            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model(**inputs)
                audio = outputs.waveform.squeeze().cpu().numpy()

            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return audio

        except Exception as e:
            console.log(f"[red]VITS synthesis error: {e}[/red]")
            return None


class EnglishTTS:
    """
    English TTS using VITS.
    High-quality neural speech synthesis.
    """

    def __init__(self, model_name: str = VITS_MODEL_NAME, device: str = "cuda"):
        console.log("[cyan]Initializing English TTS...[/cyan]")
        self.vits = VITSTTS(model_name=model_name, device=device)

        if not self.vits.available:
            console.log("[red]✗ TTS not available![/red]")

    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to speech using VITS.

        Args:
            text: Text to synthesize (English)
            output_path: Path to save audio file

        Returns:
            True if successful
        """
        return self.vits.synthesize(text, output_path)

    def synthesize_to_array(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text directly to numpy array.

        Args:
            text: Text to synthesize

        Returns:
            Audio array or None
        """
        return self.vits.synthesize_to_array(text)


# Alias for backward compatibility
CombinedTTS = EnglishTTS


# =============================================================================
# Test
# =============================================================================

def test_tts():
    """Test TTS."""
    console.log("[bold]TEST VITS TTS[/bold]")

    tts = EnglishTTS()

    test_text = "Hello, I am an intelligent voice assistant. How can I help you today?"
    output_path = "/tmp/test_tts_vits.wav"

    success = tts.synthesize(test_text, output_path)

    if success:
        console.log(f"[green]✓ TTS success! Output: {output_path}[/green]")
    else:
        console.log("[red]✗ TTS failed[/red]")


if __name__ == "__main__":
    test_tts()
