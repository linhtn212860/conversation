"""
Conformer ASR (Automatic Speech Recognition) Wrapper
Speech-to-text for English using NeMo Conformer-CTC
"""
import numpy as np
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore")

from .config import SAMPLE_RATE, ASR_MAX_AUDIO_LENGTH, CONFORMER_MODEL_NAME


class ConformerASR:
    """
    Wrapper for NeMo Conformer-CTC ASR model.

    Conformer combines:
    - Convolution for local patterns
    - Self-attention for global context
    - CTC decoding for streaming capability
    - Excellent accuracy and speed
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Conformer ASR.

        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model = None
        self.sample_rate = SAMPLE_RATE
        self.model_name = CONFORMER_MODEL_NAME

        self._load_model()

    def _load_model(self):
        """Load NeMo Conformer model."""
        print(f"🔄 Loading Conformer ASR ({self.model_name})...")

        try:
            import nemo.collections.asr as nemo_asr

            # Load pretrained model from NGC
            self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name=self.model_name
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()

            self.model.eval()

            print(f"✅ Conformer ASR loaded ({self.model_name})!")

        except ImportError:
            print("❌ NeMo not installed. Install with: pip install nemo_toolkit[asr]")
            raise

    def transcribe(self, audio: Union[np.ndarray, "torch.Tensor"], sr: int = None) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (float32, mono)
            sr: Sample rate

        Returns:
            Transcribed text
        """
        if sr is None:
            sr = self.sample_rate

        # Convert tensor to numpy if needed
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()

        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Truncate if too long
        max_samples = ASR_MAX_AUDIO_LENGTH * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # NeMo expects audio as list of numpy arrays
        # Transcribe with Conformer
        import torch
        with torch.no_grad():
            # Transcribe - model.transcribe returns list of Hypothesis objects
            transcriptions = self.model.transcribe(
                audio=[audio],
                batch_size=1
            )

            if transcriptions and len(transcriptions) > 0:
                # NeMo returns Hypothesis object, access .text attribute
                hypothesis = transcriptions[0]
                if hasattr(hypothesis, 'text'):
                    return hypothesis.text.strip()
                elif isinstance(hypothesis, str):
                    return hypothesis.strip()
                else:
                    # Fallback - convert to string
                    return str(hypothesis).strip()
            return ""

    def transcribe_streaming(self, audio_chunks: list, sr: int = None) -> str:
        """
        Transcribe from list of audio chunks.

        Args:
            audio_chunks: List of audio arrays
            sr: Sample rate

        Returns:
            Transcribed text
        """
        if len(audio_chunks) == 0:
            return ""

        full_audio = np.concatenate(audio_chunks)
        return self.transcribe(full_audio, sr)


# Alias for backward compatibility
WhisperASR = ConformerASR


# Test function
def test_asr():
    """Test ASR with sample audio."""
    print("Testing Conformer ASR...")

    # Initialize
    import torch
    asr = ConformerASR(device="cuda" if torch.cuda.is_available() else "cpu")

    # Create test audio (3 seconds of sine wave)
    t = np.linspace(0, 3, int(SAMPLE_RATE * 3))
    test_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Transcribe
    text = asr.transcribe(test_audio)
    print(f"Transcription: '{text}'")
    print("✅ Conformer ASR test complete!")


if __name__ == "__main__":
    test_asr()
