"""
Whisper ASR (Automatic Speech Recognition) Wrapper
Speech-to-text for English using OpenAI Whisper Large V3 Turbo
"""
import numpy as np
from typing import Optional, Union
import warnings
import os

# Disable all unnecessary logging and telemetry
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

from .config import SAMPLE_RATE, ASR_MAX_AUDIO_LENGTH, WHISPER_MODEL_NAME

class WhisperASR:
    """
    Wrapper for OpenAI Whisper ASR model using faster-whisper.

    Whisper Large V3 Turbo features:
    - State-of-the-art accuracy (WER ~7.4%)
    - 6x faster than Whisper Large V3
    - 809M parameters (optimized)
    - Excellent for English transcription
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Whisper ASR.

        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model = None
        self.sample_rate = SAMPLE_RATE
        self.model_name = WHISPER_MODEL_NAME

        self._load_model()

    def _load_model(self):
        """Load Whisper model using faster-whisper."""
        print(f"Loading Whisper ASR ({self.model_name})...")

        try:
            from faster_whisper import WhisperModel

            # Load model with optimizations
            # compute_type: int8 for speed, float16 for accuracy on GPU
            compute_type = "float16" if self.device == "cuda" else "int8"

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
                download_root="pretrained_models/whisper"
            )

            print(f"Whisper ASR loaded ({self.model_name})!")

        except ImportError as e:
            print(f"faster-whisper not installed or import error: {e}")
            print("Install with: pip install faster-whisper")
            raise

    def transcribe(self, audio: Union[np.ndarray, "torch.Tensor"], sr: int = None) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (float32, mono)
            sr: Sample rate (default: 16000 Hz)

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

        try:
            # Transcribe using faster-whisper
            # language="en" for English-only (faster)
            # beam_size=5 for better accuracy (default: 5)
            # vad_filter=True to filter out non-speech
            segments, info = self.model.transcribe(
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

            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            # Join and clean
            full_text = " ".join(text_parts).strip()
            return full_text

        except Exception as e:
            print(f"Whisper transcription error: {e}")
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

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)
        return self.transcribe(full_audio, sr)


# Alias for backward compatibility
ConformerASR = WhisperASR


# Test function
def test_asr():
    """Test Whisper ASR with sample audio."""
    print("Testing Whisper Large V3 Turbo ASR...")

    # Initialize
    import torch
    asr = WhisperASR(device="cuda" if torch.cuda.is_available() else "cpu")

    # Create test audio (3 seconds of sine wave)
    t = np.linspace(0, 3, int(SAMPLE_RATE * 3))
    test_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Transcribe
    text = asr.transcribe(test_audio)
    print(f"Transcription: '{text}'")
    print("Whisper ASR test complete!")


if __name__ == "__main__":
    test_asr()
