"""
Silero VAD (Voice Activity Detection) Wrapper
Phát hiện khi người dùng đang nói để kích hoạt ASR
"""
import torch
import numpy as np
from typing import Optional, Tuple, List
from collections import deque

from .config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_SPEECH_PAD_MS,
    SILERO_VAD_REPO
)


class SileroVAD:
    """
    Wrapper cho Silero VAD model.
    
    Silero VAD là model nhẹ (~2MB) và nhanh (<1ms/chunk).
    Dùng để phát hiện khi nào có tiếng nói trong audio stream.
    """
    
    def __init__(self, threshold: float = VAD_THRESHOLD):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Ngưỡng xác suất để coi là speech (0-1)
        """
        self.threshold = threshold
        self.model = None
        self.sample_rate = SAMPLE_RATE
        
        # State tracking
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_samples = 0
        
        # Timing in samples
        self.min_speech_samples = int(SAMPLE_RATE * VAD_MIN_SPEECH_DURATION_MS / 1000)
        self.min_silence_samples = int(SAMPLE_RATE * VAD_MIN_SILENCE_DURATION_MS / 1000)
        self.speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000)
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model từ torch.hub."""
        print("🔄 Loading Silero VAD model...")
        
        # Load model from torch hub
        self.model, utils = torch.hub.load(
            repo_or_dir=SILERO_VAD_REPO,
            model='silero_vad',
            force_reload=False,
            onnx=False  # Use PyTorch for simplicity
        )
        
        # Get helper functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils
        
        print("✅ Silero VAD loaded!")
    
    def reset_state(self):
        """Reset VAD state cho conversation mới."""
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_samples = 0
        self.model.reset_states()
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Xử lý một chunk audio và phát hiện speech.
        
        Args:
            audio_chunk: Audio data (float32, normalized -1 to 1)
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - bool: True nếu phát hiện kết thúc speech segment
                - Optional[np.ndarray]: Complete speech segment nếu detected, None otherwise
        """
        # Convert to tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        is_speech = speech_prob >= self.threshold
        
        if is_speech:
            # Currently speaking
            self.speech_buffer.append(audio_chunk)
            self.silence_samples = 0
            self.is_speaking = True
            
        else:
            # Silence
            if self.is_speaking:
                self.speech_buffer.append(audio_chunk)
                self.silence_samples += len(audio_chunk)
                
                # Check if silence is long enough to end speech
                if self.silence_samples >= self.min_silence_samples:
                    # Speech segment complete
                    if len(self.speech_buffer) > 0:
                        total_samples = sum(len(chunk) for chunk in self.speech_buffer)
                        
                        if total_samples >= self.min_speech_samples:
                            # Valid speech segment
                            complete_audio = np.concatenate(self.speech_buffer)
                            self.reset_state()
                            return True, complete_audio
                    
                    # Too short, discard
                    self.reset_state()
        
        return False, None
    
    def get_speech_segments(self, audio: np.ndarray) -> List[dict]:
        """
        Phát hiện tất cả speech segments trong audio file.
        Dùng cho batch processing, không phải streaming.
        
        Args:
            audio: Complete audio array
        
        Returns:
            List of speech segments with start/end timestamps
        """
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS
        )
        
        return timestamps


class VADIterator:
    """
    Iterator cho real-time VAD processing.
    Dùng trong WebSocket streaming.
    """
    
    def __init__(self, vad: SileroVAD, buffer_size: int = 512):
        self.vad = vad
        self.buffer_size = buffer_size
        self.audio_buffer = deque(maxlen=buffer_size * 100)  # ~3 seconds buffer
    
    def feed(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed audio chunk và nhận speech segment khi complete.
        
        Returns:
            Complete speech segment hoặc None
        """
        is_complete, segment = self.vad.process_chunk(audio_chunk)
        
        if is_complete:
            return segment
        return None
    
    def reset(self):
        """Reset iterator state."""
        self.vad.reset_state()
        self.audio_buffer.clear()


# Test function
def test_vad():
    """Test VAD với sample audio."""
    import torchaudio
    
    vad = SileroVAD()
    
    # Create test audio (sine wave + silence)
    duration = 3  # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Speech-like signal (0.5-2s) surrounded by silence
    audio = np.zeros_like(t)
    audio[int(0.5 * SAMPLE_RATE):int(2 * SAMPLE_RATE)] = 0.5 * np.sin(2 * np.pi * 440 * t[int(0.5 * SAMPLE_RATE):int(2 * SAMPLE_RATE)])
    
    # Process in chunks
    chunk_size = 512
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size].astype(np.float32)
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        is_complete, segment = vad.process_chunk(chunk)
        if is_complete:
            print(f"✅ Speech segment detected: {len(segment)} samples ({len(segment)/SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    test_vad()
