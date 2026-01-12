"""
Silero VAD (Voice Activity Detection) Wrapper
Phát hiện khi người dùng đang nói để kích hoạt ASR
"""
import torch
import numpy as np    
from typing import Optional, Tuple, List
from collections import deque # Deque dùng cho buffer (hàng đợi) hiệu suất cao

from .config import (
    SAMPLE_RATE,  # Tần số mẫu (16kHz)
    VAD_THRESHOLD, # ngưỡng xác suất để coi là tiếng nói 0.5
    VAD_MIN_SPEECH_DURATION_MS, # độ dài tối thiểu của một đoạn tiếng nói (ms) (250)
    VAD_MIN_SILENCE_DURATION_MS, # độ dài im lặng để coi là ngắt câu (300)
    VAD_SPEECH_PAD_MS, # khoảng đệm trước/sau câu nói (100)
    SILERO_VAD_REPO # dường dẫn repo chứa model Silero VAD
)


class SileroVAD:   
    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.threshold = threshold # ngưỡng xác suất ( >0.5 là có tiếng nói)
        self.model = None  # biến chứa model Silero VAD (load sau)
        self.sample_rate = SAMPLE_RATE # tần số mẫu (16kHz)
        
        # State tracking
        self.is_speaking = False # trạng thái hiện tại (đang nói hay im lặng)
        self.speech_buffer = [] # buffer chứa các chunk audio (để tạo thành đoạn audio hoàn chỉnh)
        self.silence_samples = 0 # số lượng mẫu im lặng
        
        # Timing in samples - quy đổi thời gian (ms) sang số lượng mẫu (samples)
        self.min_speech_samples = int(SAMPLE_RATE * VAD_MIN_SPEECH_DURATION_MS / 1000) # độ dài tối thiểu của một đoạn tiếng nói (ms) 250ms -> 4000 samples
        self.min_silence_samples = int(SAMPLE_RATE * VAD_MIN_SILENCE_DURATION_MS / 1000) # độ dài im lặng để coi là ngắt câu (300ms -> 4800 samples)
        self.speech_pad_samples = int(SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000) # khoảng đệm trước/sau câu nói (100ms -> 1600 samples)
        
        self._load_model() # hàm load model Silero VAD
    
    def _load_model(self):
        """Load Silero VAD model từ torch.hub."""
        print("🔄 Loading Silero VAD model...")
        
        # Load model from torch hub
        self.model, utils = torch.hub.load( # load model Silero VAD từ torch.hub
            repo_or_dir=SILERO_VAD_REPO, # repo chứa model Silero VAD
            model='silero_vad', # tên model Silero VAD
            force_reload=False, # không reload model nếu đã load trước đó
            onnx=False  # không sử dụng ONNX, sử dụng PyTorch
        )
        
        # Get helper functions
        (
            self.get_speech_timestamps, # hàm lấy timestamps của các segment speech
            self.save_audio, # hàm lưu audio
            self.read_audio, # hàm đọc audio
            self.VADIterator, # iterator để duyệt qua các chunk audio
            self.collect_chunks # hàm thu thập các chunk audio
        ) = utils
        
        print("✅ Silero VAD loaded!")
    
    def reset_state(self): #
        """Reset VAD state cho conversation mới."""
        self.is_speaking = False # reset về im lặng
        self.speech_buffer = [] #reset buffer chứa các chunk audio về rỗng
        self.silence_samples = 0 # reset số lượng mẫu im lặng
        self.model.reset_states() # reset state của model Silero VAD
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Xử lý một chunk audio và phát hiện speech.
        Xử lý từng chút một, dùng cho streaming
        
        Args:
            audio_chunk: Audio data (float32, normalized -1 to 1)
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - bool: True nếu phát hiện kết thúc speech segment
                - Optional[np.ndarray]: Complete speech segment nếu detected, None otherwise
        """
        # Convert to tensor đảm bảo đầu vào là tensor
        if isinstance(audio_chunk, np.ndarray): 
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk
        
        # Get speech probability
        #chạy model Silero VAD để lấy xác suất là tiếng nói của audio_chunk (từ 0 đến 1)
        speech_prob = self.model(audio_tensor, self.sample_rate).item() 
        
        is_speech = speech_prob >= self.threshold # nếu xác suất >= ngưỡng xác suất thì coi là tiếng nói
        
        if is_speech: # nếu là tiếng nói
            # Currently speaking
            self.speech_buffer.append(audio_chunk) # thêm audio_chunk vào buffer
            self.silence_samples = 0 # reset số lượng mẫu im lặng
            self.is_speaking = True # set trạng thái là đang nói
            
        else:
            # Silence, im lặng khi đang nói dở (ngắt câu) thì self.is_speaking = True vì các chunk trước đó đã phát hiện speech.
            if self.is_speaking: 
                self.speech_buffer.append(audio_chunk) # vẫn phải lưu đoạn im lặng này vào buffer để câu nói liền mạch, không bị ngắt đột ngột
                self.silence_samples += len(audio_chunk) # đếm thời gian im lặng
                
                # Check if silence is long enough to end speech
                if self.silence_samples >= self.min_silence_samples:
                    # Speech segment complete
                    if len(self.speech_buffer) > 0:
                        total_samples = sum(len(chunk) for chunk in self.speech_buffer) # cộng tất cả các gói tin chunk trong speech_buffer lại xem tổng cộng đoạn hội thoại này dài bao nhiêu
                        
                        if total_samples >= self.min_speech_samples: # so sánh tổng độ dài với min_speech_samples
                            # Valid speech segment
                            complete_audio = np.concatenate(self.speech_buffer) # nối tất cả các gói tin chunk trong speech_buffer lại thành một array
                            self.reset_state() # reset lại các biến đếm về 0 để bắt câu nói tiếp theo
                            return True, complete_audio #trả về True, basoa hiệu đã xong câu, kèm theo file âm thanh để gửi đi nhân dạng ASR
                    
                    # Too short, discard
                    self.reset_state() #nếu đoạn âm thanh quá ngắn, reset lại các biến đếm để vứt toàn bộ buffer đi, quay lại trạng thái chờ im lặng
        
        return False, None
    
    def get_speech_segments(self, audio: np.ndarray) -> List[dict]:
        """
        Phát hiện tất cả speech segments trong audio file.
        Dùng cho batch processing, không phải streaming.
        Nghĩa là ném 1 file ghi âm dài, nó sẽ trả về danh sách các vị trí bắt đầu và kết thúc của tất cả các câu nói trong đó
        Dùng để cắt nhỏ file âm thanh thành các câu thoại để train model, xử lí hội thoại offline
        
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
    
    def __init__(self, vad: SileroVAD, buffer_size: int = 512): # buffer_size: kích thước bộ đệm, số lượng samples, tương đương 32ms âm thanh
        self.vad = vad
        self.buffer_size = buffer_size
        self.audio_buffer = deque(maxlen=buffer_size * 100)  # ~3 seconds buffer, giới hạn bộ nhớ tối đa cho hàng đợi dự phòng.
                                                            # nghĩa là hàng đợi này chứa được tối đa 100 gói tin chuẩn
                                                            # nếu vượt quá 100 gói tin, hàng đợi sẽ tự động xóa gói tin đầu tiên
                                                            # để đảm bảo bộ nhớ không bị tràn
    def feed(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed audio chunk và nhận speech segment khi complete.
        
        Returns:
            Complete speech segment hoặc None
        """
        is_complete, segment = self.vad.process_chunk(audio_chunk) # ủy quyền việc phân tích khó cho self.vad.process_chunk
        
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
