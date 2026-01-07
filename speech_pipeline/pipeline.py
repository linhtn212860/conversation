"""
Speech-to-Speech Pipeline
Connects VAD -> ASR -> LLM -> TTS into a complete pipeline
"""
import torch
import numpy as np
import os
from typing import Optional, Generator, Callable
from dataclasses import dataclass
from enum import Enum
import time

from .vad import SileroVAD, VADIterator
from .asr import WhisperASR
from .llm import QwenLLM
from .tts import EnglishTTS
from .config import SAMPLE_RATE, TTS_OUTPUT_SAMPLE_RATE


class PipelineState(Enum):
    """Trạng thái của pipeline."""
    IDLE = "idle"                    # Đang đợi input
    LISTENING = "listening"          # Đang nghe (VAD active)
    PROCESSING = "processing"        # Đang xử lý (ASR/LLM/TTS)
    SPEAKING = "speaking"            # Đang phát audio output


@dataclass
class PipelineResult:
    """Kết quả từ pipeline processing."""
    state: PipelineState
    transcription: Optional[str] = None
    response_text: Optional[str] = None
    audio_chunk: Optional[np.ndarray] = None
    processing_time_ms: float = 0


class SpeechToSpeechPipeline:
    """
    End-to-end Speech-to-Speech Pipeline.
    
    Flow:
    Audio Input -> VAD -> ASR -> LLM -> TTS -> Audio Output
    
    Tối ưu cho low latency:
    - VAD streaming: Phát hiện speech realtime
    - LLM streaming: Response từng token
    - TTS streaming: Audio từng sentence
    """
    
    def __init__(
        self,
        device: str = "cuda",
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            device: "cuda" hoặc "cpu"
            on_state_change: Callback khi state thay đổi
            on_transcription: Callback khi có transcription
            on_response: Callback khi có LLM response
            on_audio: Callback khi có audio output
        """
        self.device = device
        self.state = PipelineState.IDLE
        
        # Callbacks
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_audio = on_audio
        
        # Components (lazy loading)
        self._vad = None
        self._asr = None
        self._llm = None
        self._tts = None
        
        # VAD iterator for streaming
        self._vad_iterator = None
    
    @property
    def vad(self) -> SileroVAD:
        """Lazy load VAD."""
        if self._vad is None:
            self._vad = SileroVAD()
            self._vad_iterator = VADIterator(self._vad)
        return self._vad
    
    @property
    def asr(self) -> WhisperASR:
        """Lazy load ASR."""
        if self._asr is None:
            self._asr = WhisperASR(device=self.device)
        return self._asr
    
    @property
    def llm(self) -> QwenLLM:
        """Lazy load LLM."""
        if self._llm is None:
            self._llm = QwenLLM(device=self.device)
        return self._llm
    
    @property
    def tts(self) -> EnglishTTS:
        """Lazy load TTS."""
        if self._tts is None:
            self._tts = EnglishTTS()
        return self._tts
    
    def load_all_models(self):
        """Pre-load tất cả models."""
        print("=" * 50)
        print("Loading all models...")
        print("=" * 50)
        
        _ = self.vad
        _ = self.asr
        _ = self.llm
        _ = self.tts
        
        print("=" * 50)
        print("✅ All models loaded!")
        print("=" * 50)
    
    def _set_state(self, state: PipelineState):
        """Set state và trigger callback."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(state)
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[PipelineResult]:
        """
        Process một audio chunk từ microphone.
        Dùng cho streaming mode.
        
        Args:
            audio_chunk: Audio data (float32, 16kHz)
        
        Returns:
            PipelineResult nếu có output, None otherwise
        """
        self._set_state(PipelineState.LISTENING)
        
        # Feed to VAD
        if self._vad_iterator is None:
            _ = self.vad  # Initialize
        
        speech_segment = self._vad_iterator.feed(audio_chunk)
        
        if speech_segment is not None:
            # Speech segment complete - process it
            return self.process_speech(speech_segment)
        
        return None
    
    def process_speech(self, audio: np.ndarray) -> PipelineResult:
        """
        Process một speech segment hoàn chỉnh.
        
        Flow: Audio -> ASR -> LLM -> TTS
        
        Args:
            audio: Complete speech audio
        
        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()
        
        self._set_state(PipelineState.PROCESSING)
        
        # Step 1: ASR - Speech to Text
        transcription = self.asr.transcribe(audio)
        
        if self.on_transcription:
            self.on_transcription(transcription)
        
        if not transcription.strip():
            return PipelineResult(
                state=PipelineState.IDLE,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Step 2: LLM - Generate response
        response_text = self.llm.generate(transcription)
        
        if self.on_response:
            self.on_response(response_text)
        
        # Step 3: TTS - Text to Speech
        self._set_state(PipelineState.SPEAKING)

        # TTS needs output path - create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        success = self.tts.synthesize(response_text, temp_path)

        if success:
            import soundfile as sf
            audio_output, _ = sf.read(temp_path)
            os.unlink(temp_path)
        else:
            audio_output = np.array([], dtype=np.float32)
        
        if self.on_audio:
            self.on_audio(audio_output)
        
        self._set_state(PipelineState.IDLE)
        
        return PipelineResult(
            state=PipelineState.IDLE,
            transcription=transcription,
            response_text=response_text,
            audio_chunk=audio_output,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def process_speech_streaming(self, audio: np.ndarray) -> Generator[PipelineResult, None, None]:
        """
        Process speech với streaming output.
        Yield từng audio chunk khi có.
        
        Args:
            audio: Complete speech audio
        
        Yields:
            PipelineResult với từng audio chunk
        """
        start_time = time.time()
        
        self._set_state(PipelineState.PROCESSING)
        
        # Step 1: ASR
        transcription = self.asr.transcribe(audio)
        
        if self.on_transcription:
            self.on_transcription(transcription)
        
        yield PipelineResult(
            state=PipelineState.PROCESSING,
            transcription=transcription,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        if not transcription.strip():
            self._set_state(PipelineState.IDLE)
            return
        
        # Step 2 & 3: LLM streaming -> TTS streaming
        self._set_state(PipelineState.SPEAKING)
        
        # Collect response text in chunks for TTS
        response_buffer = ""
        full_response = ""
        
        for token in self.llm.generate_streaming(transcription):
            response_buffer += token
            full_response += token
            
            # Check if we have a complete sentence to synthesize
            if self._has_complete_sentence(response_buffer):
                sentence, response_buffer = self._extract_sentence(response_buffer)

                # TTS for this sentence
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name

                success = self.tts.synthesize(sentence, temp_path)

                if success:
                    import soundfile as sf
                    audio_chunk, _ = sf.read(temp_path)
                    os.unlink(temp_path)
                else:
                    audio_chunk = np.array([], dtype=np.float32)
                
                if self.on_audio and len(audio_chunk) > 0:
                    self.on_audio(audio_chunk)
                
                yield PipelineResult(
                    state=PipelineState.SPEAKING,
                    response_text=sentence,
                    audio_chunk=audio_chunk,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Process remaining buffer
        if response_buffer.strip():
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            success = self.tts.synthesize(response_buffer, temp_path)

            if success:
                import soundfile as sf
                audio_chunk, _ = sf.read(temp_path)
                os.unlink(temp_path)
            else:
                audio_chunk = np.array([], dtype=np.float32)
            
            if self.on_audio and len(audio_chunk) > 0:
                self.on_audio(audio_chunk)
            
            yield PipelineResult(
                state=PipelineState.SPEAKING,
                response_text=response_buffer,
                audio_chunk=audio_chunk,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        if self.on_response:
            self.on_response(full_response)
        
        self._set_state(PipelineState.IDLE)
    
    def _has_complete_sentence(self, text: str) -> bool:
        """Check if text contains a complete sentence."""
        import re
        return bool(re.search(r'[.!?;]\s*$', text.strip()))
    
    def _extract_sentence(self, text: str) -> tuple:
        """Extract first complete sentence from text."""
        import re
        match = re.search(r'^(.*?[.!?;])\s*(.*)$', text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return text, ""
    
    def reset(self):
        """Reset pipeline state."""
        if self._vad_iterator:
            self._vad_iterator.reset()
        if self._llm:
            self._llm.clear_history()
        self._set_state(PipelineState.IDLE)


# Test function
def test_pipeline():
    """Test pipeline với simulated audio."""
    print("Testing Speech-to-Speech Pipeline...")
    
    # Callbacks
    def on_state(state):
        print(f"  State: {state.value}")
    
    def on_transcription(text):
        print(f"  Transcription: {text}")
    
    def on_response(text):
        print(f"  Response: {text}")
    
    def on_audio(audio):
        print(f"  Audio: {len(audio)} samples")
    
    # Create pipeline
    pipeline = SpeechToSpeechPipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        on_state_change=on_state,
        on_transcription=on_transcription,
        on_response=on_response,
        on_audio=on_audio
    )
    
    # Load all models
    pipeline.load_all_models()
    
    # Simulate audio input (random noise - won't produce meaningful results)
    test_audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.1
    
    # Process
    result = pipeline.process_speech(test_audio)
    print(f"Processing time: {result.processing_time_ms:.0f}ms")
    
    print("✅ Pipeline test complete!")


if __name__ == "__main__":
    test_pipeline()
