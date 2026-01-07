"""
Speech-to-Speech Pipeline Configuration - English Voice Chatbot
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Model paths
QWEN_MODEL_PATH = MODELS_DIR / "Qwen_Qwen3-8B"
SILERO_VAD_REPO = "snakers4/silero-vad"

# ASR Model - Conformer (NeMo or similar)
ASR_MODEL_TYPE = "conformer"
CONFORMER_MODEL_NAME = "nvidia/stt_en_conformer_ctc_large"  # NeMo Conformer model
# Options:
# - nvidia/stt_en_conformer_ctc_small
# - nvidia/stt_en_conformer_ctc_medium
# - nvidia/stt_en_conformer_ctc_large (CURRENT)

# TTS Model - VITS
TTS_MODEL_TYPE = "vits"
VITS_MODEL_NAME = "facebook/mms-tts-eng"  # VITS-based English TTS
# Options:
# - facebook/mms-tts-eng (Massively Multilingual Speech)
# - Custom VITS checkpoint

# =============================================================================
# AUDIO SETTINGS
# =============================================================================
SAMPLE_RATE = 16000  # Hz - required by most models
CHUNK_DURATION_MS = 30  # ms - for VAD processing
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # samples per chunk

# =============================================================================
# VAD SETTINGS (Silero)
# =============================================================================
VAD_THRESHOLD = 0.5  # Speech probability threshold
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum speech segment
VAD_MIN_SILENCE_DURATION_MS = 300  # Silence to end speech
VAD_SPEECH_PAD_MS = 100  # Padding around speech

# =============================================================================
# ASR SETTINGS (Conformer)
# =============================================================================
ASR_MAX_AUDIO_LENGTH = 30  # seconds - max audio for single inference
ASR_LANGUAGE = "en"  # English

# =============================================================================
# LLM SETTINGS (Qwen3-8B)
# =============================================================================
LLM_MAX_NEW_TOKENS = 1024  # Lower for faster response
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_USE_4BIT = False  # Set to True if limited VRAM

# =============================================================================
# TTS SETTINGS (VITS)
# =============================================================================
TTS_OUTPUT_SAMPLE_RATE = 16000  # VITS output sample rate
TTS_RATE = "+0%"  # Speech rate (can be +20% or -20% for faster/slower)

# =============================================================================
# WEBSOCKET SETTINGS
# =============================================================================
WS_HOST = "0.0.0.0"
WS_PORT = 8765
WS_MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB

# =============================================================================
# OPTIMIZATION (Low Latency Priority)
# =============================================================================
# Optimization for real-time voice conversation:
# - Use NeMo Conformer-CTC for ASR (fast and accurate)
# - VITS for TTS (neural, high quality)
# - No LLM quantization with 48GB VRAM (faster inference)
# - Smaller chunk sizes for responsive streaming
# - Lower max tokens for quicker LLM response
OPTIMIZE_FOR_LATENCY = True
