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

# ASR Model - Whisper Large V3 Turbo
ASR_MODEL_TYPE = "whisper"
WHISPER_MODEL_NAME = "large-v3-turbo"  # Whisper Large V3 Turbo (OpenAI)
# Options:
# - large-v3-turbo (CURRENT - 6x faster than large-v3, 809M params, WER ~7.4%)
# - large-v3 (highest accuracy, but slower)
# - medium (balanced, good for most use cases)
# - small (fast, lower accuracy)
# Note: Using faster-whisper for optimized inference

# TTS Model - Kokoro
TTS_MODEL_TYPE = "kokoro"
TTS_MODEL_NAME = "hexgrad/Kokoro-82M"  # Kokoro-82M - Fast and lightweight TTS
# Options:
# - hexgrad/Kokoro-82M (CURRENT - 82M params, fast)
# - facebook/mms-tts-eng (VITS-based, multilingual)

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
LLM_MAX_NEW_TOKENS = 512  # Lower for faster response in real-time conversation
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_USE_4BIT = True  # Set to True if limited VRAM

# =============================================================================
# TTS SETTINGS (Kokoro)
# =============================================================================
TTS_OUTPUT_SAMPLE_RATE = 24000  # Kokoro output sample rate (24kHz)
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
