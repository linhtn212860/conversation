# English Voice Conversation Chatbot

A real-time voice conversation chatbot with English speech recognition and synthesis.

## Architecture

```
Audio Input → VAD → Whisper ASR → Qwen3-8B LLM → Edge-TTS → Audio Output
```

### Components

1. **VAD (Voice Activity Detection)**: Silero VAD - detects when user is speaking
2. **ASR (Speech Recognition)**: OpenAI Whisper - converts English speech to text
3. **LLM (Language Model)**: Qwen3-8B - generates conversational responses
4. **TTS (Text-to-Speech)**: Microsoft Edge-TTS - synthesizes natural English speech

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_speech.txt
```

### 2. Download Models

The models will auto-download on first run:
- **Whisper**: Downloads from HuggingFace (base model ~290MB)
- **Silero VAD**: Downloads via torch.hub (~2MB)
- **Qwen3-8B**: Should already be in `models/Qwen_Qwen3-8B/`
- **Edge-TTS**: No model download needed (cloud-based)

## Configuration

Edit [speech_pipeline/config.py](speech_pipeline/config.py) to customize:

### ASR Settings
```python
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v3
# Recommended: "base" for real-time (74M params, good balance)
```

### TTS Voice
```python
EDGE_TTS_VOICE = "en-US-JennyNeural"  # Natural female voice
# Other options:
# - "en-US-GuyNeural" (male)
# - "en-GB-SoniaNeural" (British female)
# - "en-AU-NatashaNeural" (Australian female)
```

## Usage

### Test Individual Components

```bash
# Test ASR (Whisper)
python -m speech_pipeline.asr

# Test TTS (Edge-TTS)
python -m speech_pipeline.tts

# Test LLM
python -m speech_pipeline.llm
```

### Run Full Pipeline

```bash
python -m speech_pipeline.pipeline
```

### Start WebSocket Server

```bash
python run_server.py
```

Then connect from a client (browser, mobile app, etc.) to `ws://localhost:8765`

## Performance

With 48GB VRAM and optimized settings:
- **ASR Latency**: ~200-500ms (Whisper base)
- **LLM Latency**: ~1-2s for typical response (Qwen3-8B full precision)
- **TTS Latency**: ~500ms-1s (Edge-TTS)
- **Total RTT**: ~2-4 seconds

### Optimization Tips

1. **Faster ASR**: Use `WHISPER_MODEL_SIZE = "tiny"` (39M params, ~100ms)
2. **Faster LLM**: Reduce `LLM_MAX_NEW_TOKENS` in config
3. **Streaming**: Enable streaming mode for lower perceived latency

## Model Sizes

| Component | Model | Parameters | VRAM | Speed |
|-----------|-------|------------|------|-------|
| VAD | Silero | 2M | ~10MB | Very Fast |
| ASR | Whisper-tiny | 39M | ~150MB | Very Fast |
| ASR | Whisper-base | 74M | ~290MB | **Fast** ⭐ |
| ASR | Whisper-small | 244M | ~970MB | Medium |
| LLM | Qwen3-8B | 8B | ~16GB | Fast (FP16) |
| TTS | Edge-TTS | Cloud | N/A | Fast |

⭐ **Recommended configuration for real-time conversation**

## API Example

```python
from speech_pipeline.pipeline import SpeechToSpeechPipeline
import soundfile as sf

# Initialize pipeline
pipeline = SpeechToSpeechPipeline(device="cuda")
pipeline.load_all_models()

# Load audio file
audio, sr = sf.read("user_speech.wav")

# Process
result = pipeline.process_speech(audio)

print(f"User said: {result.transcription}")
print(f"Bot replied: {result.response_text}")

# Save audio response
sf.write("bot_response.wav", result.audio_chunk, sr)
```

## Troubleshooting

### Issue: Whisper model download fails
**Solution**: Download manually from HuggingFace
```bash
python -c "from transformers import WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-base')"
```

### Issue: Edge-TTS not working
**Solution**: Check internet connection (Edge-TTS requires internet)
```bash
edge-tts --list-voices
```

### Issue: Out of memory
**Solution**: Use smaller Whisper model or enable 4-bit quantization for LLM
```python
# In config.py
WHISPER_MODEL_SIZE = "tiny"
LLM_USE_4BIT = True
```

## Features

✅ Real-time voice activity detection
✅ High-quality English speech recognition (Whisper)
✅ Natural conversation with Qwen3-8B
✅ Natural English voice synthesis (Edge-TTS)
✅ Web search integration (optional)
✅ WebSocket support for remote clients
✅ Low latency optimized for real-time conversation

## License

See main project LICENSE
