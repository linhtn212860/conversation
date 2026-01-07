# English Voice Chatbot Summary

This repository implements an English speech-to-speech chatbot:

- **VAD**: Silero VAD
- **ASR**: OpenAI Whisper (via Hugging Face Transformers)
- **LLM**: Qwen3-8B (local weights)
- **TTS**: Microsoft Edge-TTS (English voices)

## Files Modified

### Core Components

1. **[speech_pipeline/asr.py](speech_pipeline/asr.py)**
   - Uses `openai/whisper-base` model
   - Configured for English language transcription

2. **[speech_pipeline/tts.py](speech_pipeline/tts.py)**
   - Uses `EdgeTTSWrapper`
   - Created `EnglishTTS` class
   - Default voice: `en-US-JennyNeural` (natural female)

3. **[speech_pipeline/llm.py](speech_pipeline/llm.py)**
   - Updated system prompt for English conversation
   - Optimized for spoken responses (no markdown)
   - Updated search keywords for English queries
   - Emphasis on brief, conversational responses

4. **[speech_pipeline/config.py](speech_pipeline/config.py)**
   - Added `WHISPER_MODEL_SIZE` configuration
   - Updated `EDGE_TTS_VOICE` for English
   - Added comments explaining model size trade-offs
   - Tuned for low-latency English voice chat

5. **[speech_pipeline/pipeline.py](speech_pipeline/pipeline.py)**
   - Updated imports to use `WhisperASR` and `EnglishTTS`
   - Modified TTS calls to handle file-based synthesis
   - Added proper cleanup for temporary audio files

### Configuration & Documentation

6. **[requirements_speech.txt](requirements_speech.txt)**
   - Updated comments for Whisper
   - Kept `edge-tts` as primary TTS

7. **[ENGLISH_VOICE_CHATBOT.md](ENGLISH_VOICE_CHATBOT.md)** ✨ NEW
   - Complete usage guide
   - Installation instructions
   - Configuration options
   - Performance benchmarks
   - Troubleshooting guide

8. **[test_english_chatbot.py](test_english_chatbot.py)** ✨ NEW
   - Component testing script
   - Tests VAD, ASR, TTS, LLM individually
   - Full pipeline test
   - Clear pass/fail reporting

## Key Notes

- Whisper models auto-download on first run (via Transformers)
- Edge-TTS requires an internet connection
- You can reduce VRAM/RAM usage by switching Whisper size and enabling 4-bit quantization

## Testing

Run the comprehensive test suite:

```bash
python test_english_chatbot.py
```

This will test:
- Package installations
- VAD (Voice Activity Detection)
- ASR (Whisper)
- TTS (Edge-TTS)
- LLM (Qwen3-8B)
- Full pipeline integration

## Next Steps

1. **Test the system**:
   ```bash
   python test_english_chatbot.py
   ```

2. **Install missing packages** (if any):
   ```bash
   pip install -r requirements_speech.txt
   ```

3. **Run a quick test**:
   ```bash
   # Test TTS
   python -m speech_pipeline.tts

   # Test ASR (after Whisper downloads)
   python -m speech_pipeline.asr
   ```

4. **Start the server**:
   ```bash
   python run_server.py
   ```

## Troubleshooting

### Common Issues

1. **Whisper model not found**
   - Models auto-download on first run (~290MB for base)
   - Check internet connection

2. **Edge-TTS not working**
   - Requires internet connection (cloud-based)
   - Test with: `edge-tts --list-voices`

3. **Out of memory**
   - Use smaller Whisper: `WHISPER_MODEL_SIZE = "tiny"`
   - Enable 4-bit quantization: `LLM_USE_4BIT = True`

## Notes

- The system is optimized for English conversation
- Qwen3-8B can understand other languages, but prompts and TTS are tuned for English

## Resources

- [Whisper Documentation](https://github.com/openai/whisper)
- [Edge-TTS Documentation](https://github.com/rany2/edge-tts)
- [Qwen Models](https://huggingface.co/Qwen)
- [Silero VAD](https://github.com/snakers4/silero-vad)
