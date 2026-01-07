# Speech-to-Speech System with ReSpeaker - Project Summary

## 🎯 Overview

Hệ thống Speech-to-Speech hoàn chỉnh với ReSpeaker 4 Mic Array, bao gồm:
- **ASR**: NeMo Conformer-CTC (English)
- **LLM**: Qwen3-8B (Conversational AI)
- **TTS**: VITS (English)
- **Echo Cancellation**: Hardware AEC + Software Suppression

## 📁 Project Structure

```
qwen_8b/
├── speech_pipeline/          # Core pipeline
│   ├── asr.py               # Conformer ASR
│   ├── llm.py               # Qwen3-8B LLM
│   ├── tts.py               # VITS TTS
│   ├── vad.py               # Silero VAD
│   ├── pipeline.py          # Main pipeline
│   ├── websocket_server.py  # WebSocket server
│   └── config.py            # Configuration
│
├── run_server.py            # Server entry point
│
├── s2s_client.py            # Standard client (for regular mic)
├── s2s_client_respeaker_simple.py    # ReSpeaker client (RECOMMENDED)
├── s2s_client_respeaker_pyaudio.py   # Alternative PyAudio version
├── run_respeaker_client.sh           # Quick launcher
│
├── test_connection.py       # Connection test
├── test_respeaker_audio.py  # Audio test
│
├── README_RESPEAKER.md      # Full documentation
├── QUICKSTART_RESPEAKER.md  # Quick start
├── RESTART_GUIDE.md         # Troubleshooting
└── SUMMARY.md              # This file
```

## 🚀 Quick Start

### Terminal 1: Server
```bash
conda activate speech_env
python run_server.py
```

### Terminal 2: Client
```bash
./run_respeaker_client.sh
```

## 🎤 ReSpeaker Client Features

### Echo Cancellation (Dual-Layer)
1. **Hardware Layer**: ReSpeaker built-in AEC
   - Channel 0 pre-processed
   - Hardware-level echo removal

2. **Software Layer**: Playback-aware suppression
   - Monitor audio playback state
   - Reduce mic gain to 10% during AI speech
   - Prevent false speech detection

### Audio Pipeline
```
ReSpeaker (6ch) → Channel 0 (AEC) → arecord →
→ Client VAD → WebSocket → Server →
→ ASR → LLM → TTS → aplay → Speaker
```

### Key Parameters
```python
SAMPLE_RATE = 16000
DEVICE = "hw:3,0"
CHANNELS = 6
PROCESSED_CHANNEL = 0  # Hardware AEC

OUTPUT_GAIN = 2.0       # Speaker volume
MIC_GAIN = 1.5          # Mic sensitivity
SUPPRESSION_GAIN = 0.1  # Suppression during playback
```

## 🔧 Configuration

### Adjust Echo Cancellation
Edit `s2s_client_respeaker_simple.py`:

```python
# Stronger suppression (reduce echo more)
SUPPRESSION_GAIN = 0.05  # from 0.1

# Lower speaker volume (less echo)
OUTPUT_GAIN = 1.5  # from 2.0

# Higher mic gain (more sensitive)
MIC_GAIN = 2.0  # from 1.5
```

### Adjust VAD Sensitivity
In client code, find:
```python
threshold = 0.05 if self.player.is_suppressing else 0.02
```

Change to:
```python
# More sensitive (picks up quieter speech)
threshold = 0.03 if self.player.is_suppressing else 0.01

# Less sensitive (ignore quiet sounds)
threshold = 0.08 if self.player.is_suppressing else 0.03
```

## 📊 Performance

### Latency (End-to-End)
- ASR: ~100-200ms (Conformer)
- LLM: ~500-1000ms (Qwen3-8B)
- TTS: ~200-400ms (VITS)
- **Total: ~800-1600ms**

### Models
- **Conformer**: nvidia/stt_en_conformer_ctc_large (~120M params)
- **Qwen3-8B**: 8B parameters
- **VITS**: facebook/mms-tts-eng
- **Silero VAD**: lightweight

### Hardware Requirements
- GPU: NVIDIA (CUDA) - 24GB+ VRAM recommended
- RAM: 16GB+
- ReSpeaker 4 Mic Array v2.0

## 🐛 Common Issues

### 1. Connection Timeout
```bash
# Check server
ps aux | grep run_server

# Restart server
pkill -f "python.*run_server"
python run_server.py
```

### 2. Echo Still Present
- Increase `SUPPRESSION_GAIN` to 0.05 or lower
- Decrease `OUTPUT_GAIN` to 1.5 or lower
- Move speaker farther from ReSpeaker
- Check room acoustics

### 3. Audio Not Playing
```bash
# Test speaker
speaker-test -t wav -c 1

# Check ALSA
aplay -l
```

### 4. ReSpeaker Not Found
```bash
# Check device
arecord -l | grep ReSpeaker

# Update DEVICE if card number changed
# e.g., card 2: DEVICE = "hw:2,0"
```

### 5. Low/No Transcription
```bash
# Test ReSpeaker audio
python test_respeaker_audio.py

# Check channel energy
# Adjust MIC_GAIN if needed
```

## 📚 Documentation

- **QUICKSTART_RESPEAKER.md** - 5-minute quick start
- **README_RESPEAKER.md** - Complete documentation
- **RESTART_GUIDE.md** - Restart & troubleshooting
- **SUMMARY.md** - This file (project overview)

## 🔄 Typical Workflow

1. **Development**
   ```bash
   conda activate speech_env
   python run_server.py --preload  # Preload models
   ```

2. **Testing**
   ```bash
   python test_connection.py       # Test server
   python test_respeaker_audio.py  # Test ReSpeaker
   ```

3. **Production**
   ```bash
   # Terminal 1
   python run_server.py

   # Terminal 2
   ./run_respeaker_client.sh
   ```

## 💡 Tips

1. **Better Echo Cancellation**
   - Keep speaker volume moderate
   - Position speaker away from ReSpeaker
   - Use directional speaker if possible
   - Adjust `SUPPRESSION_GAIN` to find sweet spot

2. **Lower Latency**
   - Use smaller `CHUNK_SIZE` (512 or 256)
   - Reduce `LLM_MAX_NEW_TOKENS` in config
   - Use GPU for all models

3. **Better Transcription**
   - Speak clearly and face ReSpeaker
   - Minimize background noise
   - Adjust `MIC_GAIN` if too quiet
   - Check channel 0 is used (hardware AEC)

4. **Debugging**
   - Check logs in both terminals
   - Use `test_connection.py` first
   - Test audio with `test_respeaker_audio.py`
   - Monitor with `ps aux | grep python`

## 🎓 Architecture Notes

### Why arecord/aplay?
- PyAudio has issues with 6-channel ReSpeaker
- ALSA tools are more stable for multi-channel devices
- Direct kernel interface, less overhead
- No Python audio library crashes

### Why Channel 0?
- ReSpeaker processes channel 0 with hardware AEC
- Other channels are raw microphone input
- Channel 0 = best quality for echo cancellation

### Why Playback Suppression?
- Even with hardware AEC, some echo remains
- Monitor playback state
- Reduce mic gain during AI speech
- Prevents false speech detection from echo

## 📝 TODO / Future Improvements

- [ ] Add DTLN-aec software AEC (optional)
- [ ] Support multiple ReSpeaker devices
- [ ] Add noise cancellation (beyond Silero VAD)
- [ ] Web interface for configuration
- [ ] Docker container for easy deployment
- [ ] Real-time latency monitoring
- [ ] Recording/replay for debugging

## 🙏 Credits

- **NeMo**: NVIDIA NeMo Toolkit
- **Qwen**: Alibaba Qwen LLM
- **VITS**: Conditional Variational Autoencoder TTS
- **Silero**: Silero VAD
- **ReSpeaker**: Seeed Studio

## 📄 License

MIT

---

**Last Updated**: 2026-01-07
**Version**: 1.0
**Status**: ✅ Production Ready
