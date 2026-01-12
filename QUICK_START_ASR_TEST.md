# Quick Start - ASR Testing

## 🚀 Test ASR với Microphone (Recommended)

```bash
# Test Whisper với streaming output (mặc định)
python test_asr_realtime.py --model whisper

# Nói vào microphone → Xem từng từ xuất hiện!
# Press Ctrl+C để xem summary
```

---

## 📋 Output Example

```
╔══════════════════════════════════════╗
║   ASR Real-time Testing Tool         ║
║   Model: whisper | Device: cuda      ║
║   Streaming: ✅ Enabled              ║
╚══════════════════════════════════════╝

Loading Silero VAD...
✓ VAD loaded
Loading Whisper Large V3 Turbo...
✓ Whisper loaded

════════════════════════════════════════
   READY TO TEST!
════════════════════════════════════════

🎤 Listening...

>>> Speaking detected...

📊 Audio: 3.50s
Whisper: Hello, this is a test of the streaming feature.
         ↑ Từng segment xuất hiện ngay khi có!
⏱️  0.342s | RTF: 0.098x | 10.2x real-time
🎤 Ready for next...
```

---

## 🎯 Options

### Compare Models
```bash
python test_asr_realtime.py --model compare
```
Test cả Whisper và SpeechBrain cùng lúc.

### Disable Streaming
```bash
python test_asr_realtime.py --model whisper --no-streaming
```
In full result một lần (không stream).

### Use CPU
```bash
python test_asr_realtime.py --model whisper --device cpu
```

---

## 📊 Understanding Metrics

**RTF (Real-time Factor)**:
- RTF = transcription_time / audio_duration
- RTF = 0.1 → **10x faster** than real-time
- Lower is better!

**Example**:
- Audio: 10s
- Transcription: 1s
- RTF: 0.1x (10x real-time)

---

## 💡 Tips

1. **Clear environment**: Quiet room, good mic
2. **Speak naturally**: Normal pace and volume
3. **Pause briefly**: 0.8s silence to trigger processing
4. **Try long sentences**: See streaming effect better

---

## 🔧 Troubleshooting

### No audio detected?
```bash
# Check mic
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Empty transcription?
- Speak louder
- Check mic not muted
- Ensure speech > 0.5s

---

## 📚 More Info

- [ASR_STREAMING_DEMO.md](ASR_STREAMING_DEMO.md) - Detailed streaming guide
- [ASR_TESTING_GUIDE.md](evaluation/ASR_TESTING_GUIDE.md) - Complete testing guide
- [WHISPER_MIGRATION_COMPLETE.md](WHISPER_MIGRATION_COMPLETE.md) - Technical details

---

**Start testing now:**
```bash
python test_asr_realtime.py --model whisper
```

🎤 Speak and watch the magic! ✨
