# ASR Testing Guide

Hướng dẫn test và đánh giá các ASR models với microphone thực tế.

---

## 🎯 Tools Available

### 1. **test_asr_realtime.py** - Real-time Microphone Testing ⭐
Test ASR models với voice của bạn qua microphone, giống như s2s_client.py

**Features:**
- ✅ Real-time VAD (Voice Activity Detection)
- ✅ Auto-detect khi bạn bắt đầu/dừng nói
- ✅ Test Whisper hoặc SpeechBrain hoặc compare cả hai
- ✅ Show performance metrics (time, RTF)
- ✅ Summary statistics khi kết thúc

**Usage:**
```bash
# Test Whisper Large V3 Turbo (recommended)
python test_asr_realtime.py --model whisper

# Test SpeechBrain Conformer
python test_asr_realtime.py --model speechbrain

# Compare both models
python test_asr_realtime.py --model compare

# Use CPU instead of GPU
python test_asr_realtime.py --model whisper --device cpu
```

### 2. **test_asr_benchmark.py** - Benchmark Tool
Comprehensive benchmarking với audio files hoặc synthetic audio

**Features:**
- ✅ Test multiple Whisper sizes (large-v3-turbo, medium, small)
- ✅ Compare với SpeechBrain
- ✅ Calculate WER (Word Error Rate) với ground truth
- ✅ Load custom audio files
- ✅ Generate synthetic test audio

**Usage:**
```bash
# Benchmark with synthetic audio
python test_asr_benchmark.py --duration 10

# Benchmark with your audio file
python test_asr_benchmark.py --audio test.wav --text "expected transcription"

# Test specific model only
python test_asr_benchmark.py --model whisper --audio test.wav

# Compare all models
python test_asr_benchmark.py --model all --audio test.wav --text "ground truth"
```

---

## 🚀 Quick Start

### Recommended: Real-time Testing

```bash
# 1. Activate environment
conda activate speech_env  # or your environment

# 2. Run real-time test
python test_asr_realtime.py --model whisper

# 3. Speak into your microphone
#    - System will auto-detect when you speak
#    - Wait 0.8s after speaking to trigger transcription
#    - Press Ctrl+C to see summary

# 4. Try comparing models
python test_asr_realtime.py --model compare
```

---

## 📊 Understanding the Output

### Real-time Test Output

```
🎤 Listening...
>>> Speaking detected...
Processing 3.50s audio...
  Testing whisper...
✓ whisper: "Hello, this is a test of the ASR system."
  Time: 0.342s | RTF: 0.098x
Ready for next audio...
```

**Metrics Explained:**
- **Time**: Total transcription time
- **RTF (Real-time Factor)**: transcription_time / audio_duration
  - RTF < 1.0 = Faster than real-time ✅
  - RTF = 0.1 = 10x faster than real-time
  - Lower is better

### Summary Table

```
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Test# ┃ Duration ┃ whisper Text     ┃ Time     ┃ RTF    ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ 1     │ 3.50s    │ Hello, this is...│ 0.342s   │ 0.098x │
│ 2     │ 2.80s    │ Testing one tw...│ 0.281s   │ 0.100x │
└───────┴──────────┴──────────────────┴──────────┴────────┘

AVERAGES:
  whisper: 0.312s | RTF: 0.099x
```

---

## 🎤 Testing Tips

### For Best Results:

1. **Clear Audio Environment**
   - Quiet room (minimize background noise)
   - Good microphone quality
   - Speak clearly and at normal volume

2. **Natural Speech**
   - Speak naturally, not too fast or slow
   - Complete sentences work best
   - Pause briefly (0.8s) after speaking

3. **What to Test:**
   - Short phrases (2-5 seconds)
   - Long sentences (5-10 seconds)
   - Different accents/speaking styles
   - Technical terms
   - Numbers and dates

### Example Test Sentences:

```
1. "Hello, how are you today?"
2. "The quick brown fox jumps over the lazy dog."
3. "I need to schedule a meeting for tomorrow at 3 PM."
4. "Can you help me with the artificial intelligence project?"
5. "Testing one two three four five six seven eight nine ten."
```

---

## 🔧 Troubleshooting

### Issue: No audio detected

**Check:**
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python -c "import sounddevice as sd; import numpy as np; sd.rec(48000, samplerate=16000, channels=1); sd.wait(); print('OK')"
```

### Issue: "sounddevice not installed"

```bash
pip install sounddevice
```

### Issue: Transcription is empty

**Possible causes:**
- Speak louder
- Check microphone is not muted
- Audio too short (< 0.5s)
- No speech detected by VAD

### Issue: Slow transcription

**Solutions:**
```bash
# Use smaller Whisper model
# Edit test_asr_realtime.py, change model size in _load_models():
# "large-v3-turbo" -> "medium" or "small"

# Or use CPU with int8
python test_asr_realtime.py --model whisper --device cpu
```

---

## 📈 Benchmark Results Reference

### Expected Performance (RTX 3090 GPU)

| Model | WER | RTF | Load Time | Note |
|-------|-----|-----|-----------|------|
| Whisper large-v3-turbo | ~7.4% | 0.05-0.15x | ~3-5s | **Recommended** |
| Whisper medium | ~8-9% | 0.03-0.10x | ~2-3s | Faster |
| Whisper small | ~9-11% | 0.02-0.05x | ~1-2s | Fastest |
| SpeechBrain Conformer | ~8-9% | 0.20-0.40x | ~8-12s | Slower |

**RTF = 0.1x** means processing is **10x faster** than real-time
- 10s audio → 1s transcription time

---

## 🎯 Comparing Models

### Test Scenario: Compare Whisper vs SpeechBrain

```bash
python test_asr_realtime.py --model compare
```

### What to Look For:

1. **Accuracy**
   - Which model transcribes more accurately?
   - Compare output text side by side

2. **Speed**
   - Check RTF values
   - Lower RTF = faster processing

3. **Consistency**
   - Test multiple times
   - Check if results are consistent

4. **Edge Cases**
   - Background noise handling
   - Accents
   - Technical terms
   - Multiple speakers

---

## 📝 Example Session

```bash
$ python test_asr_realtime.py --model compare

╔══════════════════════════════════════╗
║   ASR Real-time Testing Tool         ║
║   Model: compare | Device: cuda      ║
╚══════════════════════════════════════╝

Loading Silero VAD...
✓ VAD loaded
Loading Whisper Large V3 Turbo...
✓ Whisper loaded
Loading SpeechBrain Conformer...
✓ SpeechBrain loaded

════════════════════════════════════════
   READY TO TEST!
════════════════════════════════════════

- Speak naturally (VAD will detect speech)
- System will process when you stop speaking
- Press Ctrl+C to see summary and exit

🎤 Listening...

>>> Speaking detected...

Processing 3.20s audio...
  Testing whisper...
✓ whisper: "Hello, this is a real-time test."
  Time: 0.298s | RTF: 0.093x
  Testing speechbrain...
✓ speechbrain: "HELLO THIS IS A REAL TIME TEST"
  Time: 0.891s | RTF: 0.278x
Ready for next audio...

🎤 Listening...

^C
Stopping...

════════════════════════════════════════
TEST SUMMARY
════════════════════════════════════════

┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Test# ┃ Duration ┃ whisper Text        ┃ Time    ┃ RTF   ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ 1     │ 3.20s    │ Hello, this is a... │ 0.298s  │ 0.093x│
└───────┴──────────┴─────────────────────┴─────────┴───────┘

AVERAGES:
  whisper: 0.298s | RTF: 0.093x
  speechbrain: 0.891s | RTF: 0.278x

✅ Whisper is ~3x faster!
```

---

## 🎓 Advanced Usage

### Custom VAD Threshold

Edit `test_asr_realtime.py`:
```python
SPEECH_THRESHOLD = 0.5  # Default
# Lower = more sensitive (detect quieter speech)
# Higher = less sensitive (only loud speech)
```

### Custom Silence Duration

```python
max_silence = int(SAMPLE_RATE * 0.8)  # 800ms
# Increase for slower speakers
# Decrease for faster response
```

### Test with Pre-recorded Audio

Use benchmark tool:
```bash
# Record your voice first
python -c "
import sounddevice as sd
import soundfile as sf
audio = sd.rec(5*16000, samplerate=16000, channels=1, dtype='float32')
sd.wait()
sf.write('my_voice.wav', audio, 16000)
"

# Then benchmark
python test_asr_benchmark.py --audio my_voice.wav --text "your expected text"
```

---

## 📚 Related Files

- [WHISPER_MIGRATION_COMPLETE.md](WHISPER_MIGRATION_COMPLETE.md) - Whisper migration details
- [speech_pipeline/asr.py](speech_pipeline/asr.py) - ASR implementation
- [s2s_client.py](s2s_client.py) - Full speech-to-speech client

---

## 💡 Tips for Evaluation

### To Fairly Compare Models:

1. **Use Same Audio**
   - Record test samples
   - Use benchmark tool with same files

2. **Multiple Tests**
   - Run at least 5-10 samples
   - Average the results

3. **Different Scenarios**
   - Quiet room
   - Background noise
   - Different speakers
   - Various accents

4. **Metrics to Track**
   - Transcription accuracy (manual check)
   - Speed (RTF)
   - Load time
   - Memory usage (use `nvidia-smi` for GPU)

---

**Happy Testing! 🎉**

For issues or questions, check [WHISPER_MIGRATION_COMPLETE.md](WHISPER_MIGRATION_COMPLETE.md)
