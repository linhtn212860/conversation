# ASR Streaming Output Demo

## ✨ New Feature: Streaming Transcription

ASR models bây giờ sẽ **in từng từ ngay khi transcribe**, không đợi hết câu!

---

## 🎬 Example Output

### With Streaming (Default):

```
🎤 Listening...
>>> Speaking detected...

📊 Audio: 3.50s
Whisper: Hello, this is a test of the streaming transcription feature.
⏱️  0.342s | RTF: 0.098x | 10.2x real-time
🎤 Ready for next...
```

**Chú ý:** Từng từ sẽ xuất hiện dần dần trên màn hình khi model transcribe!

### Without Streaming:

```
🎤 Listening...
>>> Speaking detected...

📊 Audio: 3.50s
Whisper: Hello, this is a test of the streaming transcription feature.
⏱️  0.342s | RTF: 0.098x | 10.2x real-time
🎤 Ready for next...
```

Cả câu xuất hiện cùng lúc sau khi transcribe xong.

---

## 🚀 Usage

### Enable Streaming (Default):

```bash
python test_asr_realtime.py --model whisper
```

### Disable Streaming:

```bash
python test_asr_realtime.py --model whisper --no-streaming
```

---

## 🎯 How It Works

### Streaming Mode:

1. **Faster-whisper segments**: Model generates transcription theo từng segment
2. **Print immediately**: Mỗi segment được in ra ngay khi có
3. **Real-time feel**: Người dùng thấy từng từ xuất hiện như typing effect

```python
# Internal code
for segment in segments:
    text = segment.text
    print(text, end="", flush=True)  # Print immediately!
```

### Benefits:

- ✅ **Visual feedback**: Thấy progress real-time
- ✅ **Faster perceived speed**: Cảm giác nhanh hơn
- ✅ **Better UX**: Như ChatGPT streaming responses
- ✅ **Early results**: Đọc được kết quả ngay cả khi chưa xong

---

## 📊 Performance

Streaming **không làm chậm** model:
- RTF vẫn như cũ (~0.1x)
- Chỉ thay đổi cách hiển thị
- Memory usage không đổi

---

## 🎨 Compare Models with Streaming

```bash
# Compare Whisper vs SpeechBrain
python test_asr_realtime.py --model compare

# Output:
# Whisper: [words stream in...]
# SpeechBrain: [full result at once]
```

**Note:** SpeechBrain không có streaming API, nên vẫn in full result.

---

## 💡 Tips

1. **For demos**: Use streaming (looks impressive!)
2. **For benchmarks**: Use `--no-streaming` (cleaner output)
3. **For debugging**: Streaming helps see where model gets stuck

---

## 🔍 Technical Details

### Whisper Segments:

Whisper chia audio thành segments (mỗi segment ~2-5 giây):

```
Audio: "Hello, this is a test of the streaming feature."

Segments:
1. "Hello,"
2. " this is a test"
3. " of the streaming feature."
```

Mỗi segment được transcribe và in ra ngay lập tức!

### VAD Integration:

VAD parameters can affect segment boundaries:
- `min_speech_duration_ms=250`: Minimum 250ms per segment
- `min_silence_duration_ms=100`: 100ms silence splits segments

---

**Try it now:**
```bash
python test_asr_realtime.py --model whisper
# Speak: "This is a really long sentence to see the streaming effect"
# Watch words appear one by one! 🎉
```
