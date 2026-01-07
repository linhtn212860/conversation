# ReSpeaker Speech-to-Speech Client

Speech-to-Speech client cho ReSpeaker 4 Mic Array với echo cancellation.

## Tính năng

✅ **Hardware AEC**: Sử dụng channel 0 của ReSpeaker (có AEC sẵn)
✅ **Playback Suppression**: Tự động giảm mic khi AI đang nói
✅ **Simple VAD**: Energy-based voice activity detection
✅ **No PyAudio**: Sử dụng `arecord`/`aplay` trực tiếp (ổn định hơn)
✅ **Gain Control**: Tăng volume loa và mic tự động

## Yêu cầu

### Hardware
- ReSpeaker 4 Mic Array v2.0 (USB)
- Speaker kết nối với máy tính

### Software

**IMPORTANT**: Client và Server phải chạy trong cùng conda environment (`speech_env`)

```bash
# ALSA tools (thường có sẵn trên Linux)
sudo apt-get install alsa-utils

# Activate environment
conda activate speech_env

# Python packages (nếu chưa có)
pip install websockets numpy rich
```

## Sử dụng

### 1. Khởi động server

Terminal 1:
```bash
python run_server.py
```

Server sẽ load các models:
- Conformer ASR (NeMo)
- Qwen3-8B LLM
- VITS TTS
- Silero VAD

### 2. Chạy ReSpeaker client

Terminal 2:
```bash
# Cách 1: Dùng script launcher (tự động dùng đúng environment)
./run_respeaker_client.sh

# Cách 2: Activate environment trước
conda activate speech_env
python s2s_client_respeaker_simple.py

# Cách 3: Dùng conda run (không cần activate)
conda run -n speech_env python s2s_client_respeaker_simple.py

# Cách 4: Với custom server URL
./run_respeaker_client.sh --url ws://192.168.1.100:8765/ws
```

**Note**: Script launcher `./run_respeaker_client.sh` tự động sử dụng đúng conda environment.

### 3. Nói chuyện!

- Nói tự nhiên vào ReSpeaker
- Không cần nhấn nút
- Hệ thống tự động phát hiện giọng nói
- AI sẽ trả lời qua speaker

## Cấu trúc hệ thống

```
┌─────────────────┐
│  ReSpeaker      │
│  (6 channels)   │
│  Channel 0: AEC │
└────────┬────────┘
         │ arecord
         ▼
┌─────────────────────────────┐
│  ReSpeaker Client           │
│  - Audio capture            │
│  - Simple VAD               │
│  - Playback suppression     │
│  - WebSocket communication  │
└────────┬────────────────────┘
         │ WebSocket
         ▼
┌─────────────────────────────┐
│  Speech-to-Speech Server    │
│  - Conformer ASR            │
│  - Qwen3-8B LLM            │
│  - VITS TTS                 │
└────────┬────────────────────┘
         │ Audio response
         ▼
┌─────────────────┐
│  Speaker        │
│  (aplay)        │
└─────────────────┘
```

## Cấu hình

Chỉnh sửa trong `s2s_client_respeaker_simple.py`:

```python
# Audio settings
SAMPLE_RATE = 16000          # Hz
DEVICE = "hw:3,0"            # ReSpeaker ALSA device
CHANNELS = 6                 # ReSpeaker channels
PROCESSED_CHANNEL = 0        # Channel with hardware AEC
CHUNK_SIZE = 1024           # Buffer size

# Gain settings
OUTPUT_GAIN = 2.0           # Speaker volume (2x)
MIC_GAIN = 1.5              # Mic sensitivity (1.5x)
SUPPRESSION_GAIN = 0.1      # Mic reduction during playback (10%)
```

## Troubleshooting

### ReSpeaker không được phát hiện

```bash
# Kiểm tra ReSpeaker
arecord -l | grep ReSpeaker

# Output mong muốn:
# card 3: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: USB Audio [USB Audio]
```

Nếu card number khác 3, cập nhật `DEVICE` trong code:
```python
DEVICE = "hw:X,0"  # X = card number
```

### Không có audio output

```bash
# Kiểm tra speaker
aplay -l

# Test speaker
speaker-test -t wav -c 1
```

### Echo vẫn còn

1. **Tăng playback suppression**:
```python
SUPPRESSION_GAIN = 0.05  # Giảm từ 0.1 xuống 0.05
```

2. **Điều chỉnh VAD threshold**:
```python
# Trong code, tìm dòng:
threshold = 0.05 if self.player.is_suppressing else 0.02
# Tăng lên:
threshold = 0.08 if self.player.is_suppressing else 0.03
```

3. **Giảm speaker volume**:
```python
OUTPUT_GAIN = 1.5  # Giảm từ 2.0 xuống 1.5
```

### Latency cao

- Giảm `CHUNK_SIZE` xuống 512 hoặc 256
- Đảm bảo server chạy trên GPU
- Kiểm tra network latency nếu dùng remote server

## Files

- `s2s_client_respeaker_simple.py` - Client chính (khuyên dùng)
- `s2s_client_respeaker_pyaudio.py` - Client dùng PyAudio (có thể unstable)
- `run_respeaker_client.sh` - Script launcher
- `run_server.py` - Server entry point

## Kiến trúc Echo Cancellation

Client sử dụng 2-layer echo cancellation:

1. **Hardware Layer**: ReSpeaker's built-in AEC
   - Xử lý ở hardware level
   - Channel 0 đã được xử lý
   - Loại bỏ echo cơ bản

2. **Software Layer**: Playback-aware suppression
   - Monitor playback state
   - Giảm mic gain khi AI đang nói
   - Ngăn false speech detection

## Performance

- ASR latency: ~100-200ms (Conformer)
- LLM latency: ~500-1000ms (Qwen3-8B)
- TTS latency: ~200-400ms (VITS)
- Total: ~800-1600ms (end-to-end)

## Tham khảo

- [ReSpeaker 4 Mic Array](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)
- [NeMo Conformer ASR](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html)
- [Qwen3-8B](https://github.com/QwenLM/Qwen)
- [VITS TTS](https://github.com/jaywalnut310/vits)

## License

MIT
