# ReSpeaker Quick Start

Hướng dẫn nhanh để chạy Speech-to-Speech với ReSpeaker.

## Setup 1 lần

```bash
# 1. Cài ALSA tools (nếu chưa có)
sudo apt-get install alsa-utils

# 2. Kiểm tra ReSpeaker
arecord -l | grep ReSpeaker
# Output: card 3: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)]

# 3. Activate environment
conda activate speech_env
```

## Chạy hệ thống

### Terminal 1: Server
```bash
conda activate speech_env
python run_server.py
```

Đợi đến khi thấy: `✅ All models loaded!`

### Terminal 2: Client
```bash
./run_respeaker_client.sh
```

Hoặc:
```bash
conda activate speech_env
python s2s_client_respeaker_simple.py
```

## Sử dụng

1. Đợi thấy: `🎤 Listening...`
2. Nói vào ReSpeaker
3. Đợi AI trả lời qua speaker
4. Ctrl+C để thoát

## Troubleshooting

### Client không kết nối được
```bash
# Kiểm tra server đang chạy
ps aux | grep run_server

# Kiểm tra port
netstat -tlnp | grep 8765
```

### Không có audio output
```bash
# Test speaker
speaker-test -t wav -c 1
```

### Echo vẫn còn
Chỉnh trong `s2s_client_respeaker_simple.py`:
```python
SUPPRESSION_GAIN = 0.05  # Giảm xuống để suppress nhiều hơn
OUTPUT_GAIN = 1.5        # Giảm volume loa
```

### ReSpeaker không được phát hiện
```bash
# Xem card number
arecord -l

# Update DEVICE trong code nếu khác card 3
# Ví dụ card 2: DEVICE = "hw:2,0"
```

## Files chính

- `s2s_client_respeaker_simple.py` - Client (dùng arecord/aplay)
- `run_respeaker_client.sh` - Launcher script
- `run_server.py` - Server
- `README_RESPEAKER.md` - Full documentation

## Architecture

```
ReSpeaker → Client → WebSocket → Server → Speaker
  (AEC)    (VAD+Suppress)      (ASR+LLM+TTS)
```

## Performance

- End-to-end latency: ~800-1600ms
- Echo cancellation: Hardware AEC + Playback suppression
- Models: Conformer (ASR) + Qwen3-8B (LLM) + VITS (TTS)
