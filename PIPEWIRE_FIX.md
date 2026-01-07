# ReSpeaker PipeWire Configuration Fix

## Problem
The ReSpeaker client was failing with "Incomplete read: got 0 bytes" errors because:
1. PipeWire was managing the ReSpeaker device, blocking direct ALSA hardware access
2. The default PipeWire profile only exposed 3 channels (surround-21) instead of all 6 channels
3. `arecord` couldn't access `hw:3,0` directly due to PipeWire lock

## Solution
Switch to PipeWire-native audio capture using the 5.1 surround profile:

### 1. Configure PipeWire Profile (One-time setup)
```bash
# Find your ReSpeaker card number
pactl list cards short | grep ReSpeaker

# Switch to 5.1 surround profile (replace 2515 with your card number)
pactl set-card-profile 2515 output:iec958-stereo+input:analog-surround-51
```

This enables all 6 channels:
- Channel 0: Front-Left (with hardware AEC)
- Channel 1: Front-Right
- Channel 2: Center
- Channel 3: LFE
- Channel 4: Rear-Left
- Channel 5: Rear-Right

### 2. Client Changes
The client now uses:
- **`parec`** (PulseAudio/PipeWire record) instead of `arecord`
- **PipeWire device name**: `alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-51`
- All 6 channels accessible, extracting channel 0 for hardware AEC

## Verification

### Check Current Profile
```bash
pactl list cards | grep -A 5 "ReSpeaker" | grep "Active Profile"
```

Should show: `output:iec958-stereo+input:analog-surround-51`

### Test Audio Capture
```bash
# Should capture audio successfully (Ctrl+C to stop)
parec --device=alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-51 \
      --rate=16000 --channels=6 --format=s16le --raw | wc -c
```

### Check Available Channels
```bash
pactl list sources | grep -A 20 "ReSpeaker" | grep "Channels:"
```

Should show: `Channels: 6`

## Usage

### Start Client
```bash
bash run_respeaker_client.sh
```

Or directly:
```bash
conda run -n speech_env python s2s_client_respeaker_simple.py
```

### What Should Happen
1. Client connects to WebSocket server
2. `parec` subprocess starts and captures 6-channel audio from ReSpeaker
3. Channel 0 (hardware AEC) is extracted and sent to server
4. Audio playback works through PipeWire
5. Playback suppression prevents echo

## Persistence

The PipeWire profile change persists across reboots. If needed, add to startup:
```bash
# Add to ~/.bashrc or startup script
pactl set-card-profile alsa_card.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00 \
      output:iec958-stereo+input:analog-surround-51
```

## Troubleshooting

### Profile Resets
If profile resets to 3 channels:
```bash
# Re-apply 5.1 profile
CARD=$(pactl list cards short | grep ReSpeaker | awk '{print $1}')
pactl set-card-profile $CARD output:iec958-stereo+input:analog-surround-51
```

### Device Name Changes
If device name changes (USB port change, etc.):
```bash
# Find new device name
pactl list sources short | grep ReSpeaker

# Update DEVICE in s2s_client_respeaker_simple.py
```

### PipeWire Not Running
If PipeWire isn't running:
```bash
systemctl --user status pipewire pipewire-pulse
systemctl --user start pipewire pipewire-pulse
```

## Why This Works

### PipeWire vs Direct ALSA
- **Before**: Tried to access `hw:3,0` directly → PipeWire blocked it
- **After**: Use PipeWire's native interface → Proper audio routing

### Channel Mapping
PipeWire's 5.1 surround profile maps ReSpeaker's 6 microphones to standard 5.1 channels:
- ReSpeaker Mic 1 → Front-Left (with AEC)
- ReSpeaker Mic 2 → Front-Right
- ReSpeaker Mic 3 → Center
- ReSpeaker Mic 4 → LFE
- ReSpeaker Mic 5 → Rear-Left
- ReSpeaker Mic 6 → Rear-Right

Channel 0 (Front-Left) has hardware AEC enabled in firmware, making it the best choice for speech recognition.

## Technical Details

### Old Approach (Failed)
```python
# arecord with hw:3,0
cmd = ['arecord', '-D', 'hw:3,0', '-f', 'S16_LE', '-r', '16000', '-c', '6', '-t', 'raw']
# Result: Incomplete reads, 0 bytes
```

### New Approach (Working)
```python
# parec with PipeWire device
cmd = [
    'parec',
    '--device', 'alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-51',
    '--rate', '16000',
    '--channels', '6',
    '--format', 's16le',
    '--raw'
]
# Result: Successful audio capture
```

## References
- PipeWire documentation: https://pipewire.org/
- PulseAudio tools: `parec`, `pactl`
- ReSpeaker 4 Mic Array: https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array/
