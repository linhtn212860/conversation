#!/bin/bash
# Test if ReSpeaker client connects and stays connected

echo "================================"
echo "ReSpeaker Client Connection Test"
echo "================================"
echo ""

# Check server
echo "1. Checking if server is running..."
if ps aux | grep -q "[r]un_server.py"; then
    echo "   ✓ Server is running"
else
    echo "   ✗ Server is NOT running"
    echo "   Start it with: conda run -n speech_env python run_server.py"
    exit 1
fi
echo ""

# Check PipeWire profile
echo "2. Checking PipeWire profile..."
PROFILE=$(pactl list cards | grep -A 100 "ReSpeaker" | grep "Active Profile" | head -1)
if echo "$PROFILE" | grep -q "surround-51"; then
    echo "   ✓ Profile is surround-51 (6 channels)"
else
    echo "   ! Profile is not surround-51"
    echo "   Current: $PROFILE"
    echo "   Fixing..."
    CARD=$(pactl list cards short | grep ReSpeaker | awk '{print $1}')
    pactl set-card-profile $CARD output:iec958-stereo+input:analog-surround-51
    echo "   ✓ Profile set to surround-51"
fi
echo ""

# Check device
echo "3. Checking ReSpeaker device..."
if pactl list sources short | grep -q "analog-surround-51"; then
    CHANNELS=$(pactl list sources | grep -A 20 "surround-51" | grep "Sample Specification" | grep -o "[0-9]ch")
    echo "   ✓ Device available ($CHANNELS)"
else
    echo "   ✗ Device not found"
    exit 1
fi
echo ""

# Test parec
echo "4. Testing audio capture (2 seconds)..."
BYTES=$(timeout 2 parec --device=alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-51 --rate=16000 --channels=6 --format=s16le --raw 2>/dev/null | wc -c)
if [ "$BYTES" -gt 100000 ]; then
    echo "   ✓ Audio capture working (captured $BYTES bytes)"
else
    echo "   ✗ Audio capture failed (only $BYTES bytes)"
    exit 1
fi
echo ""

echo "================================"
echo "All checks passed!"
echo "================================"
echo ""
echo "Starting client..."
echo "Press Ctrl+C to stop"
echo ""

# Run client
conda run -n speech_env python s2s_client_respeaker_simple.py
