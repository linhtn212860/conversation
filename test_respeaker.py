#!/usr/bin/env python3
"""
Quick test script to verify ReSpeaker hardware
"""
import sounddevice as sd
import numpy as np
import time

DEVICE_INDEX = 8
CHANNELS = 6
SAMPLE_RATE = 16000
DURATION = 3

print("=" * 60)
print("ReSpeaker Hardware Test")
print("=" * 60)
print()

# Show device info
device_info = sd.query_devices(DEVICE_INDEX)
print(f"Device: {device_info['name']}")
print(f"Max Input Channels: {device_info['max_input_channels']}")
print(f"Max Output Channels: {device_info['max_output_channels']}")
print(f"Default Sample Rate: {device_info['default_samplerate']}")
print()

print(f"Testing {DURATION}s recording with {CHANNELS} channels...")
print("Say something!")
print()

try:
    # Record audio
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        device=DEVICE_INDEX
    )
    sd.wait()

    print("✓ Recording successful!")
    print()

    # Show statistics for each channel
    print("Channel statistics:")
    for ch in range(CHANNELS):
        channel_data = recording[:, ch]
        max_val = np.abs(channel_data).max()
        mean_val = np.abs(channel_data).mean()
        print(f"  Channel {ch}: max={max_val:5d}, mean={mean_val:6.1f}")

    print()
    print("✓ ReSpeaker test passed!")
    print()
    print("Recommended settings:")
    print("  - Use channel 0 (has hardware AEC)")
    print("  - dtype: 'int16'")
    print("  - samplerate: 16000")
    print("  - blocksize: 2048 or larger")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
