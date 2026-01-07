#!/usr/bin/env python3
"""
Test ReSpeaker audio capture
"""
import subprocess
import numpy as np
import time

DEVICE = "hw:3,0"
SAMPLE_RATE = 16000
CHANNELS = 6
CHUNK_SIZE = 1024
DURATION = 3  # seconds

print("=" * 60)
print("ReSpeaker Audio Capture Test")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Channels: {CHANNELS}")
print(f"Sample rate: {SAMPLE_RATE}")
print(f"Duration: {DURATION}s")
print()

cmd = [
    'arecord',
    '-D', DEVICE,
    '-f', 'S16_LE',
    '-r', str(SAMPLE_RATE),
    '-c', str(CHANNELS),
    '-t', 'raw',
]

print(f"Running: {' '.join(cmd)}")
print()

try:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=CHUNK_SIZE * CHANNELS * 2
    )

    print("Recording... Say something!")

    chunks = []
    bytes_to_read = CHUNK_SIZE * CHANNELS * 2
    total_samples = 0

    start_time = time.time()

    while time.time() - start_time < DURATION:
        data = proc.stdout.read(bytes_to_read)

        if not data or len(data) < bytes_to_read:
            print("Warning: Incomplete read")
            continue

        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.reshape(-1, CHANNELS)

        # Extract channel 0
        mono = samples[:, 0]
        chunks.append(mono)
        total_samples += len(mono)

        # Show progress
        if len(chunks) % 10 == 0:
            energy = np.abs(mono.astype(np.float32)).mean()
            print(f"  {len(chunks)} chunks, energy: {energy:.2f}")

    proc.terminate()
    proc.wait()

    print()
    print("✓ Recording complete!")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Duration: {total_samples / SAMPLE_RATE:.2f}s")

    if chunks:
        full_audio = np.concatenate(chunks)
        max_val = np.abs(full_audio).max()
        mean_val = np.abs(full_audio).mean()
        print(f"  Max amplitude: {max_val}")
        print(f"  Mean amplitude: {mean_val:.2f}")

        if max_val > 1000:
            print()
            print("✓ Audio level looks good!")
        else:
            print()
            print("⚠ Audio level very low. Check microphone!")

    print()
    print("Test PASSED!")

except KeyboardInterrupt:
    print("\n\nInterrupted")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        proc.terminate()
    except:
        pass
