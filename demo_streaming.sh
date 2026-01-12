#!/bin/bash
# Quick demo of ASR streaming feature

echo "========================================"
echo "  ASR Streaming Demo"
echo "========================================"
echo ""
echo "This will test Whisper with streaming output."
echo "Speak into your microphone and watch words appear!"
echo ""
echo "Press Ctrl+C to stop and see summary."
echo ""
read -p "Press Enter to start..."

python test_asr_realtime.py --model whisper
