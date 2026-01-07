#!/bin/bash
# Quick launcher for ReSpeaker client
# ReSpeaker 4 Mic Array via PipeWire (5.1 surround profile)

echo "Starting ReSpeaker S2S Client..."
echo "Device: ReSpeaker 4 Mic Array (6 channels via PipeWire)"
echo ""

# Run with correct conda environment
conda run -n speech_env python s2s_client_respeaker_simple.py "$@"
