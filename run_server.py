#!/usr/bin/env python3
"""
Run Speech-to-Speech Server
Entry point for the WebSocket API
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from speech_pipeline.websocket_server import run_server, set_pipeline
from speech_pipeline.config import WS_HOST, WS_PORT


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Speech WebSocket Server")
    parser.add_argument("--host", default=WS_HOST, help=f"Host to bind (default: {WS_HOST})")
    parser.add_argument("--port", type=int, default=WS_PORT, help=f"Port to bind (default: {WS_PORT})")
    parser.add_argument("--preload", action="store_true", help="Preload all models on startup")
    
    args = parser.parse_args()
    
    if args.preload:
        print("Preloading models...")
        from speech_pipeline.pipeline import SpeechToSpeechPipeline
        preloaded = SpeechToSpeechPipeline(device="cuda")
        preloaded.load_all_models()
        set_pipeline(preloaded)  # Share with server
    
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
