#!/usr/bin/env python3
"""
Quick test script for English Voice Chatbot
Tests each component individually and the full pipeline
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from speech_pipeline.config import SAMPLE_RATE, WHISPER_MODEL_SIZE, EDGE_TTS_VOICE
from rich.console import Console

console = Console()


def test_imports():
    """Test if all required packages are installed."""
    console.print("\n[bold cyan]Testing imports...[/bold cyan]")

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'edge_tts': 'Edge-TTS',
        'soundfile': 'SoundFile',
        'numpy': 'NumPy',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            console.print(f"  ✓ {name}")
        except ImportError:
            console.print(f"  ✗ {name} [red](missing)[/red]")
            missing.append(package)

    if missing:
        console.print(f"\n[red]Missing packages: {', '.join(missing)}[/red]")
        console.print("Install with: pip install -r requirements_speech.txt")
        return False

    console.print("[green]✓ All imports successful[/green]")
    return True


def test_vad():
    """Test Voice Activity Detection."""
    console.print("\n[bold cyan]Testing VAD (Silero)...[/bold cyan]")

    try:
        from speech_pipeline.vad import SileroVAD

        vad = SileroVAD()

        # Silero VAD requires exactly 512 samples per chunk for 16kHz
        chunk_size = 512
        test_audio = np.random.randn(chunk_size).astype(np.float32) * 0.1

        # Use process_chunk method
        has_speech, segment = vad.process_chunk(test_audio)

        console.print(f"  Chunk size: {chunk_size} samples")
        console.print(f"  Speech detected: {has_speech}")
        if segment is not None:
            console.print(f"  Segment length: {len(segment)} samples")
        console.print("[green]✓ VAD working[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ VAD failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_asr():
    """Test ASR (Whisper)."""
    console.print("\n[bold cyan]Testing ASR (Whisper)...[/bold cyan]")
    console.print(f"  Model: whisper-{WHISPER_MODEL_SIZE}")

    try:
        from speech_pipeline.asr import WhisperASR

        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"  Device: {device}")

        asr = WhisperASR(device=device)

        # Test with silence (should return empty or noise)
        test_audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.01
        text = asr.transcribe(test_audio)

        console.print(f"  Transcription (random noise): '{text[:50]}...'")
        console.print("[green]✓ ASR working[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ ASR failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_tts():
    """Test TTS (Edge-TTS)."""
    console.print("\n[bold cyan]Testing TTS (Edge-TTS)...[/bold cyan]")
    console.print(f"  Voice: {EDGE_TTS_VOICE}")

    try:
        from speech_pipeline.tts import EnglishTTS
        import tempfile
        import os

        tts = EnglishTTS()

        if not tts.edge_tts.available:
            console.print("[red]✗ Edge-TTS not available[/red]")
            return False

        # Test synthesis
        test_text = "Hello, this is a test of the text to speech system."

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        success = tts.synthesize(test_text, temp_path)

        if success and os.path.exists(temp_path):
            size = os.path.getsize(temp_path)
            console.print(f"  Generated audio: {size} bytes")
            os.unlink(temp_path)
            console.print("[green]✓ TTS working[/green]")
            return True
        else:
            console.print("[red]✗ TTS synthesis failed[/red]")
            return False

    except Exception as e:
        console.print(f"[red]✗ TTS failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_llm():
    """Test LLM (Qwen3-8B)."""
    console.print("\n[bold cyan]Testing LLM (Qwen3-8B)...[/bold cyan]")

    try:
        from speech_pipeline.llm import QwenLLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"  Device: {device}")
        console.print("  [yellow]Loading model (this may take a minute)...[/yellow]")

        llm = QwenLLM(device=device)

        # Test generation
        test_prompt = "Hello! How are you?"
        console.print(f"  Prompt: '{test_prompt}'")

        response = llm.generate(test_prompt)

        console.print(f"  Response: '{response[:100]}...'")
        console.print("[green]✓ LLM working[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ LLM failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline."""
    console.print("\n[bold cyan]Testing Full Pipeline...[/bold cyan]")

    try:
        from speech_pipeline.pipeline import SpeechToSpeechPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = SpeechToSpeechPipeline(device=device)

        console.print("  [yellow]Loading all models (this may take 1-2 minutes)...[/yellow]")
        pipeline.load_all_models()

        console.print("[green]✓ Full pipeline working[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    console.print("[bold]English Voice Chatbot - Component Tests[/bold]")
    console.print("=" * 60)

    results = {}

    # Test imports first
    if not test_imports():
        console.print("\n[red]Cannot proceed without required packages[/red]")
        return

    # Test each component
    results['VAD'] = test_vad()
    results['ASR'] = test_asr()
    results['TTS'] = test_tts()
    results['LLM'] = test_llm()

    # Test full pipeline if all components work
    if all(results.values()):
        results['Pipeline'] = test_full_pipeline()

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Test Summary:[/bold]")

    for component, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {component}: {status}")

    all_passed = all(results.values())

    if all_passed:
        console.print("\n[bold green]All tests passed! Your English voice chatbot is ready.[/bold green]")
        console.print("\nNext steps:")
        console.print("  1. Run the server: python run_server.py")
        console.print("  2. Or test directly: python -m speech_pipeline.pipeline")
    else:
        console.print("\n[bold red]Some tests failed. Please check the errors above.[/bold red]")


if __name__ == "__main__":
    main()
