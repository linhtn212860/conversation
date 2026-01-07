"""
Download all models required for Speech-to-Speech pipeline
"""
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speech_pipeline.config import (
    MODELS_DIR,
    SILERO_VAD_REPO,
    CONFORMER_MODEL_NAME,
    VITS_MODEL_NAME
)


def download_silero_vad():
    """Download Silero VAD model."""
    print("=" * 50)
    print("📥 Downloading Silero VAD...")
    print("=" * 50)

    import torch

    # This will download and cache the model
    model, utils = torch.hub.load(
        repo_or_dir=SILERO_VAD_REPO,
        model='silero_vad',
        force_reload=False
    )

    print(f"✅ Silero VAD downloaded from: {SILERO_VAD_REPO}")


def download_conformer_asr():
    """Download NeMo Conformer ASR model."""
    print("=" * 50)
    print("📥 Downloading Conformer ASR...")
    print("=" * 50)

    try:
        import nemo.collections.asr as nemo_asr

        print(f"Model: {CONFORMER_MODEL_NAME}")

        # Download and cache the model
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name=CONFORMER_MODEL_NAME
        )

        print(f"✅ Conformer ASR downloaded: {CONFORMER_MODEL_NAME}")
        print(f"   Model size: ~{model.num_weights / 1e6:.1f}M parameters")

    except ImportError:
        print("❌ NeMo not installed!")
        print("   Install with: pip install nemo_toolkit[asr]")
        raise
    except Exception as e:
        print(f"❌ Failed to download Conformer: {e}")
        raise


def download_vits_tts():
    """Download VITS TTS model."""
    print("=" * 50)
    print("📥 Downloading VITS TTS...")
    print("=" * 50)

    try:
        from transformers import VitsModel, VitsTokenizer

        print(f"Model: {VITS_MODEL_NAME}")

        # Download model and tokenizer
        print("  Downloading model...")
        model = VitsModel.from_pretrained(VITS_MODEL_NAME)

        print("  Downloading tokenizer...")
        tokenizer = VitsTokenizer.from_pretrained(VITS_MODEL_NAME)

        print(f"✅ VITS TTS downloaded: {VITS_MODEL_NAME}")

    except ImportError:
        print("❌ Transformers not installed!")
        print("   Install with: pip install transformers")
        raise
    except Exception as e:
        print(f"❌ Failed to download VITS: {e}")
        raise


def main():
    """Download all models."""
    print("=" * 60)
    print("🔄 DOWNLOADING ALL MODELS FOR SPEECH-TO-SPEECH PIPELINE")
    print("=" * 60)
    print()

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download each model
    try:
        download_silero_vad()
        print()

        download_conformer_asr()
        print()

        download_vits_tts()
        print()

        print("=" * 60)
        print("✅ ALL MODELS DOWNLOADED!")
        print("=" * 60)
        print()
        print("Downloaded models:")
        print(f"  ✓ Silero VAD (from {SILERO_VAD_REPO})")
        print(f"  ✓ Conformer ASR ({CONFORMER_MODEL_NAME})")
        print(f"  ✓ VITS TTS ({VITS_MODEL_NAME})")
        print()
        print("Note:")
        print("  - Qwen3-8B model should already be at:")
        print(f"    {MODELS_DIR / 'Qwen_Qwen3-8B'}")
        print()
        print("To start the server, run:")
        print("  python run_server.py")

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ DOWNLOAD FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Please ensure you have installed all dependencies:")
        print("  pip install nemo_toolkit[asr] transformers torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
