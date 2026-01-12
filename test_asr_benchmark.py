#!/usr/bin/env python3
"""
ASR Model Benchmark & Evaluation Script
Test and compare different ASR models with real audio samples
"""
import os
import sys
import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


class ASRBenchmark:
    """Benchmark different ASR models."""

    def __init__(self, device: str = "cuda"):
        """Initialize benchmark."""
        self.device = device
        self.results = {}

    def generate_test_audio(self, duration: float = 5.0, sr: int = 16000) -> np.ndarray:
        """
        Generate synthetic test audio with speech-like characteristics.

        Args:
            duration: Duration in seconds
            sr: Sample rate

        Returns:
            Audio array
        """
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)

        # Create speech-like pattern with multiple formants
        audio = np.zeros(samples, dtype=np.float32)

        # Add formants (typical for human speech)
        formants = [500, 1500, 2500]  # Hz
        for formant in formants:
            audio += 0.1 * np.sin(2 * np.pi * formant * t)

        # Add some amplitude modulation (like speech envelope)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
        audio *= envelope

        # Normalize
        audio = audio / np.abs(audio).max() * 0.5

        return audio

    def load_real_audio(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load real audio file.

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate

        Returns:
            (audio, sample_rate)
        """
        try:
            audio, sr = sf.read(file_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != target_sr:
                # Simple resampling
                import scipy.signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = scipy.signal.resample(audio, num_samples)

            # Ensure float32
            audio = audio.astype(np.float32)

            return audio, target_sr

        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None

    def benchmark_whisper(self, audio: np.ndarray, model_size: str = "large-v3-turbo") -> Dict:
        """
        Benchmark Whisper model.

        Args:
            audio: Audio array
            model_size: Model size (tiny, base, small, medium, large, large-v3-turbo)

        Returns:
            Results dict
        """
        print(f"\n{'='*60}")
        print(f"Testing Whisper {model_size}")
        print(f"{'='*60}")

        try:
            from faster_whisper import WhisperModel

            # Load model
            print("Loading model...")
            start_load = time.time()

            compute_type = "float16" if self.device == "cuda" else "int8"
            model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                download_root="pretrained_models/whisper"
            )

            load_time = time.time() - start_load
            print(f"✓ Load time: {load_time:.2f}s")

            # Warm-up run
            print("Warm-up run...")
            _ = model.transcribe(audio[:16000], language="en", beam_size=1)

            # Actual transcription
            print("Transcribing...")
            start_trans = time.time()

            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )

            # Collect segments
            text_parts = []
            segment_times = []
            for segment in segments:
                text_parts.append(segment.text)
                segment_times.append((segment.start, segment.end))

            trans_time = time.time() - start_trans
            text = " ".join(text_parts).strip()

            # Calculate metrics
            audio_duration = len(audio) / 16000
            rtf = trans_time / audio_duration  # Real-time factor

            results = {
                "model": f"Whisper {model_size}",
                "text": text,
                "load_time": load_time,
                "transcription_time": trans_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "rtfx": 1.0 / rtf if rtf > 0 else 0,
                "segments": len(text_parts),
                "segment_times": segment_times,
                "language": info.language if hasattr(info, 'language') else "en",
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else 1.0
            }

            # Print results
            print(f"\n📊 Results:")
            print(f"  Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            print(f"  Audio duration: {audio_duration:.2f}s")
            print(f"  Transcription time: {trans_time:.3f}s")
            print(f"  RTF: {rtf:.3f}x (lower is better)")
            print(f"  RTFx: {results['rtfx']:.1f}x faster than real-time")
            print(f"  Segments: {len(text_parts)}")
            if hasattr(info, 'language'):
                print(f"  Language: {info.language} ({info.language_probability:.2%})")

            return results

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"model": f"Whisper {model_size}", "error": str(e)}

    def benchmark_speechbrain(self, audio: np.ndarray) -> Dict:
        """
        Benchmark SpeechBrain Conformer model.

        Args:
            audio: Audio array

        Returns:
            Results dict
        """
        print(f"\n{'='*60}")
        print(f"Testing SpeechBrain Conformer")
        print(f"{'='*60}")

        try:
            # Apply torchaudio patch
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ["soundfile"]

            from speechbrain.inference.ASR import EncoderDecoderASR

            # Load model
            print("Loading model...")
            start_load = time.time()

            model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-conformer-transformerlm-librispeech",
                savedir="pretrained_models/speechbrain_asr-conformer-transformerlm-librispeech",
                run_opts={"device": self.device}
            )

            load_time = time.time() - start_load
            print(f"✓ Load time: {load_time:.2f}s")

            # Transcribe
            print("Transcribing...")
            start_trans = time.time()

            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            predictions, tokens = model.transcribe_batch(
                audio_tensor,
                torch.tensor([1.0])
            )

            trans_time = time.time() - start_trans
            text = predictions[0] if predictions else ""

            # Calculate metrics
            audio_duration = len(audio) / 16000
            rtf = trans_time / audio_duration

            results = {
                "model": "SpeechBrain Conformer",
                "text": text,
                "load_time": load_time,
                "transcription_time": trans_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "rtfx": 1.0 / rtf if rtf > 0 else 0
            }

            # Print results
            print(f"\n📊 Results:")
            print(f"  Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            print(f"  Audio duration: {audio_duration:.2f}s")
            print(f"  Transcription time: {trans_time:.3f}s")
            print(f"  RTF: {rtf:.3f}x")
            print(f"  RTFx: {results['rtfx']:.1f}x faster than real-time")

            return results

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"model": "SpeechBrain Conformer", "error": str(e)}

    def compare_models(self, audio: np.ndarray, ground_truth: str = None):
        """
        Compare multiple ASR models.

        Args:
            audio: Audio array
            ground_truth: Ground truth transcription (for WER calculation)
        """
        print("\n" + "="*80)
        print("ASR MODEL COMPARISON")
        print("="*80)

        results = []

        # Test Whisper models
        for model_size in ["large-v3-turbo", "medium", "small"]:
            result = self.benchmark_whisper(audio, model_size)
            results.append(result)
            time.sleep(1)  # Cool down

        # Test SpeechBrain
        # result = self.benchmark_speechbrain(audio)
        # results.append(result)

        # Summary comparison
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)

        print(f"\n{'Model':<30} {'Load Time':<12} {'Trans Time':<12} {'RTF':<10} {'RTFx':<10}")
        print("-" * 80)

        for result in results:
            if "error" not in result:
                print(f"{result['model']:<30} "
                      f"{result['load_time']:>10.2f}s "
                      f"{result['transcription_time']:>10.3f}s "
                      f"{result['rtf']:>8.3f}x "
                      f"{result['rtfx']:>8.1f}x")

        # Transcription comparison
        print(f"\n{'Model':<30} Transcription")
        print("-" * 80)
        for result in results:
            if "error" not in result and result['text']:
                text = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
                print(f"{result['model']:<30} {text}")

        # Calculate WER if ground truth provided
        if ground_truth:
            print("\n" + "="*80)
            print("WORD ERROR RATE (WER)")
            print("="*80)
            print(f"Ground truth: {ground_truth}")
            print()

            for result in results:
                if "error" not in result and result['text']:
                    wer = self.calculate_wer(ground_truth, result['text'])
                    print(f"{result['model']:<30} WER: {wer:.2%}")

        return results

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate.

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            WER as float (0.0 to 1.0)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Simple Levenshtein distance for words
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # deletion
                        d[i][j-1] + 1,    # insertion
                        d[i-1][j-1] + 1   # substitution
                    )

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
        return wer


def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(description="ASR Model Benchmark")
    parser.add_argument("--audio", type=str, help="Path to audio file (WAV)")
    parser.add_argument("--text", type=str, help="Ground truth transcription")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--duration", type=float, default=10.0, help="Test audio duration (if no file)")
    parser.add_argument("--model", type=str, default="all",
                       help="Model to test: whisper, speechbrain, or all")

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ASRBenchmark(device=args.device)

    # Load or generate audio
    if args.audio:
        print(f"Loading audio from: {args.audio}")
        audio, sr = benchmark.load_real_audio(args.audio)
        if audio is None:
            print("Failed to load audio file")
            return
    else:
        print(f"Generating test audio ({args.duration}s)...")
        audio = benchmark.generate_test_audio(duration=args.duration)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / 16000:.2f}s")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Run benchmark
    if args.model == "all":
        results = benchmark.compare_models(audio, ground_truth=args.text)
    elif args.model == "whisper":
        results = benchmark.benchmark_whisper(audio, "large-v3-turbo")
    elif args.model == "speechbrain":
        results = benchmark.benchmark_speechbrain(audio)
    else:
        print(f"Unknown model: {args.model}")
        return

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
