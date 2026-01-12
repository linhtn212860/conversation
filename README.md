# Speech-to-Speech (S2S) Pipeline (End-to-End)

This repo runs a real-time-ish **speech → text → LLM → speech** pipeline exposed over **WebSocket (FastAPI + Uvicorn)**.

Core entry points:
- Server: `run_server.py`
- Client (mic + speaker): `s2s_client.py`
- Pipeline implementation: `speech_pipeline/`

---

## 1) End-to-end architecture

**High-level flow (default client):**

1. **Mic capture (16 kHz)** in the client.
2. **Client-side VAD** (Silero VAD via `torch.hub`) segments speech.
3. When a speech segment ends, client sends the **full utterance** to the server over WebSocket.
4. Server runs:
   - **ASR** (Whisper via `faster-whisper`) → transcription
   - **LLM** (Qwen3-8B from local folder) → response text
   - **TTS** (Kokoro ONNX) → response audio (24 kHz)
5. Server streams back:
   - transcription text
   - response text
   - response audio (PCM int16, base64)

---

## 2) Key modules

### `speech_pipeline/pipeline.py`
Implements the orchestration layer `SpeechToSpeechPipeline`.

- `process_audio_chunk(audio_chunk)`: feeds chunk to **server-side** VAD (`VADIterator`). When a speech segment completes it calls `process_speech(...)`.
- `process_speech(audio)`: non-streaming pipeline
  - ASR `transcribe(...)`
  - LLM `generate(...)`
  - TTS `synthesize(text, wav_path)`
  - returns a `PipelineResult` with transcription, response text, and full audio.
- `process_speech_streaming(audio)`: “semi-streaming” output
  - ASR first
  - LLM token streaming (`generate_streaming`)
  - TTS per sentence: whenever a sentence boundary is detected, synthesize that sentence and yield an audio chunk.

### `speech_pipeline/websocket_server.py`
WebSocket API + connection management.

- WebSocket endpoint: `ws://<host>:<port>/ws`
- Two audio input modes:
  - **`audio`**: one complete utterance (client already segmented speech). Server uses `process_speech_streaming` and returns `response_chunk` + audio chunks.
  - **`audio_chunk`**: streaming chunks (server segments speech using its own VAD). When the server detects end-of-utterance it runs `process_speech` and returns one response.

### `speech_pipeline/asr.py`
ASR wrapper `WhisperASR` (aliased as `ConformerASR` for backward compatibility).

- Uses `faster_whisper.WhisperModel`
- Downloads/caches whisper weights under `pretrained_models/whisper`

### `speech_pipeline/llm.py`
LLM wrapper `QwenLLM`.

- Loads tokenizer + model from `models/Qwen_Qwen3-8B`
- Optional 4-bit quantization via `bitsandbytes`
- Supports:
  - non-streaming: `generate(prompt)`
  - streaming: `generate_streaming(prompt)`
- Optional web search augmentation (DuckDuckGo) if `ddgs` / `duckduckgo_search` is installed.

### `speech_pipeline/tts.py`
TTS wrapper `EnglishTTS` using Kokoro ONNX.

- Expects local files:
  - `models/kokoro/kokoro-v1.0.onnx`
  - `models/kokoro/voices-v1.0.bin`
- Produces 24 kHz audio.

### `run_server.py`
Thin CLI entrypoint.

- `python run_server.py` starts the FastAPI server.
- `--preload` optionally loads all models at startup (otherwise first connection triggers lazy load).

### `s2s_client.py`
Interactive client:

- Records microphone audio at **16 kHz** (`sounddevice`).
- Uses Silero VAD locally to segment speech.
- Sends full utterance to server (`type: "audio"`).
- Plays server audio responses and **suppresses mic** while playback is active to reduce feedback.

---

## 3) WebSocket protocol (what the server expects)

Connect to:

- `ws://localhost:8765/ws`

### Client → Server messages

- **Full utterance (most common, used by `s2s_client.py`)**

```json
{ "type": "audio", "data": "<base64 PCM int16>", "sample_rate": 16000 }
```

- **Streaming chunks (alternative)**

```json
{ "type": "audio_chunk", "data": "<base64 PCM int16>", "sample_rate": 16000 }
```

- **Text-only**

```json
{ "type": "text", "text": "Hello" }
```

- **Reset / ping**

```json
{ "type": "reset" }
```

```json
{ "type": "ping" }
```

### Server → Client messages

- On connect:

```json
{ "type": "connected", "message": "...", "sample_rate": 16000 }
```

- Transcription:

```json
{ "type": "transcription", "text": "..." }
```

- Response text:

```json
{ "type": "response", "text": "..." }
```

- Streaming response chunks (when using `audio` mode):

```json
{ "type": "response_chunk", "text": "..." }
```

- Audio:

```json
{ "type": "audio", "data": "<base64 PCM int16>", "sample_rate": 24000 }
```

- Errors / done:

```json
{ "type": "error", "message": "..." }
```

```json
{ "type": "processing_complete", "processing_time_ms": 1234.5 }
```

---

## 4) Quick start

### 4.1 Install deps

Recommended (create a venv first):

```bash
pip install -r requirements_speech.txt
pip install sounddevice rich kokoro-onnx
```

Notes:
- `sounddevice` is required for the **client**.
- `kokoro-onnx` is required for the **TTS** implementation in `speech_pipeline/tts.py`.

### 4.2 Provide models

This repo expects local model files (these are often large and may be gitignored):

- Qwen model folder:
  - `models/Qwen_Qwen3-8B/` (HuggingFace-format directory)
- Kokoro ONNX files:
  - `models/kokoro/kokoro-v1.0.onnx`
  - `models/kokoro/voices-v1.0.bin`

Whisper weights are downloaded automatically at runtime into:
- `pretrained_models/whisper/`

### 4.3 Run server

```bash
python run_server.py --host 0.0.0.0 --port 8765
```

Optional preload (slower startup, faster first request):

```bash
python run_server.py --preload
```

### 4.4 Run client

```bash
python s2s_client.py --url ws://localhost:8765/ws
```

---

## 5) Runtime notes / troubleshooting

- **Sample rates**:
  - Client mic input: 16 kHz
  - Server TTS output: 24 kHz
- **GPU vs CPU**:
  - Server defaults to `device="cuda"` in the pipeline. If you want CPU-only, you’ll need to adjust where `SpeechToSpeechPipeline(device="cuda")` is constructed.
- **First run can be slow** due to model downloads / cache warmup.
- If you see import errors:
  - Whisper: `pip install faster-whisper`
  - WebSocket server: `pip install fastapi uvicorn websockets`
  - TTS: `pip install kokoro-onnx`

