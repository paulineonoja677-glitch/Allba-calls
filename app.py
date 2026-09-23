import io
import os
import wave
from pathlib import Path

from flask import Flask, request, send_file, jsonify
from piper import PiperVoice

app = Flask(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
DEFAULT_MODEL = os.environ.get("PIPER_MODEL", "en_US-lessac-medium.onnx")

_voice_cache = {}


def get_voice(model_name: str) -> PiperVoice:
    """Load a Piper voice model, caching it after first load."""
    if model_name in _voice_cache:
        return _voice_cache[model_name]

    model_path = MODEL_DIR / model_name
    config_path = MODEL_DIR / f"{model_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    _voice_cache[model_name] = voice
    return voice


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "piper-tts-server",
            "endpoints": {
                "/health": "GET - health check",
                "/voices": "GET - list available voice models",
                "/synthesize": "POST {text, voice?, speaker?} - returns WAV audio",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/voices", methods=["GET"])
def voices():
    if not MODEL_DIR.exists():
        return jsonify({"voices": []})
    models = sorted(p.name for p in MODEL_DIR.glob("*.onnx"))
    return jsonify({"voices": models, "default": DEFAULT_MODEL})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True) or request.form

    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    model_name = data.get("voice") or DEFAULT_MODEL
    speaker_id = data.get("speaker")
    if speaker_id is not None:
        try:
            speaker_id = int(speaker_id)
        except (TypeError, ValueError):
            speaker_id = None

    try:
        voice = get_voice(model_name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        synth_kwargs = {}
        if speaker_id is not None:
            synth_kwargs["speaker_id"] = speaker_id
        voice.synthesize(text, wav_file, **synth_kwargs)

    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="speech.wav",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
