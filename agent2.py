import os
import io
import re
import sys
import csv
import time
import wave
import random
import base64
import winsound
import httpx
import numpy as np
import sounddevice as sd
from datetime import datetime
from scipy.io.wavfile import write
from dotenv import load_dotenv
from sarvamai import SarvamAI
from groq import Groq

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env.local")

SARVAM_API_KEY = os.environ["SARVAM_API_KEY"]
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "").strip()
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "").strip()
SARVAM_VOICE_ID = os.environ.get("SARVAM_VOICE_ID", "").strip()   # optional Sarvam clone

sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ---------- Config (Telugu only) ----------
LANGUAGE_CODE = "te-IN"
STT_MODEL = "saarika:v2.5"

LLM_PROVIDER = "sarvam"                         # "sarvam" or "groq"
SARVAM_LLM_MODEL = "sarvam-105b-conversations"
GROQ_LLM_MODEL = "openai/gpt-oss-120b"
LLM_MAX_TOKENS = 200

# TTS engine: "cartesia" (your Cartesia clone) | "sarvam_clone" | "sarvam" (default voice)
TTS_PROVIDER = "cartesia"

CARTESIA_MODEL = "sonic-3.5"        # change to "sonic-3" if you get a language/model error
CARTESIA_VERSION = "2026-03-01"
CARTESIA_LANGUAGE = "te"
CARTESIA_SAMPLE_RATE = 24000

# Default Sarvam voice (also used as fallback)
TTS_MODEL = "bulbul:v3"
TTS_SPEAKER = "shubh"
TTS_SAMPLE_RATE = 24000
TTS_PACE = 1.1
TTS_MAX_CHARS = 1000
CLONE_MAX_CHARS = 300

DEFAULT_CITY = "Visakhapatnam"

RECORD_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10
SILENCE_SECONDS = 0.8
NO_SPEECH_TIMEOUT = 4
SILENCE_THRESHOLD = 500

INPUT_AUDIO = "input.wav"
OUTPUT_AUDIO = "reply.wav"
LOG_FILE = "log.txt"

MAX_LLM_RETRIES = 2
MIN_TELUGU_RATIO = 0.7
FALLBACK_REPLY = "క్షమించండి, దయచేసి మళ్లీ చెప్పండి."
WEATHER_ERROR_REPLY = "క్షమించండి, వాతావరణ సమాచారం ఇప్పుడు అందుబాటులో లేదు."

SYSTEM_PROMPT = (
    "మీరు ఒక స్నేహపూర్వకమైన వాయిస్ అసిస్టెంట్. "
    "వినియోగదారు అడిగిన ప్రశ్నకు మీకు తెలిసిన సమాచారంతో నేరుగా సమాధానం ఇవ్వండి. "
    "సమాధానం ఒకటి లేదా రెండు చిన్న వాక్యాల్లో ఉండాలి. పరిచయాలు, అనవసర వివరాలు వద్దు. "
    "ఎల్లప్పుడూ తెలుగు లిపిలో మాత్రమే సమాధానం ఇవ్వండి; కన్నడ, హిందీ, ఇంగ్లీష్ అక్షరాలు వాడకండి. "
    "సాధారణ జ్ఞానం, వివరణలు, సలహాలు, గణితం, అనువాదం వంటి ప్రశ్నలకు తప్పకుండా సమాధానం ఇవ్వండి. "
    "ఈ రోజు వార్తలు, లైవ్ స్కోర్లు, ధరల వంటి ప్రత్యక్ష సమాచారం అవసరమైన ప్రశ్నలకు మాత్రమే "
    "'ప్రస్తుత సమాచారం నా దగ్గర లేదు' అని చెప్పండి. "
    "మార్క్‌డౌన్, బుల్లెట్లు, ఎమోజీలు వాడకండి."
)

# ---------- Cost meter settings ----------
USD_INR = 90.0                 # update to the current exchange rate
TARGET_PRICE_PER_MIN = 1.0     # Rs you plan to charge per minute
COST_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_log.csv")

STT_INR_PER_HOUR = 30.0        # Sarvam speech-to-text

# Rs per 1000 characters, by the TTS voice that actually produced the audio
TTS_INR_PER_1000_CHARS = {
    "cartesia": 49.0 / 1250.0 * USD_INR,   # Startup plan: $49 per 1.25M credits (about Rs 3.53 / 1000 chars)
    "sarvam": 3.0,                         # Bulbul v3
    "sarvam_clone": 3.0,                   # ASSUMPTION: check your Sarvam plan
}

# Rs per 1M tokens: (input, output). Change if you use a different model.
LLM_INR_PER_M_TOKENS = {
    "sarvam": (29.28, 73.2),                       # sarvam-105b
    "groq": (0.15 * USD_INR, 0.75 * USD_INR),      # gpt-oss-120b
}
FALLBACK_LLM_TOKENS = (800, 120)   # used only if the API returns no token usage

_turn = {}
_session = {"inr": 0.0, "conv_seconds": 0.0, "turns": 0}


# ---------- Cost meter ----------
def meter_reset():
    _turn.clear()
    _turn.update({
        "llm_calls": 0,
        "llm_in": 0,
        "llm_out": 0,
        "llm_usage_missing": False,
        "tts_provider": TTS_PROVIDER,
    })


def meter_add_llm(resp):
    usage = getattr(resp, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None)
    tokens_out = getattr(usage, "completion_tokens", None)
    if tokens_in is None or tokens_out is None:
        _turn["llm_usage_missing"] = True
        return
    _turn["llm_in"] += tokens_in
    _turn["llm_out"] += tokens_out


def wav_seconds(source) -> float:
    """Duration of a WAV file path or WAV bytes, in seconds."""
    try:
        f = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        with wave.open(f, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def meter_report(user_secs: float, agent_secs: float, speech_text: str):
    # STT
    stt_inr = user_secs / 3600.0 * STT_INR_PER_HOUR

    # LLM (zero on weather turns, since no LLM call is made)
    llm_in, llm_out = _turn["llm_in"], _turn["llm_out"]
    note = ""
    if _turn["llm_calls"] and _turn["llm_usage_missing"] and not (llm_in or llm_out):
        llm_in, llm_out = FALLBACK_LLM_TOKENS
        note = ", estimated"
    rate_in, rate_out = LLM_INR_PER_M_TOKENS.get(LLM_PROVIDER, (0.0, 0.0))
    llm_inr = (llm_in * rate_in + llm_out * rate_out) / 1_000_000.0

    # TTS (priced by the voice that actually produced the audio)
    provider = _turn["tts_provider"]
    chars = len(speech_text)
    tts_inr = chars / 1000.0 * TTS_INR_PER_1000_CHARS.get(provider, 3.0)

    turn_inr = stt_inr + llm_inr + tts_inr
    _session["inr"] += turn_inr
    _session["conv_seconds"] += user_secs + agent_secs
    _session["turns"] += 1

    conv_min = _session["conv_seconds"] / 60.0
    per_min = _session["inr"] / conv_min if conv_min > 0 else 0.0
    status = "OVER target" if per_min > TARGET_PRICE_PER_MIN else "within target"

    print(
        f"[Cost] STT Rs{stt_inr:.3f} ({user_secs:.1f}s) | "
        f"LLM Rs{llm_inr:.3f} ({llm_in} in/{llm_out} out{note}) | "
        f"TTS Rs{tts_inr:.3f} ({chars} chars, {provider}) | Turn Rs{turn_inr:.3f}"
    )
    print(
        f"[Cost] Session Rs{_session['inr']:.2f} over {conv_min:.2f} min of speech "
        f"= Rs{per_min:.2f}/min (target Rs{TARGET_PRICE_PER_MIN:.2f}) -> {status}"
    )
    if agent_secs > 0 and chars > 0:
        print(f"[Calib] {chars / agent_secs:.1f} characters per second of agent speech")

    try:
        new_file = not os.path.exists(COST_LOG)
        with open(COST_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "time", "user_secs", "agent_secs", "stt_inr", "llm_provider",
                    "llm_in", "llm_out", "llm_inr", "tts_provider", "tts_chars",
                    "tts_inr", "turn_inr",
                ])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(user_secs, 2), round(agent_secs, 2), round(stt_inr, 4), LLM_PROVIDER,
                llm_in, llm_out, round(llm_inr, 4), provider, chars,
                round(tts_inr, 4), round(turn_inr, 4),
            ])
    except PermissionError:
        print("[Cost] cost_log.csv is open in another program, so this turn was not saved. Close it and continue.")


# ---------- 0. Recording (auto-stop) and playback ----------
def record(path: str = INPUT_AUDIO, fs: int = RECORD_SAMPLE_RATE):
    block = int(fs * 0.1)
    max_blocks = int(MAX_RECORD_SECONDS / 0.1)
    silence_limit = int(SILENCE_SECONDS / 0.1)
    no_speech_limit = int(NO_SPEECH_TIMEOUT / 0.1)

    frames = []
    started = False
    silent_blocks = 0

    print("Speak now...")
    with sd.InputStream(samplerate=fs, channels=1, dtype="int16", blocksize=block) as stream:
        for i in range(max_blocks):
            data, _overflow = stream.read(block)
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

            if rms > SILENCE_THRESHOLD:
                started = True
                silent_blocks = 0
            elif started:
                silent_blocks += 1
                if silent_blocks >= silence_limit:
                    break
            elif i >= no_speech_limit:
                break

    audio = np.concatenate(frames)
    write(path, fs, audio)
    print("Recording done.")


def play_audio(path: str):
    """Play a 16-bit WAV through sounddevice; fall back to winsound."""
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if sw != 2:
            raise ValueError(f"unsupported sample width {sw}")
        data = np.frombuffer(frames, dtype=np.int16)
        if ch > 1:
            data = data.reshape(-1, ch)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"sounddevice playback failed ({e}); trying winsound...")
        winsound.PlaySound(path, winsound.SND_FILENAME)


# ---------- 1. STT (Sarvam, Telugu) ----------
def speech_to_text(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = sarvam.speech_to_text.transcribe(
            file=f,
            model=STT_MODEL,
            language_code=LANGUAGE_CODE,
        )
    return resp.transcript


# ---------- 2a. Weather (real forecast, no LLM) ----------
WEATHER_KEYWORDS = ["వాతావరణ", "వెదర్", "వర్షం", "వర్షాలు", "ఉష్ణోగ్రత", "ఎండ"]

CITY_ALIASES = {
    "విశాఖ": "Visakhapatnam",
    "వైజాగ్": "Visakhapatnam",
    "హైదరాబాద్": "Hyderabad",
    "విజయవాడ": "Vijayawada",
    "గుంటూరు": "Guntur",
    "తిరుపతి": "Tirupati",
    "కాకినాడ": "Kakinada",
    "రాజమండ్రి": "Rajahmundry",
    "వరంగల్": "Warangal",
    "బెంగళూరు": "Bengaluru",
    "చెన్నై": "Chennai",
    "ముంబై": "Mumbai",
    "ఢిల్లీ": "Delhi",
}

_geo_cache = {}


def is_weather_question(text: str) -> bool:
    return any(k in text for k in WEATHER_KEYWORDS)


def detect_day_offset(text: str) -> int:
    if "ఎల్లుండి" in text:
        return 2
    if "రేపు" in text:
        return 1
    return 0


def detect_city(text: str) -> str:
    for te_name, en_name in CITY_ALIASES.items():
        if te_name in text:
            return en_name
    return DEFAULT_CITY


def _geocode(city_en: str):
    if city_en in _geo_cache:
        return _geo_cache[city_en]
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_en, "count": 1, "language": "te", "format": "json"},
        timeout=5,
    )
    results = r.json().get("results")
    if not results:
        raise ValueError(f"City not found: {city_en}")
    top = results[0]
    info = (top["latitude"], top["longitude"], top.get("name", city_en))
    _geo_cache[city_en] = info
    return info


def _weather_desc(code: int) -> str:
    if code == 0:
        return "ఆకాశం నిర్మలంగా ఉంటుంది"
    if code in (1, 2):
        return "పాక్షికంగా మేఘావృతంగా ఉంటుంది"
    if code == 3:
        return "మేఘావృతంగా ఉంటుంది"
    if code in (45, 48):
        return "పొగమంచు కమ్ముకుని ఉంటుంది"
    if code in (51, 53, 55, 56, 57):
        return "తేలికపాటి చినుకులు పడతాయి"
    if code in (61, 63, 66, 67):
        return "వర్షం పడుతుంది"
    if code == 65:
        return "భారీ వర్షం పడుతుంది"
    if code in (71, 73, 75, 77, 85, 86):
        return "మంచు కురుస్తుంది"
    if code in (80, 81):
        return "జల్లులు పడతాయి"
    if code == 82:
        return "భారీ జల్లులు పడతాయి"
    if code in (95, 96, 99):
        return "ఉరుములతో కూడిన వర్షం పడుతుంది"
    return "వాతావరణం మారుతూ ఉంటుంది"


def get_weather_reply(user_text: str) -> str:
    day = detect_day_offset(user_text)
    city_en = detect_city(user_text)
    try:
        lat, lon, place = _geocode(city_en)
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=5,
        )
        daily = r.json()["daily"]
        code = int(daily["weather_code"][day])
        tmax = round(daily["temperature_2m_max"][day])
        tmin = round(daily["temperature_2m_min"][day])
        rain = daily["precipitation_probability_max"][day]
    except Exception as e:
        print(f"Weather error: {e}")
        return WEATHER_ERROR_REPLY

    day_word = {0: "ఈ రోజు", 1: "రేపు", 2: "ఎల్లుండి"}[day]
    reply = (
        f"{day_word} {place}లో {_weather_desc(code)}. "
        f"ఉష్ణోగ్రత {tmin} నుండి {tmax} డిగ్రీల సెల్సియస్ మధ్య ఉంటుంది."
    )
    if rain is not None:
        reply += f" వర్షం పడే అవకాశం {round(rain)} శాతం."
    return reply


# ---------- 2b. LLM (Sarvam or Groq) for everything else ----------
def telugu_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    telugu = [c for c in letters if "\u0c00" <= c <= "\u0c7f"]
    return len(telugu) / len(letters)


def _call_llm(messages: list) -> str:
    _turn["llm_calls"] = _turn.get("llm_calls", 0) + 1

    if LLM_PROVIDER == "sarvam":
        resp = sarvam.chat.completions(
            model=SARVAM_LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            reasoning_effort=None,
        )
    else:
        resp = groq_client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=400,
            reasoning_effort="low",
        )

    meter_add_llm(resp)
    return (resp.choices[0].message.content or "").strip()


def get_llm_reply(user_text: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_text})

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            reply = _call_llm(messages)
        except Exception as e:
            print(f"LLM error (attempt {attempt}): {e}")
            continue

        ratio = telugu_ratio(reply)
        if reply and ratio >= MIN_TELUGU_RATIO:
            return reply
        print(f"Reply was not clean Telugu (ratio {ratio:.2f}), retrying ({attempt}/{MAX_LLM_RETRIES})...")

    return FALLBACK_REPLY


def get_reply(user_text: str, history: list | None = None) -> str:
    if is_weather_question(user_text):
        return get_weather_reply(user_text)
    return get_llm_reply(user_text, history)


# ---------- Human-like fillers ----------
FILLER_START = ["ఆ...", "అంటే...", "చూడండి,", "సరే,", "అవునండి,"]
FILLER_MID = ["అలాగే,", "ఇంకా,"]
FILLER_START_PROB = 0.6   # chance of a filler at the start (0 = never, 1 = always)
FILLER_MID_PROB = 0.5     # chance of a filler between sentences

_last_filler = None


def _pick_filler(options: list[str]) -> str:
    """Pick a random filler, avoiding the one used just before."""
    global _last_filler
    choices = [f for f in options if f != _last_filler] or options
    filler = random.choice(choices)
    _last_filler = filler
    return filler


def humanize(text: str) -> str:
    """Add natural Telugu filler words so the voice sounds less robotic."""
    if text in (FALLBACK_REPLY, WEATHER_ERROR_REPLY):
        return text

    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    parts = []

    if random.random() < FILLER_START_PROB:
        parts.append(_pick_filler(FILLER_START))

    for i, s in enumerate(sentences):
        if i > 0 and random.random() < FILLER_MID_PROB:
            parts.append(_pick_filler(FILLER_MID))
        parts.append(s)

    return " ".join(parts)


# ---------- 3. TTS ----------
def _split_text(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(current) + len(s) + 1 <= max_chars:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _merge_wavs(wav_bytes_list: list[bytes]) -> bytes:
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as out_wav:
        for i, data in enumerate(wav_bytes_list):
            with wave.open(io.BytesIO(data), "rb") as w:
                if i == 0:
                    out_wav.setparams(w.getparams())
                out_wav.writeframes(w.readframes(w.getnframes()))
    return out.getvalue()


def _rebuild_wav(raw: bytes, sample_rate: int) -> bytes:
    """Take WAV bytes with a possibly odd header and rewrap the PCM in a clean 16-bit mono WAV."""
    idx = raw.find(b"data")
    if idx == -1:
        raise RuntimeError("No 'data' chunk found in audio")
    pcm = raw[idx + 8:]
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if len(pcm) == 0:
        raise RuntimeError("Audio data is empty")

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out.getvalue()


def _tts_sarvam_default(text: str, pace: float) -> bytes:
    wavs = []
    for chunk in _split_text(text, TTS_MAX_CHARS):
        resp = sarvam.text_to_speech.convert(
            text=chunk,
            language_code=LANGUAGE_CODE,
            speaker=TTS_SPEAKER,
            model=TTS_MODEL,
            pace=pace,
            speech_sample_rate=TTS_SAMPLE_RATE,
        )
        wavs.append(base64.b64decode(resp.audios[0]))
    return _merge_wavs(wavs)


def _tts_sarvam_clone(text: str) -> bytes:
    if not SARVAM_VOICE_ID:
        raise RuntimeError("SARVAM_VOICE_ID is not set in .env.local")
    wavs = []
    for chunk in _split_text(text, CLONE_MAX_CHARS):
        r = httpx.post(
            "https://api.sarvam.ai/voices/clone",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={
                "voice_id": (None, SARVAM_VOICE_ID),
                "text": (None, chunk),
                "language_code": (None, LANGUAGE_CODE),
            },
            timeout=60,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        if isinstance(d.get("data"), dict):
            d = d["data"]
        b64 = d.get("audio") or "".join(d.get("audios", []))
        if not b64:
            raise RuntimeError(f"No audio in response: {str(d)[:200]}")
        audio = base64.b64decode(b64)
        if not audio.startswith(b"RIFF"):
            raise RuntimeError("Cloned voice audio is not WAV format")
        wavs.append(audio)
    return _merge_wavs(wavs)


def _tts_cartesia(text: str) -> bytes:
    if not CARTESIA_API_KEY or not CARTESIA_VOICE_ID:
        raise RuntimeError("CARTESIA_API_KEY or CARTESIA_VOICE_ID is missing in .env.local")

    r = httpx.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "Cartesia-Version": CARTESIA_VERSION,
            "X-API-Key": CARTESIA_API_KEY,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "language": CARTESIA_LANGUAGE,
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": CARTESIA_SAMPLE_RATE,
            },
        },
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    raw = r.content
    if len(raw) < 100:
        raise RuntimeError(f"Response too small to be audio: {raw[:100]!r}")
    if not raw.startswith(b"RIFF"):
        raise RuntimeError(f"Response is not a WAV file: {raw[:100]!r}")

    return _rebuild_wav(raw, CARTESIA_SAMPLE_RATE)


def text_to_speech(text: str, pace: float = TTS_PACE) -> bytes:
    try:
        if TTS_PROVIDER == "cartesia":
            audio = _tts_cartesia(text)
            _turn["tts_provider"] = "cartesia"
            return audio
        if TTS_PROVIDER == "sarvam_clone":
            audio = _tts_sarvam_clone(text)
            _turn["tts_provider"] = "sarvam_clone"
            return audio
    except Exception as e:
        print(f"{TTS_PROVIDER} voice failed ({e}). Using default Sarvam voice.")

    _turn["tts_provider"] = "sarvam"
    return _tts_sarvam_default(text, pace)


# ---------- Full pipeline ----------
def run_pipeline(input_audio: str = INPUT_AUDIO, output_audio: str = OUTPUT_AUDIO, history: list | None = None):
    meter_reset()
    t0 = time.time()
    user_secs = wav_seconds(input_audio)

    transcript = speech_to_text(input_audio)
    t_stt = time.time()
    print(f"User: {transcript}")

    if not transcript or not transcript.strip():
        print("Could not hear anything. Please try again.")
        meter_report(user_secs, 0.0, "")      # the STT call was still billed
        return None, None

    reply = get_reply(transcript, history)
    t_llm = time.time()
    print(f"Assistant: {reply}")

    speech_text = humanize(reply)             # reply + fillers, used only for speaking
    print(f"Spoken: {speech_text}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {transcript}\nAssistant: {reply}\nSpoken: {speech_text}\n\n")

    audio = text_to_speech(speech_text)
    with open(output_audio, "wb") as f:
        f.write(audio)
    t_tts = time.time()

    print(
        f"[Timing] STT {t_stt - t0:.1f}s | Reply {t_llm - t_stt:.1f}s | "
        f"TTS {t_tts - t_llm:.1f}s | Total before audio plays {t_tts - t0:.1f}s"
    )

    agent_secs = wav_seconds(audio)
    meter_report(user_secs, agent_secs, speech_text)

    play_audio(output_audio)
    return transcript, reply                  # history keeps the clean reply, without fillers


# ---------- Main loop ----------
if __name__ == "__main__":
    print(f"TTS provider: {TTS_PROVIDER} | LLM provider: {LLM_PROVIDER}")
    history = []
    while True:
        cmd = input("\nPress Enter to speak (q to quit): ")
        if cmd.strip().lower() == "q":
            break

        record()
        transcript, reply = run_pipeline(history=history)

        # Don't store failures or "I don't know" replies, or the model starts copying them
        if (
            transcript
            and reply
            and reply != FALLBACK_REPLY
            and "తెలియదు" not in reply
            and "నా దగ్గర లేదు" not in reply
        ):
            history += [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": reply},
            ]
            history = history[-6:]