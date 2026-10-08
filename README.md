# Telugu Voice Assistant

This project is a Windows desktop voice assistant for Telugu conversations. It records speech from the microphone, converts it to text with Sarvam AI, generates a short Telugu response with Sarvam AI or Groq, converts the response back to speech, and plays it through the speakers.

Weather questions are handled separately using the Open-Meteo geocoding and forecast APIs. The assistant also keeps a short conversation history and records timing, estimated usage, and cost information for each turn.

## Features

- Telugu speech-to-text using Sarvam AI (`saarika:v2.5`)
- Telugu responses using Sarvam AI or Groq
- Text-to-speech using Cartesia, Sarvam default voice, or Sarvam cloned voice
- Weather forecasts for supported cities, including today, tomorrow, and the day after tomorrow
- Microphone recording with automatic silence detection
- WAV audio playback on Windows
- Conversation log in `log.txt`
- Usage and estimated cost log in `cost_log.csv`

## Requirements

- Windows
- Python 3.10 or newer
- A working microphone and speakers
- API keys for:
  - Sarvam AI
  - Groq
  - Cartesia, when using the default Cartesia text-to-speech configuration

The project uses these Python packages:

- `httpx`
- `numpy`
- `scipy`
- `sounddevice`
- `python-dotenv`
- `sarvamai`
- `groq`

## Setup

Open PowerShell in the project directory:

```powershell
cd "D:\Outpero\Outpero ScreenShots\ai-voice-agent-2\ai-voice-agent-2"
```

Create and activate a virtual environment if you do not already have one:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install httpx numpy scipy sounddevice python-dotenv sarvamai groq
```

If PowerShell blocks script activation, activate the environment for the current terminal session with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Environment variables

Create a file named `.env.local` in the project directory:

```env
SARVAM_API_KEY=your_sarvam_api_key
GROQ_API_KEY=your_groq_api_key

# Required when TTS_PROVIDER is "cartesia"
CARTESIA_API_KEY=your_cartesia_api_key
CARTESIA_VOICE_ID=your_cartesia_voice_id

# Optional when using a Sarvam cloned voice
SARVAM_VOICE_ID=your_sarvam_voice_id
```

Do not commit `.env.local` or share its contents. If real API keys have been placed in a local copy of this project, rotate them in the provider dashboards and replace them with new keys.

## Run the assistant

With the virtual environment activated, run:

```powershell
python agent2.py
```

The program displays the configured LLM and TTS providers and then prompts:

```text
Press Enter to speak (q to quit):
```

1. Press **Enter** and speak into the microphone.
2. Stop speaking and wait for the assistant to process the recording.
3. Listen to the Telugu response.
4. Press **Enter** again for another turn.
5. Enter **q** to quit.

The assistant writes the current recording and response to `input.wav` and `reply.wav`. These files are overwritten on the next turn.

## Configuration

The main settings are near the top of `agent2.py`:

```python
LLM_PROVIDER = "sarvam"   # "sarvam" or "groq"
TTS_PROVIDER = "cartesia" # "cartesia", "sarvam_clone", or "sarvam"
DEFAULT_CITY = "Visakhapatnam"
```

Change these values to select the language model, speech provider, or default weather city. The assistant is configured for Telugu (`te-IN`) and validates that generated responses contain predominantly Telugu script.

## Optional recording utility

`record.py` is a small standalone utility that records five seconds of microphone audio to `input.wav`:

```powershell
python record.py
```

This utility does not run the full speech-recognition or response pipeline. Use `agent2.py` for the complete assistant.

## Generated files

| File | Description |
| --- | --- |
| `input.wav` | Latest microphone recording |
| `reply.wav` | Latest generated speech |
| `log.txt` | Conversation and spoken-response log |
| `cost_log.csv` | Estimated per-turn usage and cost data |

These files are runtime artifacts and may contain private audio or conversation data. Review them before sharing the project.

## Troubleshooting

### No microphone input

- Confirm that Windows has microphone permission enabled for Python or your terminal.
- Check that the intended input device is selected as the Windows default recording device.
- Close applications that may be exclusively using the microphone.

### Audio playback fails

- Confirm that speakers or headphones are connected and selected as the Windows output device.
- Check that `sounddevice` can access the output device.
- The program attempts to fall back to Windows `winsound` playback.

### API authentication errors

- Confirm that `.env.local` is in the same directory as `agent2.py`.
- Check that each variable name matches the names shown above.
- Verify that the API key is active and has sufficient credits.

### Weather requests fail

Weather data comes from Open-Meteo and does not require an API key. A network connection is required. If the service is unavailable, the assistant returns a Telugu weather-error message.

## Notes

- The application makes external API requests for transcription, language generation, text-to-speech, geocoding, and weather data.
- API usage may incur charges depending on your provider plans.
- The cost meter uses configurable estimates in `agent2.py`; update those rates if your provider pricing changes.
