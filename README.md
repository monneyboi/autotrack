# AutoTrack

Voice-controlled Octatrack sampler using AI. Speak commands naturally, and AutoTrack translates them into MIDI messages to control your Elektron Octatrack.

## Features

- Voice-activated recording with automatic silence detection
- Natural language command processing using AI
- Full Octatrack MIDI implementation for sample triggering, parameter control, and more
- Text-to-speech feedback on command execution

## Requirements

- Python 3.12+
- CUDA-capable GPU (for Whisper speech recognition)
- MIDI interface connected to Octatrack
- Microphone for voice input

## Installation

```bash
# Clone the repository
git clone https://github.com/username/autotrack.git
cd autotrack

# Install dependencies
pip install -e .
```

## Usage

```bash
# Start voice command listener
python -m main listen --device "your-audio-device" --midi-port "your-midi-port"

# Record audio sample
python -m main record --duration 10 --output sample.wav
```

## Configuration

Create a `.env` file with your API credentials:

```
OPENAI_API_KEY=your_api_key
```

## License

MIT