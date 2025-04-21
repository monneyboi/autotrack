import click
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


@click.group()
def cli():
    """AutoTrack CLI tool for audio transcription."""
    pass


@cli.command()
def listen():
    """Listen for spoken commands and transcribe them."""
    # Load the whisper model
    model_size = "base"
    print(f"Loading {model_size} model...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Configure audio parameters
    sample_rate = 16000
    duration = 5  # seconds to record
    
    print("Listening for commands... (Press Ctrl+C to exit)")
    
    try:
        while True:
            print("Speak now...")
            # Record audio from microphone
            audio_data = sd.rec(
                int(duration * sample_rate), 
                samplerate=sample_rate, 
                channels=1,
                device="pipewire",
                dtype=np.float32
            )
            sd.wait()  # Wait until recording is finished
            
            # Normalize audio data
            audio_data = audio_data.flatten()
            
            # Transcribe audio
            print("Transcribing...")
            segments, _ = model.transcribe(audio_data, beam_size=5)
            
            # Print transcription
            transcription = " ".join([segment.text for segment in segments])
            print(f"Transcription: {transcription}")
            
            print("\nListening for next command...")
    
    except KeyboardInterrupt:
        print("Stopped listening.")


if __name__ == "__main__":
    cli()