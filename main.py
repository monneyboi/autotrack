import click
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from datetime import datetime


@click.group()
def cli():
    """AutoTrack CLI tool for audio transcription."""
    pass


def record_audio(duration=5, sample_rate=48000, device="hw:2,0"):
    """Record audio from microphone and return mono audio data.

    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        device: Audio device to use

    Returns:
        Mono audio data as numpy array
    """
    print("Speak now...")
    # Record audio from microphone
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,  # Using 2 channels
        device=device,  # Use the specified device
        dtype=np.float32,
    )
    sd.wait()  # Wait until recording is finished

    # Convert stereo to mono by averaging channels
    return audio_data.mean(axis=1)


@cli.command()
def listen():
    """Listen for spoken commands and transcribe them."""
    # Load the whisper model
    model_size = "medium"
    print(f"Loading {model_size} model...")
    model = whisper.load_model(model_size, device="cuda")

    print("Listening for commands... (Press Ctrl+C to exit)")

    try:
        while True:
            # Record audio
            audio_data = record_audio()
            
            # Save to temporary file for whisper
            sf.write("temp_audio.wav", audio_data, 48000)

            # Transcribe audio
            print("Transcribing...")
            result = model.transcribe("temp_audio.wav", fp16=True)

            # Print transcription
            print(f"Transcription: {result['text']}")

            print("\nListening for next command...")

    except KeyboardInterrupt:
        print("Stopped listening.")


@cli.command()
@click.option("--duration", default=5, help="Recording duration in seconds")
@click.option("--sample-rate", default=48000, help="Audio sample rate")
@click.option("--device", default="hw:2,0", help="Audio device to use")
@click.option(
    "--output", "-o", help="Output WAV file path (default: auto-generated filename)"
)
def record(duration, sample_rate, device, output):
    """Record audio from microphone and save to a WAV file."""
    try:
        print(f"Recording {duration} seconds of audio from device {device}...")
        audio_data = record_audio(duration, sample_rate, device)

        # Generate filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"recording_{timestamp}.wav"

        # Save to WAV file
        sf.write(output, audio_data, sample_rate)
        print(f"Audio saved to: {output}")

        # Print audio statistics
        print("\nAudio Statistics:")
        print(f"Shape: {audio_data.shape}")
        print(f"Duration: {duration} seconds")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Min value: {audio_data.min():.6f}")
        print(f"Max value: {audio_data.max():.6f}")
        print(f"Mean value: {audio_data.mean():.6f}")

        return audio_data

    except KeyboardInterrupt:
        print("Recording stopped.")


if __name__ == "__main__":
    cli()
