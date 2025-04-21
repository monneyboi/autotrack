import click
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from datetime import datetime
import time
from typing import List, Tuple, Optional, Any


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


def detect_voice_activity(
    audio_chunk: np.ndarray, threshold: float = 0.01, min_duration: int = 10
) -> bool:
    """Detect if there is voice activity in the audio chunk.

    Args:
        audio_chunk: Audio data chunk
        threshold: Energy threshold for voice detection
        min_duration: Minimum number of samples above threshold to consider as voice

    Returns:
        True if voice activity detected, False otherwise
    """
    # Calculate absolute values and check if they're above threshold
    abs_data = np.abs(audio_chunk)
    # Count samples above threshold
    active_samples = np.sum(abs_data > threshold)

    return active_samples > min_duration


def record_with_vad(
    max_silence_duration: float = 1.5,
    chunk_duration: float = 0.5,
    sample_rate: int = 48000,
    device: str = "hw:2,0",
) -> np.ndarray:
    """Record audio with voice activity detection.

    Args:
        max_silence_duration: Maximum silence duration in seconds before stopping
        chunk_duration: Duration of each audio chunk in seconds
        sample_rate: Audio sample rate
        device: Audio device to use

    Returns:
        Recorded audio data as numpy array
    """
    print("Listening... (speak to start recording)")

    chunk_size = int(chunk_duration * sample_rate)
    silence_chunks = int(max_silence_duration / chunk_duration)

    audio_chunks: List[np.ndarray] = []
    silent_chunks_count = 0
    recording_started = False

    # Create stream for real-time audio
    with sd.InputStream(
        samplerate=sample_rate,
        channels=2,
        device=device,
        blocksize=chunk_size,
        dtype=np.float32,
    ) as stream:

        while True:
            # Read audio chunk
            audio_block, _ = stream.read(chunk_size)
            mono_block = audio_block.mean(axis=1)

            # Check for voice activity
            voice_detected = detect_voice_activity(mono_block)

            if voice_detected:
                if not recording_started:
                    print("Voice detected! Recording...")
                    recording_started = True

                audio_chunks.append(mono_block)
                silent_chunks_count = 0
            elif recording_started:
                # Still append audio during silence to capture pauses between words
                audio_chunks.append(mono_block)
                silent_chunks_count += 1

                # Stop if silence is too long
                if silent_chunks_count >= silence_chunks:
                    print("Silence detected. Stopping recording.")
                    break

            # Allow starting the recording with Ctrl+C if needed
            if (
                not recording_started and len(audio_chunks) > 100
            ):  # safety to prevent infinite loop
                print("No voice detected. Please speak or press Ctrl+C to stop.")
                audio_chunks = []  # Reset to avoid memory buildup

    if not audio_chunks:
        return np.array([])

    # Concatenate all audio chunks
    return np.concatenate(audio_chunks)


@cli.command()
@click.option("--max-silence", default=1.5, help="Maximum silence duration in seconds")
@click.option("--threshold", default=0.01, help="Threshold for voice detection")
@click.option("--sample-rate", default=48000, help="Audio sample rate")
@click.option("--device", default="hw:2,0", help="Audio device to use")
def listen(max_silence, threshold, sample_rate, device):
    """Listen for spoken commands and transcribe them when you finish speaking."""
    # Load the whisper model
    model_size = "medium"
    print(f"Loading {model_size} model...")
    model = whisper.load_model(model_size, device="cuda")

    print("Listening for commands... (Press Ctrl+C to exit)")
    print(f"Voice detection: {threshold} threshold, {max_silence}s max silence")

    try:
        while True:
            # Set the voice detection threshold in the detector function
            detect_voice_activity.__defaults__ = (
                threshold,
                10,
            )  # Update default threshold

            # Record audio with voice activity detection
            audio_data = record_with_vad(
                max_silence_duration=max_silence, sample_rate=sample_rate, device=device
            )

            if len(audio_data) > 0:
                # Save to temporary file for whisper
                sf.write("temp_audio.wav", audio_data, sample_rate)

                # Transcribe audio
                print("Transcribing...")
                result = model.transcribe("temp_audio.wav", fp16=True)

                # Print transcription
                print(f"Transcription: {result['text']}")

            print("\nWaiting for next command...")

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
