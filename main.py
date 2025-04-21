import click
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import rtmidi
import json
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
import time
from typing import List, Tuple, Optional, Any, Dict, Union
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("autotrack")

# Initialize OpenAI client
client = OpenAI(base_url="https://api.deepseek.com")

# MIDI setup
midi_out = rtmidi.MidiOut()


def setup_midi(
    port_name: str = "Steinberg UR22mkII:Steinberg UR22mkII MIDI 1 24:0",
) -> bool:
    """Set up MIDI output to the specified port.

    Args:
        port_name: Name of the MIDI port to connect to

    Returns:
        True if connected successfully, False otherwise
    """
    logger.info("Setting up MIDI connection...")
    available_ports = midi_out.get_ports()
    logger.debug(f"Available MIDI ports: {available_ports}")

    if not available_ports:
        logger.error("No MIDI ports available")
        return False

    for i, port in enumerate(available_ports):
        if port_name in port:
            midi_out.open_port(i)
            logger.info(f"Connected to MIDI port: {port}")
            return True

    logger.error(f"Could not find MIDI port: {port_name}")
    return False


def send_midi_note(note: int, velocity: int = 100, channel: int = 0) -> None:
    """Send MIDI note on message followed by note off.

    Args:
        note: MIDI note number (0-127)
        velocity: Note velocity (0-127)
        channel: MIDI channel (0-15)
    """
    logger.debug(f"Sending MIDI note: {note}, velocity: {velocity}, channel: {channel}")
    # Note on message
    midi_out.send_message([0x90 + channel, note, velocity])
    # Short delay
    time.sleep(0.1)
    # Note off message
    midi_out.send_message([0x80 + channel, note, 0])


def send_midi_cc(cc: int, value: int, channel: int = 0) -> None:
    """Send MIDI Control Change message.

    Args:
        cc: Control Change number (0-127)
        value: Control Change value (0-127)
        channel: MIDI channel (0-15)
    """
    logger.debug(f"Sending MIDI CC: {cc}, value: {value}, channel: {channel}")
    midi_out.send_message([0xB0 + channel, cc, value])


def process_command_with_openai(command: str) -> Dict[str, Any]:
    """Process voice command with OpenAI to determine the Octatrack action.

    Args:
        command: Transcribed voice command

    Returns:
        Response from OpenAI with action details
    """
    logger.info(f"Processing command with OpenAI: '{command}'")

    # Read the Octatrack MIDI reference to provide context
    with open("/home/johan/Projects/autotrack/OCTA.md", "r") as f:
        octa_midi_reference = f.read()

    # Define function schema for OpenAI to use
    functions = [
        {
            "type": "function",
            "function": {
                "name": "send_midi_note",
                "description": "Send a MIDI note message to the Octatrack",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "integer",
                            "description": "MIDI note number (0-127)",
                        },
                        "velocity": {
                            "type": "integer",
                            "description": "Note velocity (0-127)",
                            "default": 100,
                        },
                        "channel": {
                            "type": "integer",
                            "description": "MIDI channel (0-15)",
                            "default": 0,
                        },
                    },
                    "required": ["note"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_midi_cc",
                "description": "Send a MIDI Control Change message to the Octatrack",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cc": {
                            "type": "integer",
                            "description": "Control Change number (0-127)",
                        },
                        "value": {
                            "type": "integer",
                            "description": "Control Change value (0-127)",
                        },
                        "channel": {
                            "type": "integer",
                            "description": "MIDI channel (0-15)",
                            "default": 0,
                        },
                    },
                    "required": ["cc", "value"],
                },
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an assistant that controls an Octatrack sampler via MIDI.
You know how to interpret user voice commands and translate them to MIDI messages.
Here is the MIDI reference for the Octatrack that you should use to decide what MIDI messages to send:

{octa_midi_reference}

Consider the user's request carefully and decide how to execute it on the Octatrack.
You can only respond by calling the provided functions to send MIDI messages.
Explain what you're doing in your response and why you chose the specific MIDI messages.""",
                },
                {"role": "user", "content": command},
            ],
            tools=functions,
            tool_choice="auto",
        )

        logger.debug(f"OpenAI response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing command with OpenAI: {e}")
        return {"error": str(e)}


def execute_openai_actions(response: Any) -> str:
    """Execute the MIDI actions recommended by OpenAI.

    Args:
        response: OpenAI API response object

    Returns:
        String describing the actions taken
    """
    try:
        message = response.choices[0].message
        content = message.content or ""

        # Check if there's a function call
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing {function_name} with args: {function_args}")

                if function_name == "send_midi_note":
                    send_midi_note(
                        note=function_args.get("note", 60),
                        velocity=function_args.get("velocity", 100),
                        channel=function_args.get("channel", 0),
                    )
                elif function_name == "send_midi_cc":
                    send_midi_cc(
                        cc=function_args.get("cc", 0),
                        value=function_args.get("value", 0),
                        channel=function_args.get("channel", 0),
                    )

            return content
        else:
            logger.warning("No tool calls found in the response")
            return content or "No action taken"

    except Exception as e:
        logger.error(f"Error executing OpenAI actions: {e}")
        return f"Error executing command: {str(e)}"


def record_audio(duration=5, sample_rate=48000, device="hw:2,0"):
    """Record audio from microphone and return mono audio data.

    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        device: Audio device to use

    Returns:
        Mono audio data as numpy array
    """
    logger.info("Recording audio...")
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
    logger.info("Listening for voice with VAD...")
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
                    logger.info("Voice detected, recording started")
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
                    logger.info("Silence detected, stopping recording")
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


@click.group()
def cli():
    """AutoTrack CLI tool for audio transcription and Octatrack control."""
    pass


@cli.command()
@click.option("--max-silence", default=1.5, help="Maximum silence duration in seconds")
@click.option("--threshold", default=0.01, help="Threshold for voice detection")
@click.option("--sample-rate", default=48000, help="Audio sample rate")
@click.option("--device", default="hw:2,0", help="Audio device to use")
@click.option(
    "--midi-port",
    default="Steinberg UR22mkII:Steinberg UR22mkII MIDI 1 24:0",
    help="MIDI port to use",
)
def listen(max_silence, threshold, sample_rate, device, midi_port):
    """Listen for spoken commands, transcribe them, and execute them on the Octatrack."""
    # Load the whisper model
    model_size = "medium"
    logger.info(f"Loading whisper {model_size} model...")
    print(f"Loading {model_size} model...")
    model = whisper.load_model(model_size, device="cuda")

    # Set up MIDI connection
    if not setup_midi(midi_port):
        logger.error("Failed to set up MIDI connection. Exiting.")
        return

    logger.info("Starting voice command listener")
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
                logger.info("Transcribing audio...")
                print("Transcribing...")
                result = model.transcribe("temp_audio.wav", fp16=True)
                transcription = result["text"]

                # Print transcription
                logger.info(f"Transcription: {transcription}")
                print(f"Transcription: {transcription}")

                # Process the command and send MIDI
                logger.info("Processing command with OpenAI...")
                print("Processing command...")
                response = process_command_with_openai(transcription)

                # Execute the recommended actions
                action_result = execute_openai_actions(response)
                logger.info(f"Action taken: {action_result}")
                print(f"\nOctatrack action: {action_result}")

            print("\nWaiting for next command...")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping")
        print("Stopped listening.")

        # Close MIDI port
        if midi_out.is_port_open():
            midi_out.close_port()
            logger.info("Closed MIDI port")


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
        logger.info(f"Recording {duration} seconds of audio from device {device}")
        print(f"Recording {duration} seconds of audio from device {device}...")
        audio_data = record_audio(duration, sample_rate, device)

        # Generate filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"recording_{timestamp}.wav"

        # Save to WAV file
        sf.write(output, audio_data, sample_rate)
        logger.info(f"Audio saved to: {output}")
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
        logger.info("Recording stopped by user")
        print("Recording stopped.")


if __name__ == "__main__":
    cli()
