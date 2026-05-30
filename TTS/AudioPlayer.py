import sounddevice as sd
import soundfile as sf
from io import BytesIO
import threading


class AudioPlayer:
    def __init__(self, samplerate=32000, channels=1):
        self.samplerate = samplerate
        self.channels = channels

    def play_audio(self, audio_bytes, stop_event=None):
        """
        Plays audio bytes from a queue. Blocks until the queue is finished.
        Optional stop_event allows external interruption.
        """
        with sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32"
        ) as stream:
            data, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
            stream.write(data)