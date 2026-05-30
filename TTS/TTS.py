from .GPTSoVITS.Inference import GptSovits
import queue, threading

class TTS:
    def __init__(self):
        self.tts_engine = GptSovits()
        self.current_voice = "Mona"

    def synthesize(self, text, streaming_mode = False):
        """Synthesize text to speech using the current voice"""
        return self.tts_engine.synthesize(text, streaming_mode)
    