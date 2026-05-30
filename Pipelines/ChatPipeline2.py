import re
import threading
import queue
from LLM.LLM import LLM
from TTS.TTS import TTS
from STT.STT import STT
from TTS.AudioPlayer import AudioPlayer
from util.Logger import logging
logger = logging.getLogger(__name__)



class ChatPipeline:
    def __init__(self, llm: LLM, tts: TTS, stt: STT, samplerate=32000, channels=1):
        self.llm = llm      # LLM instance (sync generator only)
        self.tts = tts      # TTS engine (sync)
        self.stt = stt      # STT engine (sync)
        self.audio_player = AudioPlayer(samplerate=samplerate, channels=channels)
        self.input_type = "text"
        self.output_type = "voice"

        # communication
        self.token_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.exit_flag = False

        # Stop event for clean shutdown / barge-in
        self.stop_event = threading.Event()

        # Threads
        self.llm_thread = None
        self.tts_thread = None
        self.audio_thread = None

    # --- Helper: batch streaming into sentences ---
    @staticmethod
    def _sentence_batcher(chunk_generator):
        buffer = ""
        for chunk in chunk_generator:
            buffer += chunk
            sentences = re.split(r'(?<=[.!?])\s+', buffer)
            buffer = sentences.pop() if sentences else ""
            for s in sentences:
                yield s
        if buffer.strip():
            yield buffer.strip()

    # --- LLM worker ---
    def _generate_response_thread(self, user_input):   
        try:
            stream = self.llm.llm_chat(text=user_input, stream=True)
            for token in stream:
                if self.stop_event.is_set():
                    break
                self.token_queue.put(token)

        finally:
            # Signal end
            self.token_queue.put(None)

    # --- TTS worker ---
    def _process_response_thread(self):
        buffer = ""
        print(f"{self.llm.name}: ", end="", flush=True)
        while True:
            token = self.token_queue.get()
            if token is None:
                if buffer.strip():
                    audio_bytes = self.tts.synthesize(buffer.strip())
                    self.audio_queue.put(audio_bytes)
                break
            
            print(token, end="", flush=True)  # Optional: print as we go
            if self.output_type == "text":
                continue  # Skip TTS if output is text


            # Build buffer for TTS sentence-level splitting
            buffer += token
            sentences = re.split(r'(?<=[.!?])\s+', buffer)
            buffer = sentences.pop() if sentences else ""
            for s in sentences:
                audio_bytes = self.tts.synthesize(s)
                self.audio_queue.put(audio_bytes)
        # Signal audio thread to stop
        self.audio_queue.put(None)
        print()

    # --- Audio playback worker ---
    def _audio_thread(self):
        # Blocks until all audio is played
        while self.output_type == "voice":
                audio_bytes = self.audio_queue.get()
                if audio_bytes is None:
                    break
                self.audio_player.play_audio(audio_bytes, stop_event=self.stop_event)
    
    def get_user_input(self):
        print("User: ", end="", flush=True)
        if self.input_type == "text":
            user_input = input()
        elif self.input_type == "voice":
            user_audio, sr = self.stt.record_audio()
            user_input = self.stt.transcribe(user_audio, sr)
            print(user_input)

        # Exit detection
        if user_input.strip().lower() in ["exit", "quit", "q", "bye", "goodbye"]:
            self.exit_flag = True
            user_input = "bye"

        return user_input
    
    def clear_communication(self):
        self.token_queue = queue.Queue()
        self.audio_queue = queue.Queue()
    # --- Run the full pipeline ---
    def run(self):
        while not self.exit_flag:
            # get user input
            user_input = self.get_user_input()

            # Reset stop event
            self.clear_communication()
            self.stop_event.clear()
            

            # Start threads
            self.llm_thread = threading.Thread(target=self._generate_response_thread, args=(user_input,), daemon=True)
            self.tts_thread = threading.Thread(target=self._process_response_thread)
            self.audio_thread = threading.Thread(target=self._audio_thread)

            self.llm_thread.start()
            self.tts_thread.start()
            self.audio_thread.start()

            self.llm_thread.join()
            self.audio_thread.join()
            self.tts_thread.join()
        