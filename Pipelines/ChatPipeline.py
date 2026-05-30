import re
import threading
import queue
from LLM.LLM import LLM
from TTS.TTS import TTS
from STT.STT import STT
from TTS.AudioPlayer import AudioPlayer
import sys


class ChatPipeline:
    def __init__(self, llm: LLM, tts: TTS, stt: STT, samplerate=32000, channels=1):
        self.llm = llm      # LLM instance (sync generator only)
        self.tts = tts      # TTS engine (sync)
        self.stt = stt      # STT engine (sync)
        self.audio_player = AudioPlayer(samplerate=samplerate, channels=channels)

        # Queues for communication
        self.print_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # Stop event for clean shutdown / barge-in
        self.stop_event = threading.Event()

        # Threads
        self.llm_thread = None
        self.print_thread = None
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
    def _llm_thread(self, user_input):
        buffer = ""
        try:
            stream = self.llm.llm_chat(text=user_input, stream=True)
            for token in stream:
                if self.stop_event.is_set():
                    break

                # Build buffer for TTS sentence-level splitting
                buffer += token
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                buffer = sentences.pop() if sentences else ""
                for s in sentences:
                    self.print_queue.put(s)
                    self.tts_queue.put(s)

        finally:
            if buffer.strip():
                self.print_queue.put(buffer)
                self.tts_queue.put(buffer.strip())

            # Signal end
            self.print_queue.put(None)
            self.tts_queue.put(None)

    # --- Printer worker ---
    def _print_thread(self):
        print(f"{self.llm.name}: ", end="", flush=True)
        while True:
            segment = self.print_queue.get()
            if segment is None:
                break
            print(segment, end="", flush=True)
        print()  # newline at end

    # --- TTS worker ---
    def _tts_thread(self):
        while True:
            sentence = self.tts_queue.get()
            if sentence is None:
                break
            audio_bytes = self.tts.synthesize(sentence)
            self.audio_queue.put(audio_bytes)
        # Signal audio thread to stop
        self.audio_queue.put(None)

    # --- Audio playback worker ---
    def _audio_thread(self):
        # Blocks until all audio is played
        self.audio_player.play_audio_from_queue(self.audio_queue, stop_event=self.stop_event)
    
    def clear_queues(self):
        self.print_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
    # --- Run the full pipeline ---
    def run(self):
        # Record user audio
        print("User: ", end="", flush=True)
        user_audio, sr = self.stt.record_audio()
        user_input = self.stt.transcribe(user_audio, sr)
        print(user_input)

        # Reset stop event
        self.clear_queues()
        self.stop_event.clear()
        

        # Start threads
        self.llm_thread = threading.Thread(target=self._llm_thread, args=(user_input,))
        self.print_thread = threading.Thread(target=self._print_thread)
        #self.tts_thread = threading.Thread(target=self._tts_thread)
        #self.audio_thread = threading.Thread(target=self._audio_thread)

        self.llm_thread.start()
        self.print_thread.start()
        #self.tts_thread.start()
        #self.audio_thread.start()

        

        # !!ORDER MATTER!! Wait for all threads to finish 
        self.llm_thread.join()
        print(f"[LLM thread finished]")  # Debug print for LLM thread completion
        #self.tts_thread.join()
        #print(f"[TTS thread finished]")  # Debug print for TTS thread completion
        #self.audio_thread.join()
        #print(f"[Audio thread finished]")  # Debug print for Audio thread completion
        self.print_thread.join()
        print(f"[Print thread finished]")  # Debug print for Print thread completion

        # Exit detection
        user_input_clean = re.sub(r'(?<=[.!?])\s+', "", user_input.lower())
        if user_input_clean in ["exit", "quit", "q", "bye"]:
            return 0