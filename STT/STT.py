import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import torch
import torchaudio
import time
from util.Logger import logging

logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class STT:
    def __init__(self, model_size="small", device=None, silence_threshold=0.01, silence_duration=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.target_sr = 16000
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def record_audio(self, sample_rate=16000, block_size=1024):
        """
        Record audio until speech is detected and followed by silence.
        Returns float32 NumPy array.
        """
        audio_buffer = []

        def callback(indata, frames, time_info, status):
            audio_buffer.append(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=block_size,
            callback=callback
        )

        stream.start()
        first_speech_detected = False
        silence_start_time = None

        while True:
            if len(audio_buffer) == 0:
                time.sleep(0.01)
                continue

            current_chunk = audio_buffer[-1].flatten()
            rms = np.sqrt(np.max(current_chunk ** 2))

            if not first_speech_detected:
                if rms > self.silence_threshold:
                    first_speech_detected = True
                    #print("Speech detected!")
            else:
                if rms < self.silence_threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= self.silence_duration:
                        #print("Silence detected. Stopping recording.")
                        break
                else:
                    silence_start_time = None  # reset if speech continues

            time.sleep(0.01)

        stream.stop()
        stream.close()

        audio_array = np.concatenate(audio_buffer).flatten()
        return audio_array, sample_rate

    def resample_audio(self, audio_array, original_sr):
        if original_sr == self.target_sr:
            return audio_array
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio_tensor, orig_freq=original_sr, new_freq=self.target_sr)
        return resampled.squeeze(0).numpy()

    def transcribe(self, audio_array, sample_rate, language="en", beam_size=1, vad_filter=True):
        audio_array = self.resample_audio(audio_array, sample_rate)
        segments, _ = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter
        )
        transcription = " ".join(segment.text for segment in segments)
        return transcription

# Example usage
if __name__ == "__main__":
    stt = STT(model_size="small")
    audio, sr = stt.record_audio()
    text = stt.transcribe(audio, sr)
    print("Transcription:", text)