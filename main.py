from LLM.LLM import LLM
from TTS.TTS import TTS
from STT.STT import STT
from Pipelines.ChatPipeline2 import ChatPipeline
from util.Logger import logger
from util.func import load_settings


model_path = "LLM/models/Qwen2.5-VL-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
mmproj_path = "LLM/models/Qwen2.5-VL-7b/mmproj-F16.gguf"
settings = load_settings("settings.json")


llm = LLM("Sora", model_path=model_path, system_prompt=settings["system"], type="text", verbose=False)
llm.load_model(gpu_layers=8)
tts = TTS()
stt = STT()

pipeline = ChatPipeline(llm, tts, stt)

if __name__ == "__main__":
    pipeline.run()