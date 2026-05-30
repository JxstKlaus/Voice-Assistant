from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
import base64
from PIL import Image
from io import BytesIO
from util.Logger import logging

logger = logging.getLogger(__name__)

def image_to_base64_data_uri(file_path, max_size=(512, 512)):
    """
    Opens an image, optionally resizes it to reduce GPU memory usage,
    then converts to base64 data URI.
    """
    with Image.open(file_path) as img:
        img.thumbnail(max_size)  # Resize to fit max_size
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    

class LLM:
    def __init__(self, name, model_path, system_prompt="", mmproj_path=None, type="text", verbose=False):
        self.model_path = model_path
        self.type = type
        self.system_prompt = system_prompt
        self.mmproj_path = mmproj_path
        self.verbose = verbose
        self.llm = None
        self.name = name

    def unload_model(self):
        if self.llm:
            del self.llm
            self.llm = None

    def load_model(self, gpu_layers=-1):
        if self.type == "text":
            self.llm = Llama(model_path=self.model_path,
                             n_ctx=4096,
                             n_gpu_layers=gpu_layers,
                             verbose=self.verbose
                            )
        elif self.type == "vision":
            chat_handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path, verbose=self.verbose)
            self.llm = Llama(model_path=self.model_path,
                             chat_handler=chat_handler,
                             n_ctx=4096,
                             n_gpu_layers=gpu_layers,
                             verbose=self.verbose,
                            )
            

        else:
            logger.error(f"Unsupported type: {self.type}")
            raise ValueError(f"Unsupported type: {self.type}. Supported types are 'text' and 'vision'.")

        logger.info(f"Model loaded from {self.model_path} with type {self.type}")

        # Warmup inference
        self.llm_chat("Hi", max_tokens=8)

        
    def llm_chat(self, text, history = None, image_uri = None, stream=True, max_tokens=128, temperature=0.9):
        """
        stream=True: Returns a generator that yields chunks of the response\n
        stream=False: Returns the final string
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)

        if self.type == "vision" and image_uri is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": text
            })

        stop = ["<|im_end|>", "<|im_start|>"]
        response = self.llm.create_chat_completion(
            messages = messages,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
            temperature=temperature
        )

        if stream:
        # streaming mode: yield chunks
            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
                else:
                    pass
        else:
            # non-streaming mode: return final string
            return response["choices"][0]["message"]["content"]