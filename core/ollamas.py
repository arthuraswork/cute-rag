from enum import Enum

class OllamaRAGModels(str, Enum):
    LIGHT_GEMMA_3_4B = "gemma3:4b"
    LIGHT_PHI_4_MINI = "phi4-mini"
    LIGHT_LLAMA_3_2_3B = "llama3.2:3b"
    LIGHT_QWEN_2_5_7B = "qwen2.5:7b"
    LIGHT_MISTRAL_7B = "mistral:7b"
    MEDIUM_LLAMA_3_1_8B = "llama3.1:8b"
    MEDIUM_GEMMA_3_12B = "gemma3:12b"
    MEDIUM_QWEN_2_5_14B = "qwen2.5:14b"
    HEAVY_LLAMA_3_3_70B = "llama3.3:70b"
    HEAVY_QWEN_2_5_32B = "qwen2.5:32b"
    HEAVY_DEEPSEEK_R1_14B = "deepseek-r1:14b"