from enum import Enum
from enum import unique

# 定义常量检查点
from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils import WEIGHTS_NAME


@unique
class QuantizationMethod(str, Enum):

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"


class RopeScaling(str, Enum):
    """RoPE scaling strategies."""

    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


class AttentionFunction(str, Enum):
    """Attention function types."""

    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"


class EngineName(str, Enum):
    """Inference engine names."""

    HF = "huggingface"


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}


MEMORY_THRESHOLD = 1024**2  # 1MB threshold for warmup

IGNORE_INDEX = -100
