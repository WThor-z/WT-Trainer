"""Template module for handling chat templates and tokenization."""

from copy import deepcopy
from dataclasses import dataclass
import logging
import re

from typing_extensions import override
from transformers import PreTrainedTokenizer  # noqa: F401

from ..const import Role
from .formatter import EmptyFormatter, SLOTS
from .formatter import StringFormatter
from .formatter import Formatter  # noqa: F401
from wt_trainer.args import DataArguments  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """Template class for handling chat templates and tokenization.

    This class provides methods for encoding messages, handling special tokens,
    and managing chat templates for different models.
    """

    format_user: Formatter  # 用户消息格式
    format_assistant: Formatter  # 助手信息格式
    format_system: Formatter  # 系统消息格式
    format_observation: Formatter  # 观察信息格式
    format_prefix: Formatter  # 对话前缀格式
    default_system: str  # 默认系统消息，当没有提供特定系统消息时使用
    stop_words: str | None  # 模型生成时的停止词列表，当模型生成这些词时会停止生成
    thought_words: tuple[str, str]  # 定义思考过程的开始和结束标记，用于支持推理过程
    efficient_eos: bool  # 是否使用高效的EOS处理方式
    replace_eos: bool  # 是否替换默认的EOS标记
    replace_jinja_template: bool  # 替换tokenizer中的Jinja模板
    enable_thinking: bool | None  # 启用思考过程，用于支持推理能力

    def encode_oneturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system)
        return [
            (encoded_messages[i], encoded_messages[i + 1])
            for i in range(0, len(encoded_messages), 2)
        ]

    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(
            f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL
        )
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _convert_elements_to_ids(
        self, tokenizer: PreTrainedTokenizer, elements: SLOTS
    ) -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(
                    f"Input must be string, set[str] or dict[str, str], got {type(elem)}"
                )

        return token_ids

    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, str]],
        system: str | None,
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            else:
                raise NotImplementedError(f"Unexpected role: {message['role']}")

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: PreTrainedTokenizer, eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: PreTrainedTokenizer) -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(
                {"additional_special_tokens": stop_words}, replace_additional_special_tokens=False
            )
            logger.info(f"Add {','.join(stop_words)} to stop words.")
            if num_added_tokens > 0:
                logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(
        slots: SLOTS, tokenizer: PreTrainedTokenizer, placeholder: str = "content"
    ) -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + Template._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + Template._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return " + ".join(slot_items)

    def _get_jinja_template(self, tokenizer: PreTrainedTokenizer) -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(
            self.format_system.apply(), tokenizer, placeholder="system_message"
        )
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += (
                "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"
            )

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template

    def fix_jinja_template(self, tokenizer: PreTrainedTokenizer) -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info(f"Cannot add this chat template to tokenizer: {e}.")


@dataclass
class ReasoningTemplate(Template):
    """A template that adds thought to assistant message.

    This template is used for models that support reasoning or chain-of-thought
    processing, where the assistant can provide intermediate thinking steps.
    """

    @override
    def encode_oneturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system)
        if (
            self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = (
                        self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]
                    )

        return [
            (encoded_messages[i], encoded_messages[i + 1])
            for i in range(0, len(encoded_messages), 2)
        ]


TEMPLATES: dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Formatter | None = None,
    format_assistant: Formatter | None = None,
    format_system: Formatter | None = None,
    format_observation: Formatter | None = None,
    format_prefix: Formatter | None = None,
    default_system: str = "",
    stop_words: list[str] | None = None,
    thought_words: tuple[str, str] | None = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    enable_thinking: bool | None = True,
    template_class: type[Template] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)  # type: ignore

    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>", "</think>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        enable_thinking=enable_thinking,
    )


def parse_template(tokenizer: PreTrainedTokenizer) -> Template:
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )[
        len(prefix) :
    ]  # type: ignore

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [
        {"role": "user", "content": "{{content}}"},
        {"role": "assistant", "content": "{{content}}"},
    ]
    assistant_slot = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]
    template_class = ReasoningTemplate if "<think>" in assistant_slot else Template
    assistant_slot = (
        assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")
    )  # remove thought tags

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return template_class(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_observation=StringFormatter(slots=[user_slot]),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>", "</think>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
    )


def get_template_and_fix_tokenizer(
    tokenizer: PreTrainedTokenizer, data_args: DataArguments
) -> Template:
    r"""Get chat template and fixes the tokenizer."""
    if data_args.template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning(
                "`template` was not specified, try parsing the chat template from the tokenizer."
            )
            template = parse_template(tokenizer)
        else:
            logger.warning("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if data_args.template not in TEMPLATES:
            raise ValueError(f"Template {data_args.template} does not exist.")

        template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.default_system is not None:
        logger.info(f"Using default system message: {data_args.default_system}.")
        template.default_system = data_args.default_system

    template.enable_thinking = data_args.enable_thinking
    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    ),
    replace_jinja_template=True,
)


register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["System: {{content}}", {"eos_token"}, "\n"]),
    replace_jinja_template=True,
)


register_template(
    name="glm4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(
        slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
)


register_template(
    name="llama4",
    format_user=StringFormatter(
        slots=[
            "<|header_start|>user<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot|>"]),
    format_system=StringFormatter(
        slots=["<|header_start|>system<|header_end|>\n\n{{content}}<|eot|>"]
    ),
    format_observation=StringFormatter(
        slots=[
            "<|header_start|>ipython<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot|>", "<|eom|>"],
    replace_eos=True,
)


# copied from chatml template
register_template(
    name="qwen",
    format_user=StringFormatter(
        slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(
        slots=[
            "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        ]
    ),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from qwen template
register_template(
    name="qwen3",
    format_user=StringFormatter(
        slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(
        slots=[
            "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        ]
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)
