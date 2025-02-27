from typing import List

class ChatFormat:
    def __init__(self, chat_prefix: List[int], chat_suffix: List[int], always_suffix=False):
        self.chat_prefix = chat_prefix
        self.chat_suffix = chat_suffix
        self.always_suffix = always_suffix

    def prepare_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        return self.chat_prefix + prompt_tokens + adv_tokens + self.chat_suffix

    def prepare_prefix_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        if not self.always_suffix:
            return self.chat_prefix + prompt_tokens + adv_tokens
        else:
            return self.chat_prefix + prompt_tokens + self.chat_suffix + adv_tokens


class SamplerChatFormat:
    def __init__(self, slice=0):
        self.slice = slice

    def prepare_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        return prompt_tokens + adv_tokens

    def prepare_prefix_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        return prompt_tokens + adv_tokens[self.slice:] 