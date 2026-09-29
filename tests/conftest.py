"""标准库替身客户端与公共 fixture。绝不联网：所有 OpenAI 调用都落到 FakeClient。"""
from types import SimpleNamespace

import pytest

from backend.services.ai_service import AIService


class FakeClient:
    """顶替 svc.client。

    - chat_script / transcribe_script: 按调用顺序消费的脚本。
      元素是 str -> 作为 message.content 返回；
      元素是 Exception 实例 -> 原样抛出；
      元素是其他对象 -> 原样返回（转写用）。
    - chat_calls: 记录每次 chat.completions.create 的 model/messages/temperature。
    - transcribe_calls: 记录每次 audio.transcriptions.create 的关键字参数。
    """

    def __init__(self, chat_script=None, transcribe_script=None):
        self.chat_script = list(chat_script or [])
        self.transcribe_script = list(transcribe_script or [])
        self.chat_calls = []
        self.transcribe_calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._chat_create)
        )
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe_create)
        )

    @property
    def call_count(self):
        return len(self.chat_calls)

    async def _chat_create(self, *, model, messages, temperature, **kwargs):
        self.chat_calls.append(
            {"model": model, "messages": messages, "temperature": temperature}
        )
        if not self.chat_script:
            raise AssertionError("FakeClient: 意外的 chat.completions.create 调用")
        action = self.chat_script.pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=action))]
        )

    async def _transcribe_create(self, **kwargs):
        self.transcribe_calls.append(kwargs)
        if not self.transcribe_script:
            raise AssertionError("FakeClient: 意外的 transcriptions.create 调用")
        action = self.transcribe_script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def make_transcript(text, segments):
    """构造 whisper verbose_json 风格的返回对象。"""
    return SimpleNamespace(text=text, segments=segments)


@pytest.fixture
def svc():
    service = AIService()
    service.client = FakeClient()
    return service
