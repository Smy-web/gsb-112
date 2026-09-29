"""T1-T8 契约测试。断言全部落在可观察产出上。"""
import logging
import sys
import types
from types import SimpleNamespace

import pytest

import backend.services.ai_service as ai_module
from backend.services.ai_service import (
    AIService,
    AIServiceError,
    ROLE_WHITELIST,
    SPEAKER_NAMES,
)
from tests.conftest import FakeClient, make_transcript

# 模拟 openai SDK 401 异常的 str()，里面带着密钥片段
LEAKY_401 = (
    "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
    "sk-abc1****xyz', 'type': 'invalid_request_error', 'param': None, "
    "'code': 'invalid_api_key'}}"
)


# ---------------- T1 失败不许伪装成正文，凭据不许外泄 ----------------

async def test_t1_generate_summary_success_returns_plain_text(svc):
    svc.client.chat_script = ["会议讨论了木构腐朽与彩绘剥落。"]
    result = await svc.generate_meeting_summary(
        "甲：大殿木构腐朽。", {"name": "大雄宝殿"}
    )
    assert result == "会议讨论了木构腐朽与彩绘剥落。"
    (call,) = svc.client.chat_calls
    assert call["model"] == "gpt-4o"
    assert call["temperature"] == 0.7
    assert "大殿木构腐朽" in call["messages"][0]["content"]
    assert "大雄宝殿" in call["messages"][0]["content"]


@pytest.mark.parametrize(
    "method,args",
    [
        ("generate_meeting_summary", ("对话内容", None)),
        ("generate_fundraising_copy", ("纪要", {"name": "观音殿"})),
        ("generate_merit_list", ("募捐文案",)),
    ],
)
async def test_t1_generate_failure_raises_and_sanitizes(svc, method, args):
    """负例钉点：若改回 return f'...{str(e)}'，pytest.raises 立刻变红；
    若脱敏被去掉，密钥片段断言立刻变红。"""
    svc.client.chat_script = [RuntimeError(LEAKY_401)]
    with pytest.raises(AIServiceError) as exc_info:
        await getattr(svc, method)(*args)
    message = str(exc_info.value)
    assert "失败" in message  # 可读原因
    assert "401" in message  # 保留非敏感上下文
    assert "abc1" not in message
    assert "xyz" not in message
    assert "sk-abc1****xyz" not in message


async def test_t1_generate_failure_does_not_leak_configured_key(svc, monkeypatch):
    monkeypatch.setattr(
        ai_module.settings, "OPENAI_API_KEY", "sk-live-secret-key-123"
    )
    svc.client.chat_script = [
        RuntimeError("401 Incorrect API key provided: sk-live-secret-key-123")
    ]
    with pytest.raises(AIServiceError) as exc_info:
        await svc.generate_merit_list("募捐文案")
    assert "sk-live-secret-key-123" not in str(exc_info.value)


# ---------------- T2 transcribe_audio 的失败必须让上游停得下来 ----------------

async def test_t2_transcribe_success_shape(svc, tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake-audio")
    svc.client.transcribe_script = [
        make_transcript(
            "全文转写",
            [
                {"start": 0.0, "end": 1.5, "text": "第一段"},
                {"start": 1.5, "end": 3.0, "text": "第二段"},
            ],
        )
    ]
    result = await svc.transcribe_audio(str(audio))
    assert set(result.keys()) == {"transcript", "segments"}
    assert result["transcript"] == "全文转写"
    assert len(result["segments"]) == 2
    for seg in result["segments"]:
        assert set(seg.keys()) == {"start", "end", "text"}
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)
    assert result["segments"][0] == {"start": 0.0, "end": 1.5, "text": "第一段"}
    (call,) = svc.client.transcribe_calls
    assert call["model"] == "whisper-1"


async def test_t2_transcribe_missing_file_raises(svc, tmp_path):
    with pytest.raises(AIServiceError) as exc_info:
        await svc.transcribe_audio(str(tmp_path / "gone.wav"))
    assert "转写" in str(exc_info.value)
    assert svc.client.transcribe_calls == []


async def test_t2_transcribe_api_error_raises_sanitized(svc, tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake-audio")
    svc.client.transcribe_script = [RuntimeError(LEAKY_401)]
    with pytest.raises(AIServiceError) as exc_info:
        await svc.transcribe_audio(str(audio))
    message = str(exc_info.value)
    assert "abc1" not in message
    assert "xyz" not in message


# ---------------- T3 diarize_speakers 不许把哨兵混进分段列表 ----------------

async def test_t3_diarize_unavailable_returns_empty_and_warns(svc, caplog):
    """本环境没有 pyannote，必然走 ImportError 路径。"""
    with caplog.at_level(logging.WARNING):
        result = await svc.diarize_speakers("any.wav")
    assert result == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "pyannote" in warnings[0].getMessage()


async def test_t3_diarize_with_pyannote_returns_segments(svc, monkeypatch):
    turn = SimpleNamespace(start=0.0, end=4.5)

    class FakeDiarization:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield turn, None, "SPEAKER_00"

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, path, num_speakers=None):
            return FakeDiarization()

    fake_pkg = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    fake_pkg.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    result = await svc.diarize_speakers("any.wav")
    assert result == [{"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"}]
    for seg in result:
        assert set(seg.keys()) == {"start", "end", "speaker"}
        assert "error" not in seg


def test_t3_empty_diarization_aligns_to_unknown(svc):
    aligned = svc.align_transcript_with_speakers(
        [
            {"start": 0.0, "end": 1.0, "text": "第一句"},
            {"start": 1.0, "end": 2.0, "text": "第二句"},
        ],
        [],
    )
    assert len(aligned) == 2
    assert [seg["speaker"] for seg in aligned] == ["UNKNOWN", "UNKNOWN"]


# ---------------- T4 identify_speaker_roles 的返回值必须落在白名单里 ----------------

async def test_t4_exact_and_whitespace_roles_normalized(svc):
    svc.client.chat_script = ["工匠", "  居士\n"]
    roles = await svc.identify_speaker_roles(
        [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "木构"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "发心"},
        ]
    )
    assert roles == {"SPEAKER_00": "工匠", "SPEAKER_01": "居士"}
    assert set(roles.values()) <= set(ROLE_WHITELIST)


async def test_t4_verbose_sentence_normalized_to_whitelist(svc):
    """负例钉点：若改回模型原文照存，长句断言立刻变红。"""
    svc.client.chat_script = [
        "这个人应该是负责木作的老工匠师傅，语气比较像住持",
        "法师",
        "……",
    ]
    roles = await svc.identify_speaker_roles(
        [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02", "text": "c"},
        ]
    )
    # 句中「工匠」出现位置早于「住持」，按出现先后取最前者
    assert roles["SPEAKER_00"] == "工匠"
    assert roles["SPEAKER_01"] == "其他"  # 白名单外的词
    assert roles["SPEAKER_02"] == "其他"  # 只有标点
    assert set(roles.values()) <= set(ROLE_WHITELIST)


async def test_t4_request_failure_warns_and_falls_back(svc, caplog):
    svc.client.chat_script = [RuntimeError(LEAKY_401)]
    with caplog.at_level(logging.WARNING):
        roles = await svc.identify_speaker_roles(
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "a"}]
        )
    assert roles == {"SPEAKER_00": "其他"}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "SPEAKER_00" in message
    assert "401" in message
    assert "abc1" not in message
    assert "xyz" not in message


# ---------------- T5 UNKNOWN 不发请求 ----------------

async def test_t5_unknown_speaker_makes_no_request(svc):
    svc.client.chat_script = ["住持"]
    roles = await svc.identify_speaker_roles(
        [
            {"start": 0.0, "end": 1.0, "speaker": "UNKNOWN", "text": "大家好"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "阿弥陀佛"},
        ]
    )
    assert svc.client.call_count == 1
    assert roles == {"SPEAKER_00": "住持"}
    assert "UNKNOWN" not in roles
    prompt = svc.client.chat_calls[0]["messages"][0]["content"]
    assert "UNKNOWN" not in prompt


async def test_t5_only_unknown_makes_zero_requests(svc):
    roles = await svc.identify_speaker_roles(
        [{"start": 0.0, "end": 1.0, "speaker": "UNKNOWN", "text": "大家好"}]
    )
    assert svc.client.call_count == 0
    assert roles == {}


# ---------------- T6 align_transcript_with_speakers 输入容错与输出形状 ----------------

def test_t6_align_success_shape_order_and_error_sentinel_skipped(svc):
    transcript = [
        {"start": 0.0, "end": 4.0, "text": "第一段"},
        {"start": 5.0, "end": 9.0, "text": "第二段"},
        {"start": 20.0, "end": 25.0, "text": "第三段"},
    ]
    speakers = [
        {"error": "No module named 'pyannote'"},  # 哨兵必须被跳过
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    aligned = svc.align_transcript_with_speakers(transcript, speakers)
    assert len(aligned) == 3
    assert [seg["text"] for seg in aligned] == ["第一段", "第二段", "第三段"]
    assert aligned[0]["speaker"] == "SPEAKER_00"
    assert aligned[1]["speaker"] == "SPEAKER_01"
    assert aligned[2]["speaker"] == "UNKNOWN"
    for seg in aligned:
        assert set(seg.keys()) == {"start", "end", "speaker", "text"}
        assert isinstance(seg["start"], (int, float))
        assert isinstance(seg["end"], (int, float))
        f"{seg['start']:.1f}"  # 上游拼 timestamp 的格式化必须可用


def test_t6_missing_text_raises_readable_error(svc):
    with pytest.raises(AIServiceError) as exc_info:
        svc.align_transcript_with_speakers(
            [
                {"start": 0.0, "end": 1.0, "text": "正常段"},
                {"start": 1.0, "end": 2.0},  # 人工补录缺 text
            ],
            [],
        )
    message = str(exc_info.value)
    assert "text" in message
    assert "1" in message  # 指出是第几段


def test_t6_non_numeric_time_raises_readable_error(svc):
    with pytest.raises(AIServiceError) as exc_info:
        svc.align_transcript_with_speakers(
            [{"start": "12.5", "end": 13.0, "text": "字符串时间"}], []
        )
    message = str(exc_info.value)
    assert "start" in message
    assert "12.5" in message


# ---------------- T7 SPEAKER_NAMES 要覆盖任意 SPEAKER_<数字> ----------------

def test_t7_existing_speaker_names_preserved():
    assert SPEAKER_NAMES == {
        "SPEAKER_00": "甲",
        "SPEAKER_01": "乙",
        "SPEAKER_02": "丙",
        "SPEAKER_03": "丁",
        "SPEAKER_04": "戊",
    }


async def test_t7_speaker_05_gets_chinese_alias_in_prompt(svc):
    svc.client.chat_script = ["工匠"]
    await svc.identify_speaker_roles(
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_05", "text": "木构"}]
    )
    prompt = svc.client.chat_calls[0]["messages"][0]["content"]
    assert "SPEAKER_05" not in prompt
    assert "己" in prompt  # 天干第六位


async def test_t7_non_pyannote_speaker_kept_verbatim(svc):
    svc.client.chat_script = ["居士"]
    await svc.identify_speaker_roles(
        [{"start": 0.0, "end": 1.0, "speaker": "法师A", "text": "发心"}]
    )
    prompt = svc.client.chat_calls[0]["messages"][0]["content"]
    assert "法师A" in prompt


# ---------------- T8 离线 import 必须成功 ----------------

def test_t8_module_singleton_and_client_attr():
    assert isinstance(ai_module.ai_service, AIService)
    assert hasattr(ai_module.ai_service, "client")
    assert ai_module.ai_service.client is not None
    assert isinstance(ai_module.SPEAKER_NAMES, dict)
