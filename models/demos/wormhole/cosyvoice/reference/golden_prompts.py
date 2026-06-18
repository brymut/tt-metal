# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sample texts for the reference golden path, organized by language and mode.

Each language has 3 strings:
  - text:        the tts text to synthesize
  - prompt_text: matching transcript of a reference audio (used for zero_shot only)
  - instruct:    an instruction string (used for instruct only)

Whisper language tags are prepended for non-English languages to match the
official CosyVoice examples (e.g. ``<|zh|>...``, ``<|jp|>...``). The tag is
encoded as a single special token (no whitespace between ``|`` markers) and is
stripped from the encoded text length since the LLM consumes it normally.
"""

# (lang, text, prompt_text, instruct)
SAMPLES = {
    "en": (
        "Hello world, this is a test of English synthesis.",
        "Hello there, this is a sample reference sentence.",
        "Speak in a calm and friendly tone.",
    ),
    "zh": (
        "<|zh|>今天天气真好, 我们一起去公园散步吧。",
        "<|zh|>你好, 这是一段中文的参考语音。",
        "<|zh|>请用温柔舒缓的语气朗读。",
    ),
    "ja": (
        "<|jp|>こんにちは、世界の皆さん、お元気ですか?",
        "<|jp|>こんにちは、こちらは日本語の参考音声です。",
        "<|jp|>落ち着いた丁寧な声で話してください。",
    ),
    "yue": (
        "<|yue|>你好嗎, 我哋而家去邊度食飯呀?",
        "<|yue|>你好, 呢段係粵語嘅參考語音。",
        "<|yue|>用輕鬆愉快嘅語氣講嘢。",
    ),
    "ko": (
        "<|ko|>안녕하세요, 오늘 날씨가 정말 좋네요.",
        "<|ko|>안녕, 이건 한국어 참조 음성이야.",
        "<|ko|>차분하고 친근한 톤으로 말해주세요.",
    ),
}

LANGUAGES = list(SAMPLES.keys())
MODES = ["sft", "zero_shot", "cross_lingual", "instruct"]


def case_id(mode: str, lang: str) -> str:
    return f"{mode}_{lang}"


def text_for(mode: str, lang: str) -> str:
    return SAMPLES[lang][0]


def prompt_text_for(mode: str, lang: str) -> str:
    """Prompt transcript for zero_shot / instruct (cross_lingual has no prompt_text in LLM)."""
    return SAMPLES[lang][1]


def instruct_text_for(mode: str, lang: str) -> str:
    return SAMPLES[lang][2]
