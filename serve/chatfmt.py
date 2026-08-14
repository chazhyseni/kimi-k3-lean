"""chatfmt.py -- convert OpenAI Chat Completions messages <-> token ids.

OpenAI's Chat Completions format is a list of messages with roles
(system, user, assistant, tool). We flatten them into a single prompt
suitable for the article's engine, which has no native chat template.

The article's model expects:

  <|system|>...\n<|user|>...\n<|assistant|>

The exact token sequences are a host-side convention. For now we use a
simple, unambiguous format:

  <|im_start|>system\n{system_content}<|im_end|>\n
  <|im_start|>user\n{user_content}<|im_end|>\n
  <|im_start|>assistant\n{assistant_content}<|im_end|>\n
  ...

If your downstream fine-tunes used a different chat template, override
`format_chat` with one that emits the matching tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .engine import Engine


IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"


@dataclass
class Message:
    """One OpenAI Chat Completions message."""
    role: str           # "system" | "user" | "assistant" | "tool" | "developer"
    content: str
    name: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        role = (d.get("role") or "user").lower()
        # content can be a string or a list of parts (vision, multimodal).
        # For the lean engine we only handle string content.
        c = d.get("content")
        if isinstance(c, list):
            text_parts = [
                p.get("text", "") for p in c
                if isinstance(p, dict) and p.get("type") in ("text", None)
            ]
            content = "\n".join(text_parts)
        else:
            content = c or ""
        return cls(role=role, content=content, name=d.get("name"))


def format_chat(messages: Iterable[Message]) -> str:
    """Flatten messages into a single prompt string.

    Returns the text (not token ids); tokenization is done by the caller
    so the same string can be inspected or rerouted.
    """
    out: list[str] = []
    for m in messages:
        # Tool messages become a system note; tool-calling is not yet
        # wired in this lean build. Future work: parse tool_calls blocks.
        role = m.role
        if role == "tool":
            role = "system"
        if role not in ("system", "user", "assistant", "developer"):
            # Unknown role -> treat as user to avoid dropping content.
            role = "user"

        out.append(f"{IM_START}{role}\n{m.content}{IM_END}\n")

    # End with assistant turn so the model continues.
    out.append(f"{IM_START}assistant\n")
    return "".join(out)


def flatten(messages: list[dict]) -> str:
    """Convenience: list of dicts -> prompt string."""
    return format_chat(Message.from_dict(d) for d in messages)


def build_prompt(
    engine: Engine,
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Build a token-id prompt from OpenAI-shape messages.

    `add_generation_prompt=True` is the OpenAI Chat Completions behavior:
    the prompt ends with `<|im_start|>assistant\n` so the model produces
    the assistant's reply.
    """
    text = flatten(messages)
    if not add_generation_prompt and not text.endswith(IM_END + "\n"):
        # Caller asked for a closing assistant token already; nothing to do.
        pass
    return engine.tokenize(text)


def parse_message_delta(delta_text: str) -> str | None:
    """Pass-through for now; future: filter control tokens from streams.

    Returns None if the delta should not be emitted to the user
    (e.g. role labels). Returns the cleaned text otherwise.
    """
    if delta_text is None:
        return None
    return delta_text


__all__ = ["Message", "format_chat", "flatten", "build_prompt", "parse_message_delta",
           "IM_START", "IM_END"]