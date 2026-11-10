"""Classification backend interface and implementations."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from re import compile as re_compile, sub, DOTALL
from typing import IO

_THINK_RE = re_compile(r'<think>.*?</think>', flags=DOTALL)


def parse_category(raw: str, categories: list[str]) -> str:
    """Extract the first valid category word from a model response.

    Strips <think>…</think> blocks first (DeepSeek-R1 / reasoning models),
    then returns the first word that matches a known category.  Falls back to
    'manual-review' if no match is found.
    """
    text = _THINK_RE.sub('', raw)
    for word in text.split():
        candidate = sub(r'[^a-zA-Z-]', '', word).lower()
        if candidate in categories:
            return candidate
    return "manual-review"


@dataclass
class ClassificationResult:
    """Result returned by a classification backend."""
    category: str
    labels: list[str]
    scores: list[float]
    reasoning: str | None = None


class ClassifierBackend:
    """Base class for classification backends."""

    def classify(self, text: str, categories: list[str], **kwargs: object) -> ClassificationResult:
        raise NotImplementedError


class ZeroShotBackend(ClassifierBackend):
    """HuggingFace zero-shot NLI classification."""

    def __init__(self, model: str, multi_label: bool,
                 positive: list[str], negative: list[str], architectures: list[str],
                 compare_model: str | None = None) -> None:
        from transformers import pipeline as _pipeline

        self.classifier = _pipeline("zero-shot-classification", model=model)
        self.multi_label = multi_label
        self.positive = positive
        self.negative = negative
        self.architectures = architectures
        self.compare_classifier = None
        if compare_model:
            self.compare_classifier = _pipeline("zero-shot-classification", model=compare_model)

    def classify(self, text: str, categories: list[str], **kwargs: object) -> ClassificationResult:
        from bug_classifier.main import get_category, compare_category

        result = self.classifier(text, categories, multi_label=self.multi_label)
        category = get_category(result, self.positive, self.negative, self.architectures, self.multi_label)

        if self.compare_classifier and sum(1 for c in self.positive if c in category) >= 1:
            compare_result = self.compare_classifier(text, categories, multi_label=self.multi_label)
            category = compare_category(compare_result, category, self.positive)

            result['labels'] = result['labels'] + ['SPLIT'] + compare_result['labels']
            result['scores'] = result['scores'] + [0] + compare_result['scores']

        return ClassificationResult(
            category=category,
            labels=result['labels'],
            scores=result['scores'],
        )


class OllamaBackend(ClassifierBackend):
    """Local LLM classification via Ollama."""

    def __init__(self, model: str, preamble: str) -> None:
        self.model = model
        self.preamble = preamble

    def classify(self, text: str, categories: list[str], **kwargs: object) -> ClassificationResult:
        from ollama import chat

        response = chat(self.model, [{'role': 'user', 'content': text + "\n" + self.preamble}])
        raw = response['message']['content']
        category = parse_category(raw, categories)

        return ClassificationResult(
            category=category,
            labels=[],
            scores=[],
            reasoning=raw,
        )


class AnthropicBackend(ClassifierBackend):
    """Classification via the Anthropic Messages API."""

    def __init__(self, model: str, preamble: str, max_tokens: int = 1024) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.model = model
        self.preamble = preamble
        self.max_tokens = max_tokens

    def classify(self, text: str, categories: list[str], **kwargs: object) -> ClassificationResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.preamble,
            messages=[{"role": "user", "content": text}],
        )
        raw = next(
            (t for b in response.content if isinstance(t := getattr(b, "text", None), str)),
            "",
        )
        category = parse_category(raw, categories)

        return ClassificationResult(
            category=category,
            labels=[],
            scores=[],
            reasoning=raw,
        )


class ClassificationError(RuntimeError):
    """A backend failed to produce a model response; nothing should be recorded."""


class PiBackend(ClassifierBackend):
    """Classification via pi coding agent in RPC mode.

    Pi runs with a private agent directory that only links the user's
    auth.json, so their AGENTS.md, APPEND_SYSTEM.md, settings and extensions
    never reach the classifier prompt.
    """

    _stdin: IO[str]
    _lines: queue.Queue[str | None]

    def __init__(self, model: str, preamble: str, binary: str = "pi", thinking: str = "off",
                 attempts: int = 4, retry_delay: float = 60.0, timeout: float = 600.0) -> None:
        import os
        import shutil
        import tempfile
        from pathlib import Path

        pi_bin = shutil.which(binary)
        if pi_bin is None:
            raise RuntimeError(
                f"pi binary not found on PATH: {binary}. "
                "Run via `nix run .#bug-classifier-full` or enter `nix develop .#full`."
            )

        self._home = tempfile.TemporaryDirectory(prefix="bug-classifier-pi-")
        source = Path(os.environ.get("PI_CODING_AGENT_DIR") or "~/.pi/agent").expanduser()
        auth = source / "auth.json"
        if auth.exists():
            # A link rather than a copy, so refreshed OAuth tokens reach the real file.
            os.symlink(auth.absolute(), Path(self._home.name) / "auth.json")
        self._env = {**os.environ, "PI_CODING_AGENT_DIR": self._home.name}
        self._command = [
            pi_bin,
            "--mode", "rpc",
            "--no-session",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--thinking", thinking,
            "--model", model,
            "--system-prompt", preamble,
        ]
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._start()

    def _start(self) -> None:
        # Run outside the repository so no project .pi/ configuration is picked up.
        self.proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=self._home.name,
            env=self._env,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._stdin = self.proc.stdin
        # Read on a thread so that waiting for pi can time out; None marks EOF.
        lines: queue.Queue[str | None] = queue.Queue()

        def read(stdout: IO[str] = self.proc.stdout) -> None:
            for line in stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read, daemon=True).start()
        self._lines = lines

    def _send(self, command: dict[str, object]) -> None:
        import json

        try:
            self._stdin.write(json.dumps(command) + "\n")
            self._stdin.flush()
        except OSError:
            raise ClassificationError("pi exited unexpectedly") from None

    def _events(self, deadline: float):
        import json

        while True:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                # A hung pi is killed; the retry starts a fresh process.
                self.proc.kill()
                self.proc.wait(timeout=5)
                raise ClassificationError(f"pi did not finish within {self.timeout:g}s") from None
            if line is None:
                raise ClassificationError("pi exited unexpectedly")
            try:
                event = json.loads(line.rstrip("\n"))
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event

    def _wait_response(self, command: str, deadline: float) -> None:
        for event in self._events(deadline):
            if event.get("type") == "response" and event.get("command") in (command, "parse"):
                if event.get("success") is False:
                    raise ClassificationError(f"pi rejected {command}: {event.get('error')}")
                return

    def _prompt(self, text: str) -> str:
        # Start every attempt from an empty session, even after a failed one.
        deadline = time.monotonic() + self.timeout
        self._send({"type": "new_session"})
        self._wait_response("new_session", deadline)
        self._send({"type": "prompt", "message": text})
        self._wait_response("prompt", deadline)

        # agent_end can be followed by pi's own retries; only agent_settled is final.
        last: dict[str, object] | None = None
        for event in self._events(deadline):
            if event.get("type") == "agent_end":
                for msg in reversed(event.get("messages") or []):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        last = msg
                        break
            elif event.get("type") == "agent_settled":
                if event.get("aborted"):
                    raise ClassificationError("pi aborted the run")
                break

        if last is None:
            raise ClassificationError("pi returned no assistant message")
        if last.get("stopReason", "stop") != "stop":
            raise ClassificationError(
                f"pi stopped with {last.get('stopReason')}: {last.get('errorMessage', 'no details')}")
        content = last.get("content")
        for block in reversed(content if isinstance(content, list) else []):
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
            if isinstance(block, str):
                return block
        raise ClassificationError("pi returned no text")

    def classify(self, text: str, categories: list[str], **kwargs: object) -> ClassificationResult:
        import sys

        attempt = 1
        while True:
            try:
                raw = self._prompt(text)
            except ClassificationError as error:
                if attempt >= self.attempts:
                    raise
                delay = self.retry_delay * 2 ** (attempt - 1)
                attempt += 1
                print(f"pi: {error}; retry {attempt}/{self.attempts} in {delay:.0f}s",
                      file=sys.stderr)
                if self.proc.poll() is not None:
                    self._start()
                time.sleep(delay)
                continue
            break

        return ClassificationResult(
            category=parse_category(raw, categories),
            labels=[],
            scores=[],
            reasoning=raw,
        )

    def close(self) -> None:
        """Terminate the pi subprocess and remove its agent directory."""
        if hasattr(self, 'proc') and self.proc.poll() is None:
            self._stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        if hasattr(self, '_home'):
            self._home.cleanup()

    def __del__(self) -> None:
        self.close()
