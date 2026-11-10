import json
from pathlib import Path
import pytest
import sys
from typing import Any
from unittest.mock import patch, MagicMock
from bug_classifier.backend import (
    ClassifierBackend,
    ClassificationResult,
)


POSITIVE = ['semantic', 'TCG', 'assembly', 'architecture', 'mistranslation', 'register', 'user-level']
NEGATIVE = ['boot', 'network', 'kvm', 'vnc', 'graphic', 'device', 'socket', 'debug', 'files', 'PID', 'permissions', 'performance', 'kernel', 'peripherals', 'VMM', 'hypervisor', 'virtual', 'other']
ARCHITECTURES = ['x86', 'arm', 'risc-v', 'i386', 'ppc']
CATEGORIES = POSITIVE + NEGATIVE + ARCHITECTURES


class TestClassificationResult:
    def test_defaults(self):
        r = ClassificationResult(category="network", labels=["network"], scores=[0.9])
        assert r.category == "network"
        assert r.reasoning is None

    def test_with_reasoning(self):
        r = ClassificationResult(category="boot", labels=[], scores=[], reasoning="because")
        assert r.reasoning == "because"


class TestClassifierBackend:
    def test_base_raises(self):
        backend = ClassifierBackend()
        with pytest.raises(NotImplementedError):
            backend.classify("text", ["a", "b"])


class TestZeroShotBackend:
    def _make_backend(self, primary_result: object, multi_label: bool = False, compare_result: object = None):
        """Create a ZeroShotBackend with mocked transformers.pipeline."""
        primary_pipe = MagicMock(return_value=primary_result)
        compare_pipe = MagicMock(return_value=compare_result) if compare_result else None

        pipes = iter([primary_pipe] + ([compare_pipe] if compare_pipe else []))
        mock_pipeline = MagicMock(side_effect=lambda *a, **kw: next(pipes))

        # Inject a fake transformers module so the import inside __init__ works
        fake_transformers = MagicMock()
        fake_transformers.pipeline = mock_pipeline
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            from bug_classifier.backend import ZeroShotBackend
            backend = ZeroShotBackend(
                model="test-model",
                multi_label=multi_label,
                positive=POSITIVE,
                negative=NEGATIVE,
                architectures=ARCHITECTURES,
                compare_model="compare-model" if compare_result else None,
            )

        return backend

    def test_single_label_returns_highest(self):
        backend = self._make_backend({
            'labels': ['network', 'boot', 'kvm'],
            'scores': [0.95, 0.5, 0.3],
        })

        result = backend.classify("some bug text", CATEGORIES)
        assert result.category == 'network'
        assert result.labels == ['network', 'boot', 'kvm']
        assert result.scores == [0.95, 0.5, 0.3]
        assert result.reasoning is None

    def test_multi_label_positive_with_arch(self):
        backend = self._make_backend(
            {'labels': ['mistranslation', 'x86', 'boot'], 'scores': [0.9, 0.85, 0.3]},
            multi_label=True,
        )

        result = backend.classify("some bug text", CATEGORIES)
        assert result.category == 'mistranslation-x86'

    def test_multi_label_all_low_returns_none(self):
        backend = self._make_backend(
            {'labels': ['network', 'boot'], 'scores': [0.7, 0.6]},
            multi_label=True,
        )

        result = backend.classify("some bug text", CATEGORIES)
        assert result.category == 'none'

    def test_compare_model_keeps_positive(self):
        backend = self._make_backend(
            {'labels': ['semantic', 'boot', 'network'], 'scores': [0.9, 0.5, 0.3]},
            compare_result={'labels': ['semantic', 'boot'], 'scores': [0.9, 0.5]},
        )

        result = backend.classify("some bug text", CATEGORIES)
        assert result.category == 'semantic'
        assert 'SPLIT' in result.labels

    def test_compare_model_returns_review(self):
        backend = self._make_backend(
            {'labels': ['semantic', 'boot', 'network'], 'scores': [0.9, 0.5, 0.3]},
            compare_result={'labels': ['boot', 'network'], 'scores': [0.5, 0.3]},
        )

        result = backend.classify("some bug text", CATEGORIES)
        assert result.category == 'review'


class TestOllamaBackend:
    def _mock_chat(self, content: str):
        """Patch ollama.chat to return a canned response."""
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = {'message': {'content': content}}
        return patch.dict(sys.modules, {"ollama": fake_ollama}), fake_ollama.chat

    def test_valid_category(self):
        ctx, _ = self._mock_chat('The category is network')
        with ctx:
            from bug_classifier.backend import OllamaBackend as OB
            backend = OB(model="test-model", preamble="classify this")
            result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'network'
        assert result.labels == []
        assert result.scores == []
        assert result.reasoning == 'The category is network'

    def test_unknown_category_falls_back(self):
        ctx, _ = self._mock_chat('I think this is a banana')
        with ctx:
            from bug_classifier.backend import OllamaBackend as OB
            backend = OB(model="test-model", preamble="classify this")
            result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'manual-review'

    def test_strips_non_alpha(self):
        ctx, _ = self._mock_chat('Result: **boot**!')
        with ctx:
            from bug_classifier.backend import OllamaBackend as OB
            backend = OB(model="test-model", preamble="classify this")
            result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'boot'

    def test_preamble_appended_to_text(self):
        ctx, mock_chat = self._mock_chat('network')
        with ctx:
            from bug_classifier.backend import OllamaBackend as OB
            backend = OB(model="test-model", preamble="MY PREAMBLE")
            backend.classify("bug text", CATEGORIES)

        content = mock_chat.call_args[0][1][0]['content']
        assert "bug text" in content
        assert "MY PREAMBLE" in content


class TestAnthropicBackend:
    def _make_backend(self, response_text: str):
        """Create an AnthropicBackend with a mocked Anthropic client."""
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_text)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        fake_anthropic = MagicMock()
        fake_anthropic.Anthropic.return_value = mock_client

        ctx = patch.dict(sys.modules, {"anthropic": fake_anthropic})
        ctx.start()
        from bug_classifier.backend import AnthropicBackend
        backend = AnthropicBackend(model="claude-test", preamble="classify this")
        ctx.stop()

        return backend, mock_client

    def test_valid_category(self):
        backend, _ = self._make_backend('The category is network')
        result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'network'
        assert result.labels == []
        assert result.scores == []
        assert result.reasoning == 'The category is network'

    def test_unknown_category_falls_back(self):
        backend, _ = self._make_backend('I think this is a banana')
        result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'manual-review'

    def test_strips_non_alpha(self):
        backend, _ = self._make_backend('Result: **boot**!')
        result = backend.classify("some bug", CATEGORIES)

        assert result.category == 'boot'

    def test_preamble_sent_as_system(self):
        backend, mock_client = self._make_backend('network')
        backend.classify("bug text", CATEGORIES)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs['system'] == 'classify this'
        assert call_kwargs['messages'][0]['content'] == 'bug text'
        assert call_kwargs['messages'][0]['role'] == 'user'

    def test_model_passed_through(self):
        backend, mock_client = self._make_backend('network')
        backend.classify("bug text", CATEGORIES)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs['model'] == 'claude-test'


class TestPiBackend:
    """Tests for PiBackend using a mocked subprocess."""

    NEW_SESSION = json.dumps({"type": "response", "command": "new_session", "success": True})
    PROMPT = json.dumps({"type": "response", "command": "prompt", "success": True})
    SETTLED = json.dumps({"type": "agent_settled", "aborted": False})

    def _agent_end(self, text: str, stop_reason: str = "stop", error: str | None = None):
        """Build an agent_end JSON line with the given assistant text."""
        message: dict[str, object] = {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stopReason": stop_reason,
        }
        if error:
            message["errorMessage"] = error
        return json.dumps({"type": "agent_end", "messages": [message]})

    def _run(self, *agent_ends: str) -> list[str]:
        """The events of one successful or failed prompt."""
        return [self.NEW_SESSION, self.PROMPT, *agent_ends, self.SETTLED]

    def _proc(self, stdout_lines: list[str]):
        proc = MagicMock()
        proc.stdin = MagicMock()
        # stdout is iterated line-by-line; simulate with a list iterator
        proc.stdout = iter(line + "\n" for line in stdout_lines)
        proc.poll.return_value = None
        return proc

    @pytest.fixture(autouse=True)
    def _agent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.agent_dir = tmp_path / "agent"
        self.agent_dir.mkdir()
        (self.agent_dir / "auth.json").write_text("{}")
        (self.agent_dir / "AGENTS.md").write_text("personal instructions")
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(self.agent_dir))

    def _make_backend(self, *processes: list[str], **kwargs: Any):
        """Create a PiBackend whose successive pi processes print the given lines."""
        procs = [self._proc(lines) for lines in processes]
        mock_popen = MagicMock(side_effect=procs)
        kwargs.setdefault("retry_delay", 0)
        with patch("shutil.which", return_value="/usr/bin/pi"), \
             patch("subprocess.Popen", mock_popen), \
             patch("time.sleep"):
            from bug_classifier.backend import PiBackend
            backend = PiBackend(model="test-model", preamble="classify this", **kwargs)
        return backend, procs, mock_popen

    def _classify(self, backend: Any, mock_popen: MagicMock):
        with patch("subprocess.Popen", mock_popen), patch("time.sleep"):
            return backend.classify("some bug", CATEGORIES)

    def test_valid_category(self):
        backend, _, popen = self._make_backend(self._run(self._agent_end("The category is network")))
        result = self._classify(backend, popen)

        assert result.category == "network"
        assert result.labels == []
        assert result.scores == []
        assert result.reasoning == "The category is network"

    def test_unknown_category_falls_back(self):
        backend, _, popen = self._make_backend(self._run(self._agent_end("I think this is a banana")))
        assert self._classify(backend, popen).category == "manual-review"

    def test_strips_non_alpha(self):
        backend, _, popen = self._make_backend(self._run(self._agent_end("Result: **boot**!")))
        assert self._classify(backend, popen).category == "boot"

    def test_non_json_lines_are_skipped(self):
        lines = self._run(self._agent_end("network"))
        lines[2:2] = ["some debug log output", "not valid json at all"]
        backend, _, popen = self._make_backend(lines)
        assert self._classify(backend, popen).category == "network"

    def test_string_content_block(self):
        """agent_end with plain string content blocks (not dict)."""
        event = json.dumps({
            "type": "agent_end",
            "messages": [{"role": "assistant", "content": ["boot"], "stopReason": "stop"}],
        })
        backend, _, popen = self._make_backend(self._run(event))
        assert self._classify(backend, popen).category == "boot"

    def test_starts_new_session_before_prompt(self):
        backend, procs, popen = self._make_backend(self._run(self._agent_end("network")))
        self._classify(backend, popen)

        writes = [json.loads(call.args[0]) for call in procs[0].stdin.write.call_args_list]
        assert writes == [{"type": "new_session"}, {"type": "prompt", "message": "some bug"}]

    def test_waits_for_settled_after_pi_retry(self):
        """pi retries transient errors itself; the answer is the run after the retry."""
        lines = self._run(self._agent_end("", "error", "429 rate limited"),
                          self._agent_end("network"))
        backend, _, popen = self._make_backend(lines)
        assert self._classify(backend, popen).category == "network"

    def test_error_is_retried(self):
        backend, _, popen = self._make_backend(
            self._run(self._agent_end("", "error", "usage limit")) + self._run(self._agent_end("boot")))
        assert self._classify(backend, popen).category == "boot"
        assert popen.call_count == 1

    def test_persistent_error_raises(self):
        from bug_classifier.backend import ClassificationError
        lines = self._run(self._agent_end("", "error", "usage limit"))
        backend, _, popen = self._make_backend(lines * 2, attempts=2)
        with pytest.raises(ClassificationError, match="usage limit"):
            self._classify(backend, popen)

    def test_truncated_response_raises(self):
        from bug_classifier.backend import ClassificationError
        backend, _, popen = self._make_backend(self._run(self._agent_end("netw", "length")),
                                               attempts=1)
        with pytest.raises(ClassificationError, match="length"):
            self._classify(backend, popen)

    def test_empty_response_raises(self):
        from bug_classifier.backend import ClassificationError
        event = json.dumps({"type": "agent_end", "messages": [
            {"role": "assistant", "content": [], "stopReason": "stop"}]})
        backend, _, popen = self._make_backend(self._run(event), attempts=1)
        with pytest.raises(ClassificationError, match="no text"):
            self._classify(backend, popen)

    def test_aborted_run_raises(self):
        from bug_classifier.backend import ClassificationError
        lines = self._run(self._agent_end("network"))
        lines[-1] = json.dumps({"type": "agent_settled", "aborted": True})
        backend, _, popen = self._make_backend(lines, attempts=1)
        with pytest.raises(ClassificationError, match="aborted"):
            self._classify(backend, popen)

    def test_rejected_prompt_raises(self):
        from bug_classifier.backend import ClassificationError
        lines = [self.NEW_SESSION, json.dumps(
            {"type": "response", "command": "prompt", "success": False, "error": "No API key"})]
        backend, _, popen = self._make_backend(lines, attempts=1)
        with pytest.raises(ClassificationError, match="No API key"):
            self._classify(backend, popen)

    def test_exited_pi_is_restarted(self):
        backend, procs, popen = self._make_backend(
            [self.NEW_SESSION, self.PROMPT], self._run(self._agent_end("network")))
        procs[0].poll.return_value = 1
        assert self._classify(backend, popen).category == "network"
        assert popen.call_count == 2

    def test_hung_pi_is_killed_and_restarted(self):
        import threading
        released = threading.Event()

        def hang():
            yield self.NEW_SESSION + "\n"
            released.wait()

        hung = self._proc([])
        hung.stdout = hang()

        def kill():
            hung.poll.return_value = -9
            released.set()

        hung.kill.side_effect = kill
        answering = self._proc(self._run(self._agent_end("network")))
        popen = MagicMock(side_effect=[hung, answering])
        with patch("shutil.which", return_value="/usr/bin/pi"), patch("subprocess.Popen", popen):
            from bug_classifier.backend import PiBackend
            backend = PiBackend(model="test-model", preamble="classify this",
                                retry_delay=0, timeout=0.2)

        assert self._classify(backend, popen).category == "network"
        hung.kill.assert_called_once()
        assert backend.proc is answering
        assert popen.call_count == 2

    def test_popen_flags(self):
        _, _, popen = self._make_backend([], thinking="low")

        args = popen.call_args[0][0]
        assert args[0] == "/usr/bin/pi"
        assert "--mode" in args and "rpc" in args
        assert "--no-session" in args
        assert "--no-tools" in args
        assert "--no-extensions" in args
        assert "--no-skills" in args
        assert "--no-context-files" in args
        assert args[args.index("--thinking") + 1] == "low"
        assert args[args.index("--model") + 1] == "test-model"
        assert args[args.index("--system-prompt") + 1] == "classify this"

    def test_binary_is_looked_up(self):
        with patch("shutil.which", return_value=None) as which:
            from bug_classifier.backend import PiBackend
            with pytest.raises(RuntimeError, match="pi binary not found on PATH: pi"):
                PiBackend(model="test-model", preamble="classify this", binary="pi")
        which.assert_called_once_with("pi")

    def test_agent_dir_holds_only_linked_auth(self):
        import os
        backend, _, popen = self._make_backend([])
        kwargs = popen.call_args[1]
        home = Path(kwargs["env"]["PI_CODING_AGENT_DIR"])

        assert kwargs["cwd"] == str(home)
        assert home != self.agent_dir
        assert os.listdir(home) == ["auth.json"]
        assert (home / "auth.json").resolve() == (self.agent_dir / "auth.json").resolve()
        backend.close()
        assert not home.exists()

    def test_close_terminates_subprocess(self):
        backend, procs, _ = self._make_backend([])
        backend.close()

        procs[0].stdin.close.assert_called_once()
        procs[0].terminate.assert_called_once()
        procs[0].wait.assert_called_once_with(timeout=5)

    def test_close_noop_when_already_exited(self):
        backend, procs, _ = self._make_backend([])
        procs[0].poll.return_value = 0  # already exited
        backend.close()

        procs[0].terminate.assert_not_called()
