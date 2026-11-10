import pytest
import os
import tempfile
from bug_classifier.main import get_category, compare_category, write_output
from pathlib import Path
from time import monotonic
from typing import Any
from unittest.mock import patch


POSITIVE = ['semantic', 'TCG', 'assembly', 'architecture', 'mistranslation', 'register', 'user-level']
NEGATIVE = ['boot', 'network', 'kvm', 'vnc', 'graphic', 'device', 'socket', 'debug', 'files', 'PID', 'permissions', 'performance', 'kernel', 'peripherals', 'VMM', 'hypervisor', 'virtual', 'other']
ARCHITECTURES = ['x86', 'arm', 'risc-v', 'i386', 'ppc']


class TestGetCategorySingleLabel:
    """Tests with multi_label=False: always returns the highest-scoring label."""

    def test_returns_highest(self):
        classification = {
            'labels': ['network', 'boot', 'kvm'],
            'scores': [0.95, 0.5, 0.3]
        }
        assert get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, False) == 'network'

    def test_positive_category(self):
        classification = {
            'labels': ['mistranslation', 'boot', 'kvm'],
            'scores': [0.95, 0.5, 0.3]
        }
        assert get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, False) == 'mistranslation'


class TestGetCategoryMultiLabel:
    """Tests with multi_label=True: applies threshold logic."""

    def test_all_low_scores(self):
        classification = {
            'labels': ['network', 'boot'],
            'scores': [0.7, 0.6]
        }
        assert get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, True) == 'none'

    def test_small_spread(self):
        classification = {
            'labels': ['network', 'boot'],
            'scores': [0.9, 0.85]
        }
        assert get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, True) == 'unknown'

    def test_negative_high_score(self):
        classification = {
            'labels': ['network', 'semantic', 'boot'],
            'scores': [0.95, 0.85, 0.3]
        }
        assert get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, True) == 'network'

    def test_positive_with_arch(self):
        classification = {
            'labels': ['mistranslation', 'x86', 'boot'],
            'scores': [0.9, 0.85, 0.3]
        }
        result = get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, True)
        assert result == 'mistranslation-x86'

    def test_arch_then_positive(self):
        classification = {
            'labels': ['x86', 'assembly', 'boot'],
            'scores': [0.9, 0.85, 0.3]
        }
        result = get_category(classification, POSITIVE, NEGATIVE, ARCHITECTURES, True)
        assert result == 'assembly-x86'


class TestCompareCategory:
    def test_positive_high_score_keeps_category(self):
        classification = {
            'labels': ['semantic', 'boot'],
            'scores': [0.9, 0.5]
        }
        assert compare_category(classification, 'semantic', POSITIVE) == 'semantic'

    def test_no_match_returns_review(self):
        classification = {
            'labels': ['boot', 'network'],
            'scores': [0.5, 0.3]
        }
        assert compare_category(classification, 'semantic', POSITIVE) == 'review'


class TestWriteOutput:
    def test_creates_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = monotonic()
            write_output("bug text", "network", ["network", "boot"], [0.9, 0.5], "bug123", tmpdir, start)
            out_path = os.path.join(tmpdir, "network", "bug123")
            assert os.path.exists(out_path)
            content = open(out_path).read()
            assert "network: 0.900" in content
            assert "bug text" in content

    def test_creates_reasoning_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = monotonic()
            write_output("bug text", "network", [], [], "bug123", tmpdir, start, reasoning="because reasons")
            reasoning_path = os.path.join(os.path.dirname(tmpdir), "reasoning", "network", "bug123")
            assert os.path.exists(reasoning_path)
            assert "because reasons" in open(reasoning_path).read()


class TestMainFailures:
    def _run_main(self, tmp_path: Path, outcomes: list[str | None]) -> tuple[str, list[str]]:
        from bug_classifier.backend import ClassificationError, ClassificationResult

        bugs = tmp_path / "bugs"
        bugs.mkdir()
        for i in range(len(outcomes)):
            (bugs / f"bug{i}").write_text(f"bug {i}")
        preamble = tmp_path / "preamble.txt"
        preamble.write_text("classify")
        output = tmp_path / "output"


        class FakeBackend:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def classify(self, text: str, categories: list[str], **kwargs: Any) -> ClassificationResult:
                outcome = outcomes[int(text.split()[1])]
                if outcome is None:
                    raise ClassificationError("usage limit")
                return ClassificationResult(category=outcome, labels=[], scores=[])

        argv = ["bug-classifier", "--backend", "ollama", "--preamble", str(preamble),
                "-i", str(bugs), "-o", str(output), "--positive", "boot", "--negative", "other",
                "--architectures", "x86"]
        from bug_classifier import main as main_module
        listing = main_module.list_files_recursive
        with patch("sys.argv", argv), patch("bug_classifier.backend.OllamaBackend", FakeBackend), \
             patch.object(main_module, "list_files_recursive",
                          lambda *a: sorted(listing(*a))):
            from bug_classifier.main import main
            with pytest.raises(SystemExit) as exit_info:
                main()
        written = sorted(p.name for p in output.rglob("*") if p.is_file()) if output.exists() else []
        return str(exit_info.value.code), written

    def test_failed_bug_is_not_written(self, tmp_path: Path):
        code, written = self._run_main(tmp_path, ["boot", None, "other"])
        assert "1 bugs failed" in code
        assert written == ["bug0", "bug2"]

    def test_stops_after_three_consecutive_failures(self, tmp_path: Path):
        code, written = self._run_main(tmp_path, ["boot", None, None, None, "other"])
        assert "stopped after 3 consecutive failures" in code
        assert written == ["bug0"]
