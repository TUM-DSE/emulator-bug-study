from __future__ import annotations

from argparse import ArgumentParser
from datetime import timedelta
from os import makedirs, path
import sys
from time import monotonic
from typing import Any

from buglib import install_error_handler, list_files_recursive


def write_output(
    text: str,
    category: str,
    labels: list[str],
    scores: list[float],
    identifier: str,
    output_dir: str,
    start_time: float,
    reasoning: str | None = None,
) -> None:
    print(f"Category: {category}, Time: {timedelta(seconds=monotonic() - start_time)}")
    file_path = path.join(output_dir, category, identifier)
    makedirs(path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        for label, score in zip(labels, scores):
            if label == "SPLIT":
                file.write("--------------------\n")
            else:
                file.write(f"{label}: {score:.3f}\n")

        file.write("\n")
        file.write(text)

    if reasoning:
        reasoning_dir = path.join(path.dirname(output_dir), "reasoning")
        file_path = path.join(reasoning_dir, category, identifier)
        makedirs(path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            file.write(reasoning)


def get_category(
    classification: dict[str, Any],
    positive_categories: list[str],
    negative_categories: list[str],
    architectures: list[str],
    multi_label: bool,
) -> str:
    highest_category: str = classification['labels'][0]

    if not multi_label:
        return highest_category

    if all(i < 0.8 for i in classification["scores"]):
        return "none"
    elif classification["scores"][0] - classification["scores"][-1] <= 0.2:
        return "unknown"

    result = highest_category
    arch: str | None = None
    pos: str | None = None
    for label, score in zip(classification["labels"], classification["scores"]):
        if label in negative_categories and (not arch and not pos or score >= 0.92):
            return label

        if label in positive_categories and not pos and score > 0.8:
            pos = label
            if not arch:
                result = label
            else:
                result = label + "-" + arch

        if label in architectures and not arch and score > 0.8:
            arch = label
            if pos:
                result = pos + "-" + label

    return result


def compare_category(
    classification: dict[str, Any],
    category: str,
    positive_categories: list[str],
) -> str:
    for label, score in zip(classification["labels"], classification["scores"]):
        if label in positive_categories and score >= 0.85:
            return category
        if label in category and score >= 0.85:
            return category

    return "review"


def main() -> None:
    install_error_handler()
    parser = ArgumentParser(prog='bug-classifier')
    parser.add_argument('-i', '--input-dir', required=True, action='append', help="Input directory containing bug files (repeatable)")
    parser.add_argument('-o', '--output-dir', default='output', help="Output directory (default: output)")

    parser.add_argument('--backend', choices=['zero-shot', 'ollama', 'anthropic', 'pi'], default='zero-shot', help="Classification backend (default: zero-shot)")
    parser.add_argument('--model', type=str, help="Model name (default depends on backend)")
    parser.add_argument('--preamble', type=str, help="Path to preamble/prompt file (required for ollama)")
    parser.add_argument('--compare', nargs='?', const="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", type=str, help="Second model for cross-validation (zero-shot only)")
    parser.add_argument('--pi-binary', default='pi', help="pi executable to run (pi backend only, default: pi)")
    parser.add_argument('--thinking', default='off', choices=['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'], help="Thinking level (pi backend only, default: off)")
    parser.add_argument('--timeout', type=float, default=600, help="Seconds to wait for each pi response before restarting pi (pi backend only, default: 600)")
    parser.add_argument('-m', '--multi-label', action='store_true', help="Enable multi-label classification (zero-shot only)")

    parser.add_argument('--config', type=str, help="Path to TOML category config file (overrides --positive/--negative/--architectures)")
    parser.add_argument('--positive', nargs='+', default=['semantic', 'TCG', 'assembly', 'architecture', 'mistranslation', 'register', 'user-level'], help="Positive category labels")
    parser.add_argument('--negative', nargs='+', default=['boot', 'network', 'kvm', 'vnc', 'graphic', 'device', 'socket', 'debug', 'files', 'PID', 'permissions', 'performance', 'kernel', 'peripherals', 'VMM', 'hypervisor', 'virtual', 'other'], help="Negative category labels")
    parser.add_argument('--architectures', nargs='+', default=['x86', 'arm', 'risc-v', 'i386', 'ppc'], help="Architecture labels")
    args = parser.parse_args()

    if args.config:
        from bug_classifier.config import load_config
        cfg = load_config(args.config)
        positive_categories = cfg.positive
        negative_categories = cfg.negative
        architectures = cfg.architectures
    else:
        positive_categories = args.positive
        negative_categories = args.negative
        architectures = args.architectures
    categories = positive_categories + negative_categories + architectures

    start_time = monotonic()

    from bug_classifier.backend import ClassificationError, ClassifierBackend
    backend: ClassifierBackend

    if args.backend == 'zero-shot':
        from bug_classifier.backend import ZeroShotBackend
        model = args.model or "facebook/bart-large-mnli"
        backend = ZeroShotBackend(
            model=model,
            multi_label=args.multi_label,
            positive=positive_categories,
            negative=negative_categories,
            architectures=architectures,
            compare_model=args.compare,
        )
        print(f"The model {model} will be used")
        if args.compare:
            print(f"The comparison model {args.compare} will be used")

    elif args.backend == 'ollama':
        from bug_classifier.backend import OllamaBackend
        if not args.preamble:
            parser.error("--preamble is required when using --backend ollama")
        model = args.model or "deepseek-r1:7b"
        with open(args.preamble, "r") as file:
            preamble = file.read()
        backend = OllamaBackend(model=model, preamble=preamble)
        print(f"The model {model} will be used")

    elif args.backend == 'anthropic':
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit(
                "error: --backend anthropic requires the ANTHROPIC_API_KEY "
                "environment variable to be set."
            )
        from bug_classifier.backend import AnthropicBackend
        if not args.preamble:
            parser.error("--preamble is required when using --backend anthropic")
        model = args.model or "claude-sonnet-4-20250514"
        with open(args.preamble, "r") as file:
            preamble = file.read()
        backend = AnthropicBackend(model=model, preamble=preamble)
        print(f"The model {model} will be used")

    elif args.backend == 'pi':
        import shutil
        if shutil.which(args.pi_binary) is None:
            sys.exit(
                f"error: --backend pi requires the `{args.pi_binary}` coding agent to be installed.\n"
                "Run the classifier via `nix run .#bug-classifier-full` or enter "
                "`nix develop .#full` and re-run."
            )
        from bug_classifier.backend import PiBackend
        if not args.preamble:
            parser.error("--preamble is required when using --backend pi")
        model = args.model or "claude-haiku-4-5"
        with open(args.preamble, "r") as file:
            preamble = file.read()
        backend = PiBackend(model=model, preamble=preamble, binary=args.pi_binary,
                            thinking=args.thinking, timeout=args.timeout)
        print(f"The model {model} will be used via {args.pi_binary} with thinking {args.thinking}")

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    processed_bugs = list_files_recursive(args.output_dir, True)

    bugs: list[str] = []
    for input_dir in args.input_dir:
        bugs = bugs + list_files_recursive(input_dir)

    print(f"{len(bugs)} number of bugs will be processed")
    failures = consecutive = 0
    for i, bug in enumerate(bugs):
        print(f"Bug: {bug}, Number: {i+1},", end=" ")

        if path.basename(bug) in processed_bugs:
            print("skipped")
            continue

        with open(bug, "r") as file:
            text = file.read()

        try:
            result = backend.classify(text, categories)
        except ClassificationError as error:
            # Write nothing, so that resuming the run classifies this bug again.
            print(f"failed: {error}")
            failures += 1
            consecutive += 1
            if consecutive == 3:
                break
            continue
        consecutive = 0
        write_output(text, result.category, result.labels, result.scores,
                     path.basename(bug), args.output_dir, start_time, result.reasoning)

    end_time = monotonic()
    print(timedelta(seconds=end_time - start_time))
    if failures:
        stopped = " (stopped after 3 consecutive failures)" if consecutive == 3 else ""
        sys.exit(f"error: {failures} bugs failed{stopped}; rerun to retry them")


if __name__ == "__main__":
    main()
