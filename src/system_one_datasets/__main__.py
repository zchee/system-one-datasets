"""Command line entry point: ``python -m system_one_datasets [run|report|build|validate|push] ...``."""

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from system_one_datasets import decisions
from system_one_datasets.build import JEV_BENCH_REVISION, build
from system_one_datasets.card import HF_REPO_ID
from system_one_datasets.client import SystemOneClient
from system_one_datasets.push import push
from system_one_datasets.report import render_report
from system_one_datasets.rows import load_records
from system_one_datasets.runner import run
from system_one_datasets.validate import (
    LIVE_SAMPLE_ROWS,
    data_files,
    live_check,
    render_summary,
    sample_records,
    validate_file,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger("system_one_datasets")

API_KEY_ENV = "TYPESAFE_API_KEY"


@dataclass(frozen=True, slots=True)
class Backend:
    """Default connection settings for a named backend."""

    base_url: str
    model: str
    api_key_env: str | None


BACKENDS: dict[str, Backend] = {
    "jev": Backend(base_url="https://api.typesafe.ai", model="jev-1.13.0", api_key_env=API_KEY_ENV),
    "local": Backend(base_url="http://localhost:3000", model="openjev", api_key_env=None),
}


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m system_one_datasets", description="Run the built datasets on a backend."
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--base-url", help="override the backend's default base URL")
    parser.add_argument("--model", help="override the backend's default model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="dataset root built by `build` (default: data/); rows are read from <suite>/<config>/<split>.jsonl",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="config name under --data; repeatable (default: every config that has the split)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N rows per config")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-attempt HTTP timeout in seconds")
    parser.add_argument("--out", type=Path, required=True, help="results JSONL (appended; completed ids are skipped)")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m system_one_datasets report", description="Markdown report over results JSONL files."
    )
    parser.add_argument("results", nargs="+", type=Path, help="results JSONL files")
    parser.add_argument("--out", type=Path, help="write the report here instead of stdout")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m system_one_datasets build",
        description="Regenerate the dataset files and manifest.yaml deterministically from pinned sources.",
    )
    parser.add_argument("--out", type=Path, required=True, help="dataset root, e.g. data/")
    parser.add_argument(
        "--jev-bench-revision",
        default=JEV_BENCH_REVISION,
        help="Praveenrajus/jev-bench commit, branch, or tag (default: the pinned commit)",
    )
    parser.add_argument(
        "--decisions-revision",
        default=decisions.DATASET_REVISION,
        help="samatv256/jev-decisions-v1 commit, branch, or tag (default: the pinned commit)",
    )
    return parser


def _validate_parser() -> argparse.ArgumentParser:
    local = BACKENDS["local"]
    parser = argparse.ArgumentParser(
        prog="python -m system_one_datasets validate",
        description="Check every <suite>/<config>/<split>.jsonl under a dataset root. Exits 1 on any structural "
        "error or, with --live, on any failed live request.",
    )
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/")
    parser.add_argument(
        "--live",
        action="store_true",
        help=f"also POST the first {LIVE_SAMPLE_ROWS} test rows of each config to a /v1/systemone backend",
    )
    parser.add_argument("--base-url", default=local.base_url, help=f"live backend origin (default: {local.base_url})")
    parser.add_argument("--model", default=local.model, help=f"model sent in live requests (default: {local.model})")
    parser.add_argument(
        "--api-key-env", help="environment variable holding a bearer token for the live backend (default: none)"
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds (default: 300)")
    return parser


def _push_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m system_one_datasets push",
        description="Validate the dataset root, then upload its card, manifest, and split files to a Hugging Face "
        "dataset repository. Uses the local `hf auth login` credentials.",
    )
    parser.add_argument("--repo", default=HF_REPO_ID, help=f"dataset repository id (default: {HF_REPO_ID})")
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset root (default: data/)")
    parser.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no network writes")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="also delete remote .jsonl files that are not in --data (default: never delete remote files)",
    )
    return parser


def _cmd_push(argv: Sequence[str]) -> int:
    args = _push_parser().parse_args(argv)
    return push(args.data, args.repo, public=args.public, dry_run=args.dry_run, prune=args.prune)


def _cmd_build(argv: Sequence[str]) -> int:
    args = _build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    build(args.out, args.jev_bench_revision, args.decisions_revision)
    logger.info("wrote %s", args.out / "manifest.yaml")
    return 0


def _cmd_validate(argv: Sequence[str]) -> int:
    args = _validate_parser().parse_args(argv)
    files = data_files(args.root)
    if not files:
        logger.error("no <suite>/<config>/<split>.jsonl files under %s", args.root)
        return 2
    results = [validate_file(path) for path in files]
    failed = any(r.errors for r in results)
    live = None
    if args.live:
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not api_key:
            logger.error("--api-key-env %s is not set", args.api_key_env)
            return 2
        live = {}
        with SystemOneClient(args.base_url, args.model, api_key=api_key, timeout=args.timeout, max_retries=1) as client:
            for r in results:
                key = f"{r.suite}/{r.config}"
                if r.split == "test" and not r.errors:
                    live[key] = live_check(key, sample_records(r.path), client)
        failed = failed or any(lr.failures for lr in live.values())
    sys.stdout.write(render_summary(results, live))
    return 1 if failed else 0


def _split_files(data: Path, split: str, configs: Sequence[str] | None) -> list[Path]:
    """Row files ``<suite>/<config>/<split>.jsonl`` under ``data``, in a stable order.

    Args:
        data: Dataset root written by ``build``.
        split: Split name.
        configs: Config names to keep, or ``None`` for every config that has the split.

    Returns:
        Matching file paths sorted by suite then config.
    """
    wanted = set(configs) if configs else None
    return sorted(path for path in data.glob(f"*/*/{split}.jsonl") if wanted is None or path.parent.name in wanted)


def _cmd_run(argv: Sequence[str]) -> int:
    args = _run_parser().parse_args(argv)
    backend = BACKENDS[args.backend]
    api_key = None
    if backend.api_key_env:
        api_key = os.environ.get(backend.api_key_env)
        if not api_key:
            logger.error("backend %s needs the %s environment variable", args.backend, backend.api_key_env)
            return 2
    files = _split_files(args.data, args.split, args.config)
    if not files:
        logger.error("no %s split found under %s for configs %s", args.split, args.data, args.config or "(all)")
        return 2
    records = [record for path in files for record in load_records(path)[: args.limit]]
    with SystemOneClient(
        args.base_url or backend.base_url, args.model or backend.model, api_key=api_key, timeout=args.timeout
    ) as client:
        summary = run(records, client, args.out, progress=not args.no_progress)
    logger.info(
        "%s: %d succeeded, %d failed, %d skipped (already done)",
        args.out,
        summary.succeeded,
        summary.failed,
        summary.skipped,
    )
    return 1 if summary.failed else 0


def _cmd_report(argv: Sequence[str]) -> int:
    args = _report_parser().parse_args(argv)
    missing = [str(p) for p in args.results if not p.is_file()]
    if missing:
        logger.error("results file(s) not found: %s", ", ".join(missing))
        return 2
    text = render_report(args.results)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to ``run`` (the default), ``report``, ``build``, ``validate``, or ``push``.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    commands = {"report": _cmd_report, "build": _cmd_build, "validate": _cmd_validate, "push": _cmd_push}
    if argv[:1] and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    if argv[:1] == ["run"]:
        argv = argv[1:]
    return _cmd_run(argv)


if __name__ == "__main__":
    sys.exit(main())
