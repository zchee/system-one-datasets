"""Materialize ``data/``: canonical JSONL rows per ``<suite>/<config>/<split>`` plus ``manifest.yaml``.

The build is deterministic: sources are pinned to commit shas, sampling is seeded, and the manifest carries no
timestamps, so rebuilding at the same revisions reproduces every file byte for byte.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download
import orjson
import yaml

from system_one_datasets import decisions
from system_one_datasets.card import DESCRIPTION, render_card
from system_one_datasets.data import DATASET_ID as JEV_BENCH_ID, load_jev_bench, resolve_revision
from system_one_datasets.rows import CANONICAL_FIELDS, NO_SOFT_LABEL, encode_row
from system_one_datasets.validate import CHARS_PER_TOKEN, TOKEN_WARN_LIMIT


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from system_one_datasets.schema import JSONValue


logger = logging.getLogger(__name__)

# Commit of Praveenrajus/jev-bench (v0.1.1) the committed data/ was built from.
JEV_BENCH_REVISION = "18f88da81c28c2bec55edc31f63f2afdfba109ea"
JEV_BENCH_SPLITS: tuple[str, ...] = ("test", "validation")
AGENT_ACTION_SUITE = "agent_action"


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """Where one jev-bench config lands under ``data/``."""

    suite: str
    config: str


JEV_BENCH_LAYOUT: tuple[BenchConfig, ...] = (
    BenchConfig("moderation", "civil_comments"),
    BenchConfig("moderation", "measuring_hate_speech"),
    BenchConfig("moderation", "go_emotions"),
    BenchConfig("quality", "helpsteer2_helpfulness"),
    BenchConfig("quality", "stsb"),
)

JEV_BENCH_TRANSFORMATIONS: tuple[str, ...] = (
    "state and question are JSON text in the source and here; they are decoded and re-encoded compactly "
    "(orjson, key order preserved), so the decoded values are identical to the source's and can be sent to "
    "POST /v1/systemone as-is.",
    "options: choice = criteria keys in order; score = level indices '0'..'K-1'; noul = ['0', '1'].",
    "soft_label normalized to {option: probability}: noul scalar P(yes) p -> {'0': 1 - p, '1': p}; score array -> "
    "keyed by level index; choice object -> every option, absent options 0.0. Stored as JSON text; the text "
    "'null' when the source has none.",
    "id and label copied from the source; rows kept in source order; no rows added, dropped, or rewritten.",
    "upstream = the original dataset jev-bench built this config from (hf_id in jev-bench's manifest.json); "
    "license = that dataset's license as jev-bench states it, without the parenthetical note.",
)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Write rows as JSONL (orjson, one row per line) and describe the file.

    Args:
        path: Output file; parent directories are created.
        rows: Rows to write.

    Returns:
        Row count, rows with a soft label, and the file's SHA-256.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = soft = 0
    with path.open("wb") as f:
        for row in rows:
            line = orjson.dumps(row) + b"\n"
            f.write(line)
            digest.update(line)
            count += 1
            soft += row.get("soft_label") != NO_SOFT_LABEL
    return {"rows": count, "rows_with_soft_label": soft, "sha256": digest.hexdigest()}


def _license_id(text: str) -> str:
    """``'cc-by-sa-4.0 (STS Benchmark)'`` -> ``'cc-by-sa-4.0'``."""
    return text.split(" (", 1)[0].strip()


def _jev_bench_sources(revision: str) -> dict[str, dict[str, JSONValue]]:
    path = hf_hub_download(JEV_BENCH_ID, "manifest.json", repo_type="dataset", revision=revision)
    manifest = orjson.loads(Path(path).read_bytes())
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, dict):
        msg = f"{JEV_BENCH_ID}@{revision} manifest.json has no sources object"
        raise TypeError(msg)
    return sources


def build_jev_bench(out: Path, revision: str) -> list[dict[str, object]]:
    """Export the configured jev-bench configs and return their manifest entries.

    Args:
        out: Dataset root.
        revision: jev-bench commit sha.

    Returns:
        One manifest entry per config.

    Raises:
        KeyError: A configured config is missing from jev-bench's own manifest.
    """
    sources = _jev_bench_sources(revision)
    entries: list[dict[str, object]] = []
    for layout in JEV_BENCH_LAYOUT:
        info = sources.get(layout.config)
        if not isinstance(info, dict):
            msg = f"{layout.config} is not in {JEV_BENCH_ID}@{revision} manifest.json"
            raise KeyError(msg)
        license_text = str(info.get("license"))
        license_id = _license_id(license_text)
        kinds: Counter[str] = Counter()
        splits: dict[str, object] = {}
        for split in JEV_BENCH_SPLITS:
            records = load_jev_bench(layout.config, split=split, revision=revision)
            kinds.update(r.kind for r in records)
            rel = f"{layout.suite}/{layout.config}/{split}.jsonl"
            upstream = str(info.get("hf_id"))
            rows = (encode_row(r, layout.suite, JEV_BENCH_ID, revision, upstream, license_id) for r in records)
            splits[split] = {"file": rel, **write_jsonl(out / rel, rows)}
            logger.info("%s: %d rows", rel, len(records))
        notes = [*JEV_BENCH_TRANSFORMATIONS]
        if info.get("notes"):
            notes.append(f"jev-bench build note: {info['notes']}")
        entries.append({
            "config": layout.config,
            "suite": layout.suite,
            "kinds": sorted(kinds),
            "description": info.get("description"),
            "splits": splits,
            "source": {"hf_id": JEV_BENCH_ID, "hf_config": layout.config, "revision": revision},
            "upstream": {"hf_id": info.get("hf_id"), "hf_config": info.get("hf_config")},
            "license": license_id,
            "license_as_stated_by_source": license_text,
            "soft_labels": bool(info.get("has_soft_labels")),
            "transformations": notes,
        })
    return entries


def build_agent_action(out: Path, revision: str) -> dict[str, object]:
    """Select the jev-decisions-v1 choice subset, write it, and return its manifest entry.

    Args:
        out: Dataset root.
        revision: jev-decisions-v1 commit sha.

    Returns:
        The manifest entry.
    """
    shard = decisions.download_test_shard(revision)
    total = decisions.source_row_count(shard)
    selected, stats = decisions.select(decisions.iter_source_rows(shard))
    stats.excluded["not_choice_eligible"] += total - stats.source_rows
    rel = f"{AGENT_ACTION_SUITE}/{decisions.CONFIG}/{decisions.SPLIT}.jsonl"
    rows = (
        encode_row(d.record, AGENT_ACTION_SUITE, decisions.DATASET_ID, revision, d.upstream, d.license)
        for d in selected
    )
    split_info = write_jsonl(out / rel, rows)
    logger.info("%s: %d rows", rel, len(selected))
    buckets = [name for name, _, _ in decisions.BUCKETS]
    return {
        "config": decisions.CONFIG,
        "suite": AGENT_ACTION_SUITE,
        "kinds": ["choice"],
        "description": "Next-action selection for tool-using agents: given the visible agent state, which of the "
        "available tools/actions should be called next?",
        "splits": {decisions.SPLIT: {"file": rel, **split_info}},
        "source": {
            "hf_id": decisions.DATASET_ID,
            "hf_config": "default",
            "hf_split": decisions.SPLIT,
            "file": decisions.TEST_SHARD,
            "revision": revision,
        },
        "upstream": [
            {"hf_id": "nvidia/Nemotron-SFT-Agentic-v2"},
            {"hf_id": "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"},
            {"hf_id": "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"},
            {"hf_id": "nvidia/Open-SWE-Traces"},
        ],
        "license": "per row",
        "licenses": sorted(stats.selected_by_license),
        "license_rule": f"Rows whose upstream is {decisions.SPDX_UPSTREAM} carry the SPDX id of the source "
        "repository (provenance.source_metadata_json.repo_license) in license; an Open-SWE-Traces row without a "
        f"usable SPDX id would keep {decisions.LICENSE} and is counted in selection.license_fallbacks. Rows from "
        f"the Nemotron datasets carry no repository license and are {decisions.LICENSE}. Every row is also subject "
        "to the CC BY 4.0 attribution terms of jev-decisions-v1 and its NVIDIA upstream datasets.",
        "license_as_stated_by_source": "cc-by-4.0 (dataset card). All four NVIDIA upstream cards list CC BY 4.0; "
        "some also list Apache-2.0/MIT (and BSD for Open-SWE-Traces). Open-SWE-Traces records carry a "
        "per-repository SPDX license (provenance.source_metadata_json.repo_license). See the source's "
        "SOURCE_LICENSES.md before redistribution.",
        "soft_labels": False,
        "transformations": [
            "Source: the test partition only (data/test/test-00000-of-00001.parquet, one shard), read in Arrow "
            "batches from the Hugging Face cache.",
            "Eligibility: training.choice_eligible is true; 2..255 candidates; ordered_targets empty; candidate "
            "names non-empty and unique; target.candidate_id matches exactly one candidate whose name equals "
            "target.action_name.",
            f"Token cap: rows whose state+question exceed ~{TOKEN_WARN_LIMIT} tokens (serialized JSON chars / "
            f"{CHARS_PER_TOKEN}, the validator's warning threshold) are excluded before sampling; no row is "
            "truncated.",
            "question.type = 'choice'; question.instructions = the fixed Choice question the publisher uses in its "
            f"general-clean-50k config: {decisions.INSTRUCTIONS!r}. The record carries no per-row question text; "
            "the user's task stays in state.user_goal.",
            "question.criteria = {candidate.name: candidate.description, or null when empty}, in source candidate "
            "order. Parameter schemas, candidate metadata, and target arguments are not included.",
            "label = target.action_name (the matched candidate's name); options = criteria keys in order; "
            "soft_label = null.",
            "state = {system, user_goal, environment, history}: system and user_goal copied as strings (omitted "
            "when null/empty; kept even when they repeat a history turn); environment = decoded "
            "state.environment_json (omitted when null/empty); history = [{role, content}] with content = decoded "
            "payload_json. Serialized JSON that does not parse is kept as the raw string (count below).",
            "upstream = the record's source field (the NVIDIA dataset it was derived from); license = see "
            "license_rule.",
            "id = 'jev_decisions_v1/test/<source id>'. Labels, provenance, and training fields are never copied "
            "into state.",
            "Sampling: candidate-count buckets 2, 3-4, 5-8, 9-16, 17+; the 1,000 rows are split across buckets in "
            "proportion to rows available after the token cap (largest-remainder method); within a bucket, rows "
            f"with the smallest sha256('{decisions.SEED}:<source id>') are taken. Selection is independent of "
            "source row order. Rows are written by bucket, then by that key.",
        ],
        "selection": {
            "seed": decisions.SEED,
            "size": decisions.SAMPLE_SIZE,
            "source_rows": total,
            "excluded": dict(sorted(stats.excluded.items())),
            "buckets": [
                {
                    "bucket": name,
                    "eligible": stats.eligible[name],
                    "within_token_cap": stats.within_cap[name],
                    "selected": stats.allocation.get(name, 0),
                }
                for name in buckets
            ],
            "by_upstream": [
                {
                    "hf_id": upstream,
                    "eligible": stats.eligible_by_upstream[upstream],
                    "within_token_cap": stats.within_cap_by_upstream[upstream],
                    "selected": stats.selected_by_upstream[upstream],
                }
                for upstream in sorted(stats.eligible_by_upstream)
            ],
            "by_license": [
                {"license": license_id, "selected": count}
                for license_id, count in sorted(stats.selected_by_license.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "by_upstream_and_license": [
                {"hf_id": upstream, "license": license_id, "selected": count}
                for (upstream, license_id), count in sorted(Counter((d.upstream, d.license) for d in selected).items())
            ],
            "license_fallbacks": stats.license_fallbacks,
            "selected_trajectories": stats.selected_trajectories,
            "max_rows_per_trajectory": stats.max_rows_per_trajectory,
            "json_decode_fallbacks": stats.decode_fallbacks,
        },
    }


def build(out: Path, jev_bench_revision: str, decisions_revision: str) -> dict[str, object]:
    """Build every config and write ``manifest.yaml`` and the dataset card ``README.md``.

    Args:
        out: Dataset root.
        jev_bench_revision: jev-bench branch, tag, or commit.
        decisions_revision: jev-decisions-v1 branch, tag, or commit.

    Returns:
        The manifest that was written.
    """
    jev_sha = resolve_revision(JEV_BENCH_ID, jev_bench_revision)
    decisions_sha = resolve_revision(decisions.DATASET_ID, decisions_revision)
    configs = [*build_jev_bench(out, jev_sha), build_agent_action(out, decisions_sha)]
    manifest: dict[str, object] = {
        "name": "system-one-datasets",
        "description": DESCRIPTION,
        "format": "JSON Lines, one row per line, at <suite>/<config>/<split>.jsonl. state, question, and "
        "soft_label are JSON-encoded strings (soft_label is the text 'null' when absent); every other field "
        "is a plain JSON value.",
        "row_fields": list(CANONICAL_FIELDS),
        "wire_request": "{'state': json.loads(row['state']), 'model': <model>, "
        "'questions': {<any id>: json.loads(row['question'])}}",
        "rebuild": "uv run python -m system_one_datasets build --out data/",
        "validate": "uv run python -m system_one_datasets validate data/",
        "configs": configs,
    }
    text = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, allow_unicode=True, width=110)
    (out / "manifest.yaml").write_text(text, encoding="utf-8")
    (out / "README.md").write_text(render_card(configs), encoding="utf-8")
    return manifest
