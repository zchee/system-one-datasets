"""Hugging Face dataset card (``data/README.md``), generated from the build's manifest entries."""

from collections.abc import Mapping, Sequence

import yaml


HF_REPO_ID = "zchee/system-one-datasets"
# The Hub rejects a relative license_link; it must be an https URL, so point at the manifest on the Hub itself.
LICENSE_LINK = f"https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/manifest.yaml"
DESCRIPTION = "Typed-decision datasets for System One models, normalized to the /v1/systemone wire format"
TAGS: tuple[str, ...] = ("system-one", "typed-decisions", "calibration", "noul", "choice", "score")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list | tuple) else ()


def _size_category(total_rows: int) -> str:
    for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")):
        if total_rows < bound:
            return label
    return "n>1M"


def _split_rows(entry: Mapping[str, object], split: str) -> int | None:
    info = _mapping(_mapping(entry.get("splits")).get(split))
    rows = info.get("rows")
    return rows if isinstance(rows, int) else None


def _upstream_ids(entry: Mapping[str, object]) -> list[str]:
    upstream = entry.get("upstream")
    items = _sequence(upstream) if isinstance(upstream, list) else [upstream]
    return [str(_mapping(item).get("hf_id")) for item in items if _mapping(item).get("hf_id")]


def frontmatter(configs: Sequence[Mapping[str, object]]) -> str:
    """Card metadata: license, tags, size, and one ``configs`` entry per config with its data files.

    Args:
        configs: Manifest config entries.

    Returns:
        YAML text between ``---`` fences, block-style sequences only.
    """
    total = sum(_split_rows(e, s) or 0 for e in configs for s in _mapping(e.get("splits")))
    meta: dict[str, object] = {
        "pretty_name": "System One Datasets",
        "license": "other",
        "license_name": "mixed-upstream-licenses",
        "license_link": LICENSE_LINK,
        "language": ["en"],
        "task_categories": ["text-classification"],
        "tags": list(TAGS),
        "size_categories": [_size_category(total)],
        "configs": [
            {
                "config_name": str(entry.get("config")),
                "data_files": [
                    {"split": split, "path": str(_mapping(info).get("file"))}
                    for split, info in _mapping(entry.get("splits")).items()
                ],
            }
            for entry in configs
        ],
    }
    body = yaml.safe_dump(meta, sort_keys=False, default_flow_style=False, allow_unicode=True, width=110)
    return f"---\n{body}---\n"


def _config_table(configs: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| config | suite | kind | test | validation | soft labels | upstream | license |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for entry in configs:
        test, validation = _split_rows(entry, "test"), _split_rows(entry, "validation")
        kinds = ", ".join(str(k) for k in _sequence(entry.get("kinds")))
        lines.append(
            f"| `{entry.get('config')}` | {entry.get('suite')} | {kinds} "
            f"| {f'{test:,}' if test is not None else '-'} | {f'{validation:,}' if validation is not None else '-'} "
            f"| {'yes' if entry.get('soft_labels') else 'no'} | {', '.join(_upstream_ids(entry))} "
            f"| {_license_cell(entry)} |"
        )
    return "\n".join(lines)


def _license_cell(entry: Mapping[str, object]) -> str:
    licenses = [str(x) for x in _sequence(entry.get("licenses"))]
    return f"per row: {', '.join(licenses)}" if licenses else str(entry.get("license"))


def _spdx_rows(configs: Sequence[Mapping[str, object]], upstream: str) -> str:
    counts = [
        _mapping(item)
        for entry in configs
        for item in _sequence(_mapping(entry.get("selection")).get("by_upstream_and_license"))
        if _mapping(item).get("hf_id") == upstream
    ]
    counts.sort(key=lambda item: (-int(str(item.get("selected"))), str(item.get("license"))))
    return ", ".join(f"{item.get('license')} ({item.get('selected')} rows)" for item in counts) or "none"


def _jev_bench_attribution(configs: Sequence[Mapping[str, object]]) -> str:
    lines = []
    for entry in configs:
        source = _mapping(entry.get("source"))
        if source.get("hf_id") != "Praveenrajus/jev-bench":
            continue
        upstream = ", ".join(f"[{u}](https://huggingface.co/datasets/{u})" for u in _upstream_ids(entry))
        lines.append(f"  - `{entry.get('config')}`: {upstream}, {entry.get('license_as_stated_by_source')}.")
    return "\n".join(lines)


def _revision(configs: Sequence[Mapping[str, object]], hf_id: str) -> str:
    for entry in configs:
        source = _mapping(entry.get("source"))
        if source.get("hf_id") == hf_id:
            return str(source.get("revision"))
    return "unknown"


def render_card(configs: Sequence[Mapping[str, object]]) -> str:
    """Render the complete dataset card.

    Args:
        configs: Manifest config entries, in the order they appear in the manifest.

    Returns:
        Markdown with YAML frontmatter.
    """
    jev_bench_rev = _revision(configs, "Praveenrajus/jev-bench")
    decisions_rev = _revision(configs, "samatv256/jev-decisions-v1")
    return f"""{frontmatter(configs)}
# System One Datasets

{DESCRIPTION}.

Every row is one typed decision (`noul`, `choice`, or `score`) whose `state` and `question`, once decoded, are
the body of a `POST /v1/systemone` request, the API served by TypeSafe's Jev and by open reimplementations such
as openjev. Use the rows for evaluation, calibration, regression tests, or training data selection.

This dataset is not affiliated with or endorsed by TypeSafe, OpenJev, NVIDIA, or the authors of the upstream
datasets.

## Configs

{_config_table(configs)}

`choice` rows (`go_emotions`, `jev_decisions_v1`) are the subset the most backends can answer; the OpenAI
Decisions API preview accepts only choice-shaped questions (its schema was unpublished as of 2026-09-30).

`jev_decisions_v1` is a 1,000-row sample of the test partition of `samatv256/jev-decisions-v1`: given an
agent's visible state (system prompt, user goal, conversation and tool history), which of the available tools
should it call next? The sample is stratified by the number of candidate tools (2, 3-4, 5-8, 9-16, 17+) and
excludes records whose state is longer than about 32k tokens. Its rows contain source code and some
non-English text. `manifest.yaml` documents every mapping decision, the exclusions, and the per-bucket and
per-source counts.

## Row schema

| field | type | meaning |
|---|---|---|
| `id` | string | Row id, unique within its split. |
| `suite` | string | `moderation`, `quality`, or `agent_action`. |
| `config` | string | Config name. |
| `kind` | string | `noul`, `choice`, or `score`. |
| `state` | string (JSON) | JSON text of the request `state` (a string, object, or array). |
| `question` | string (JSON) | JSON text of the wire-format Question object (`type`, `instructions`, `criteria`). |
| `options` | list of strings | `choice`: criteria keys; `score`: levels `"0"`..`"K-1"`; `noul`: `["0", "1"]`. |
| `label` | string | Gold option; one of `options`. |
| `soft_label` | string (JSON) | JSON text of `{{option: probability}}` from annotators, or `null` when absent. |
| `source` | string | Hugging Face dataset the row was loaded from. |
| `source_revision` | string | Commit sha of `source`. |
| `upstream` | string | Original dataset the row's content comes from (`source` is derived from it). |
| `license` | string | License of the row's content; for Open-SWE-Traces rows, the SPDX id of the source repository. |

`state`, `question`, and `soft_label` are JSON strings so that every config shares one flat schema: `criteria`
has different keys in every row, and nested columns would be merged into one sparse struct by the loader.

## Loading

```python
import json

from datasets import load_dataset

ds = load_dataset("{HF_REPO_ID}", "go_emotions", split="test")
row = ds[0]
state = json.loads(row["state"])
question = json.loads(row["question"])
soft_label = json.loads(row["soft_label"])  # None when absent
```

## Sending a row to a /v1/systemone backend

```python
import httpx

body = {{"state": state, "model": "jev-1.13.0", "questions": {{"q": question}}}}
response = httpx.post("https://api.typesafe.ai/v1/systemone", json=body, headers={{"authorization": "Bearer ..."}})
answer = response.json()["answers"]["q"]
# noul: answer["noul"] is P(option "1"); choice and score: answer["probabilities"] is keyed by the row's options.
```

## Sources, attribution, and licenses

Rows keep the licenses of their upstream datasets. This card's `license: other` means the licenses differ per
config; check each row's `license` field and `manifest.yaml` before redistributing. No rights are granted
beyond the upstream terms.

- [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1 (revision
  `{jev_bench_rev}`), published as `license: other` (mixed, per upstream dataset). The moderation and quality
  configs are exported from it unchanged. Its upstream datasets and their licenses:
{_jev_bench_attribution(configs)}
  - `stsb` is CC BY-SA 4.0: derivatives must be shared under the same license. `civil_comments` is CC0.
- [samatv256/jev-decisions-v1](https://huggingface.co/datasets/samatv256/jev-decisions-v1) (revision
  `{decisions_rev}`), CC BY 4.0. It is derived from four datasets developed by NVIDIA; per its
  `SOURCE_LICENSES.md`, each card lists CC BY 4.0, with additional terms as noted:
  - [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2): CC BY
    4.0; the card also lists Apache 2.0 and MIT.
  - [nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1):
    CC BY 4.0.
  - [nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1):
    CC BY 4.0; the card also lists Apache 2.0 and MIT.
  - [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces): CC BY 4.0; the card
    also lists MIT, Apache 2.0, BSD 2-Clause, and BSD 3-Clause, and each source record carries the SPDX
    license of its repository. Rows from Open-SWE-Traces carry that SPDX id in `license`:
    {_spdx_rows(configs, "nvidia/Open-SWE-Traces")}. They remain subject to the CC BY 4.0 attribution terms
    above.

  NVIDIA is the developer of the upstream data, not the publisher or endorser of jev-decisions-v1 or of this
  dataset.

## Rebuilding

The rows are generated by `python -m system_one_datasets build --out data/` in the
[system-one-datasets](https://github.com/zchee/system-one-datasets) repository from the pinned revisions above,
with a fixed seed; rebuilding at the same revisions reproduces every file byte for byte.
"""
