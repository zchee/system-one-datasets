# system-one-datasets

Typed-decision datasets for System One models, normalized to the /v1/systemone wire format.

Every row is one typed decision (`noul`, `choice`, or `score`) whose `state` and `question`, once decoded, are
the body of a `POST /v1/systemone` request, the API served by TypeSafe `jev-1.13.0`, openjev / openjev-MLX,
and other System One backends. Evaluation and benchmarking are two uses; calibration, regression tests, and
training-data selection are others.

This repository holds:

- `data/`: the rows (`<suite>/<config>/<split>.jsonl`), `manifest.yaml` (where every row came from and how
  it was transformed), and `README.md`, the Hugging Face dataset card. `data/` is the root of the Hub
  dataset repository. The `.jsonl` files are not committed to git: run `build` to regenerate them
  byte-for-byte from the pinned sources, or load them from the Hub.
- `src/system_one_datasets/`: the `build` that regenerates `data/` from pinned Hugging Face revisions, the
  `validate` checker, the `push` uploader, and a loader that decodes rows into wire-ready records.

## Row schema

One JSON object per line, keys in this order:

| field | type | meaning |
|---|---|---|
| `id` | string | Row id, unique within its file (e.g. `civil_comments/test/71007`). |
| `suite` | string | `moderation`, `quality`, or `agent_action`; matches the directory. |
| `config` | string | Config name; matches the directory. |
| `kind` | string | `noul`, `choice`, or `score`; equals the question's `type`. |
| `state` | string (JSON) | JSON text of the request `state` (a string, object, or array). |
| `question` | string (JSON) | JSON text of the wire-format Question (`type`, `instructions`, `criteria`). |
| `options` | list of strings | Option keys of the predicted distribution. `choice`: `criteria` keys in order. `score`: level indices `"0"`..`"K-1"`. `noul`: `["0", "1"]` (no, yes). |
| `label` | string | Gold option; one of `options`. |
| `soft_label` | string (JSON) | JSON text of the annotator distribution `{option: probability}`, or the text `null` when the row has none. |
| `source` | string | Hugging Face dataset id the row was loaded from. |
| `source_revision` | string | Commit sha of that dataset. |
| `upstream` | string | Original dataset the row's content comes from (`source` is derived from it), e.g. `google/civil_comments` or `nvidia/Open-SWE-Traces`. |
| `license` | string | License of the row's content (see [Licensing](#licensing)); per row for `jev_decisions_v1`. |

`state`, `question`, and `soft_label` are JSON-encoded strings, the convention `Praveenrajus/jev-bench` uses.
The Hugging Face `datasets` JSON loader infers one struct per column across all rows; `criteria` has
different keys in every agent-action row, so nested objects would be merged into one sparse struct and come
back with null entries for options that are not the row's own. String columns keep one flat schema for every
config.

## Contents

| suite | config | kind | test | validation | license | soft labels | upstream |
|---|---|---|---:|---:|---|---|---|
| moderation | `civil_comments` | noul | 2,000 | 500 | cc0-1.0 | yes | google/civil_comments |
| moderation | `measuring_hate_speech` | score (3 levels) | 1,000 | 500 | cc-by-4.0 | yes | ucberkeley-dlab/measuring-hate-speech |
| moderation | `go_emotions` | choice (28 options) | 1,000 | 500 | apache-2.0 | yes | google-research-datasets/go_emotions |
| quality | `helpsteer2_helpfulness` | score (5 levels) | 1,000 | 500 | cc-by-4.0 | no | nvidia/HelpSteer2 |
| quality | `stsb` | score (6 levels) | 1,000 | 500 | cc-by-sa-4.0 | no | sentence-transformers/stsb |
| agent_action | `jev_decisions_v1` | choice (2..67 options) | 1,000 | - | per row: cc-by-4.0 or the repository's SPDX id | no | 4 NVIDIA agent datasets |

The moderation and quality configs are exported unchanged from
[`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1. `jev_decisions_v1`
is a 1,000-row sample of the test partition of
[`samatv256/jev-decisions-v1`](https://huggingface.co/datasets/samatv256/jev-decisions-v1): given an
agent's visible state, which available tool should it call next? The sample is stratified by the number of
candidate tools (2, 3-4, 5-8, 9-16, 17+) and excludes records longer than about 32k tokens.
`data/manifest.yaml` records how each record became a question, what was excluded and why, and how many rows
each bucket and each upstream dataset contributed.

### Choice-only subset

Rows with `kind == "choice"` (`go_emotions` and `jev_decisions_v1`) are the subset the most backends can
answer. The OpenAI Decisions API preview accepts only choice-shaped questions; its schema was unpublished as
of 2026-09-30, so this repository has no adapter for it.

## Using the rows

```python
from pathlib import Path

from system_one_datasets import SystemOneClient, load_records
from system_one_datasets.client import answer_distribution

record = load_records(Path("data/moderation/go_emotions/test.jsonl"))[0]

# Request body: {"state": record.state, "model": ..., "questions": {"q": record.question}}
with SystemOneClient("http://localhost:3000", "openjev") as client:
    evaluation = client.evaluate(record.state, {"q": record.question})

probabilities = answer_distribution(evaluation.answers["q"], record.kind, record.options)
predicted = max(probabilities, key=probabilities.__getitem__)
```

Without this package, `json.loads(row["state"])` and `json.loads(row["question"])` give the same values. From
the Hub: `load_dataset("zchee/system-one-datasets", "go_emotions", split="test")`.

## Building, validating, publishing

```sh
uv run python -m system_one_datasets build --out data/
uv run python -m system_one_datasets validate data/
uv run python -m system_one_datasets validate data/ --live --base-url http://localhost:3000
uv run python -m system_one_datasets push --repo zchee/system-one-datasets --dry-run
```

`build` downloads the pinned source revisions (`--jev-bench-revision` and `--decisions-revision` override
them), rewrites every split file, and writes `manifest.yaml` (with a SHA-256 per file) and the dataset card.
Sampling uses a fixed seed (20260930) and does not depend on source row order, and neither the manifest nor
the card has timestamps, so a rebuild at the same revisions produces identical files. The jev-decisions-v1
step downloads one 569 MB Parquet shard into the Hugging Face cache.

`validate` decodes every row and checks it: `kind` and the question's `type`; `choice` criteria has 2..255
entries whose keys equal `options` in order; `score` criteria has 2..10 levels; `noul` options are
`["0", "1"]`; `label` is an option; `soft_label` keys are options and its values sum to 0.98..1.02; ids are
unique per file; `suite` and `config` match the directory. It warns, without failing, when `state` plus
`question` exceeds about 32k tokens (serialized characters / 3). `--live` sends the first 3 test rows of each
config to a backend and checks for HTTP 200 and an answer of the question's kind (`--model` and
`--api-key-env` for backends that need them). The command exits 1 on any structural error or failed live
request.

`push` runs the structural validation first and refuses to upload if it fails. It then creates the dataset
repository (private unless `--public`; the visibility of an existing repository is not changed) and uploads
`README.md`, `manifest.yaml`, and the split files in one commit. By default it never deletes remote files;
`--prune` also deletes remote `.jsonl` files that are not in `--data`, so splits removed by a rebuild do not
remain. Credentials come from `hf auth login`. `--dry-run` prints the target, visibility, prune mode, and file
list with sizes, and makes no network writes.

Tests: `uv run pytest`. Two live tests are skipped by default: `RUN_LIVE_TESTS=1` runs the TypeSafe API test
(needs `TYPESAFE_API_KEY`), and `RUN_LOCAL_LIVE_TESTS=1` runs the local-backend test (`SYSTEMONE_BASE_URL`,
default `http://localhost:3000`; `SYSTEMONE_MODEL`, default `openjev`).

## Licensing

Rows keep the licenses of their upstream datasets; this repository adds no terms and grants no rights beyond
them. Check each row's `license` and the `license_as_stated_by_source` entries in `data/manifest.yaml` before
redistributing.

- jev-bench is published as `license: other` (mixed, per upstream dataset); each config carries its upstream
  dataset's license, which this repository copies into `license`. `stsb` is CC BY-SA 4.0 (share-alike);
  `civil_comments` is CC0.
- `jev_decisions_v1` is CC BY 4.0, with attribution to its publisher and to NVIDIA as the upstream data
  developer. Rows derived from Open-SWE-Traces carry the SPDX license of their source repository in `license`
  (MIT, Apache-2.0, BSD-3-Clause, BSD-2-Clause); the other rows are `cc-by-4.0`. `upstream` names each row's
  NVIDIA source dataset, and `manifest.yaml` counts rows per upstream and per license. Read the source's
  [`SOURCE_LICENSES.md`](https://huggingface.co/datasets/samatv256/jev-decisions-v1/blob/main/SOURCE_LICENSES.md)
  before redistributing.

This repository and its datasets are not affiliated with or endorsed by TypeSafe, OpenJev, or NVIDIA. The code
is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)).
