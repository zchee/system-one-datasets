---
pretty_name: System One Datasets
license: other
license_name: mixed-upstream-licenses
license_link: https://huggingface.co/datasets/zchee/system-one-datasets/blob/main/manifest.yaml
language:
- en
task_categories:
- text-classification
tags:
- system-one
- typed-decisions
- calibration
- noul
- choice
- score
size_categories:
- 1K<n<10K
configs:
- config_name: civil_comments
  data_files:
  - split: test
    path: moderation/civil_comments/test.jsonl
  - split: validation
    path: moderation/civil_comments/validation.jsonl
- config_name: measuring_hate_speech
  data_files:
  - split: test
    path: moderation/measuring_hate_speech/test.jsonl
  - split: validation
    path: moderation/measuring_hate_speech/validation.jsonl
- config_name: go_emotions
  data_files:
  - split: test
    path: moderation/go_emotions/test.jsonl
  - split: validation
    path: moderation/go_emotions/validation.jsonl
- config_name: helpsteer2_helpfulness
  data_files:
  - split: test
    path: quality/helpsteer2_helpfulness/test.jsonl
  - split: validation
    path: quality/helpsteer2_helpfulness/validation.jsonl
- config_name: stsb
  data_files:
  - split: test
    path: quality/stsb/test.jsonl
  - split: validation
    path: quality/stsb/validation.jsonl
- config_name: jev_decisions_v1
  data_files:
  - split: test
    path: agent_action/jev_decisions_v1/test.jsonl
---

# System One Datasets

Typed-decision datasets for System One models, normalized to the /v1/systemone wire format.

Every row is one typed decision (`noul`, `choice`, or `score`) whose `state` and `question`, once decoded, are
the body of a `POST /v1/systemone` request, the API served by TypeSafe's Jev and by open reimplementations such
as openjev. Use the rows for evaluation, calibration, regression tests, or training data selection.

This dataset is not affiliated with or endorsed by TypeSafe, OpenJev, NVIDIA, or the authors of the upstream
datasets.

## Configs

| config | suite | kind | test | validation | soft labels | upstream | license |
|---|---|---|---:|---:|---|---|---|
| `civil_comments` | moderation | noul | 2,000 | 500 | yes | google/civil_comments | cc0-1.0 |
| `measuring_hate_speech` | moderation | score | 1,000 | 500 | yes | ucberkeley-dlab/measuring-hate-speech | cc-by-4.0 |
| `go_emotions` | moderation | choice | 1,000 | 500 | yes | google-research-datasets/go_emotions | apache-2.0 |
| `helpsteer2_helpfulness` | quality | score | 1,000 | 500 | no | nvidia/HelpSteer2 | cc-by-4.0 |
| `stsb` | quality | score | 1,000 | 500 | no | sentence-transformers/stsb | cc-by-sa-4.0 |
| `jev_decisions_v1` | agent_action | choice | 1,000 | - | no | nvidia/Nemotron-SFT-Agentic-v2, nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1, nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1, nvidia/Open-SWE-Traces | per row: Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT, cc-by-4.0 |

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
| `soft_label` | string (JSON) | JSON text of `{option: probability}` from annotators, or `null` when absent. |
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

ds = load_dataset("zchee/system-one-datasets", "go_emotions", split="test")
row = ds[0]
state = json.loads(row["state"])
question = json.loads(row["question"])
soft_label = json.loads(row["soft_label"])  # None when absent
```

## Sending a row to a /v1/systemone backend

```python
import httpx

body = {"state": state, "model": "jev-1.13.0", "questions": {"q": question}}
response = httpx.post("https://api.typesafe.ai/v1/systemone", json=body, headers={"authorization": "Bearer ..."})
answer = response.json()["answers"]["q"]
# noul: answer["noul"] is P(option "1"); choice and score: answer["probabilities"] is keyed by the row's options.
```

## Sources, attribution, and licenses

Rows keep the licenses of their upstream datasets. This card's `license: other` means the licenses differ per
config; check each row's `license` field and `manifest.yaml` before redistributing. No rights are granted
beyond the upstream terms.

- [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1 (revision
  `18f88da81c28c2bec55edc31f63f2afdfba109ea`), published as `license: other` (mixed, per upstream dataset). The moderation and quality
  configs are exported from it unchanged. Its upstream datasets and their licenses:
  - `civil_comments`: [google/civil_comments](https://huggingface.co/datasets/google/civil_comments), cc0-1.0.
  - `measuring_hate_speech`: [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech), cc-by-4.0.
  - `go_emotions`: [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), apache-2.0.
  - `helpsteer2_helpfulness`: [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), cc-by-4.0.
  - `stsb`: [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb), cc-by-sa-4.0 (STS Benchmark).
  - `stsb` is CC BY-SA 4.0: derivatives must be shared under the same license. `civil_comments` is CC0.
- [samatv256/jev-decisions-v1](https://huggingface.co/datasets/samatv256/jev-decisions-v1) (revision
  `c12aadf1f01c72616bfab0b02480e21806397669`), CC BY 4.0. It is derived from four datasets developed by NVIDIA; per its
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
    MIT (183 rows), Apache-2.0 (100 rows), BSD-3-Clause (25 rows), BSD-2-Clause (6 rows). They remain subject to the CC BY 4.0 attribution terms
    above.

  NVIDIA is the developer of the upstream data, not the publisher or endorser of jev-decisions-v1 or of this
  dataset.

## Rebuilding

The rows are generated by `python -m system_one_datasets build --out data/` in the
[system-one-datasets](https://github.com/zchee/system-one-datasets) repository from the pinned revisions above,
with a fixed seed; rebuilding at the same revisions reproduces every file byte for byte.
