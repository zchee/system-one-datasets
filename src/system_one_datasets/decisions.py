"""Choice-question subset of ``samatv256/jev-decisions-v1`` (agent action selection).

Each selected source record becomes one wire-format ``choice`` question: the state is the record's visible agent
state, the options are the candidate actions available at the decision, and the label is the target action.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import heapq
from pathlib import Path
import re
from typing import TYPE_CHECKING, cast

from huggingface_hub import hf_hub_download
import orjson
import pyarrow.parquet as pq

from system_one_datasets.schema import BenchRecord, JSONValue
from system_one_datasets.validate import CHOICE_MAX_OPTIONS, TOKEN_WARN_LIMIT, estimate_tokens


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import pyarrow as pa


DATASET_ID = "samatv256/jev-decisions-v1"
# Commit of the dataset repository the committed data/ was built from.
DATASET_REVISION = "c12aadf1f01c72616bfab0b02480e21806397669"
TEST_SHARD = "data/test/test-00000-of-00001.parquet"
LICENSE = "cc-by-4.0"
# Upstream whose records carry a per-repository SPDX license in ``provenance.source_metadata_json.repo_license``.
SPDX_UPSTREAM = "nvidia/Open-SWE-Traces"
# A single SPDX license id (no expressions); NOASSERTION/NONE are not licenses.
_SPDX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_NOT_A_LICENSE = frozenset({"NOASSERTION", "NONE"})
CONFIG = "jev_decisions_v1"
SPLIT = "test"
SEED = 20260930
SAMPLE_SIZE = 1000

# The publisher's own fixed Choice question from the dataset's ``general-clean-50k`` config.
INSTRUCTIONS = "Given the current state and available options,\nwhich option should be selected?"

# Candidate-count buckets: (name, min, max), inclusive.
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17+", 17, CHOICE_MAX_OPTIONS),
)

SOURCE_COLUMNS: tuple[str, ...] = (
    "id",
    "source",
    "state",
    "candidates",
    "target",
    "ordered_targets",
    "provenance.trajectory_id",
    "provenance.source_metadata_json",
    "training.choice_eligible",
)


def bucket_of(n_candidates: int) -> str | None:
    """Return the candidate-count bucket name, or ``None`` outside 2..255.

    Args:
        n_candidates: Number of candidate actions.

    Returns:
        Bucket name.
    """
    for name, low, high in BUCKETS:
        if low <= n_candidates <= high:
            return name
    return None


def rank_key(record_id: str, seed: int = SEED) -> int:
    """Deterministic, order-independent sampling key: the seeded SHA-256 of the record id as an integer.

    Args:
        record_id: Source record id.
        seed: Sampling seed.

    Returns:
        Non-negative integer; smaller keys are selected first.
    """
    return int.from_bytes(hashlib.sha256(f"{seed}:{record_id}".encode()).digest())


def allocate(available: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across buckets proportionally to availability (largest-remainder method).

    Args:
        available: Eligible rows per bucket, in bucket order.
        total: Rows to select.

    Returns:
        Rows to take per bucket; never more than available, summing to ``min(total, sum(available))``.
    """
    pool = sum(available.values())
    if pool <= total:
        return dict(available)
    quotas = {name: total * n / pool for name, n in available.items()}
    shares = {name: int(q) for name, q in quotas.items()}
    order = sorted(available, key=lambda name: quotas[name] - shares[name], reverse=True)
    for name in order[: total - sum(shares.values())]:
        shares[name] += 1
    return shares


def decode_json_text(text: str | None) -> tuple[JSONValue, bool]:
    """Decode a serialized-JSON source field, falling back to the raw text.

    Args:
        text: Field value.

    Returns:
        The decoded value (or the raw string when it is not JSON) and whether decoding failed.
    """
    if text is None:
        return None, False
    try:
        return orjson.loads(text), False
    except orjson.JSONDecodeError:
        return text, True


def ineligible_reason(row: Mapping[str, JSONValue]) -> str | None:
    """Return why a source record cannot become a choice question, or ``None`` when it can.

    Args:
        row: Source record with the ``SOURCE_COLUMNS`` fields.

    Returns:
        Short reason slug.
    """
    training = row.get("training")
    if not isinstance(training, dict) or training.get("choice_eligible") is not True:
        return "not_choice_eligible"
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or bucket_of(len(candidates)) is None:
        return "candidate_count_outside_2_255"
    if row.get("ordered_targets"):
        return "multiple_targets"
    names = [c.get("name") for c in candidates if isinstance(c, dict)]
    if len(names) != len(candidates) or not all(isinstance(n, str) and n for n in names):
        return "candidate_without_name"
    if len(set(names)) != len(names):
        return "duplicate_candidate_names"
    target = row.get("target")
    if not isinstance(target, dict):
        return "no_target"
    matches = [c for c in candidates if isinstance(c, dict) and c.get("id") == target.get("candidate_id")]
    if len(matches) != 1 or matches[0].get("name") != target.get("action_name"):
        return "target_not_unique_in_candidates"
    return None


@dataclass(frozen=True, slots=True)
class Decision:
    """A mapped record with the metadata used for sampling and the manifest.

    Attributes:
        record: The wire-format choice record.
        bucket: Candidate-count bucket.
        upstream: Upstream dataset the source record was derived from.
        license: License of the row's content (see ``row_license``).
        license_fallback: The record is from ``SPDX_UPSTREAM`` but carries no usable SPDX id.
        trajectory: ``<upstream>/<trajectory_id>`` grouping key.
        tokens: ``estimate_tokens`` of state + question.
        decode_fallbacks: Serialized-JSON fields kept as raw strings because they did not parse.
    """

    record: BenchRecord
    bucket: str
    upstream: str
    license: str
    license_fallback: bool
    trajectory: str
    tokens: int
    decode_fallbacks: int


def row_license(upstream: str, source_metadata_json: str | None) -> tuple[str, bool]:
    """License of a record's content: the repository's SPDX id for ``SPDX_UPSTREAM`` records, else ``LICENSE``.

    Args:
        upstream: Upstream dataset of the record.
        source_metadata_json: ``provenance.source_metadata_json`` of the record.

    Returns:
        The license id, and whether an ``SPDX_UPSTREAM`` record fell back to ``LICENSE`` for lack of an SPDX id.
    """
    if upstream != SPDX_UPSTREAM:
        return LICENSE, False
    metadata, _ = decode_json_text(source_metadata_json)
    spdx = metadata.get("repo_license") if isinstance(metadata, dict) else None
    if isinstance(spdx, str) and _SPDX_ID.match(spdx) and spdx.upper() not in _NOT_A_LICENSE:
        return spdx, False
    return LICENSE, True


def map_decision(row: Mapping[str, JSONValue]) -> Decision:
    """Map an eligible source record (``ineligible_reason(row) is None``) to a choice question.

    Args:
        row: Source record with the ``SOURCE_COLUMNS`` fields.

    Returns:
        The mapped decision.

    Raises:
        ValueError: The record is not eligible.
    """
    if (reason := ineligible_reason(row)) is not None:
        msg = f"record {row.get('id')}: {reason}"
        raise ValueError(msg)
    # Shapes below are guaranteed by the source Parquet schema and by ineligible_reason().
    raw_state = cast("dict[str, JSONValue]", row["state"])
    candidates = cast("list[dict[str, JSONValue]]", row["candidates"])
    target = cast("dict[str, JSONValue]", row["target"])
    provenance = cast("dict[str, JSONValue]", row["provenance"])
    fallbacks = 0
    state: dict[str, JSONValue] = {}
    for key in ("system", "user_goal"):
        value = raw_state.get(key)
        if isinstance(value, str) and value:
            state[key] = value
    environment, failed = decode_json_text(_as_text(raw_state.get("environment_json")))
    fallbacks += failed
    if environment not in (None, "", {}, []):
        state["environment"] = environment
    history: list[JSONValue] = []
    raw_history = raw_state.get("history")
    for turn in raw_history if isinstance(raw_history, list) else []:
        if not isinstance(turn, dict):
            continue
        content, failed = decode_json_text(_as_text(turn.get("payload_json")))
        fallbacks += failed
        history.append({"role": turn.get("role"), "content": content})
    state["history"] = history
    criteria: dict[str, JSONValue] = {}
    for candidate in candidates:
        description = candidate.get("description")
        criteria[str(candidate["name"])] = description if isinstance(description, str) and description else None
    question: dict[str, JSONValue] = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}
    upstream = str(row.get("source"))
    license_id, license_fallback = row_license(upstream, _as_text(provenance.get("source_metadata_json")))
    record = BenchRecord(
        id=f"{CONFIG}/{SPLIT}/{row['id']}",
        config=CONFIG,
        kind="choice",
        state=state,
        question=question,
        options=list(criteria),
        label=str(target["action_name"]),
        soft_label=None,
    )
    return Decision(
        record=record,
        bucket=str(bucket_of(len(candidates))),
        upstream=upstream,
        license=license_id,
        license_fallback=license_fallback,
        trajectory=f"{upstream}/{provenance.get('trajectory_id')}",
        tokens=estimate_tokens(state, question),
        decode_fallbacks=fallbacks,
    )


def _as_text(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


@dataclass(slots=True)
class SelectionStats:
    """Counts reported in the manifest.

    Attributes:
        source_rows: Rows read from the source split.
        excluded: Rows dropped, by reason.
        eligible: Rows passing eligibility, by bucket (before the token cap).
        eligible_by_upstream: Rows passing eligibility, by upstream dataset.
        within_cap: Eligible rows within the token cap, by bucket.
        within_cap_by_upstream: Eligible rows within the token cap, by upstream dataset.
        allocation: Rows selected per bucket.
        selected_by_upstream: Selected rows by upstream dataset.
        selected_by_license: Selected rows by license.
        license_fallbacks: Selected ``SPDX_UPSTREAM`` rows without a usable SPDX id (licensed ``LICENSE``).
        selected_trajectories: Distinct trajectories among the selected rows.
        max_rows_per_trajectory: Largest number of selected rows sharing one trajectory.
        decode_fallbacks: Serialized-JSON fields kept as raw strings among the selected rows.
    """

    source_rows: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    eligible: Counter[str] = field(default_factory=Counter)
    eligible_by_upstream: Counter[str] = field(default_factory=Counter)
    within_cap: Counter[str] = field(default_factory=Counter)
    within_cap_by_upstream: Counter[str] = field(default_factory=Counter)
    allocation: dict[str, int] = field(default_factory=dict)
    selected_by_upstream: Counter[str] = field(default_factory=Counter)
    selected_by_license: Counter[str] = field(default_factory=Counter)
    license_fallbacks: int = 0
    selected_trajectories: int = 0
    max_rows_per_trajectory: int = 0
    decode_fallbacks: int = 0


def select(
    rows: Iterable[Mapping[str, JSONValue]],
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
    token_limit: int = TOKEN_WARN_LIMIT,
) -> tuple[list[Decision], SelectionStats]:
    """Stratified, order-independent sample of eligible records by candidate-count bucket.

    Every eligible record within ``token_limit`` is ranked by ``rank_key``; each bucket keeps its ``size`` smallest
    keys (a bounded heap, so memory does not grow with the source), and ``allocate`` decides how many of those
    each bucket contributes.

    Args:
        rows: Source records.
        size: Rows to select.
        seed: Sampling seed.
        token_limit: Records whose ``estimate_tokens`` exceeds this are excluded.

    Returns:
        Selected decisions ordered by bucket then key, and the selection statistics.
    """
    stats = SelectionStats()
    # Per bucket: a max-heap (negated keys) of the ``size`` smallest keys seen, and the decisions it holds by id.
    heaps: dict[str, list[tuple[int, str]]] = {name: [] for name, _, _ in BUCKETS}
    kept: dict[str, Decision] = {}
    for row in rows:
        stats.source_rows += 1
        reason = ineligible_reason(row)
        if reason is not None:
            stats.excluded[reason] += 1
            continue
        decision = map_decision(row)
        stats.eligible[decision.bucket] += 1
        stats.eligible_by_upstream[decision.upstream] += 1
        if decision.tokens > token_limit:
            stats.excluded["over_token_limit"] += 1
            continue
        stats.within_cap[decision.bucket] += 1
        stats.within_cap_by_upstream[decision.upstream] += 1
        heap = heaps[decision.bucket]
        item = (-rank_key(decision.record.id, seed), decision.record.id)
        if len(heap) < size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            del kept[heapq.heapreplace(heap, item)[1]]
        else:
            continue
        kept[decision.record.id] = decision
    stats.allocation = allocate({name: stats.within_cap[name] for name, _, _ in BUCKETS}, size)
    selected: list[Decision] = []
    for name, _, _ in BUCKETS:
        ranked = sorted(heaps[name], key=lambda item: (-item[0], item[1]))
        selected += [kept[record_id] for _, record_id in ranked[: stats.allocation[name]]]
    trajectories = Counter(d.trajectory for d in selected)
    stats.selected_by_upstream = Counter(d.upstream for d in selected)
    stats.selected_by_license = Counter(d.license for d in selected)
    stats.license_fallbacks = sum(d.license_fallback for d in selected)
    stats.selected_trajectories = len(trajectories)
    stats.max_rows_per_trajectory = max(trajectories.values(), default=0)
    stats.decode_fallbacks = sum(d.decode_fallbacks for d in selected)
    return selected, stats


def iter_source_rows(path: Path, columns: Sequence[str] = SOURCE_COLUMNS) -> Iterable[dict[str, JSONValue]]:
    """Stream records from a local Parquet shard in batches, yielding only ``choice_eligible`` rows.

    Ineligible rows are dropped in Arrow before conversion to Python; they are not counted by ``select``.

    Args:
        path: Parquet file.
        columns: Columns to read (nested leaves as dotted paths).

    Yields:
        Records as dictionaries.
    """
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512, columns=list(columns)):
        training = cast("pa.StructArray", batch.column("training"))
        yield from batch.filter(training.field("choice_eligible")).to_pylist()


def download_test_shard(revision: str = DATASET_REVISION) -> Path:
    """Download (or reuse from the Hugging Face cache) the test partition shard at ``revision``.

    Args:
        revision: Dataset commit.

    Returns:
        Local path of the shard.
    """
    return Path(hf_hub_download(DATASET_ID, TEST_SHARD, repo_type="dataset", revision=revision))


def source_row_count(path: Path) -> int:
    """Return the number of rows in a Parquet file from its metadata.

    Args:
        path: Parquet file.

    Returns:
        Row count.
    """
    return pq.ParquetFile(path).metadata.num_rows
