"""Loader for the ``Praveenrajus/jev-bench`` Hugging Face dataset."""

from typing import TYPE_CHECKING

import datasets
from huggingface_hub import HfApi
import orjson

from system_one_datasets.schema import KINDS, NOUL_OPTIONS, BenchRecord, JSONValue, Kind


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


DATASET_ID = "Praveenrajus/jev-bench"
PHASE1_CONFIGS: tuple[str, ...] = (
    "civil_comments",
    "measuring_hate_speech",
    "go_emotions",
    "helpsteer2_helpfulness",
    "stsb",
)


def question_options(kind: Kind, question: Mapping[str, JSONValue]) -> list[str]:
    """Return the ordered option keys of a wire-format question.

    Args:
        kind: Question type.
        question: Parsed Question object.

    Returns:
        Choice: criteria keys in order. Score: level indices ``"0".."K-1"``. Noul: ``["0", "1"]``.

    Raises:
        ValueError: ``criteria`` has the wrong shape for ``kind``.
    """
    if kind == "noul":
        return list(NOUL_OPTIONS)
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            msg = "choice criteria must be a non-empty object"
            raise ValueError(msg)
        return list(criteria)
    if not isinstance(criteria, list) or len(criteria) < 2:  # The API minimum is two levels.
        msg = "score criteria must be an array of at least two levels"
        raise ValueError(msg)
    return [str(i) for i in range(len(criteria))]


def normalize_soft_label(raw: str | None, kind: Kind, options: Sequence[str]) -> dict[str, float] | None:
    """Convert a jev-bench ``soft_label`` into a distribution keyed by ``options``.

    jev-bench stores three shapes: noul is a scalar P(yes) (the hard label is ``"1"`` when it is >= 0.5),
    score is an array indexed by level, and choice is an object keyed by option. Values are used as stored; the
    dataset's distributions already sum to 1.

    Args:
        raw: JSON text from the dataset, or ``None``/empty when the row has no soft label.
        kind: Question type.
        options: Ordered option keys of the question.

    Returns:
        Mapping of every option to its annotator probability, or ``None`` when absent.

    Raises:
        ValueError: The stored value does not match the shape expected for ``kind``.
    """
    if raw is None or not raw.strip():
        return None
    value = orjson.loads(raw)
    if value is None:
        return None
    if kind == "noul":
        if not isinstance(value, int | float):
            msg = "noul soft_label must be a number"
            raise ValueError(msg)
        no, yes = NOUL_OPTIONS
        return {no: 1.0 - float(value), yes: float(value)}
    if kind == "score":
        if not isinstance(value, list) or len(value) != len(options):
            msg = "score soft_label must be an array with one entry per level"
            raise ValueError(msg)
        return {option: float(p) for option, p in zip(options, value, strict=True)}
    if not isinstance(value, dict) or set(value) - set(options):
        msg = "choice soft_label must be an object keyed by criteria options"
        raise ValueError(msg)
    return {option: float(value.get(option, 0.0)) for option in options}


def record_from_row(row: Mapping[str, str | None], config: str) -> BenchRecord:
    """Build a ``BenchRecord`` from one raw jev-bench row.

    ``state`` and ``question`` are decoded from their JSON strings and otherwise left untouched.

    Args:
        row: Dataset row with ``id``, ``state``, ``question``, ``label``, ``soft_label`` string columns.
        config: Dataset config name.

    Returns:
        The parsed record.

    Raises:
        TypeError: The question is not a JSON object.
        ValueError: The question type is unknown or its fields are malformed.
    """
    question = orjson.loads(row["question"] or "null")
    if not isinstance(question, dict):
        msg = f"row {row['id']}: question is not an object"
        raise TypeError(msg)
    kind = question.get("type")
    if kind not in KINDS:
        msg = f"row {row['id']}: unknown question type {kind!r}"
        raise ValueError(msg)
    options = question_options(kind, question)
    label = str(row["label"])
    if label not in options:
        msg = f"row {row['id']}: label {label!r} is not an option"
        raise ValueError(msg)
    return BenchRecord(
        id=str(row["id"]),
        config=config,
        kind=kind,
        state=orjson.loads(row["state"] or "null"),
        question=question,
        options=options,
        label=label,
        soft_label=normalize_soft_label(row.get("soft_label"), kind, options),
    )


def resolve_revision(dataset_id: str, revision: str | None = None) -> str:
    """Resolve a dataset revision (branch, tag, or ``None`` for the default branch) to its commit sha.

    Args:
        dataset_id: Hugging Face dataset repository id.
        revision: Branch, tag, or commit; ``None`` means the default branch.

    Returns:
        The 40-character commit sha the Hub reports for ``revision``.

    Raises:
        ValueError: The Hub returned no sha.
    """
    sha = HfApi().dataset_info(dataset_id, revision=revision).sha
    if not sha:
        msg = f"no commit sha for {dataset_id}@{revision or 'default'}"
        raise ValueError(msg)
    return sha


def load_jev_bench(
    config: str, split: str = "test", limit: int | None = None, revision: str | None = None
) -> list[BenchRecord]:
    """Load one jev-bench config into ``BenchRecord`` objects.

    The dataset is cached under the default Hugging Face cache directory.

    Args:
        config: Dataset config name (e.g. ``civil_comments``).
        split: Dataset split.
        limit: Keep only the first ``limit`` rows when set.
        revision: Dataset commit, branch, or tag; ``None`` loads the default branch.

    Returns:
        Records in dataset order.

    Raises:
        TypeError: ``load_dataset`` returned something other than a single ``Dataset``.
    """
    ds = datasets.load_dataset(DATASET_ID, config, split=split, revision=revision)
    if not isinstance(ds, datasets.Dataset):
        msg = f"expected a Dataset for {config}/{split}, got {type(ds).__name__}"
        raise TypeError(msg)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [record_from_row(row, config) for row in ds]
