"""Record types shared by the loader, runner, metrics, and report."""

from dataclasses import dataclass
from typing import Literal


type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
type Kind = Literal["noul", "choice", "score"]

KINDS: tuple[Kind, ...] = ("noul", "choice", "score")

# Option keys for a noul question: "0" is no, "1" is yes. They match the jev-bench label strings.
NOUL_OPTIONS: tuple[str, str] = ("0", "1")


@dataclass(frozen=True, slots=True)
class BenchRecord:
    """One benchmark item: a state, a single wire-format question, and its gold label.

    Attributes:
        id: Stable row id from the dataset (e.g. ``civil_comments/test/71007``).
        config: Dataset config the row belongs to.
        kind: Question type.
        state: Parsed ``state`` value, sent to the backend unchanged.
        question: Parsed wire-format Question object, sent to the backend unchanged.
        options: Ordered option keys of the predictive distribution. Choice: criteria keys in order.
            Score: level indices as strings. Noul: ``["0", "1"]``.
        label: Gold option key (one of ``options``).
        soft_label: Annotator distribution over ``options``, or ``None`` when the row has none.
    """

    id: str
    config: str
    kind: Kind
    state: JSONValue
    question: dict[str, JSONValue]
    options: list[str]
    label: str
    soft_label: dict[str, float] | None
