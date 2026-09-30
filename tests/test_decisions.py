"""jev-decisions-v1 mapping and sampling tests on records shaped like the source Parquet rows (no network)."""

import copy
import random
from typing import TYPE_CHECKING

import orjson
import pytest

from system_one_datasets.decisions import (
    INSTRUCTIONS,
    allocate,
    bucket_of,
    ineligible_reason,
    map_decision,
    rank_key,
    row_license,
    select,
)
from system_one_datasets.rows import decode_row, encode_row
from system_one_datasets.validate import check_row


if TYPE_CHECKING:
    from system_one_datasets.schema import JSONValue

    type Row = dict[str, JSONValue]


def candidate(name: str, description: str | None) -> Row:
    """A source candidate struct."""
    return {
        "id": f"tool::{name}",
        "name": name,
        "description": description,
        "parameters_json": '{"type":"object","properties":{}}',
        "metadata_json": '{"type":"function"}',
    }


def source_row(record_id: str = "abc", n_candidates: int = 3, target: str = "send_code") -> Row:
    """A choice-eligible record modeled on a Nemotron-SFT-Agentic-v2 test row."""
    names = ["authenticate", "send_code", "cancel", *(f"extra_{i}" for i in range(max(n_candidates - 3, 0)))]
    candidates: list[JSONValue] = [candidate(n, None if n == "cancel" else f"Do {n}.") for n in names[:n_candidates]]
    return {
        "id": record_id,
        "source": "nvidia/Nemotron-SFT-Agentic-v2",
        "state": {
            "system": "You are a customer service agent.",
            "user_goal": "Renew my membership.",
            "history": [
                {"role": "system", "payload_json": '{"content":"You are a customer service agent."}'},
                {"role": "user", "payload_json": '{"content":"Renew my membership."}'},
                {"role": "assistant", "payload_json": "not json"},
            ],
            "environment_json": None,
        },
        "candidates": candidates,
        "target": {
            "candidate_id": f"tool::{target}",
            "action_name": target,
            "arguments_json": '{"email":"a@b.c"}',
            "label_json": None,
        },
        "ordered_targets": [],
        "provenance": {"trajectory_id": "152", "source_metadata_json": "{}"},
        "training": {"choice_eligible": True},
    }


def test_map_decision_builds_a_wire_choice_question() -> None:
    """State keeps system/user_goal, decodes history payloads, and never leaks the target's arguments."""
    decision = map_decision(source_row())
    record = decision.record
    assert record.id == "jev_decisions_v1/test/abc"
    assert record.kind == "choice"
    assert record.question == {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {"authenticate": "Do authenticate.", "send_code": "Do send_code.", "cancel": None},
    }
    assert record.options == ["authenticate", "send_code", "cancel"]
    assert record.label == "send_code"
    assert record.soft_label is None
    assert record.state == {
        "system": "You are a customer service agent.",
        "user_goal": "Renew my membership.",
        "history": [
            {"role": "system", "content": {"content": "You are a customer service agent."}},
            {"role": "user", "content": {"content": "Renew my membership."}},
            {"role": "assistant", "content": "not json"},
        ],
    }
    assert decision.decode_fallbacks == 1
    assert decision.bucket == "3-4"
    assert decision.trajectory == "nvidia/Nemotron-SFT-Agentic-v2/152"
    assert "a@b.c" not in orjson.dumps(record.state).decode()


def test_mapped_row_passes_the_validator() -> None:
    """The on-disk row built from a mapped decision is structurally valid and decodes back to the record."""
    record = map_decision(source_row()).record
    row = encode_row(
        record, "agent_action", "samatv256/jev-decisions-v1", "0" * 40, "nvidia/Nemotron-SFT-Agentic-v2", "cc-by-4.0"
    )
    assert check_row(row) == []
    assert row["soft_label"] == "null"
    assert decode_row(row) == record


LICENSES: dict[str, tuple[str, str | None, tuple[str, bool]]] = {
    "success: Open-SWE repository SPDX id": (
        "nvidia/Open-SWE-Traces",
        '{"repo_license":"BSD-3-Clause"}',
        ("BSD-3-Clause", False),
    ),
    "success: Nemotron rows keep the dataset license": (
        "nvidia/Nemotron-SFT-Agentic-v2",
        '{"repo_license":"MIT"}',
        ("cc-by-4.0", False),
    ),
    "fallback: Open-SWE without repo_license": ("nvidia/Open-SWE-Traces", '{"repo":"a/b"}', ("cc-by-4.0", True)),
    "fallback: Open-SWE NOASSERTION": ("nvidia/Open-SWE-Traces", '{"repo_license":"NOASSERTION"}', ("cc-by-4.0", True)),
    "fallback: Open-SWE SPDX expression": (
        "nvidia/Open-SWE-Traces",
        '{"repo_license":"MIT OR Apache-2.0"}',
        ("cc-by-4.0", True),
    ),
    "fallback: Open-SWE metadata not JSON": ("nvidia/Open-SWE-Traces", "not json", ("cc-by-4.0", True)),
    "fallback: Open-SWE metadata null": ("nvidia/Open-SWE-Traces", None, ("cc-by-4.0", True)),
}


@pytest.mark.parametrize(("upstream", "metadata", "expected"), list(LICENSES.values()), ids=list(LICENSES))
def test_row_license(upstream: str, metadata: str | None, expected: tuple[str, bool]) -> None:
    """Open-SWE rows take the repository's single SPDX id; everything else is the dataset license."""
    assert row_license(upstream, metadata) == expected


def test_open_swe_row_carries_repository_license() -> None:
    """map_decision and select propagate upstream and the per-row license."""
    row = source_row()
    row["source"] = "nvidia/Open-SWE-Traces"
    row["provenance"] = {"trajectory_id": "7", "source_metadata_json": '{"repo_license":"Apache-2.0"}'}
    decision = map_decision(row)
    assert (decision.upstream, decision.license, decision.license_fallback) == (
        "nvidia/Open-SWE-Traces",
        "Apache-2.0",
        False,
    )
    _, stats = select([row, source_row("other")], size=10, seed=7)
    assert stats.selected_by_license == {"Apache-2.0": 1, "cc-by-4.0": 1}
    assert stats.selected_by_upstream == {"nvidia/Open-SWE-Traces": 1, "nvidia/Nemotron-SFT-Agentic-v2": 1}
    assert stats.license_fallbacks == 0


def edit(row: Row, path: tuple[str, ...], value: JSONValue) -> Row:
    """Deep-copy ``row`` and set the value at ``path``."""
    out = copy.deepcopy(row)
    node = out
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = value
    return out


INELIGIBLE: dict[str, tuple[Row, str]] = {
    "not choice eligible": (edit(source_row(), ("training", "choice_eligible"), False), "not_choice_eligible"),
    "one candidate": (edit(source_row(), ("candidates",), [candidate("send_code", None)]), "candidate_count"),
    "multiple targets": (edit(source_row(), ("ordered_targets",), [{"action_name": "x"}]), "multiple_targets"),
    "duplicate names": (
        edit(source_row(), ("candidates",), [candidate("send_code", None), candidate("send_code", "again")]),
        "duplicate_candidate_names",
    ),
    "no target": (edit(source_row(), ("target",), None), "no_target"),
    "target id not a candidate": (
        edit(source_row(), ("target", "candidate_id"), "tool::missing"),
        "target_not_unique_in_candidates",
    ),
    "target name differs from candidate": (
        edit(source_row(), ("target", "action_name"), "cancel"),
        "target_not_unique_in_candidates",
    ),
}


@pytest.mark.parametrize(("row", "reason"), list(INELIGIBLE.values()), ids=list(INELIGIBLE))
def test_ineligible_records(row: Row, reason: str) -> None:
    """Each disqualifying shape is reported and map_decision refuses it."""
    got = ineligible_reason(row)
    assert got is not None
    assert got.startswith(reason)
    with pytest.raises(ValueError, match=reason):
        map_decision(row)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, None), (2, "2"), (3, "3-4"), (4, "3-4"), (5, "5-8"), (16, "9-16"), (17, "17+"), (255, "17+"), (256, None)],
)
def test_bucket_edges(n: int, expected: str | None) -> None:
    """Bucket boundaries are inclusive and 2..255 is the valid range."""
    assert bucket_of(n) == expected


def test_allocate_is_proportional_and_capped() -> None:
    """Largest-remainder shares sum to the target, follow availability, and never exceed it."""
    available = {"2": 5853, "3-4": 42950, "5-8": 13791, "9-16": 21281, "17+": 16924}
    shares = allocate(available, 1000)
    assert sum(shares.values()) == 1000
    assert shares == {"2": 58, "3-4": 426, "5-8": 137, "9-16": 211, "17+": 168}
    assert allocate({"2": 3, "3-4": 4}, 1000) == {"2": 3, "3-4": 4}
    tiny = allocate({"2": 1, "3-4": 1, "5-8": 1}, 2)
    assert sum(tiny.values()) == 2
    assert all(v <= 1 for v in tiny.values())


def test_rank_key_is_seeded_and_stable() -> None:
    """The key depends on seed and id only."""
    assert rank_key("abc", 20260930) == rank_key("abc", 20260930)
    assert rank_key("abc", 20260930) != rank_key("abc", 1)
    assert rank_key("abc", 20260930) != rank_key("abd", 20260930)


def test_select_is_order_independent_and_stratified() -> None:
    """The same records in any order give the same selection; the token cap excludes before sampling."""
    rows = [source_row(f"r{i}", n_candidates=2 if i % 4 == 0 else 3) for i in range(80)]
    first, stats = select(rows, size=10, seed=7)
    shuffled = rows.copy()
    random.Random(1).shuffle(shuffled)
    second, _ = select(shuffled, size=10, seed=7)
    assert [d.record.id for d in first] == [d.record.id for d in second]
    assert stats.allocation == {"2": 3, "3-4": 7, "5-8": 0, "9-16": 0, "17+": 0}
    assert [d.bucket for d in first] == ["2"] * 3 + ["3-4"] * 7
    # Within a bucket, rows are the smallest keys among all available rows of that bucket.
    bucket2 = sorted((rank_key(f"jev_decisions_v1/test/r{i}", 7), f"r{i}") for i in range(0, 80, 4))
    assert [d.record.id for d in first[:3]] == [f"jev_decisions_v1/test/{rid}" for _, rid in bucket2[:3]]
    assert stats.source_rows == 80
    assert stats.max_rows_per_trajectory == 10  # Every synthetic row shares one trajectory id.

    capped, cap_stats = select(rows, size=10, seed=7, token_limit=10)
    assert capped == []
    assert cap_stats.excluded["over_token_limit"] == 80
