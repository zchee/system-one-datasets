"""``push`` tests with a recording stand-in for ``HfApi`` (no network), plus the generated card's metadata."""

import io
from typing import TYPE_CHECKING

import orjson
import yaml

from system_one_datasets.card import render_card
from system_one_datasets.push import DELETE_PATTERNS, UPLOAD_PATTERNS, push, upload_files
from system_one_datasets.rows import encode_row
from system_one_datasets.schema import BenchRecord


if TYPE_CHECKING:
    from pathlib import Path


class RecordingApi:
    """Records ``create_repo``/``upload_folder`` calls instead of talking to the Hub."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(
        self, repo_id: str, *, private: bool | None = None, repo_type: str | None = None, exist_ok: bool = False
    ) -> object:
        """Record the call."""
        self.calls.append((
            "create_repo",
            {"repo_id": repo_id, "private": private, "repo_type": repo_type, "exist_ok": exist_ok},
        ))
        return None

    def upload_folder(self, **kwargs: object) -> object:
        """Record the call."""
        self.calls.append(("upload_folder", kwargs))
        return None


RECORD = BenchRecord(
    id="cfg/test/0",
    config="cfg",
    kind="noul",
    state="some state",
    question={"type": "noul", "instructions": "Is this spam?"},
    options=["0", "1"],
    label="1",
    soft_label=None,
)


def make_root(tmp_path: Path, *, valid: bool = True, card: bool = True) -> Path:
    """A dataset root with one split file, a manifest, a card, and a file that must not be uploaded."""
    root = tmp_path / "data"
    split = root / "suite" / "cfg" / "test.jsonl"
    split.parent.mkdir(parents=True)
    row = encode_row(RECORD, "suite", "owner/dataset", "0" * 40, "owner/original", "cc0-1.0")
    if not valid:
        row["label"] = "7"
    split.write_bytes(orjson.dumps(row) + b"\n")
    (root / "manifest.yaml").write_text("name: x\n", encoding="utf-8")
    if card:
        (root / "README.md").write_text("---\nlicense: other\n---\n", encoding="utf-8")
    (root / "notes.txt").write_text("local scratch\n", encoding="utf-8")
    return root


def test_upload_files_follow_the_allow_patterns(tmp_path: Path) -> None:
    """Only the card, the manifest, and split files are selected; sizes are the on-disk sizes."""
    root = make_root(tmp_path)
    files = upload_files(root)
    assert [f.path for f in files] == ["README.md", "manifest.yaml", "suite/cfg/test.jsonl"]
    assert all(f.size == (root / f.path).stat().st_size for f in files)


def test_dry_run_lists_files_and_makes_no_calls(tmp_path: Path) -> None:
    """A dry run reports target, visibility, files, and total bytes without touching the API."""
    root = make_root(tmp_path)
    api, out = RecordingApi(), io.StringIO()
    status = push(root, "someone/repo", public=False, dry_run=True, api=api, out=out)
    text = out.getvalue()
    assert status == 0
    assert api.calls == []
    assert "https://huggingface.co/datasets/someone/repo (dataset, private if created)" in text
    assert "suite/cfg/test.jsonl" in text
    assert "notes.txt" not in text
    total = sum(f.size for f in upload_files(root))
    assert f"3 files, {total:,} bytes" in text
    assert "dry run: nothing uploaded" in text
    assert "no prune: remote files are only added or overwritten, never deleted" in text
    assert "prune: remote files matching" not in text

    out = io.StringIO()
    assert push(root, "someone/repo", public=False, dry_run=True, prune=True, api=api, out=out) == 0
    assert api.calls == []
    assert "prune: remote files matching ['*.jsonl'] and absent locally are deleted" in out.getvalue()


def test_push_uploads_after_validation(tmp_path: Path) -> None:
    """A real push creates the repo (private by default) and uploads with the allow patterns, deleting nothing."""
    root = make_root(tmp_path)
    api = RecordingApi()
    assert push(root, "someone/repo", public=False, dry_run=False, api=api, out=io.StringIO()) == 0
    assert [name for name, _ in api.calls] == ["create_repo", "upload_folder"]
    assert api.calls[0][1] == {"repo_id": "someone/repo", "private": True, "repo_type": "dataset", "exist_ok": True}
    upload = api.calls[1][1]
    assert upload["repo_id"] == "someone/repo"
    assert upload["repo_type"] == "dataset"
    assert upload["folder_path"] == root
    assert upload["allow_patterns"] == list(UPLOAD_PATTERNS)
    assert upload["delete_patterns"] is None

    api = RecordingApi()
    assert push(root, "someone/repo", public=True, dry_run=False, prune=True, api=api, out=io.StringIO()) == 0
    assert api.calls[0][1]["private"] is False
    assert api.calls[1][1]["delete_patterns"] == list(DELETE_PATTERNS)


def test_push_refuses_invalid_data(tmp_path: Path) -> None:
    """Structural errors stop the push before any API call, dry run or not."""
    root = make_root(tmp_path, valid=False)
    for dry_run in (True, False):
        api, out = RecordingApi(), io.StringIO()
        assert push(root, "someone/repo", public=False, dry_run=dry_run, api=api, out=out) == 1
        assert api.calls == []
        assert "refusing to upload: structural validation" in out.getvalue()
        assert "label '7' is not an option" in out.getvalue()


def test_push_requires_card_and_manifest(tmp_path: Path) -> None:
    """A root without README.md is refused."""
    root = make_root(tmp_path, card=False)
    api, out = RecordingApi(), io.StringIO()
    assert push(root, "someone/repo", public=False, dry_run=True, api=api, out=out) == 2
    assert api.calls == []
    assert "lacks README.md" in out.getvalue()


def test_card_frontmatter_declares_every_config() -> None:
    """The card's YAML has license other, block-style lists, and one data_files entry per split."""
    configs = [
        {
            "config": "civil_comments",
            "suite": "moderation",
            "kinds": ["noul"],
            "splits": {
                "test": {"file": "moderation/civil_comments/test.jsonl", "rows": 2000},
                "validation": {"file": "moderation/civil_comments/validation.jsonl", "rows": 500},
            },
            "source": {"hf_id": "Praveenrajus/jev-bench", "revision": "a" * 40},
            "upstream": {"hf_id": "google/civil_comments"},
            "license": "cc0-1.0",
            "license_as_stated_by_source": "cc0-1.0",
            "soft_labels": True,
        },
    ]
    card = render_card(configs)
    _, front, body = card.split("---\n", 2)
    meta = yaml.safe_load(front)
    assert meta["license"] == "other"
    assert meta["size_categories"] == ["1K<n<10K"]
    assert meta["configs"] == [
        {
            "config_name": "civil_comments",
            "data_files": [
                {"split": "test", "path": "moderation/civil_comments/test.jsonl"},
                {"split": "validation", "path": "moderation/civil_comments/validation.jsonl"},
            ],
        }
    ]
    assert "[" not in front  # Block-style sequences only.
    assert "| `civil_comments` | moderation | noul | 2,000 | 500 | yes | google/civil_comments | cc0-1.0 |" in body
    assert "not affiliated with or endorsed by TypeSafe, OpenJev" in body
