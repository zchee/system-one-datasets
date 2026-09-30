"""Upload a validated ``data/`` directory to a Hugging Face dataset repository.

Authentication uses huggingface_hub's defaults (the ``hf auth login`` token or ``HF_TOKEN``); this module never
reads or prints the token.
"""

from dataclasses import dataclass
import sys
from typing import TYPE_CHECKING, Protocol

from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

from system_one_datasets.validate import data_files, render_summary, validate_file


if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO


# Files uploaded from the dataset root: the card, the manifest, and every split file.
UPLOAD_PATTERNS: tuple[str, ...] = ("README.md", "manifest.yaml", "*/*/*.jsonl")
# With ``prune``: remote files deleted in the upload commit, so splits removed by a rebuild do not linger.
DELETE_PATTERNS: tuple[str, ...] = ("*.jsonl",)
REQUIRED_FILES: tuple[str, ...] = ("README.md", "manifest.yaml")


class HubApi(Protocol):
    """The two ``huggingface_hub.HfApi`` methods ``push`` calls."""

    def create_repo(
        self, repo_id: str, *, private: bool | None = None, repo_type: str | None = None, exist_ok: bool = False
    ) -> object:
        """Create the repository, or do nothing when it exists and ``exist_ok`` is set."""
        ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        commit_message: str | None = None,
        repo_type: str | None = None,
        allow_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
    ) -> object:
        """Upload matching files of a local folder in one commit."""
        ...


@dataclass(frozen=True, slots=True)
class UploadFile:
    """One file to upload.

    Attributes:
        path: Path relative to the dataset root (also the path in the repository).
        size: Size in bytes.
    """

    path: str
    size: int


def upload_files(root: Path) -> list[UploadFile]:
    """List the files ``upload_folder`` would send, using the same pattern matcher.

    Args:
        root: Dataset root.

    Returns:
        Files in sorted order.
    """
    local = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    selected = filter_repo_objects(local, allow_patterns=list(UPLOAD_PATTERNS))
    return [UploadFile(path=rel, size=(root / rel).stat().st_size) for rel in selected]


def push(
    root: Path,
    repo_id: str,
    *,
    public: bool,
    dry_run: bool,
    prune: bool = False,
    api: HubApi | None = None,
    out: TextIO | None = None,
) -> int:
    """Validate ``root`` and upload it to ``repo_id`` (a dataset repository), or describe the upload.

    Args:
        root: Dataset root (``data/``).
        repo_id: Target repository, e.g. ``zchee/system-one-datasets``.
        public: Create the repository public; private otherwise. Visibility of an existing repository is unchanged.
        dry_run: Describe the upload without any network writes.
        prune: Delete remote ``DELETE_PATTERNS`` files that are not uploaded. Off by default, so a push from a
            partial dataset root never removes remote splits.
        api: Hub client; ``HfApi()`` when ``None``.
        out: Report stream; ``sys.stdout`` when ``None``.

    Returns:
        Process exit status: 0 on success, 1 when validation fails, 2 when required files are missing.
    """
    stream = out or sys.stdout
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        stream.write(f"refusing to upload: {root} lacks {', '.join(missing)}; run the build first\n")
        return 2
    results = [validate_file(path) for path in data_files(root)]
    if not results or any(r.errors for r in results):
        stream.write(render_summary(results))
        stream.write(f"refusing to upload: structural validation of {root} failed\n")
        return 1
    files = upload_files(root)
    total = sum(f.size for f in files)
    visibility = "public" if public else "private"
    stream.write(f"target: https://huggingface.co/datasets/{repo_id} (dataset, {visibility} if created)\n")
    if prune:
        stream.write(f"prune: remote files matching {list(DELETE_PATTERNS)} and absent locally are deleted\n")
    else:
        stream.write("no prune: remote files are only added or overwritten, never deleted\n")
    width = max(len(f"{f.size:,}") for f in files)
    stream.writelines(f"  {f.size:>{width},}  {f.path}\n" for f in files)
    stream.write(f"{len(files)} files, {total:,} bytes\n")
    if dry_run:
        stream.write("dry run: nothing uploaded\n")
        return 0
    hub = api if api is not None else HfApi()
    hub.create_repo(repo_id, private=not public, repo_type="dataset", exist_ok=True)
    hub.upload_folder(
        repo_id=repo_id,
        folder_path=root,
        commit_message=f"Upload System One datasets ({len(files)} files)",
        repo_type="dataset",
        allow_patterns=list(UPLOAD_PATTERNS),
        delete_patterns=list(DELETE_PATTERNS) if prune else None,
    )
    stream.write(f"uploaded to https://huggingface.co/datasets/{repo_id}\n")
    return 0
