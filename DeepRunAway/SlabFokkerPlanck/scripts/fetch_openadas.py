#!/usr/bin/env python3
"""Fetch user-owned OpenADAS ADF11 files and write a provenance manifest.

OpenADAS terms prohibit redistributing downloaded data with software.  The
script therefore stores files in a user-selected data directory, which should
remain outside Git, and commits only this acquisition tool and its metadata
format.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


CLASSES = ("acd", "scd", "plt", "prb")
BASE_URL = "https://open.adas.ac.uk/download/adf11"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(element: str, year: int, target: Path, classes: tuple[str, ...], force: bool) -> Path:
    element = element.lower()
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for kind in classes:
        filename = f"{kind}{year}_{element}.dat"
        url = f"{BASE_URL}/{kind}{year}/{filename}"
        output = target/filename
        if output.exists() and not force:
            action = "existing"
        else:
            partial = output.with_suffix(output.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, partial)
                partial.replace(output)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            action = "downloaded"
        records.append({
            "class": kind,
            "filename": filename,
            "url": url,
            "path": str(output.resolve()),
            "sha256": sha256(output),
            "action": action,
        })

    manifest = target/f"openadas_{element}_{year}_manifest.json"
    payload = {
        "source": "OPEN-ADAS",
        "terms_url": "https://open.adas.ac.uk/terms-and-conditions",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "element": element,
        "year": year,
        "files": records,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch user-owned OpenADAS ADF11 files")
    parser.add_argument("--element", required=True, help="OpenADAS element symbol, e.g. Ar")
    parser.add_argument("--year", required=True, type=int, help="ADF11 data year, e.g. 89 or 96")
    parser.add_argument("--data-dir", type=Path, default=Path("data/openadas"))
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--force", action="store_true", help="replace existing files explicitly")
    args = parser.parse_args()
    manifest = fetch(args.element, args.year, args.data_dir, tuple(args.classes), args.force)
    print(f"wrote provenance manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
