#!/usr/bin/env python3

#Offline catalog build step.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "catalog_raw.json"
OUT_PATH = PROJECT_ROOT / "data" / "catalog_processed.json"

# SHL's published category -> single-letter test-type code convention.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

_DURATION_RE = re.compile(r"(\d+)")


def _clean_text(value: str) -> str:
    """Collapse line-wrap whitespace artifacts left over from source
    extraction (e.g. a description that was hard-wrapped mid-sentence)."""
    if not isinstance(value, str):
        return value
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _parse_duration_minutes(duration: str) -> int | None:
    if not duration:
        return None
    match = _DURATION_RE.search(duration)
    return int(match.group(1)) if match else None


def build_one(raw: dict) -> dict | None:
    if raw.get("status") != "ok":
        return None

    name = _clean_text(raw.get("name", ""))
    url = _clean_text(raw.get("link", ""))
    if not name or not url:
        return None

    keys = raw.get("keys") or []
    codes = sorted({CATEGORY_TO_CODE[k] for k in keys if k in CATEGORY_TO_CODE})
    duration_raw = _clean_text(raw.get("duration", "") or "")

    return {
        "id": str(raw.get("entity_id")),
        "name": name,
        "url": url,
        "description": _clean_text(raw.get("description", "") or ""),
        "job_levels": raw.get("job_levels") or [],
        "languages": raw.get("languages") or [],
        "duration_raw": duration_raw,
        "duration_minutes": _parse_duration_minutes(duration_raw),
        "remote_testing": raw.get("remote") == "yes",
        "adaptive": raw.get("adaptive") == "yes",
        "categories": keys,
        "test_type_codes": codes,
    }


def main() -> int:
    if not RAW_PATH.exists():
        print(f"ERROR: raw catalog not found at {RAW_PATH}", file=sys.stderr)
        return 1

    raw_records = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    processed, skipped = [], 0
    for record in raw_records:
        item = build_one(record)
        if item is None:
            skipped += 1
            continue
        processed.append(item)

    if not processed:
        print("ERROR: zero valid records produced - aborting write.", file=sys.stderr)
        return 1

    OUT_PATH.write_text(json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(processed)} assessments to {OUT_PATH} ({skipped} skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
