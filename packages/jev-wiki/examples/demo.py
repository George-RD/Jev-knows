"""An executable lifecycle demonstration with explicitly scripted decisions.

This verifies plumbing and evidence invariants; it does not measure JEV quality.
Run from packages/jev-wiki: python examples/demo.py --root /tmp/jev-wiki-demo
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from jev_wiki.engine import Engine
from jev_wiki.provider import ScriptedProvider


def accepted(kind="fact", topic="projects"):
    return {
        f"{key}_0": {"value": value, "confidence": 0.99}
        for key, value in (("keep", "keep"), ("kind", kind), ("topic", topic))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="jev-wiki-demo-"))
    if args.root and root.exists():
        parser.error("choose a new directory so an existing memory is never changed")
    provider = ScriptedProvider(
        [
            accepted(),
            accepted(),
            {"relation": {"value": "conflict", "confidence": 0.99}},
            accepted("preference", "preferences"),
            accepted("preference", "preferences"),
        ]
    )
    engine = Engine(root, provider)
    first = engine.ingest("Orion's approved launch colour is blue.", "orion:brief", "Orion brief")
    second = engine.ingest("Orion's approved launch colour is red.", "orion:review", "Orion review")
    maintenance = engine.maintain()
    before = engine.recall("Orion launch colour", offline=True)
    assert maintenance["conflicts"] == 1 and len(before["items"]) == 2
    assert all(item["conflicts"] for item in before["items"])
    engine.ingest("Morgan prefers email updates.", "morgan:preferences", "Morgan preferences")
    engine.ingest("Morgan prefers short voice updates.", "morgan:preferences", "Morgan preferences")
    current = engine.recall("Morgan updates", offline=True)
    assert len(current["items"]) == 1 and "voice" in current["items"][0]["text"]
    engine.forget("orion:review")
    after = engine.recall("Orion launch colour", offline=True)
    assert len(after["items"]) == 1 and not after["items"][0]["conflicts"]
    assert engine.store.lint() == []
    summary = {
        "mode": "scripted demonstration; no JEV inference or quality measurement",
        "root": str(root.resolve()),
        "wiki": str((root / "wiki" / "index.md").resolve()),
        "ingested": [first, second],
        "maintenance": maintenance,
        "conflicting_evidence": before["context"],
        "revised_preference": current["context"],
        "after_forget": after["context"],
        "lint": "passed",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
