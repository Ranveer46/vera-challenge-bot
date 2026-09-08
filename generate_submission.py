#!/usr/bin/env python3
"""Generate submission.jsonl for the 30 canonical test pairs.

Loads the expanded dataset (dataset/expanded/) and test_pairs.json, calls the
same compose_message() used by the live bot's /v1/tick, and writes one JSON
line per test pair with keys: test_id, body, cta, send_as, suppression_key,
rationale -- matching challenge-brief.md section 7.2.

Usage:
    python generate_submission.py [--dataset-dir ../dataset/expanded] [--out submission.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import composer


def load_dataset(dataset_dir: Path):
    categories = {}
    for f in (dataset_dir / "categories").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        categories[data["slug"]] = data

    merchants = {}
    for f in (dataset_dir / "merchants").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        merchants[data["merchant_id"]] = data

    customers = {}
    for f in (dataset_dir / "customers").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        customers[data["customer_id"]] = data

    triggers = {}
    for f in (dataset_dir / "triggers").glob("*.json"):
        data = json.load(open(f, encoding="utf-8"))
        triggers[data["id"]] = data

    with open(dataset_dir / "test_pairs.json", encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]

    return categories, merchants, customers, triggers, pairs


async def run(dataset_dir: Path, out_path: Path) -> None:
    categories, merchants, customers, triggers, pairs = load_dataset(dataset_dir)
    print(f"Loaded {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers, {len(pairs)} test pairs")

    lines = []
    for pair in pairs:
        test_id = pair["test_id"]
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None

        if not trigger or not merchant:
            print(f"  [SKIP] {test_id}: missing trigger or merchant")
            continue
        category = categories.get(merchant.get("category_slug"))
        if not category:
            print(f"  [SKIP] {test_id}: missing category {merchant.get('category_slug')}")
            continue

        result = await composer.compose_message(category, merchant, trigger, customer, allow_wait_on_rate_limit=True)
        line = {
            "test_id": test_id,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        }
        lines.append(line)
        print(f"  [OK] {test_id} ({pair['merchant_id']} / {trigger.get('kind')}): {result['body'][:70]}...")

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(lines)} lines to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="../dataset/expanded")
    parser.add_argument("--out", default="submission.jsonl")
    args = parser.parse_args()
    asyncio.run(run(Path(args.dataset_dir).resolve(), Path(args.out).resolve()))


if __name__ == "__main__":
    main()
