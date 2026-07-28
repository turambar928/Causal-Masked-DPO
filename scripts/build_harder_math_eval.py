#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import write_jsonl


def inventory(seed: int) -> tuple[str, int]:
    shelves = 3 + seed % 5
    boxes = 4 + seed % 7
    packs = 5 + seed % 6
    sold = (seed % 4 + 1) * (3 + seed % 5)
    added = 10 + seed % 13
    answer = shelves * boxes * packs - sold + added
    question = (
        f"A store has {shelves} shelves with {boxes} boxes on each shelf. "
        f"Each box contains {packs} markers. During the morning, {sold} markers are sold, "
        f"and later {added} new markers are added. How many markers are in the store now?"
    )
    return question, answer


def tickets(seed: int) -> tuple[str, int]:
    adults = 12 + seed % 11
    children = 8 + seed % 9
    adult_price = 9 + seed % 8
    child_price = 4 + seed % 5
    coupon = 15 + (seed % 6) * 5
    snacks = 18 + seed % 17
    answer = adults * adult_price + children * child_price - coupon + snacks
    question = (
        f"A group buys {adults} adult tickets at ${adult_price} each and {children} child tickets "
        f"at ${child_price} each. They use a ${coupon} coupon, then spend ${snacks} on snacks. "
        f"How many dollars do they spend in total?"
    )
    return question, answer


def garden(seed: int) -> tuple[str, int]:
    rows = 5 + seed % 8
    per_row = 7 + seed % 9
    days = 3 + seed % 5
    daily = 4 + seed % 6
    lost = 6 + seed % 11
    answer = rows * per_row + days * daily - lost
    question = (
        f"A garden starts with {rows} rows of flowers and {per_row} flowers in each row. "
        f"For {days} days, {daily} more flowers are planted each day. Then {lost} flowers wilt. "
        f"How many flowers are left?"
    )
    return question, answer


def classroom(seed: int) -> tuple[str, int]:
    students = 18 + seed % 13
    pages = 6 + seed % 8
    extra_packets = 3 + seed % 5
    packet_pages = 9 + seed % 7
    recycled = 20 + seed % 21
    answer = students * pages + extra_packets * packet_pages - recycled
    question = (
        f"A teacher prints {pages} pages for each of {students} students. "
        f"She also prints {extra_packets} extra packets with {packet_pages} pages each. "
        f"If {recycled} pages are recycled, how many printed pages remain?"
    )
    return question, answer


def recipe(seed: int) -> tuple[str, int]:
    batches = 4 + seed % 7
    cups = 3 + seed % 5
    bought = 12 + seed % 9
    used_elsewhere = 5 + seed % 6
    answer = batches * cups + bought - used_elsewhere
    question = (
        f"A bakery uses {cups} cups of flour for each batch of muffins and makes {batches} batches. "
        f"It also buys {bought} more cups of flour, but uses {used_elsewhere} cups for bread. "
        f"How many cups of flour are accounted for by muffins and remaining added flour?"
    )
    return question, answer


def shipping(seed: int) -> tuple[str, int]:
    crates = 6 + seed % 8
    items = 11 + seed % 10
    damaged_crates = 1 + seed % 3
    damaged_each = 2 + seed % 4
    bonus = 15 + seed % 12
    answer = crates * items - damaged_crates * damaged_each + bonus
    question = (
        f"A warehouse packs {crates} crates with {items} items in each crate. "
        f"In {damaged_crates} crates, {damaged_each} items are damaged in each. "
        f"Then {bonus} replacement items arrive. How many usable items are there?"
    )
    return question, answer


TEMPLATES: list[Callable[[int], tuple[str, int]]] = [
    inventory,
    tickets,
    garden,
    classroom,
    recipe,
    shipping,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for idx in range(args.limit):
        template = TEMPLATES[idx % len(TEMPLATES)]
        question, answer = template(rng.randint(0, 10_000))
        rows.append(
            {
                "prompt": f"Question: {question}\nAnswer step by step and end with '#### <answer>'.",
                "answer": f"#### {answer}",
                "metadata": {
                    "source": "harder_template_arithmetic",
                    "row_id": idx,
                    "template": template.__name__,
                },
            }
        )

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
