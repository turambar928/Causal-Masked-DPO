#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.localization import build_cm_weights
from cmdpo.data import write_jsonl


@dataclass
class MathPair:
    question: str
    answer: int
    chosen_steps: list[str]
    rejected_steps: list[str]
    first_error_step: int
    template: str


def inventory(seed: int) -> MathPair:
    shelves = 3 + seed % 5
    boxes = 4 + seed % 7
    packs = 5 + seed % 6
    sold = (seed % 4 + 1) * (3 + seed % 5)
    added = 10 + seed % 13
    initial = shelves * boxes * packs
    answer = initial - sold + added
    wrong_initial = initial + boxes
    wrong_answer = wrong_initial - sold + added
    question = (
        f"A store has {shelves} shelves with {boxes} boxes on each shelf. "
        f"Each box contains {packs} markers. During the morning, {sold} markers are sold, "
        f"and later {added} new markers are added. How many markers are in the store now?"
    )
    return MathPair(
        question,
        answer,
        [
            f"There are {shelves} * {boxes} = {shelves * boxes} boxes.",
            f"The boxes contain {shelves * boxes} * {packs} = {initial} markers.",
            f"After selling {sold} and adding {added}, the total is {initial} - {sold} + {added} = {answer}.",
            f"#### {answer}",
        ],
        [
            f"There are {shelves} * {boxes} = {shelves * boxes} boxes.",
            f"The boxes contain {shelves * boxes} * {packs} = {wrong_initial} markers.",
            f"After selling {sold} and adding {added}, the total is {wrong_initial} - {sold} + {added} = {wrong_answer}.",
            f"#### {wrong_answer}",
        ],
        1,
        "inventory",
    )


def tickets(seed: int) -> MathPair:
    adults = 12 + seed % 11
    children = 8 + seed % 9
    adult_price = 9 + seed % 8
    child_price = 4 + seed % 5
    coupon = 15 + (seed % 6) * 5
    snacks = 18 + seed % 17
    adult_cost = adults * adult_price
    child_cost = children * child_price
    subtotal = adult_cost + child_cost
    answer = subtotal - coupon + snacks
    wrong_child_cost = child_cost + children
    wrong_subtotal = adult_cost + wrong_child_cost
    wrong_answer = wrong_subtotal - coupon + snacks
    question = (
        f"A group buys {adults} adult tickets at ${adult_price} each and {children} child tickets "
        f"at ${child_price} each. They use a ${coupon} coupon, then spend ${snacks} on snacks. "
        f"How many dollars do they spend in total?"
    )
    return MathPair(
        question,
        answer,
        [
            f"The adult tickets cost {adults} * {adult_price} = {adult_cost} dollars.",
            f"The child tickets cost {children} * {child_price} = {child_cost} dollars.",
            f"Before the coupon and snacks, the subtotal is {adult_cost} + {child_cost} = {subtotal} dollars.",
            f"After the ${coupon} coupon and ${snacks} snacks, the total is {subtotal} - {coupon} + {snacks} = {answer}.",
            f"#### {answer}",
        ],
        [
            f"The adult tickets cost {adults} * {adult_price} = {adult_cost} dollars.",
            f"The child tickets cost {children} * {child_price} = {wrong_child_cost} dollars.",
            f"Before the coupon and snacks, the subtotal is {adult_cost} + {wrong_child_cost} = {wrong_subtotal} dollars.",
            f"After the ${coupon} coupon and ${snacks} snacks, the total is {wrong_subtotal} - {coupon} + {snacks} = {wrong_answer}.",
            f"#### {wrong_answer}",
        ],
        1,
        "tickets",
    )


def garden(seed: int) -> MathPair:
    rows = 5 + seed % 8
    per_row = 7 + seed % 9
    days = 3 + seed % 5
    daily = 4 + seed % 6
    lost = 6 + seed % 11
    initial = rows * per_row
    planted = days * daily
    answer = initial + planted - lost
    wrong_planted = planted + days
    wrong_answer = initial + wrong_planted - lost
    question = (
        f"A garden starts with {rows} rows of flowers and {per_row} flowers in each row. "
        f"For {days} days, {daily} more flowers are planted each day. Then {lost} flowers wilt. "
        f"How many flowers are left?"
    )
    return MathPair(
        question,
        answer,
        [
            f"The garden starts with {rows} * {per_row} = {initial} flowers.",
            f"The new planting adds {days} * {daily} = {planted} flowers.",
            f"After {lost} flowers wilt, the total is {initial} + {planted} - {lost} = {answer}.",
            f"#### {answer}",
        ],
        [
            f"The garden starts with {rows} * {per_row} = {initial} flowers.",
            f"The new planting adds {days} * {daily} = {wrong_planted} flowers.",
            f"After {lost} flowers wilt, the total is {initial} + {wrong_planted} - {lost} = {wrong_answer}.",
            f"#### {wrong_answer}",
        ],
        1,
        "garden",
    )


def classroom(seed: int) -> MathPair:
    students = 18 + seed % 13
    pages = 6 + seed % 8
    extra_packets = 3 + seed % 5
    packet_pages = 9 + seed % 7
    recycled = 20 + seed % 21
    student_pages = students * pages
    extra_pages = extra_packets * packet_pages
    answer = student_pages + extra_pages - recycled
    wrong_extra_pages = extra_pages + packet_pages
    wrong_answer = student_pages + wrong_extra_pages - recycled
    question = (
        f"A teacher prints {pages} pages for each of {students} students. "
        f"She also prints {extra_packets} extra packets with {packet_pages} pages each. "
        f"If {recycled} pages are recycled, how many printed pages remain?"
    )
    return MathPair(
        question,
        answer,
        [
            f"The student pages total {students} * {pages} = {student_pages}.",
            f"The extra packets total {extra_packets} * {packet_pages} = {extra_pages} pages.",
            f"After recycling {recycled} pages, {student_pages} + {extra_pages} - {recycled} = {answer} pages remain.",
            f"#### {answer}",
        ],
        [
            f"The student pages total {students} * {pages} = {student_pages}.",
            f"The extra packets total {extra_packets} * {packet_pages} = {wrong_extra_pages} pages.",
            f"After recycling {recycled} pages, {student_pages} + {wrong_extra_pages} - {recycled} = {wrong_answer} pages remain.",
            f"#### {wrong_answer}",
        ],
        1,
        "classroom",
    )


def recipe(seed: int) -> MathPair:
    batches = 4 + seed % 7
    cups = 3 + seed % 5
    bought = 12 + seed % 9
    used_elsewhere = 5 + seed % 6
    muffin_flour = batches * cups
    answer = muffin_flour + bought - used_elsewhere
    wrong_muffin_flour = muffin_flour + cups
    wrong_answer = wrong_muffin_flour + bought - used_elsewhere
    question = (
        f"A bakery uses {cups} cups of flour for each batch of muffins and makes {batches} batches. "
        f"It also buys {bought} more cups of flour, but uses {used_elsewhere} cups for bread. "
        f"How many cups of flour are accounted for by muffins and remaining added flour?"
    )
    return MathPair(
        question,
        answer,
        [
            f"The muffins use {batches} * {cups} = {muffin_flour} cups of flour.",
            f"The remaining added flour is {bought} - {used_elsewhere} = {bought - used_elsewhere} cups.",
            f"Together that is {muffin_flour} + {bought - used_elsewhere} = {answer} cups.",
            f"#### {answer}",
        ],
        [
            f"The muffins use {batches} * {cups} = {wrong_muffin_flour} cups of flour.",
            f"The remaining added flour is {bought} - {used_elsewhere} = {bought - used_elsewhere} cups.",
            f"Together that is {wrong_muffin_flour} + {bought - used_elsewhere} = {wrong_answer} cups.",
            f"#### {wrong_answer}",
        ],
        0,
        "recipe",
    )


def shipping(seed: int) -> MathPair:
    crates = 6 + seed % 8
    items = 11 + seed % 10
    damaged_crates = 1 + seed % 3
    damaged_each = 2 + seed % 4
    bonus = 15 + seed % 12
    initial = crates * items
    damaged = damaged_crates * damaged_each
    answer = initial - damaged + bonus
    wrong_damaged = damaged + damaged_crates
    wrong_answer = initial - wrong_damaged + bonus
    question = (
        f"A warehouse packs {crates} crates with {items} items in each crate. "
        f"In {damaged_crates} crates, {damaged_each} items are damaged in each. "
        f"Then {bonus} replacement items arrive. How many usable items are there?"
    )
    return MathPair(
        question,
        answer,
        [
            f"The warehouse starts with {crates} * {items} = {initial} items.",
            f"The damaged items total {damaged_crates} * {damaged_each} = {damaged}.",
            f"After replacements, there are {initial} - {damaged} + {bonus} = {answer} usable items.",
            f"#### {answer}",
        ],
        [
            f"The warehouse starts with {crates} * {items} = {initial} items.",
            f"The damaged items total {damaged_crates} * {damaged_each} = {wrong_damaged}.",
            f"After replacements, there are {initial} - {wrong_damaged} + {bonus} = {wrong_answer} usable items.",
            f"#### {wrong_answer}",
        ],
        1,
        "shipping",
    )


TEMPLATES: list[Callable[[int], MathPair]] = [
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
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for idx in range(args.limit):
        template = TEMPLATES[idx % len(TEMPLATES)]
        pair = template(rng.randint(0, 1_000_000))
        prompt = f"Question: {pair.question}\nAnswer step by step and end with '#### <answer>'."
        rows.append(
            {
                "prompt": prompt,
                "chosen": "\n".join(pair.chosen_steps),
                "rejected": "\n".join(pair.rejected_steps),
                "answer": f"#### {pair.answer}",
                "rejected_steps": pair.rejected_steps,
                "first_error_step": pair.first_error_step,
                "step_weights": build_cm_weights(len(pair.rejected_steps), pair.first_error_step, args.gamma),
                "localization_confidence": 1.0,
                "metadata": {
                    "source": "harder_template_arithmetic",
                    "row_id": idx,
                    "template": pair.template,
                    "localizer": "template_oracle",
                },
            }
        )

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
