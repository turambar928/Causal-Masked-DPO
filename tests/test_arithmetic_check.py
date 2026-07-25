from __future__ import annotations

import unittest

from cmdpo.arithmetic_check import locate_arithmetic_error


class ArithmeticCheckTest(unittest.TestCase):
    def test_bad_addition_result(self) -> None:
        prompt = "Question: A box has 12 packs. Each pack contains 3 pencils. Then 2 more pencils are added."
        steps = ["12 × 3 = 36", "36 + 2 = 39", "#### 39"]
        self.assertEqual(locate_arithmetic_error(prompt, steps), 1)

    def test_bad_extra_value(self) -> None:
        prompt = "Question: A box has 13 packs. Each pack contains 4 pencils. Then 3 more pencils are added."
        steps = ["13 × 4 = 52", "52 + 4 = 56", "#### 56"]
        self.assertEqual(locate_arithmetic_error(prompt, steps), 1)


if __name__ == "__main__":
    unittest.main()
