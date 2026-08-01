from __future__ import annotations

import unittest

from cmdpo.localization import build_cm_weights, localize_from_success_rates
from cmdpo.segmentation import segment_steps
from cmdpo.verifier import extract_answer, verify_answer


class CoreTest(unittest.TestCase):
    def test_extract_answer(self) -> None:
        self.assertEqual(extract_answer("We compute it. #### 42"), "42")
        self.assertEqual(extract_answer("So the answer is \\boxed{7/2}."), "7/2")
        self.assertEqual(extract_answer("Final answer. #### 362Human: next prompt"), "362")

    def test_verify_answer(self) -> None:
        self.assertTrue(verify_answer("The answer is 3.5", "\\boxed{7/2}"))
        self.assertTrue(verify_answer("#### 1,024", "1024"))
        self.assertTrue(verify_answer("Final answer. #### 362Human: next prompt", "#### 362"))
        self.assertFalse(verify_answer("#### 5", "6"))

    def test_segment_steps(self) -> None:
        text = "1. Let x = 2.\n2. Then x + 3 = 5.\n3. Therefore the answer is 5."
        steps = segment_steps(text)
        self.assertGreaterEqual(len(steps), 2)
        self.assertIn("answer", steps[-1].lower())

    def test_weights(self) -> None:
        self.assertEqual(build_cm_weights(5, 2, 0.5), [0.0, 0.0, 1.0, 0.5, 0.25])

    def test_localize_from_success_rates(self) -> None:
        result = localize_from_success_rates([0.75, 0.75, 0.0, 0.0], gamma=0.5, tau=0.3)
        self.assertEqual(result.first_error_step, 2)
        self.assertEqual(result.step_weights, [0.0, 0.0, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
