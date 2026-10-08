import numpy as np

from scripts.summarize_stage1 import holm_adjust, mcnemar_exact, paired_bootstrap


def test_mcnemar_known_discordant_counts():
    assert mcnemar_exact(np.zeros(4), np.ones(4)) == .125
    assert mcnemar_exact(np.array([0, 1]), np.array([1, 0])) == 1.
    assert mcnemar_exact(np.ones(4), np.ones(4)) == 1.


def test_holm_adjustment_preserves_original_order():
    np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])


def test_paired_bootstrap_preserves_identical_predictions():
    assert paired_bootstrap(np.zeros((3, 20)), replicates=100) == [0., 0.]
