"""Frozen token-level controls for the position-versus-mass diagnostic."""
import hashlib
import math
import random


def control_weights(weights, variant, sample_id, seed=20260930):
    """Input excludes EOS; the collator appends EOS with the last token's weight."""
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Expected a nonempty, finite nonnegative response mask")
    mass = math.fsum(weights) + weights[-1]
    if variant == "mass_matched_uniform":
        return [mass / (len(weights) + 1)] * len(weights)
    if variant != "prefix_preserving_shuffle":
        raise ValueError(variant)
    result = list(weights)
    # Preserve prefix zeros and the last response token, hence EOS as well.
    positions = [i for i, w in enumerate(weights[:-1]) if w > 0]
    values = [weights[i] for i in positions]
    rng = random.Random(int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest(), 16))
    rng.shuffle(values)
    # Do not silently turn a nondegenerate negative control into the true mask.
    original = [weights[i] for i in positions]
    if values == original and len(set(values)) > 1:
        values = values[1:] + values[:1]
    for i, w in zip(positions, values):
        result[i] = w
    return result
