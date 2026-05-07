"""
StoppingCriteria: stopping conditions for tree growth.

Extracted verbatim from gsnh_mdt_v3.py lines 1687-1725.
"""

from dataclasses import dataclass


@dataclass
class StoppingCriteria:
    """Enhanced stopping criteria."""
    min_gain_threshold: float = 1e-7  # Very small to allow more splits
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_depth: int = 15
    purity_threshold: float = 0.99
    use_mdl: bool = False
    mdl_penalty: float = 0.5
    complexity_penalty: float = 0.0

    def should_stop(self, n_samples: int, n_pos: int, n_neg: int,
                    depth: int, gain: float) -> tuple[bool, str]:

        if depth >= self.max_depth:
            return True, f"MAX_DEPTH:{depth}"

        if n_samples < self.min_samples_split:
            return True, f"MIN_SPLIT:{n_samples}"

        if n_pos == 0 or n_neg == 0:
            return True, "PURE"

        purity = max(n_pos, n_neg) / n_samples
        if purity >= self.purity_threshold:
            return True, f"PURITY:{purity:.3f}"

        if gain < self.min_gain_threshold:
            return True, f"MIN_GAIN:{gain:.6f}"

        # NOTE: MDL penalty is already applied inside penalized_gain()
        # in _search_best_split. Applying it again here would double-penalize.

        adjusted = gain - self.complexity_penalty * depth
        if adjusted < 0:
            return True, "COMPLEXITY"

        return False, "CONTINUE"
