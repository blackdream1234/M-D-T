"""
Typed configuration dataclasses for GSNH-MDT.

Each field is explicitly classified as:
  - BASELINE: journal-compliant, controls core algorithm behavior
  - ENHANCED: research improvement, can be toggled independently
  - EXPERIMENTAL: unstable, under development

Parameter Mapping Table
=======================

Old Constructor Param        Config Field                          Default  Classification
─────────────────────        ─────────────                         ───────  ──────────────
stopping_criteria            ModelConfig.stopping                  SC()     BASELINE
n_bins                       ModelConfig.n_bins                    64       BASELINE
binning_strategy             ModelConfig.binning_strategy          'quant.' BASELINE
top_k_features               SearchConfig.top_k_features           15       BASELINE
use_gain_ratio               ModelConfig.use_gain_ratio            False    BASELINE
laplace_smoothing            ModelConfig.laplace_smoothing         1.0      BASELINE
search_1d                    SearchConfig.search_1d                True     BASELINE
search_2d                    SearchConfig.search_2d                True     BASELINE
search_3d                    SearchConfig.search_3d                True     BASELINE
use_supervised_binning       OptimizationConfig.use_supervised_..  True     ENHANCED
use_attention                OptimizationConfig.use_attention      True     ENHANCED
use_look_ahead               OptimizationConfig.use_look_ahead     False    ENHANCED
look_ahead_gamma             OptimizationConfig.look_ahead_gamma   0.3      ENHANCED
look_ahead_top_p             OptimizationConfig.look_ahead_top_p   5        ENHANCED
verbose                      ModelConfig.verbose                   False    BASELINE
mode                         ModelConfig.mode                      'heur.'  BASELINE
language                     ModelConfig.language                  ANY      BASELINE
limit_2d                     SearchConfig.limit_2d                 None     BASELINE
limit_3d                     SearchConfig.limit_3d                 None     BASELINE
use_binary_comparisons       SearchConfig.use_binary_comparisons   False    EXPERIMENTAL
enable_compare_literals      SearchConfig.enable_compare_literals  False    EXPERIMENTAL
prune                        ModelConfig.prune                     False    ENHANCED
prune_alpha                  ModelConfig.prune_alpha               0.01     ENHANCED

Precedence Rules
================
1. If `from_config(config)` is used, ALL parameters come from the config.
2. The legacy 22-param constructor is never altered.
3. `from_config()` converts config fields into the exact legacy parameter
   values, then calls the unchanged constructor.
4. No implicit merging — config is the single source when used.
"""

from dataclasses import dataclass, field
from typing import Optional

from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily


@dataclass
class SearchConfig:
    """Controls which search arities and optimizations are enabled.

    Classification:
        search_1d, search_2d, search_3d, top_k_features,
        limit_2d, limit_3d: BASELINE
        use_binary_comparisons, enable_compare_literals: EXPERIMENTAL
    """
    # BASELINE
    search_1d: bool = True
    search_2d: bool = True
    search_3d: bool = True
    top_k_features: int = 15
    limit_2d: Optional[int] = None
    limit_3d: Optional[int] = None

    # EXPERIMENTAL
    use_binary_comparisons: bool = False
    enable_compare_literals: bool = False


@dataclass
class OptimizationConfig:
    """Enhanced features beyond baseline journal logic.

    ALL fields are ENHANCED.
    None of these affect journal compliance — only accuracy/speed.
    """
    # ENHANCED
    use_supervised_binning: bool = True
    use_attention: bool = True
    use_look_ahead: bool = False
    look_ahead_gamma: float = 0.3
    look_ahead_top_p: int = 5


@dataclass
class CalibrationConfig:
    """Probability calibration settings. ENHANCED."""
    enabled: bool = False
    method: str = 'platt'  # 'platt' or 'isotonic'


@dataclass
class ModelConfig:
    """Complete model configuration.

    Usage::

        config = ModelConfig(language=LanguageFamily.BEST_PER_NODE)
        tree = ExpertGSNHTree.from_config(config)

    Classification:
        n_bins, binning_strategy, use_gain_ratio, laplace_smoothing,
        mode, language, verbose: BASELINE
        prune, prune_alpha: ENHANCED
    """
    # BASELINE — core algorithm behavior
    n_bins: int = 64
    binning_strategy: str = 'quantile'
    use_gain_ratio: bool = False
    laplace_smoothing: float = 1.0
    mode: str = 'heuristic'
    language: LanguageFamily = LanguageFamily.ANY
    verbose: bool = False

    # ENHANCED — research improvements
    prune: bool = False
    prune_alpha: float = 0.01

    # Sub-configs
    stopping: StoppingCriteria = field(default_factory=StoppingCriteria)
    search: SearchConfig = field(default_factory=SearchConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_constructor_kwargs(self) -> dict:
        """Convert this config into the exact kwargs for ExpertGSNHTree.__init__.

        This is the ONLY conversion point. The mapping is explicit,
        deterministic, and preserves every default exactly.

        Returns:
            dict of keyword arguments for ExpertGSNHTree(...).
        """
        return {
            'stopping_criteria': self.stopping,
            'n_bins': self.n_bins,
            'binning_strategy': self.binning_strategy,
            'top_k_features': self.search.top_k_features,
            'use_gain_ratio': self.use_gain_ratio,
            'laplace_smoothing': self.laplace_smoothing,
            'search_1d': self.search.search_1d,
            'search_2d': self.search.search_2d,
            'search_3d': self.search.search_3d,
            'use_supervised_binning': self.optimization.use_supervised_binning,
            'use_attention': self.optimization.use_attention,
            'use_look_ahead': self.optimization.use_look_ahead,
            'look_ahead_gamma': self.optimization.look_ahead_gamma,
            'look_ahead_top_p': self.optimization.look_ahead_top_p,
            'verbose': self.verbose,
            'mode': self.mode,
            'language': self.language,
            'limit_2d': self.search.limit_2d,
            'limit_3d': self.search.limit_3d,
            'use_binary_comparisons': self.search.use_binary_comparisons,
            'enable_compare_literals': self.search.enable_compare_literals,
            'prune': self.prune,
            'prune_alpha': self.prune_alpha,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark scripts."""
    n_runs: int = 10
    random_state: int = 42
    skip_large: int = 5000
    data_dir: str = 'data'
