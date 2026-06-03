"""
GSNH-MDT Language Family Comparison — Professor's .dl8 Data
===========================================================

Compares different language families at depth 5 AND 7 against sklearn DT at depth 7:
- 1D (univariate)
- Horn
- Anti-Horn  
- ConjUI (AND/box)
- Square 2CNF (paper-style)
- Affine
- BEST_PER_NODE

.dl8 format:
    label f1 f2 f3 ... fn
    (space-separated integers, first column = class, rest = binary features)

Usage:
    python benchmark_dl8_languages.py
"""

import numpy as np
import os
import sys
import glob
import time
import traceback
import warnings
from collections import OrderedDict
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')


# =============================================================================
# IMPORT GSNH
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def import_gsnh():
    """Import GSNH module."""
    try:
        import types
        gsnh = types.SimpleNamespace()
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        from gsnh_mdt.tree.stopping import StoppingCriteria
        from gsnh_mdt.types import LanguageFamily
        gsnh.ExpertGSNHTree = ExpertGSNHTree
        gsnh.StoppingCriteria = StoppingCriteria
        gsnh.LanguageFamily = LanguageFamily

        # Print available enum values for debugging
        print("✓ Imported from gsnh_mdt package")
        print(f"  Available LanguageFamily values: {[e.name for e in LanguageFamily]}")
        return gsnh
    except ImportError as e:
        print(f"✗ Cannot import GSNH: {e}")
        sys.exit(1)


# =============================================================================
# .DL8 PARSER
# =============================================================================

def parse_dl8(filepath):
    """Parse .dl8 file: label f1 f2 ... fn"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(x) for x in line.split()]
            data.append(vals)

    data = np.array(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data shape: {data.shape}")

    y = data[:, 0].astype(np.int32)
    X = data[:, 1:]
    return X, y


def load_all_dl8(data_dir='data', max_binary_features=None):
    """Load all .dl8 files."""
    dl8_files = sorted(glob.glob(os.path.join(data_dir, '*.dl8')))

    if not dl8_files:
        print(f"  No .dl8 files in {os.path.abspath(data_dir)}/")
        return OrderedDict()

    datasets = OrderedDict()

    for filepath in dl8_files:
        name = os.path.basename(filepath).replace('.dl8', '')
        try:
            X, y = parse_dl8(filepath)
            n_unary = X.shape[1]

            # Make labels binary
            unique_labels = np.unique(y)
            if len(unique_labels) > 2:
                majority = np.argmax(np.bincount(y.astype(int)))
                y = (y == majority).astype(np.int32)
            elif set(unique_labels) != {0, 1}:
                y = (y == unique_labels[1]).astype(np.int32)

            # Remove zero-variance columns
            var = np.var(X, axis=0)
            valid = var > 1e-10
            if valid.sum() < X.shape[1]:
                X = X[:, valid]

            if X.shape[1] == 0:
                print(f"  ⚠ {name:<35} No valid features")
                continue

            is_binary = np.all((np.unique(X) == 0) | (np.unique(X) == 1))

            datasets[name] = {
                'X': X, 'y': y,
                'n_unary': n_unary,
                'is_binary': is_binary,
            }

            print(f"  ✓ {name:<35} n={len(y):>6}  "
                  f"d={X.shape[1]:>4}  pos={y.mean():.2%}")

        except Exception as e:
            print(f"  ✗ {name:<35} {str(e)[:60]}")

    return datasets


# =============================================================================
# EXPLANATION LENGTH
# =============================================================================

class ExplAnalyzer:
    """Compute explanation length."""

    @staticmethod
    def gsnh_expl(root, x):
        """Explanation length for one instance."""
        feats = set()
        node = root
        x2d = x.reshape(1, -1)

        while node is not None:
            if node.get('is_leaf', True) or node.get('predicate') is None:
                break

            pred = node['predicate']
            lits = pred.literals
            result = pred.evaluate(x2d)[0]

            if result:
                # TRUE branch: 1 sufficient literal
                for lit in lits:
                    if lit.evaluate(x2d)[0]:
                        feats.add(lit.feature)
                        break
                node = node.get('left')
            else:
                # FALSE branch: all features needed
                for lit in lits:
                    feats.add(lit.feature)
                node = node.get('right')

        return len(feats)

    @staticmethod
    def sklearn_expl(tree, x):
        """Explanation length for sklearn DT = path length."""
        t = tree.tree_
        nid = 0
        depth = 0
        while t.children_left[nid] != t.children_right[nid]:
            depth += 1
            if x[t.feature[nid]] <= t.threshold[nid]:
                nid = t.children_left[nid]
            else:
                nid = t.children_right[nid]
        return depth

    @staticmethod
    def avg_gsnh(tree, X):
        if hasattr(tree, 'root_') and tree.root_ is None:
            return 0.0
        # Sample 50 for efficiency
        n_samples = min(50, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)
        return np.mean([len(tree.extract_axp(X[i])) for i in idx])

    @staticmethod
    def avg_sklearn(tree, X):
        return np.mean([ExplAnalyzer.sklearn_expl(tree, X[i])
                        for i in range(len(X))])


def count_gsnh_nodes(node):
    if node is None:
        return 0
    if node.get('is_leaf', True) or node.get('predicate') is None:
        return 1
    return 1 + count_gsnh_nodes(node.get('left')) + count_gsnh_nodes(node.get('right'))


# =============================================================================
# BENCHMARK
# =============================================================================

class LanguageComparisonBenchmark:
    """
    Compare different language families at depth 5 AND 7 vs sklearn DT at depth 7.
    """

    # Define language configurations with their search parameters
    # Format: (label, enum_name_or_value, search_1d, search_2d, search_3d)
    LANGUAGE_CONFIGS = [
        ('1D', 'ONE_D', True, False, False),      # 1D only
        ('Horn', 'HORN', True, True, False),       # 1D + 2D Horn
        ('AntiHorn', 'ANTI_HORN', True, True, False),  # 1D + 2D Anti-Horn
        ('ConjUI', 'CONJ_UI', True, True, False),        # 1D + 2D ConjUI (box/AND)
        ('Square2CNF', 'SQUARE_2CNF', True, True, False),  # 1D + 2D paper-style
        ('Affine', 'AFFINE', True, True, False),   # 1D + 2D Affine
        ('BEST_PER_NODE', 'BEST_PER_NODE', True, True, True),  # Best of all
    ]

    DEPTHS = [5, 7]  # Test both depths

    def __init__(self, gsnh_module, n_runs=10, random_state=42):
        self.gsnh = gsnh_module
        self.n_runs = n_runs
        self.rs = random_state
        self.results_ = OrderedDict()
        self.LF = self.gsnh.LanguageFamily
        self._build_language_mapping()

    def _build_language_mapping(self):
        """Build mapping from config names to actual enum values."""
        self.lang_map = {}
        available_names = {e.name: e for e in self.LF}

        for label, enum_name, s1, s2, s3 in self.LANGUAGE_CONFIGS:
            if enum_name in available_names:
                self.lang_map[label] = available_names[enum_name]
            else:
                print(f"⚠ Warning: {enum_name} not found in LanguageFamily enum")
                # Try common variations
                variations = [
                    enum_name,
                    enum_name.replace('_', ''),
                    enum_name.replace('_', '_2D'),
                    'UNIVARIATE' if enum_name == 'ONE_D' else None,
                    'HORN_2D' if enum_name == 'HORN' else None,
                    'ANTI_HORN_2D' if enum_name == 'ANTI_HORN' else None,
                ]
                found = False
                for var in variations:
                    if var and var in available_names:
                        self.lang_map[label] = available_names[var]
                        print(f"  → Using {var} instead")
                        found = True
                        break
                if not found:
                    print(f"  ✗ Could not find matching enum for {label}")

    def _get_language_family(self, label):
        """Get LanguageFamily enum by label."""
        return self.lang_map.get(label)

    def run_all(self, datasets, skip_large=2000):
        """Run on all datasets."""
        total = len(datasets)

        for idx, (name, info) in enumerate(datasets.items()):
            X, y = info['X'], info['y']

            if skip_large and X.shape[1] > skip_large:
                print(f"\n[{idx+1}/{total}] {name} — "
                      f"SKIPPED (d={X.shape[1]} > {skip_large})")
                continue

            print(f"\n[{idx+1}/{total}] ", end="")

            try:
                t0 = time.time()
                self._evaluate_one(name, X, y)
                elapsed = time.time() - t0
                print(f"    ⏱ {elapsed:.1f}s")
            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                traceback.print_exc()

    def _evaluate_one(self, name, X, y):
        """Evaluate one dataset with all language families at both depths."""
        n, d = X.shape
        print(f"{name}: n={n}, d={d}, pos={y.mean():.2%}")

        ET = self.gsnh.ExpertGSNHTree
        SC = self.gsnh.StoppingCriteria

        # Adaptive settings
        if d > 500:
            n_bins = 16
            use_3d = False
            top_k = 10
        elif d > 100:
            n_bins = 32
            use_3d = False
            top_k = 12
        else:
            n_bins = 64
            use_3d = d <= 50
            top_k = 15

        # For binary features, fewer bins
        if len(np.unique(X)) <= 3:
            n_bins = min(n_bins, 8)

        # Initialize metrics for all methods
        # sklearn DT at depth 7 only
        metrics = {
            'sklearn_dt7': {'acc': [], 'size': [], 'expl': [], 'time': []},
        }
        # GSNH: each language family at both depths
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            for depth in self.DEPTHS:
                metrics[f'gsnh_{lang_label}_d{depth}'] = {'acc': [], 'size': [], 'expl': [], 'time': []}

        splitter = StratifiedShuffleSplit(
            n_splits=self.n_runs, test_size=0.2, random_state=self.rs
        )

        for run_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # ── sklearn DT (depth 7) ──
            t0 = time.time()
            dt = DecisionTreeClassifier(
                max_depth=7, min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.rs + run_idx
            )
            dt.fit(X_tr, y_tr)
            dt_time = time.time() - t0

            metrics['sklearn_dt7']['acc'].append(float((dt.predict(X_te) == y_te).mean()))
            metrics['sklearn_dt7']['size'].append(dt.tree_.node_count)
            metrics['sklearn_dt7']['expl'].append(ExplAnalyzer.avg_sklearn(dt, X_te))
            metrics['sklearn_dt7']['time'].append(dt_time)

            # ── GSNH with each language family at both depths ──
            for lang_label, _, search_1d, search_2d, search_3d in self.LANGUAGE_CONFIGS:
                lang_family = self._get_language_family(lang_label)
                if lang_family is None:
                    continue  # Skip if enum not found

                for depth in self.DEPTHS:
                    t0 = time.time()
                    gsnh_tree = ET(
                        stopping_criteria=SC(
                            max_depth=depth, min_samples_leaf=5,
                            min_samples_split=10),
                        n_bins=n_bins, top_k_features=top_k,
                        search_1d=search_1d, 
                        search_2d=search_2d, 
                        search_3d=search_3d and use_3d,
                        mode='journal',
                        language=lang_family,
                    )
                    gsnh_tree.fit(X_tr, y_tr)
                    gsnh_time = time.time() - t0

                    key = f'gsnh_{lang_label}_d{depth}'
                    metrics[key]['acc'].append(float((gsnh_tree.predict(X_te) == y_te).mean()))
                    metrics[key]['size'].append(count_gsnh_nodes(gsnh_tree.root_))
                    metrics[key]['expl'].append(ExplAnalyzer.avg_gsnh(gsnh_tree, X_te))
                    metrics[key]['time'].append(gsnh_time)

            if (run_idx + 1) % 5 == 0:
                # Show summary for depth 5 and 7 of BEST_PER_NODE
                d5_acc = metrics.get('gsnh_BEST_PER_NODE_d5', {}).get('acc', [0])[-1] if metrics.get('gsnh_BEST_PER_NODE_d5', {}).get('acc') else 0
                d7_acc = metrics.get('gsnh_BEST_PER_NODE_d7', {}).get('acc', [0])[-1] if metrics.get('gsnh_BEST_PER_NODE_d7', {}).get('acc') else 0
                dt_acc = metrics['sklearn_dt7']['acc'][-1]
                print(f"    Run {run_idx+1}/{self.n_runs}: "
                      f"DT7={dt_acc:.4f} Best_d5={d5_acc:.4f} Best_d7={d7_acc:.4f}")

        # Average results
        result = {}
        for key in metrics:
            if metrics[key]['acc']:  # Only include if we have data
                result[key] = {
                    'acc': np.mean(metrics[key]['acc']),
                    'acc_std': np.std(metrics[key]['acc']),
                    'size': np.mean(metrics[key]['size']),
                    'expl': np.mean(metrics[key]['expl']),
                    'time': np.mean(metrics[key]['time']),
                    'accs': metrics[key]['acc'],
                }

        self.results_[name] = result

        # Print per-dataset summary
        print(f"\n    {'Method':<22} {'Depth':<6} {'Accuracy':<18} {'Size':<10} {'Expl':<8} {'Time(s)':<8}")
        print(f"    {'─'*80}")

        # sklearn DT
        r = result.get('sklearn_dt7', {})
        if r:
            print(f"    {'sklearn DT':<22} {'7':<6} {r['acc']:.4f}±{r['acc_std']:.4f}  {r['size']:>7.1f}  {r['expl']:>6.2f}  {r['time']:>6.3f}")

        # GSNH variants at both depths
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            for depth in self.DEPTHS:
                key = f'gsnh_{lang_label}_d{depth}'
                r = result.get(key)
                if r:
                    label = f'GSNH-{lang_label}'
                    print(f"    {label:<22} {depth:<6} {r['acc']:.4f}±{r['acc_std']:.4f}  {r['size']:>7.1f}  {r['expl']:>6.2f}  {r['time']:>6.3f}")

    def print_table(self):
        """Print complete comparison table."""
        if not self.results_:
            print("No results.")
            return

        print(f"\n{'='*180}")
        print("LANGUAGE FAMILY COMPARISON — Depth 5 & 7 vs sklearn DT (Depth 7)")
        print(f"{'='*180}")

        # Header - split into two tables for readability
        print("\n--- DEPTH 5 COMPARISON ---")
        headers_d5 = ['Dataset', 'DT7', '1D', 'Horn', 'AntiHorn', 'SqCNF', 'Affine', 'BestPN']
        header_line = f"{headers_d5[0]:<25}" + "".join([f"{h:>10}" for h in headers_d5[1:]])
        print(header_line)
        print("─" * 100)

        # Accumulators
        all_methods = ['sklearn_dt7'] + [f'gsnh_{l[0]}_d5' for l in self.LANGUAGE_CONFIGS] + [f'gsnh_{l[0]}_d7' for l in self.LANGUAGE_CONFIGS]
        accs = {m: [] for m in all_methods}
        sizes = {m: [] for m in all_methods}
        expls = {m: [] for m in all_methods}
        times = {m: [] for m in all_methods}

        # Win counters at depth 5 vs sklearn DT7
        wins_d5 = {f'gsnh_{l[0]}_d5': 0 for l in self.LANGUAGE_CONFIGS}
        losses_d5 = {f'gsnh_{l[0]}_d5': 0 for l in self.LANGUAGE_CONFIGS}
        ties_d5 = {f'gsnh_{l[0]}_d5': 0 for l in self.LANGUAGE_CONFIGS}

        # Win counters at depth 7 vs sklearn DT7
        wins_d7 = {f'gsnh_{l[0]}_d7': 0 for l in self.LANGUAGE_CONFIGS}
        losses_d7 = {f'gsnh_{l[0]}_d7': 0 for l in self.LANGUAGE_CONFIGS}
        ties_d7 = {f'gsnh_{l[0]}_d7': 0 for l in self.LANGUAGE_CONFIGS}

        for name, r in self.results_.items():
            dt_acc = r.get('sklearn_dt7', {}).get('acc', 0)

            # Depth 5 row
            row_d5 = f"{name:<25}"
            for method in ['sklearn_dt7'] + [f'gsnh_{l[0]}_d5' for l in self.LANGUAGE_CONFIGS]:
                if method in r:
                    acc = r[method]['acc']
                    marker = ""
                    if 'd5' in method:
                        if acc > dt_acc + 0.005:
                            marker = "↑"
                            wins_d5[method] += 1
                        elif acc < dt_acc - 0.005:
                            marker = "↓"
                            losses_d5[method] += 1
                        else:
                            ties_d5[method] += 1
                    row_d5 += f"{acc:>9.4f}{marker}"
                    accs[method].append(r[method]['acc'])
                    sizes[method].append(r[method]['size'])
                    expls[method].append(r[method]['expl'])
                    times[method].append(r[method]['time'])
                else:
                    row_d5 += f"{'N/A':>10}"

            print(row_d5)

        # Average row for depth 5
        print("─" * 100)
        avg_row = f"{'AVERAGE D5':<25}"
        for method in ['sklearn_dt7'] + [f'gsnh_{l[0]}_d5' for l in self.LANGUAGE_CONFIGS]:
            if accs[method]:
                avg_acc = np.mean(accs[method])
                avg_row += f"{avg_acc:>10.4f}"
            else:
                avg_row += f"{'N/A':>10}"
        print(avg_row)

        # Depth 7 table
        print("\n--- DEPTH 7 COMPARISON ---")
        headers_d7 = ['Dataset', 'DT7', '1D', 'Horn', 'AntiHorn', 'SqCNF', 'Affine', 'BestPN']
        header_line = f"{headers_d7[0]:<25}" + "".join([f"{h:>10}" for h in headers_d7[1:]])
        print(header_line)
        print("─" * 100)

        for name, r in self.results_.items():
            dt_acc = r.get('sklearn_dt7', {}).get('acc', 0)

            # Depth 7 row
            row_d7 = f"{name:<25}"
            for method in ['sklearn_dt7'] + [f'gsnh_{l[0]}_d7' for l in self.LANGUAGE_CONFIGS]:
                if method in r:
                    acc = r[method]['acc']
                    marker = ""
                    if 'd7' in method:
                        if acc > dt_acc + 0.005:
                            marker = "↑"
                            wins_d7[method] += 1
                        elif acc < dt_acc - 0.005:
                            marker = "↓"
                            losses_d7[method] += 1
                        else:
                            ties_d7[method] += 1
                    row_d7 += f"{acc:>9.4f}{marker}"
                    accs[method].append(r[method]['acc'])
                    sizes[method].append(r[method]['size'])
                    expls[method].append(r[method]['expl'])
                    times[method].append(r[method]['time'])
                else:
                    row_d7 += f"{'N/A':>10}"

            print(row_d7)

        # Average row for depth 7
        print("─" * 100)
        avg_row = f"{'AVERAGE D7':<25}"
        for method in ['sklearn_dt7'] + [f'gsnh_{l[0]}_d7' for l in self.LANGUAGE_CONFIGS]:
            if accs[method]:
                avg_acc = np.mean(accs[method])
                avg_row += f"{avg_acc:>10.4f}"
            else:
                avg_row += f"{'N/A':>10}"
        print(avg_row)

        # Summary statistics
        print(f"\n{'='*120}")
        print("SUMMARY STATISTICS")
        print(f"{'='*120}")

        n_ds = len(self.results_)

        print(f"\n  ACCURACY (mean ± std across {n_ds} datasets):")
        print(f"    {'Method':<25} {'Depth 5':<20} {'Depth 7':<20}")
        print(f"    {'─'*70}")

        # sklearn DT
        if accs.get('sklearn_dt7'):
            print(f"    {'sklearn DT':<25} {'—':<20} {np.mean(accs['sklearn_dt7']):.4f} ± {np.std(accs['sklearn_dt7']):.4f}")

        # GSNH variants
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            key_d5 = f'gsnh_{lang_label}_d5'
            key_d7 = f'gsnh_{lang_label}_d7'
            acc_d5 = np.mean(accs[key_d5]) if accs.get(key_d5) else None
            std_d5 = np.std(accs[key_d5]) if accs.get(key_d5) else None
            acc_d7 = np.mean(accs[key_d7]) if accs.get(key_d7) else None
            std_d7 = np.std(accs[key_d7]) if accs.get(key_d7) else None

            d5_str = f"{acc_d5:.4f} ± {std_d5:.4f}" if acc_d5 is not None else "N/A"
            d7_str = f"{acc_d7:.4f} ± {std_d7:.4f}" if acc_d7 is not None else "N/A"
            print(f"    {f'GSNH-{lang_label}':<25} {d5_str:<20} {d7_str}")

        print(f"\n  TREE SIZE (average):")
        print(f"    {'Method':<25} {'Depth 5':<15} {'Depth 7':<15} {'vs DT7 D5':<12} {'vs DT7 D7':<12}")
        print(f"    {'─'*80}")

        dt_sz = np.mean(sizes['sklearn_dt7']) if sizes.get('sklearn_dt7') else 0
        print(f"    {'sklearn DT(d=7)':<25} {'—':<15} {dt_sz:>7.1f}")

        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            key_d5 = f'gsnh_{lang_label}_d5'
            key_d7 = f'gsnh_{lang_label}_d7'
            sz_d5 = np.mean(sizes[key_d5]) if sizes.get(key_d5) else None
            sz_d7 = np.mean(sizes[key_d7]) if sizes.get(key_d7) else None

            if sz_d5 is not None and sz_d7 is not None:
                comp_d5 = dt_sz / max(sz_d5, 1)
                comp_d7 = dt_sz / max(sz_d7, 1)
                print(f"    {f'GSNH-{lang_label}':<25} {sz_d5:>7.1f}        {sz_d7:>7.1f}        {comp_d5:>5.1f}×       {comp_d7:>5.1f}×")
            else:
                d5_str = f"{sz_d5:>7.1f}" if sz_d5 is not None else "N/A"
                d7_str = f"{sz_d7:>7.1f}" if sz_d7 is not None else "N/A"
                print(f"    {f'GSNH-{lang_label}':<25} {d5_str:<15} {d7_str:<15}")

        print(f"\n  EXPLANATION LENGTH (average):")
        print(f"    {'Method':<25} {'Depth 5':<15} {'Depth 7':<15}")
        print(f"    {'─'*60}")
        if expls.get('sklearn_dt7'):
            print(f"    {'sklearn DT(d=7)':<25} {'—':<15} {np.mean(expls['sklearn_dt7']):.2f}")
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            key_d5 = f'gsnh_{lang_label}_d5'
            key_d7 = f'gsnh_{lang_label}_d7'
            expl_d5 = np.mean(expls[key_d5]) if expls.get(key_d5) else None
            expl_d7 = np.mean(expls[key_d7]) if expls.get(key_d7) else None
            d5_str = f"{expl_d5:.2f}" if expl_d5 is not None else "N/A"
            d7_str = f"{expl_d7:.2f}" if expl_d7 is not None else "N/A"
            print(f"    {f'GSNH-{lang_label}':<25} {d5_str:<15} {d7_str}")

        print(f"\n  TRAINING TIME in seconds (average):")
        print(f"    {'Method':<25} {'Depth 5':<15} {'Depth 7':<15}")
        print(f"    {'─'*60}")
        if times.get('sklearn_dt7'):
            print(f"    {'sklearn DT(d=7)':<25} {'—':<15} {np.mean(times['sklearn_dt7']):.3f}s")
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            key_d5 = f'gsnh_{lang_label}_d5'
            key_d7 = f'gsnh_{lang_label}_d7'
            time_d5 = np.mean(times[key_d5]) if times.get(key_d5) else None
            time_d7 = np.mean(times[key_d7]) if times.get(key_d7) else None
            d5_str = f"{time_d5:.3f}s" if time_d5 is not None else "N/A"
            d7_str = f"{time_d7:.3f}s" if time_d7 is not None else "N/A"
            print(f"    {f'GSNH-{lang_label}':<25} {d5_str:<15} {d7_str}")

        print(f"\n  WIN/LOSS/TIE vs sklearn DT(d=7) (threshold ±0.005):")
        print(f"    {'Method':<25} {'Depth 5 W/L/T':<20} {'Depth 7 W/L/T':<20}")
        print(f"    {'─'*70}")
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            key_d5 = f'gsnh_{lang_label}_d5'
            key_d7 = f'gsnh_{lang_label}_d7'
            print(f"    {f'GSNH-{lang_label}':<25} {wins_d5[key_d5]:>2}/{losses_d5[key_d5]:>2}/{ties_d5[key_d5]:>2}             {wins_d7[key_d7]:>2}/{losses_d7[key_d7]:>2}/{ties_d7[key_d7]:>2}")

        # Statistical tests
        print(f"\n  STATISTICAL TESTS vs sklearn DT(d=7):")
        if accs.get('sklearn_dt7'):
            dt_accs = np.array(accs['sklearn_dt7'])

            for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
                print(f"\n    GSNH-{lang_label}:")

                for depth in [5, 7]:
                    key = f'gsnh_{lang_label}_d{depth}'
                    if accs.get(key):
                        gsnh_accs = np.array(accs[key])
                        d = gsnh_accs - dt_accs
                        if np.std(d) > 1e-10 and len(d) >= 3:
                            _, pt = stats.ttest_rel(gsnh_accs, dt_accs)
                            try:
                                _, pw = stats.wilcoxon(d)
                            except:
                                pw = pt
                            print(f"      Depth {depth}: Δ={d.mean():+.4f}  t-test p={pt:.4f}  Wilcoxon p={pw:.4f}")

    def generate_latex(self):
        """Generate LaTeX tables for both depths."""
        if not self.results_:
            return ""

        print(f"\n{'='*80}")
        print("LATEX TABLES")
        print(f"{'='*80}\n")

        latex_output = []

        # Table for Depth 5
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering\small")
        lines.append(r"\caption{Language Family Comparison at Depth 5 vs sklearn DT (Depth 7) on .dl8 datasets.}")
        lines.append(r"\begin{tabular}{l|ccccccc}")
        lines.append(r"\toprule")
        lines.append(r" & DT$_7$ & 1D$_5$ & Horn$_5$ & Anti-Horn$_5$ & Sq-CNF$_5$ & Affine$_5$ & Best-PN$_5$ \\ \midrule")

        for name, r in self.results_.items():
            cname = name.replace('_', r'\_')

            all_accs = [r.get('sklearn_dt7', {}).get('acc', 0)] + [r.get(f'gsnh_{l[0]}_d5', {}).get('acc', 0) for l in self.LANGUAGE_CONFIGS]
            best_acc = max([a for a in all_accs if a > 0])

            def fmt(v):
                s = f"{v:.4f}"
                return r"\textbf{" + s + "}" if abs(v - best_acc) < 0.001 and v > 0 else s

            row = f"{cname}"
            row += f" & {fmt(r.get('sklearn_dt7', {}).get('acc', 0))}"
            for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
                acc = r.get(f'gsnh_{lang_label}_d5', {}).get('acc', 0)
                row += f" & {fmt(acc)}"
            row += r" \\"
            lines.append(row)

        lines.append(r"\midrule")

        n = len(self.results_)
        avg_row = f"Average ({n})"
        avg_row += f" & {np.mean([r.get('sklearn_dt7', {}).get('acc', 0) for r in self.results_.values()]):.4f}"
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            avg_acc = np.mean([r.get(f'gsnh_{lang_label}_d5', {}).get('acc', 0) for r in self.results_.values()])
            avg_row += f" & {avg_acc:.4f}"
        avg_row += r" \\"
        lines.append(avg_row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_d5 = "\n".join(lines)
        latex_output.append("% DEPTH 5 TABLE")
        latex_output.append(latex_d5)

        # Table for Depth 7
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering\small")
        lines.append(r"\caption{Language Family Comparison at Depth 7 vs sklearn DT (Depth 7) on .dl8 datasets.}")
        lines.append(r"\begin{tabular}{l|ccccccc}")
        lines.append(r"\toprule")
        lines.append(r" & DT$_7$ & 1D$_7$ & Horn$_7$ & Anti-Horn$_7$ & Sq-CNF$_7$ & Affine$_7$ & Best-PN$_7$ \\ \midrule")

        for name, r in self.results_.items():
            cname = name.replace('_', r'\_')

            all_accs = [r.get('sklearn_dt7', {}).get('acc', 0)] + [r.get(f'gsnh_{l[0]}_d7', {}).get('acc', 0) for l in self.LANGUAGE_CONFIGS]
            best_acc = max([a for a in all_accs if a > 0])

            def fmt(v):
                s = f"{v:.4f}"
                return r"\textbf{" + s + "}" if abs(v - best_acc) < 0.001 and v > 0 else s

            row = f"{cname}"
            row += f" & {fmt(r.get('sklearn_dt7', {}).get('acc', 0))}"
            for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
                acc = r.get(f'gsnh_{lang_label}_d7', {}).get('acc', 0)
                row += f" & {fmt(acc)}"
            row += r" \\"
            lines.append(row)

        lines.append(r"\midrule")

        avg_row = f"Average ({n})"
        avg_row += f" & {np.mean([r.get('sklearn_dt7', {}).get('acc', 0) for r in self.results_.values()]):.4f}"
        for lang_label, _, _, _, _ in self.LANGUAGE_CONFIGS:
            avg_acc = np.mean([r.get(f'gsnh_{lang_label}_d7', {}).get('acc', 0) for r in self.results_.values()])
            avg_row += f" & {avg_acc:.4f}"
        avg_row += r" \\"
        lines.append(avg_row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_d7 = "\n".join(lines)
        latex_output.append("\n% DEPTH 7 TABLE")
        latex_output.append(latex_d7)

        full_latex = "\n\n".join(latex_output)
        print(full_latex)
        return full_latex


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("#   Language Family Comparison — Depth 5 & 7 — Professor's .dl8 Data   #")
    print("#" * 80)

    print("\nComparing at depth 5 AND 7:")
    print("  - 1D (univariate)")
    print("  - Horn")
    print("  - Anti-Horn")
    print("  - ConjUI (AND/box)")
    print("  - Square 2CNF (paper-style)")
    print("  - Affine")
    print("  - BEST_PER_NODE")
    print("\nAgainst sklearn Decision Tree at depth 7")

    # Import
    print("\n[1] Importing GSNH...")
    gsnh = import_gsnh()

    # Find data
    data_dir = None
    for d in [os.environ.get('GSNH_MDT_DATA_DIR'), os.environ.get('DATA_DIR'), os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')), 'data', './data', '../data']:
        if d and os.path.isdir(d) and glob.glob(os.path.join(d, '*.dl8')):
            data_dir = d
            break

    if data_dir is None:
        print("✗ No data/ directory with .dl8 files found!")
        return None

    # Load
    print(f"\n[2] Loading .dl8 files from {data_dir}/...")
    datasets = load_all_dl8(data_dir)

    if not datasets:
        print("No datasets loaded!")
        return None

    # Run
    print(f"\n[3] Benchmark: {len(datasets)} datasets × 10 runs")
    print(f"    Comparing 6 language families at depth 5 & 7 vs sklearn DT (depth 7)")

    bench = LanguageComparisonBenchmark(gsnh, n_runs=10, random_state=42)
    bench.run_all(datasets, skip_large=2000)

    # Results
    print("\n\n[4] Results...")
    bench.print_table()

    # LaTeX
    print("\n\n[5] LaTeX...")
    latex = bench.generate_latex()

    try:
        with open('language_comparison_results.tex', 'w') as f:
            f.write(latex)
        print(f"\n✓ Saved to language_comparison_results.tex")
    except Exception as e:
        print(f"\n✗ Save error: {e}")

    print("\n" + "#" * 80)
    print("#" + " " * 25 + "COMPLETE!" + " " * 24 + "#")
    print("#" * 80 + "\n")

    return bench


if __name__ == "__main__":
    bench = main()
