"""
capture_golden.py — Capture golden outputs from the monolith.

Run this ONCE against the original gsnh_mdt_v3.py to freeze baseline outputs.
The outputs are saved to golden_baselines.json and used by test_tree_regression.py.
"""

import sys
import os
import json
import numpy as np

from data_path import DATA_DIR, RS_ROOT
from golden_utils import find_dataset_file

# Add repo root so we can import the package/monolith
sys.path.insert(0, RS_ROOT)
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily
from gsnh_mdt.api import GSNHClassifier


def parse_dl8(filepath):
    """Parse .dl8 file. Format: label f1 f2 ... fn (space-separated ints)."""
    data = np.loadtxt(filepath, dtype=np.int64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


def capture_tree_golden(name, X, y, seed=42, max_depth=5):
    """Train ExpertGSNHTree on the data and capture reproducible golden outputs."""
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Binarize labels for binary classification
    classes = np.unique(y)
    if len(classes) > 2:
        majority = classes[np.argmax([np.sum(y == c) for c in classes])]
        y_train = (y_train == majority).astype(np.int32)
        y_test = (y_test == majority).astype(np.int32)

    stopping = StoppingCriteria(max_depth=max_depth, min_samples_leaf=5)
    tree = ExpertGSNHTree(
        stopping_criteria=stopping,
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    tree.fit(X_train, y_train)

    train_acc = float(tree.score(X_train, y_train))
    test_acc = float(tree.score(X_test, y_test))
    n_nodes = tree.n_nodes_
    n_leaves = tree.n_leaves_
    max_depth_reached = tree.max_depth_reached_

    # Capture predict_proba for first 3 test samples
    probas = tree.predict_proba(X_test[:3])
    probas_list = probas.tolist()

    # Root split signature
    root = tree.root_
    root_split = None
    if root and 'predicate' in root:
        root_split = str(root['predicate'])

    # AXp length for first test sample (if available)
    axp_len = None
    try:
        axp = tree.extract_axp(X_test[0])
        axp_len = len(axp)
    except Exception:
        pass

    return {
        'name': name,
        'seed': seed,
        'max_depth': max_depth,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_acc': round(train_acc, 6),
        'test_acc': round(test_acc, 6),
        'n_nodes': n_nodes,
        'n_leaves': n_leaves,
        'max_depth_reached': max_depth_reached,
        'probas_first3': probas_list,
        'root_split': root_split,
        'axp_len': axp_len,
    }


def capture_reproducibility(X, y, seed=42):
    """Verify that 2 fits with the same seed produce identical results."""
    results = []
    for _ in range(2):
        np.random.seed(seed)
        classes = np.unique(y)
        if len(classes) > 2:
            majority = classes[np.argmax([np.sum(y == c) for c in classes])]
            y_bin = (y == majority).astype(np.int32)
        else:
            y_bin = y.copy()

        n = len(y_bin)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        X_train, y_train = X[idx[:split]], y_bin[idx[:split]]
        X_test, y_test = X[idx[split:]], y_bin[idx[split:]]

        stopping = StoppingCriteria(max_depth=5, min_samples_leaf=5)
        tree = ExpertGSNHTree(
            stopping_criteria=stopping,
            n_bins=64,
            language=LanguageFamily.BEST_PER_NODE,
            verbose=False,
        )
        tree.fit(X_train, y_train)
        results.append({
            'acc': float(tree.score(X_test, y_test)),
            'n_nodes': tree.n_nodes_,
        })

    return results[0] == results[1]


def main():
    datasets = {
        'lymph': find_dataset_file('lymph'),
        'hepatitis': find_dataset_file('hepatitis'),
        'vote': find_dataset_file('vote'),
    }

    golden = {
        "_comment": "Post-theorem-boundary deterministic golden baselines."
    }
    for name, full_path in datasets.items():
        if not os.path.exists(full_path):
            print(f"  SKIP {name}: file not found at {full_path}")
            continue

        print(f"Capturing golden for {name}...")
        X, y = parse_dl8(full_path)
        result = capture_tree_golden(name, X, y)
        golden[name] = result
        print(f"  train_acc={result['train_acc']}, test_acc={result['test_acc']}, "
              f"nodes={result['n_nodes']}, leaves={result['n_leaves']}, "
              f"root='{result['root_split']}'")

        # Reproducibility check
        repro = capture_reproducibility(X, y)
        golden[name]['reproducible'] = repro
        print(f"  reproducible={repro}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'golden_baselines.json')
    with open(out_path, 'w') as f:
        json.dump(golden, f, indent=2)
    print(f"\nGolden baselines saved to {out_path}")


if __name__ == '__main__':
    main()
