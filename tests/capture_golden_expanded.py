"""
capture_golden_expanded.py — Capture expanded golden baselines.

Adds 6 more datasets and additional output fields:
- 5 predict_proba samples (not just 3)
- get_summary() fields
- AXp lengths for 3 test samples
- Deeper split info
"""

import sys
import os
import json
import numpy as np

RS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, RS_ROOT)
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily


def parse_dl8(filepath):
    data = np.loadtxt(filepath, dtype=np.int64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


def capture_expanded(name, X, y, seed=42, max_depth=7):
    """Train and capture comprehensive golden outputs."""
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

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

    train_acc = round(float(tree.score(X_train, y_train)), 6)
    test_acc = round(float(tree.score(X_test, y_test)), 6)

    # 5 predict_proba samples
    n_proba = min(5, len(X_test))
    probas = tree.predict_proba(X_test[:n_proba]).tolist()

    # Root split
    root_split = None
    if tree.root_ and 'predicate' in tree.root_:
        root_split = str(tree.root_['predicate'])

    # Arity distribution
    arity_counts = dict(tree.arity_counts_) if hasattr(tree, 'arity_counts_') else {}

    # AXp lengths for 3 test samples
    axp_lengths = []
    for i in range(min(3, len(X_test))):
        try:
            axp = tree.extract_axp(X_test[i])
            axp_lengths.append(len(axp))
        except Exception:
            axp_lengths.append(None)

    # Reproducibility
    np.random.seed(seed)
    idx2 = np.random.permutation(n)
    X_train2, y_train2 = X[idx2[:split]], y[idx2[:split]]
    X_test2, y_test2 = X[idx2[split:]], y[idx2[split:]]
    if len(classes) > 2:
        y_train2 = (y_train2 == majority).astype(np.int32)
        y_test2 = (y_test2 == majority).astype(np.int32)

    tree2 = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=max_depth, min_samples_leaf=5),
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    tree2.fit(X_train2, y_train2)
    reproduced = (
        float(tree2.score(X_test2, y_test2)) == float(tree.score(X_test, y_test))
        and tree2.n_nodes_ == tree.n_nodes_
    )

    return {
        'name': name,
        'seed': seed,
        'max_depth': max_depth,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': X.shape[1],
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_nodes': tree.n_nodes_,
        'n_leaves': tree.n_leaves_,
        'max_depth_reached': tree.max_depth_reached_,
        'arity_counts': {str(k): v for k, v in arity_counts.items()},
        'probas_first5': probas,
        'root_split': root_split,
        'axp_lengths': axp_lengths,
        'reproducible': reproduced,
    }


def main():
    datasets = {
        # Original 3:
        'lymph': os.path.join(RS_ROOT, 'data', 'lymph.dl8'),
        'hepatitis': os.path.join(RS_ROOT, 'data', 'hepatitis.dl8'),
        'vote': os.path.join(RS_ROOT, 'data', 'vote.dl8'),
        # Expanded 6:
        'tic-tac-toe': os.path.join(RS_ROOT, 'data', 'tic-tac-toe.dl8'),
        'ionosphere': os.path.join(RS_ROOT, 'data', 'ionosphere.dl8'),
        'kr-vs-kp': os.path.join(RS_ROOT, 'data', 'kr-vs-kp.dl8'),
        'mushroom': os.path.join(RS_ROOT, 'data', 'mushroom.dl8'),
        'anneal': os.path.join(RS_ROOT, 'data', 'anneal.dl8'),
        'heart-cleveland': os.path.join(RS_ROOT, 'data', 'heart-cleveland.dl8'),
    }

    golden = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"  SKIP {name}: not found at {path}")
            continue

        print(f"Capturing golden for {name}...")
        X, y = parse_dl8(path)
        result = capture_expanded(name, X, y)
        golden[name] = result
        print(f"  acc={result['train_acc']}/{result['test_acc']}, "
              f"nodes={result['n_nodes']}, arity={result['arity_counts']}, "
              f"axp={result['axp_lengths']}, repro={result['reproducible']}")

    out_path = os.path.join(os.path.dirname(__file__), 'golden_expanded.json')
    with open(out_path, 'w') as f:
        json.dump(golden, f, indent=2)
    print(f"\nExpanded golden baselines saved to {out_path} ({len(golden)} datasets)")


if __name__ == '__main__':
    main()
