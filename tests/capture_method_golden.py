"""
capture_method_golden.py — Capture method-level golden outputs for Phase C split.

Freezes: predict(), predict_proba(), extract_axp(), weak_axp_check(),
_traverse(), get_summary(), on fixed datasets and instances.
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
    return data[:, 1:].astype(np.float64), data[:, 0].astype(np.int32)


def train_tree(name, seed=42, max_depth=7):
    X, y = parse_dl8(os.path.join(RS_ROOT, 'data', f'{name}.dl8'))
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

    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=max_depth, min_samples_leaf=5),
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    tree.fit(X_train, y_train)
    return tree, X_train, y_train, X_test, y_test


def capture_method_goldens(name, seed=42, max_depth=7):
    tree, X_train, y_train, X_test, y_test = train_tree(name, seed, max_depth)
    n_test = len(X_test)

    # === predict_proba on 10 fixed samples ===
    n_pp = min(10, n_test)
    pp = tree.predict_proba(X_test[:n_pp]).tolist()

    # === predict on batch of first 10 samples ===
    pred_batch = tree.predict(X_test[:n_pp]).tolist()

    # === extract_axp on 5 samples ===
    axp_results = []
    for i in range(min(5, n_test)):
        axp = tree.extract_axp(X_test[i])
        axp_results.append(sorted(list(axp)))

    # === weak_axp_check on 3 samples with their full AXp set ===
    # Should return True when S = the full AXp
    weak_check_results = []
    for i in range(min(3, n_test)):
        axp = tree.extract_axp(X_test[i])
        y_pred = tree.predict(X_test[i].reshape(1, -1))[0]
        # Check with full explanation — must be True
        check_full = tree.weak_axp_check(X_test[i], y_pred, axp)
        # Check with empty set — should be False for non-trivial trees
        check_empty = tree.weak_axp_check(X_test[i], y_pred, set())
        weak_check_results.append({
            'instance_idx': i,
            'y_pred': int(y_pred),
            'axp_features': sorted(list(axp)),
            'check_full_axp': check_full,
            'check_empty': check_empty,
        })

    # === score on full train and test ===
    train_score = float(tree.score(X_train, y_train))
    test_score = float(tree.score(X_test, y_test))

    # === get_summary ===
    summary_str = tree.get_summary()

    return {
        'name': name,
        'seed': seed,
        'max_depth': max_depth,
        'predict_proba_10': pp,
        'predict_10': pred_batch,
        'axp_5': axp_results,
        'weak_check_3': weak_check_results,
        'train_score': round(train_score, 10),
        'test_score': round(test_score, 10),
        'summary_str': summary_str,
        'n_nodes': tree.n_nodes_,
        'n_leaves': tree.n_leaves_,
        'max_depth_reached': tree.max_depth_reached_,
    }


def main():
    datasets = ['vote', 'hepatitis', 'lymph', 'ionosphere', 'kr-vs-kp']
    golden = {}
    for name in datasets:
        path = os.path.join(RS_ROOT, 'data', f'{name}.dl8')
        if not os.path.exists(path):
            print(f"  SKIP {name}")
            continue
        print(f"Capturing method-level golden for {name}...")
        golden[name] = capture_method_goldens(name)
        g = golden[name]
        print(f"  predict_proba[0]={g['predict_proba_10'][0]}")
        print(f"  predict[0:5]={g['predict_10'][:5]}")
        print(f"  axp[0]={g['axp_5'][0]}")
        print(f"  weak_check[0]={g['weak_check_3'][0]}")
        print(f"  scores={g['train_score']}/{g['test_score']}")

    out_path = os.path.join(os.path.dirname(__file__), 'golden_methods.json')
    with open(out_path, 'w') as f:
        json.dump(golden, f, indent=2)
    print(f"\nMethod-level golden baselines: {out_path} ({len(golden)} datasets)")


if __name__ == '__main__':
    main()
