"""
Exact SAT solvers for tractable GSNH language families.

Provides Horn-SAT (forward chaining), Anti-Horn-SAT (reduction to Horn),
2-SAT (implication graph + SCC), and Affine-SAT (GF(2) Gaussian elimination).

Extracted verbatim from gsnh_mdt_v3.py lines 1330-1552.
These are BASELINE JOURNAL LOGIC — must not be modified.
"""


import numpy as np 


class ExactSATSolver:
    """Exact SAT solvers for each tractable language L.

    Used for weak-AXp checking in journal mode.
    No heuristics — complete and sound decision procedures.
    """

    @staticmethod
    def horn_sat(clauses: list[list[tuple[int, bool]]]) -> bool:
        """Exact Horn-SAT via forward chaining.

        A Horn clause is: (not a1 or ... or not ak or b) = (a1 and ... and ak -> b)
        Represented as list of (var, is_positive) where at most one is positive.
        """
        implications = []
        all_negative = []

        for clause in clauses:
            pos_lits = [(v, s) for v, s in clause if s]
            neg_lits = [(v, s) for v, s in clause if not s]

            if len(pos_lits) == 0:
                all_negative.append(set(v for v, _ in neg_lits))
            elif len(pos_lits) == 1:
                head = pos_lits[0][0]
                body = set(v for v, _ in neg_lits)
                implications.append((body, head))
            else:
                raise ValueError("Non-Horn clause in Horn-SAT")

        true_vars = set()
        changed = True

        while changed:
            changed = False
            for body, head in implications:
                if body.issubset(true_vars) and head not in true_vars:
                    true_vars.add(head)
                    changed = True

        for neg_set in all_negative:
            if neg_set.issubset(true_vars):
                return False

        return True

    @staticmethod
    def antihorn_sat(clauses: list[list[tuple[int, bool]]]) -> bool:
        """Exact Anti-Horn-SAT via reduction to Horn.

        Flip all polarities: Anti-Horn becomes Horn.
        """
        flipped = []
        for clause in clauses:
            flipped_clause = [(v, not s) for v, s in clause]
            flipped.append(flipped_clause)
        return ExactSATSolver.horn_sat(flipped)

    @staticmethod
    def two_sat(clauses: list[list[tuple[int, bool]]]) -> bool:
        """Exact 2-SAT via implication graph and SCC (Kosaraju's algorithm).

        Each clause (a or b) becomes (not a -> b) and (not b -> a).
        Unsatisfiable iff some variable and its negation are in same SCC.
        """
        vars_set = set()
        for clause in clauses:
            for v, _ in clause:
                vars_set.add(v)

        if not vars_set:
            return True

        var_list = sorted(vars_set)
        idx = {v: i for i, v in enumerate(var_list)}
        n = len(var_list)

        def lit_node(v, s):
            return 2 * idx[v] + (1 if s else 0)

        def neg_node(node):
            return node ^ 1

        g = [[] for _ in range(2 * n)]
        gr = [[] for _ in range(2 * n)]

        def add_imp(u, v):
            g[u].append(v)
            gr[v].append(u)

        for clause in clauses:
            if len(clause) == 1:
                (v, s) = clause[0]
                l = lit_node(v, s)
                add_imp(neg_node(l), l)
            elif len(clause) == 2:
                (v1, s1), (v2, s2) = clause
                l1, l2 = lit_node(v1, s1), lit_node(v2, s2)
                add_imp(neg_node(l1), l2)
                add_imp(neg_node(l2), l1)
            else:
                raise ValueError("Non-2CNF clause in 2-SAT")

        visited = [False] * (2 * n)
        order = []

        def dfs(start):
            # Iterative DFS using explicit stack to avoid RecursionError
            # on large implication graphs (>500 variables).
            stack = [(start, 0)]
            while stack:
                u, idx = stack[-1]
                if idx == 0 and not visited[u]:
                    visited[u] = True
                if idx < len(g[u]):
                    stack[-1] = (u, idx + 1)
                    w = g[u][idx]
                    if not visited[w]:
                        stack.append((w, 0))
                else:
                    stack.pop()
                    if visited[u]:
                        order.append(u)
                        # Mark as processed to avoid duplicate appends
                        visited[u] = True

        for i in range(2 * n):
            if not visited[i]:
                dfs(i)

        comp = [-1] * (2 * n)

        def rdfs(start, c):
            # Iterative reverse DFS for SCC labeling
            stack = [start]
            while stack:
                u = stack.pop()
                if comp[u] != -1:
                    continue
                comp[u] = c
                for w in gr[u]:
                    if comp[w] == -1:
                        stack.append(w)

        c = 0
        for u in reversed(order):
            if comp[u] == -1:
                rdfs(u, c)
                c += 1

        for i in range(n):
            if comp[2 * i] == comp[2 * i + 1]:
                return False

        return True

    from typing import Hashable
    @staticmethod
    def affine_sat(equations: list[tuple[set[Hashable], int]]) -> bool:
        """Exact Affine-SAT via Gaussian elimination over GF(2).

        Each equation is (vars, const) representing: XOR_{v in vars} v = const (mod 2)
        """
        if not equations:
            return True

        all_vars = set()
        for vars_set, _ in equations:
            all_vars.update(vars_set)

        if not all_vars:
            for _, c in equations:
                if c == 1:
                    return False
            return True

        var_list = sorted(all_vars)
        var_idx = {v: i for i, v in enumerate(var_list)}
        n_vars = len(var_list)
        n_eqs = len(equations)

        A = np.zeros((n_eqs, n_vars + 1), dtype=np.int8)
        for i, (vars_set, c) in enumerate(equations):
            for v in vars_set:
                A[i, var_idx[v]] = 1
            A[i, -1] = c

        row = 0
        for col in range(n_vars):
            pivot = -1
            for i in range(row, n_eqs):
                if A[i, col] == 1:
                    pivot = i
                    break

            if pivot == -1:
                continue

            if pivot != row:
                A[[row, pivot]] = A[[pivot, row]]

            for i in range(n_eqs):
                if i != row and A[i, col] == 1:
                    A[i] = (A[i] + A[row]) % 2

            row += 1

        for i in range(row, n_eqs):
            if np.all(A[i, :-1] == 0) and A[i, -1] == 1:
                return False

        return True
