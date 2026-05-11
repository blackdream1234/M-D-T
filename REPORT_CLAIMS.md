# Master's Thesis Claims

Use this exact claim in the thesis:

The proposed GSNH-MDT framework separates empirical predictive performance from theorem-certified explainability. Fixed Horn and AntiHorn modes are theorem-certified through structural threshold encoding and polynomial SAT backends. Square2CNF is theorem-certified only when paths are explicitly encoded as 2-CNF and solved using the two_sat backend. BEST_PER_NODE is treated as an empirical adaptive strategy by default and becomes theorem-certified only when every explanation path receives a Horn, AntiHorn, or 2-CNF certificate. The Coq development proves the mathematical correctness of threshold encoding, path satisfiability, weak AXp reflection, and deletion-based subset-minimal numeric AXp extraction.

## Do Not Claim

- Do not claim unrestricted mixed BEST_PER_NODE is not NP-hard.
- Do not claim all benchmark rows are theorem-certified.
- Do not claim Square2CNF is theorem-compliant unless backend=two_sat and certificate=2cnf.
- Do not claim Python is fully formally verified.
