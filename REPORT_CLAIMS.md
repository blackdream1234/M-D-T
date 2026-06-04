# Master's Thesis Claims

Use this exact claim in the thesis:

The proposed GSNH-MDT framework separates empirical predictive performance from theorem-certified explainability. Fixed Horn and AntiHorn modes are certified for the implemented structural SAT fragments over supported threshold literals. Square2CNF is theorem-certified only when paths are explicitly encoded as 2-CNF and solved using the `two_sat` backend. Affine/GF(2) has Coq-side certificate support, but Python benchmark certification remains disabled until a verified GF(2) certificate checker is implemented. BEST_PER_NODE is treated as an empirical adaptive strategy by default and becomes theorem-certified only when every checked explanation path receives an accepted Horn, AntiHorn, or 2-CNF certificate. The Python implementation is theorem-aligned and tested, but not fully formally verified or extracted from Coq.

## Do Not Claim

- Do not claim unrestricted mixed BEST_PER_NODE is polynomial or theorem-certified.
- Do not claim all benchmark rows are theorem-certified.
- Do not claim Square2CNF is theorem-compliant unless backend=`two_sat`, certificate=`2cnf`, and `theorem_mode_used=True`.
- Do not claim Affine appears in Python theorem-certified benchmark tables until verified GF(2) certificate metadata exists.
- Do not claim the Python Horn/AntiHorn checker validates full star-nested Horn/AntiHorn unless that separate validator is implemented and enabled.
- Do not claim Python is fully formally verified.
