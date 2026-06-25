# AGENTS.md

## Project rule

This repository contains a GSNH / MDT research implementation.

Do not perform large rewrites without tests.

## Architecture

Python is the experiment layer.
Rust is the high-performance algorithmic core.

Python handles:

* dataset loading
* benchmark orchestration
* sklearn comparison
* CSV output
* plotting
* reports

Rust handles:

* GSNH predicates
* candidate generation
* scoring
* pruning
* tree search
* prediction
* bitset/matrix-heavy operations

## Correctness policy

The current Python implementation is the reference oracle.

Any Rust implementation must be tested against Python before replacing behavior.

If Python and Rust disagree:

1. save the failing input
2. show expected Python output
3. show actual Rust output
4. explain the suspected cause
5. do not continue broad implementation until fixed

## Rust policy

Prefer safe Rust.
Do not use unsafe unless necessary and documented.
Keep functions small.
Use explicit types.
Avoid hidden global state.
Keep deterministic behavior.
Add unit tests for each module.

## Testing policy

After Rust changes, run:

cargo test

After Python wrapper changes, run:

maturin develop
pytest

After benchmark-related changes, run the smallest benchmark first before full benchmark execution.

## Review policy

Before final response, summarize:

* changed files
* tests run
* test output
* known limitations
* next safe step
