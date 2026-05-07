# Archived Monolith

This directory contains the original monolithic implementation of GSNH-MDT.

## Status: ARCHIVED — Do Not Modify

`gsnh_mdt_v3_reference.py` is the original 4,465-line monolith from which the
`gsnh_mdt` package was extracted. It is retained **only** for:

- Archival reference during thesis writing
- Historical comparison if behavioral questions arise
- One-time golden baseline capture (already completed)

## Do Not

- Add new features to this file
- Fix bugs in this file (fix them in the package)
- Import from this file in production or test code
- Use this file for benchmarking

## The authoritative implementation is now

```
src/gsnh_mdt/
```

All development, testing, and benchmarking must target the package.
