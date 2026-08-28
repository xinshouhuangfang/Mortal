# AGENTS.md

This file gives every new opencode session the baseline context about the
Mortal project. Read it first, then follow up with targeted exploration (`ls`,
`grep`, reading key files) instead of a blind full-project read.

## What this project is

Mortal (凡夫) is a free, open-source AI for Japanese riichi mahjong, powered by
deep reinforcement learning. Code is licensed AGPL-3.0-or-later.

## Repository layout (Rust workspace + Python engine)

- `Cargo.toml` — root Cargo workspace, resolver "3". Members: `libriichi`,
  `exe-wrapper`, `play`.
- `libriichi/` — the core. A Rust crate (`[lib] name = "riichi"`) exposing a
  Python module via **PyO3 0.25** (crate-type `cdylib` + `rlib`). Builds the
  compiled game/dataset backend. Deps: `boomphf`, `ndarray`, `numpy`, `rayon`.
  Edition 2024. Build script uses `pyo3-build-config`.
  Features: `default = ["pymod", "mimalloc"]`, plus `abi3` and
  `sp_reproduce_cpp_ver`.
- `exe-wrapper/` — small wrapper crate (likely the distributed engine binary).
- `play/` — an interactive/serving crate for playing against the AI.
- `mortal/` — **Python training/self-play engine**:
  - `train.py`, `train_loop.py`, `engine.py`, `model.py`, `dataloader.py`,
    `common.py`, `player.py`, `prelude.py`, `lr_scheduler.py`.
  - `config.py` / `config.example.toml` — config is TOML; path from env
    `MORTAL_CFG` (default `config.toml`); `device` value `auto` is resolved by
    `MORTAL_DEVICE` env or torch availability.
  - `mortal/libriichi.so` is the built pyo3 module (generated, not committed).
- `log-viewer/` — browser tool for viewing game logs (`index.example.html`,
  `render_log.py`, `files/`). Note: `files/` is excluded from typos checks.
- `docs/` — mdbook (`book.toml`) documentation; user + online + perf + ref.

## Conventions

- Rust: stable-ish 2024 edition; format with `cargo fmt`, lint `cargo clippy`.
- Python engine is driven by `config.toml`; add new knobs there and in
  `config.example.toml`.
- Typos: config in `typos.toml`; avoid introducing misspelled identifiers.
- Any code/docs here are AGPL-3.0-or-later — preserve license headers/attributions.

## Useful commands (run from repo root)

- `cargo build --release` — build; the game logic (libriichi) needs pyo3 env.
- `cargo test`.
- `cargo build -p libriichi --lib` — build/check the core crate (Python
  extension module `libriichi.so`; needs pyo3 env).
- `cargo build -p libriichi --bins --no-default-features` — build the
  standalone binaries (`stat`, `validate_logs`). Do NOT enable the default
  `pymod` feature here, or they fail to link against libpython.
- Python engine: invoked via the `mortal/` scripts (e.g. `mortal/train.py`),
  picking config through `MORTAL_CFG`.
- `mdbook serve` inside `docs/` — local docs preview.

## Working rules for the assistant

- Confirm the concrete task before diverging; if a long session drifts from the
  user's goal, surface that and re-align rather than continuing silently.
- When asked to "read the project", reuse this file and explore only what the
  current task actually needs.
