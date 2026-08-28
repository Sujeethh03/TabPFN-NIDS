#!/usr/bin/env python3
"""Environment smoke test (Build Plan step 0.3).

Runs TabPFN on the sklearn breast-cancer dataset and prints the accuracy.
This is the gate for every later phase: if it fails, the environment is
broken and no pipeline code should be written until it passes.

The test deliberately exercises the three things most likely to be wrong on
a fresh machine:

1. the package installs and imports (``pip install -e .``),
2. the TabPFN checkpoint downloads without HuggingFace authentication,
3. torch selects a working backend (MPS on Apple Silicon, else CPU).

Exit status is 0 on success and 1 on failure, so the script can gate CI or a
Makefile target rather than merely printing a warning.

Usage:
    python smoke_test.py
    python smoke_test.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time

# Minimum accuracy for the run to count as a pass. TabPFN v2 scores ~0.96 on
# this split; 0.95 leaves room for backend-level numerical variation without
# admitting a genuinely broken environment.
ACCURACY_THRESHOLD = 0.95


def _fail(message: str, hint: str = "") -> None:
    """Print a failure message with an optional remediation hint, then exit 1.

    Args:
        message: What went wrong.
        hint: An actionable next step for the user, if one is known.
    """
    print(f"\nFAILED: {message}", file=sys.stderr)
    if hint:
        print(f"\n{hint}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cpu"),
        help="Torch backend to use. 'auto' prefers MPS on Apple Silicon.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the smoke test and return a process exit code."""
    args = parse_args()

    # --- 1. the project package -------------------------------------------
    try:
        from tabpfn_nids import config
    except ImportError as exc:
        _fail(
            f"cannot import the tabpfn_nids package ({exc})",
            "The package is not installed in this interpreter. From the "
            "project root run:\n"
            "    source venv/bin/activate\n"
            "    pip install -e .",
        )

    # --- 2. third-party dependencies --------------------------------------
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        _fail(
            f"a required dependency is missing ({exc})",
            "Install the locked dependency set:\n"
            "    pip install -r requirements.txt",
        )

    config.set_seed(config.SEED)
    env = config.capture_environment(seed=config.SEED)
    device = config.resolve_device(args.device)

    print("TabPFN-NIDS environment smoke test")
    print("-" * 52)
    print(f"  python         {env.python_version} ({env.machine})")
    print(f"  torch          {env.torch_version}")
    print(f"  tabpfn         {env.tabpfn_version}")
    print(f"  scikit-learn   {env.sklearn_version}")
    print(f"  checkpoint     {config.TABPFN_CHECKPOINT}")
    print(f"  device         {device}")
    print(f"  seed           {env.seed}")
    print("-" * 52)

    # --- 3. data ----------------------------------------------------------
    print("Loading breast-cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.SEED, stratify=y
    )
    print(f"  train {X_train.shape[0]} rows, test {X_test.shape[0]} rows, "
          f"{X_train.shape[1]} features")

    if device == "cpu" and X_train.shape[0] > config.MAX_CPU_SAMPLES:
        print(
            f"  note: {X_train.shape[0]} training rows exceeds the tabpfn CPU "
            f"limit of {config.MAX_CPU_SAMPLES}; expect a warning or an error."
        )

    # --- 4. inference -----------------------------------------------------
    print(f"Running TabPFN on {device} (first run downloads the checkpoint)...")
    started = time.time()
    try:
        clf = TabPFNClassifier(
            model_path=config.TABPFN_CHECKPOINT,
            device=device,
            random_state=config.SEED,
        )
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
    except Exception as exc:  # noqa: BLE001 - the point is a helpful message
        name = type(exc).__name__
        if "GatedRepo" in name or "gated" in str(exc).lower():
            _fail(
                f"the TabPFN checkpoint download was refused ({name})",
                f"A GATED model was requested: TABPFN_CHECKPOINT is currently "
                f"'{config.TABPFN_CHECKPOINT}'.\n"
                "'auto' resolves to the gated 'Prior-Labs/tabpfn_3' repository, "
                "which needs a HuggingFace token and accepted terms.\n"
                "Set TABPFN_CHECKPOINT in tabpfn_nids/config.py back to the "
                "ungated TabPFN v2 checkpoint:\n"
                '    TABPFN_CHECKPOINT = "tabpfn-v2-classifier.ckpt"',
            )
        if device == "mps":
            _fail(
                f"TabPFN failed on the MPS backend ({name}: {exc})",
                "Retry on CPU to isolate whether this is a backend problem:\n"
                "    python smoke_test.py --device cpu",
            )
        _fail(f"TabPFN inference failed ({name}: {exc})")

    elapsed = time.time() - started
    accuracy = accuracy_score(y_test, predictions)

    # --- 5. verdict -------------------------------------------------------
    print("-" * 52)
    print(f"  Accuracy       {accuracy:.4f}")
    print(f"  Runtime        {elapsed:.2f}s")
    print("-" * 52)

    if accuracy < ACCURACY_THRESHOLD:
        _fail(
            f"accuracy {accuracy:.4f} is below the {ACCURACY_THRESHOLD} "
            "threshold; the environment runs but produces suspect results",
            "Check that the pinned checkpoint matches the one in "
            "requirements.txt and that no dependency has been upgraded.",
        )

    print("\nSmoke test PASSED. Environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
