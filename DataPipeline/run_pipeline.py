"""
DataPipeline/run_pipeline.py
============================
Master entry point — runs the full data pipeline in sequence.

Steps
-----
  collect   Step 1 – Run headless MPM simulations to collect training data
  prepare   Step 2 – GMM clustering + stratified train/val/test split
  train     Step 3 – Train one GP expert per GMM cluster
  evaluate  Step 4 – Evaluate experts; print metrics table + optional plots

Usage examples
--------------
# Full pipeline (skip collection, use existing CSV):
python run_pipeline.py --csv workspace/data.csv --out workspace/moe_ws \\
                       --steps prepare,train,evaluate

# Add targeted hard-region data then retrain hard clusters only:
python run_pipeline.py --steps collect,train \\
                       --preset splashing --n-samples 2000 \\
                       --out workspace/moe_ws --clusters 19 42 46 --force

# Full pipeline from scratch (collect → prepare → train → evaluate):
python run_pipeline.py --steps collect,prepare,train,evaluate \\
                       --n-samples 5000 --out workspace/moe_ws \\
                       --csv workspace/data.csv

# Evaluate only, generate plots:
python run_pipeline.py --steps evaluate --out workspace/moe_ws --plots
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).parent.resolve()


def _run(cmd: list, step_name: str):
    """Run a sub-process; exit on failure."""
    log.info(f"\n{'='*60}")
    log.info(f"  STEP: {step_name}")
    log.info(f"  CMD : {' '.join(str(c) for c in cmd)}")
    log.info(f"{'='*60}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        log.error(f"Step '{step_name}' failed (exit code {ret.returncode}).")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full MoE data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Pipeline control
    parser.add_argument(
        "--steps",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["prepare", "train", "evaluate"],
        metavar="STEP[,STEP...]",
        help=(
            "Comma-separated list of steps to run: "
            "collect, prepare, train, evaluate  (default: prepare,train,evaluate)"
        ),
    )

    # Paths
    parser.add_argument("--csv",  nargs="*", default=None,
                        help="Input CSV file(s) for prepare step")
    parser.add_argument("--out",  type=str,  default="workspace/moe_ws",
                        help="Workspace directory (default: workspace/moe_ws)")

    # collect_data.py args
    parser.add_argument("--n-samples",        type=int,   default=1000)
    parser.add_argument("--setups-per-param", type=int,   default=1)
    parser.add_argument("--sampler",          choices=["random", "lhs"], default="random")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--collect-out",      type=str,   default=None,
                        help="Output CSV for collect step (default: <out>/../data_collected.csv)")
    parser.add_argument("--append",           action="store_true")
    parser.add_argument("--preset",           type=str,   default=None,
                        choices=["splashing", "narrow_width", "pseudo_static", "low_eta"],
                        help="Hard-region preset for targeted collection")

    # prepare_data.py args
    parser.add_argument("--k",    type=int, default=None,
                        help="Number of GMM clusters (default from config.py)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if artifacts exist")

    # train_experts.py args
    parser.add_argument("--clusters", type=int, nargs="*", default=None,
                        help="Retrain only these cluster IDs")

    # evaluate_experts.py args
    parser.add_argument("--eval-mode", choices=["val", "test", "both"], default="both")
    parser.add_argument("--out-csv",   type=str, default=None,
                        help="Save evaluation metrics to this CSV")
    parser.add_argument("--plots",     action="store_true",
                        help="Generate parity plots during evaluation")

    args = parser.parse_args()

    out_dir = Path(args.out)
    steps   = [s.lower() for s in args.steps]

    valid = {"collect", "prepare", "train", "evaluate"}
    bad   = set(steps) - valid
    if bad:
        parser.error(f"Unknown step(s): {bad}.  Valid: {valid}")

    py = sys.executable   # use the same Python interpreter

    # ── Step 1: collect ───────────────────────────────────────────────────────
    if "collect" in steps:
        collect_out = args.collect_out
        if collect_out is None:
            collect_out = str(out_dir.parent / "data_collected.csv")

        cmd = [py, str(HERE / "collect_data.py"),
               "--n-samples",        str(args.n_samples),
               "--setups-per-param", str(args.setups_per_param),
               "--sampler",          args.sampler,
               "--seed",             str(args.seed),
               "--out",              collect_out]
        if args.append:
            cmd.append("--append")
        if args.preset:
            cmd += ["--preset", args.preset]

        _run(cmd, "collect")

        # If CSV list not given, auto-add the collected file for prepare step
        if args.csv is None:
            args.csv = [collect_out]
        elif collect_out not in args.csv:
            args.csv.append(collect_out)

    # ── Step 2: prepare ───────────────────────────────────────────────────────
    if "prepare" in steps:
        if not args.csv:
            log.error("--csv is required for the prepare step.")
            sys.exit(1)

        cmd = [py, str(HERE / "prepare_data.py"),
               "--csv"] + [str(c) for c in args.csv] + [
               "--out",  str(out_dir),
               "--seed", str(args.seed)]
        if args.k is not None:
            cmd += ["--k", str(args.k)]
        if args.force:
            cmd.append("--force")

        _run(cmd, "prepare")

    # ── Step 3: train ─────────────────────────────────────────────────────────
    if "train" in steps:
        cmd = [py, str(HERE / "train_experts.py"),
               "--data", str(out_dir)]
        if args.clusters:
            cmd += ["--clusters"] + [str(c) for c in args.clusters]
        if args.force:
            cmd.append("--force")

        _run(cmd, "train")

    # ── Step 4: evaluate ──────────────────────────────────────────────────────
    if "evaluate" in steps:
        cmd = [py, str(HERE / "evaluate_experts.py"),
               "--data", str(out_dir),
               "--mode", args.eval_mode]
        if args.out_csv:
            cmd += ["--out-csv", args.out_csv]
        if args.plots:
            cmd.append("--plots")

        _run(cmd, "evaluate")

    log.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
