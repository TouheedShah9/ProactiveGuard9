#!/usr/bin/env python3
"""
ProactiveGuard v2.0 — One-command pipeline runner.

Runs every module in the correct dependency order (each stage reads
files produced by the one before it), stopping immediately and
printing a clear error if any stage fails — rather than continuing
to run later stages against incomplete or missing input data.

Usage:
    python run_pipeline.py

To also launch the dashboard once the pipeline finishes:
    python run_pipeline.py --dashboard
"""
import subprocess
import sys
import time

PIPELINE = [
    ("v2_data_engine.py",           "Generating synthetic multi-site sensor data"),
    ("v2_digital_twin.py",          "Running physics-based 30-min risk forecasts"),
    ("v2_model.py",                 "Training XGBoost model (SMOTE + Optuna + SHAP)"),
    ("v2_zone_engine.py",           "Computing zone WBGT + coordinated risk detection"),
    ("v2_rl_scheduler.py",          "Training Q-learning proactive scheduler"),
    ("v2_safety_infrastructure.py", "Building audit trail, PDF report, FMEA docs"),
    ("v2_llm_engine.py",            "Generating three-audience LLM explanations"),
]


def run_stage(script: str, description: str) -> bool:
    print(f"\n{'='*65}")
    print(f"  {description}")
    print(f"  ({script})")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ {script} failed (exit code {result.returncode}) "
              f"after {elapsed:.1f}s.")
        print(f"   Fix the error above before continuing — later stages "
              f"depend on this one's output.")
        return False
    print(f"\n✅ {script} completed in {elapsed:.1f}s")
    return True


def main():
    print("ProactiveGuard v2.0 — Full Pipeline Runner")
    print(f"Running {len(PIPELINE)} stages in dependency order...\n")

    t_start = time.time()
    for script, description in PIPELINE:
        if not run_stage(script, description):
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  ✅ PIPELINE COMPLETE in {total/60:.1f} minutes")
    print(f"{'='*65}")
    print("\nNext step:")
    print("  streamlit run v2_dashboard.py")

    if "--dashboard" in sys.argv:
        print("\nLaunching dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "v2_dashboard.py"])


if __name__ == "__main__":
    main()
