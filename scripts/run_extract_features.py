"""CLI: extract windowed physiological features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wesad_arousal.config import ensure_dirs, load_config
from wesad_arousal.pipeline import extract_all_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract windowed features from interim data")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    ensure_dirs(cfg)
    features = extract_all_features(cfg)
    print(f"Extracted {len(features)} windows -> {cfg['paths']['processed_dir']}")


if __name__ == "__main__":
    main()
