#!/usr/bin/env python3
import argparse
from pathlib import Path
from agent_v4.collect import collect


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "decoder", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--rate", type=int, choices=(50,80,120,160,200), required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--zero-gate", action="store_true")
    p.add_argument("--stochastic-validation",action="store_true")
    a = p.parse_args()
    if not 32 <= a.frames <= 512 or a.start < 0 or a.seed < 0:
        p.error("Invalid frame count/offset/seed")
    collect(**vars(a))


if __name__ == "__main__":
    main()

