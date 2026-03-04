#!/usr/bin/env python3
"""Run sequential OFAT experiments for PPO and REINFORCE with isolated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunSpec:
    index: int
    algo: str
    config: dict
    slug: str


def _fmt_gamma(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return f"{value:.1f}"
    return f"{value:.12g}"


def _fmt_lr(value: float) -> str:
    txt = f"{value:.0e}"
    txt = txt.replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")
    return txt


def _build_slug(config: dict) -> str:
    if config["algo"] == "ppo":
        return (
            f"algo-ppo_env{config['n_envs']}_gamma{_fmt_gamma(config['gamma'])}"
            f"_lr{_fmt_lr(config['lr'])}_mb{config['ppo_mini_batches']}"
            f"_ep{config['ppo_epochs']}_seed{config['seed']}"
        )
    return (
        f"algo-reinforce_env{config['n_envs']}_gamma{_fmt_gamma(config['gamma'])}"
        f"_lr{_fmt_lr(config['lr'])}_seed{config['seed']}"
    )


def _build_run_specs(episodes: int, seed: int, save_period: int) -> list[RunSpec]:
    reinforce_base = {
        "algo": "reinforce",
        "episodes": episodes,
        "gamma": 0.999,
        "lr": 1e-3,
        "n_envs": 1,
        "seed": seed,
        "save_period": save_period,
    }
    ppo_base = {
        "algo": "ppo",
        "episodes": episodes,
        "gamma": 0.999,
        "lr": 1e-3,
        "n_envs": 1,
        "seed": seed,
        "save_period": save_period,
        "ppo_clip": 0.2,
        "ppo_epochs": 10,
        "ppo_mini_batches": 2,
    }

    specs: list[RunSpec] = []
    idx = 1

    for value in [1, 5, 20]:
        cfg = dict(reinforce_base, n_envs=value)
        specs.append(RunSpec(index=idx, algo="reinforce", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [0.999, 0.99, 1.0]:
        cfg = dict(reinforce_base, gamma=value)
        specs.append(RunSpec(index=idx, algo="reinforce", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [1e-2, 1e-3, 1e-4]:
        cfg = dict(reinforce_base, lr=value)
        specs.append(RunSpec(index=idx, algo="reinforce", config=cfg, slug=_build_slug(cfg)))
        idx += 1

    for value in [1, 5, 20]:
        cfg = dict(ppo_base, n_envs=value)
        specs.append(RunSpec(index=idx, algo="ppo", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [0.999, 0.99, 1.0]:
        cfg = dict(ppo_base, gamma=value)
        specs.append(RunSpec(index=idx, algo="ppo", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [1e-2, 1e-3, 1e-4]:
        cfg = dict(ppo_base, lr=value)
        specs.append(RunSpec(index=idx, algo="ppo", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [1, 2, 5]:
        cfg = dict(ppo_base, ppo_mini_batches=value)
        specs.append(RunSpec(index=idx, algo="ppo", config=cfg, slug=_build_slug(cfg)))
        idx += 1
    for value in [1, 5, 10]:
        cfg = dict(ppo_base, ppo_epochs=value)
        specs.append(RunSpec(index=idx, algo="ppo", config=cfg, slug=_build_slug(cfg)))
        idx += 1

    return specs


def _build_command(python_bin: str, main_py: Path, config: dict, run_dir: Path) -> list[str]:
    cmd = [
        python_bin,
        str(main_py),
        "train",
        "--algo",
        config["algo"],
        "--episodes",
        str(config["episodes"]),
        "--gamma",
        str(config["gamma"]),
        "--lr",
        str(config["lr"]),
        "--save-period",
        str(config["save_period"]),
        "--n-envs",
        str(config["n_envs"]),
        "--seed",
        str(config["seed"]),
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--tb-log-dir",
        str(run_dir / "tensorboard"),
    ]
    if config["algo"] == "ppo":
        cmd.extend(
            [
                "--ppo-clip",
                str(config["ppo_clip"]),
                "--ppo-epochs",
                str(config["ppo_epochs"]),
                "--ppo-mini-batches",
                str(config["ppo_mini_batches"]),
            ]
        )
    return cmd


def _write_summary(summary_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "index",
        "algo",
        "slug",
        "status",
        "exit_code",
        "duration_seconds",
        "run_dir",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OFAT PPO/REINFORCE experiment suite.")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed shared by all runs.")
    parser.add_argument("--save-period", type=int, default=10, help="Checkpoint save period.")
    parser.add_argument("--campaign-root", type=str, default="experiments", help="Root directory for campaigns.")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter for run commands.")
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without executing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"
    campaign_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = repo_root / args.campaign_root / campaign_ts

    run_specs = _build_run_specs(episodes=args.episodes, seed=args.seed, save_period=args.save_period)
    if len(run_specs) != 24:
        raise RuntimeError(f"Expected 24 runs, got {len(run_specs)}.")

    print(f"Planned runs: {len(run_specs)}")
    print(f"Campaign dir: {campaign_dir}")

    if args.dry_run:
        for spec in run_specs:
            run_dir = campaign_dir / spec.algo / f"{spec.index:03d}_{spec.slug}"
            cmd = _build_command(args.python_bin, main_py, spec.config, run_dir)
            print(f"[{spec.index:03d}] {shlex.join(cmd)}")
        return 0

    campaign_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for spec in run_specs:
        run_dir = campaign_dir / spec.algo / f"{spec.index:03d}_{spec.slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

        cmd = _build_command(args.python_bin, main_py, spec.config, run_dir)
        command_str = shlex.join(cmd)
        print(f"[{spec.index:03d}/{len(run_specs)}] {command_str}")

        config_payload = {
            "index": spec.index,
            "algo": spec.algo,
            "slug": spec.slug,
            "config": spec.config,
            "command": cmd,
            "command_str": command_str,
            "run_dir": str(run_dir),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        started = datetime.now().isoformat()
        t0 = time.time()
        exit_code = None
        status = "success"
        error = ""

        with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as err:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    stdout=out,
                    stderr=err,
                    text=True,
                    check=False,
                )
                exit_code = int(proc.returncode)
                if exit_code != 0:
                    status = "failed"
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                exit_code = -1
                error = str(exc)
                err.write(f"\nException while running command:\n{exc}\n")

        duration = round(time.time() - t0, 3)
        ended = datetime.now().isoformat()

        status_payload = {
            "index": spec.index,
            "algo": spec.algo,
            "slug": spec.slug,
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": duration,
            "started_at": started,
            "ended_at": ended,
            "error": error,
        }
        (run_dir / "status.json").write_text(
            json.dumps(status_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "index": spec.index,
                "algo": spec.algo,
                "slug": spec.slug,
                "status": status,
                "exit_code": exit_code,
                "duration_seconds": duration,
                "run_dir": str(run_dir),
            }
        )

    summary_path = campaign_dir / "summary.csv"
    _write_summary(summary_path, summary_rows)
    print(f"Suite finished. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
