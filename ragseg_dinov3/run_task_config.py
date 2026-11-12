import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run DINOv3 RAG-SEG from a task config.")
    parser.add_argument("--config", required=True, help="Path to a COD/SOD task JSON config.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset name from the config, comma-separated names, or 'all'.",
    )
    parser.add_argument("--result_path", default="", help="Override config result_path.")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_prior", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def add_arg(cmd, key, value):
    if value is None or value is False:
        return
    flag = f"--{key}"
    if value is True:
        cmd.append(flag)
    elif isinstance(value, (list, tuple)):
        if value:
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
    else:
        cmd.extend([flag, str(value)])


def build_command(config, dataset_name, dataset_cfg, cli_args):
    merged = {}
    merged.update(config.get("common_args", {}))
    merged.update(dataset_cfg)
    merged["dataset_name"] = dataset_name
    merged["result_path"] = cli_args.result_path or config.get("result_path", "outputs/dinov3_task")

    if cli_args.gpu_id is not None:
        merged["gpu_id"] = cli_args.gpu_id
    if cli_args.limit:
        merged["limit"] = cli_args.limit
    if cli_args.seed is not None:
        merged["seed"] = cli_args.seed
    if cli_args.eval_only:
        merged["eval_only"] = True
    if cli_args.force:
        merged["force"] = True
    if cli_args.save_prior:
        merged["save_prior"] = True

    cmd = [sys.executable, "-m", "ragseg_dinov3.evaluation_dinov3"]
    for key, value in merged.items():
        add_arg(cmd, key, value)
    return cmd


def select_datasets(config, selection):
    datasets = config.get("datasets", {})
    if not datasets:
        raise ValueError("Config has no datasets.")
    if selection == "all":
        return list(datasets.items())
    selected = []
    for name in [item.strip() for item in selection.split(",") if item.strip()]:
        if name not in datasets:
            available = ", ".join(sorted(datasets))
            raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
        selected.append((name, datasets[name]))
    return selected


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    selected = select_datasets(config, args.dataset)

    for dataset_name, dataset_cfg in selected:
        cmd = build_command(config, dataset_name, dataset_cfg, args)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[{config.get('task', 'task')}] {dataset_name}: {printable}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
