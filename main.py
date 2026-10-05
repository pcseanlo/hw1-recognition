"""Run the experiments required by the 16-824 Homework 1 handout.

Examples:
    python main.py --dry-run
    python main.py --experiments q1 q2
    python main.py --experiments detection
    python main.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CLASSIFICATION_DIR = ROOT / "q1_q2_classification"
DETECTION_DIR = ROOT / "detection"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"
Q2_CHECKPOINT = ARTIFACTS_DIR / "q2_resnet18.pt"

Q1_STEPS = ("q1-no-aug", "q1-aug")
Q2_STEPS = ("q2-train", "q2-tsne")
DETECTION_STEPS = (
    "detection-tests",
    "detection-gt",
    "detection-overfit",
    "detection-full",
    "detection-demo",
    "detection-map",
)
ALL_STEPS = Q1_STEPS + Q2_STEPS + DETECTION_STEPS


def run_process(
    name: str,
    command: list[str],
    cwd: Path,
    *,
    env_updates: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(command)
    print(f"\n[{name}] ({cwd}) $ {printable}", flush=True)
    if dry_run:
        return

    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def create_q2_tsne(output_path: Path, dry_run: bool = False) -> None:
    print(f"\n[q2-tsne] Writing {output_path}", flush=True)
    if dry_run:
        return
    if not Q2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing {Q2_CHECKPOINT}. Run --experiments q2-train first."
        )

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from sklearn.manifold import TSNE
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    import torchvision

    sys.path.insert(0, str(CLASSIFICATION_DIR))
    from voc_dataset import VOCDataset

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(VOCDataset.CLASS_NAMES))
    saved_state = torch.load(Q2_CHECKPOINT, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("resnet."): value for key, value in saved_state.items()
    }
    model.load_state_dict(state)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    dataset = VOCDataset("test", 224)
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)[:1000].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)

    features = []
    labels = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            batch_features = feature_extractor(images.to(device)).flatten(1)
            features.append(batch_features.cpu())
            labels.append(targets)

    features_np = torch.cat(features).numpy()
    labels_np = torch.cat(labels).numpy()
    points = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(features_np)

    class_colors = plt.get_cmap("tab20")(np.arange(20))[:, :3]
    active_counts = np.maximum(labels_np.sum(axis=1, keepdims=True), 1)
    point_colors = labels_np @ class_colors / active_counts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(points[:, 0], points[:, 1], c=point_colors, s=12, alpha=0.75)
    legend = [
        Line2D(
            [0], [0], marker="o", linestyle="", label=name,
            markerfacecolor=class_colors[index], markeredgecolor="none",
        )
        for index, name in enumerate(VOCDataset.CLASS_NAMES)
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("Fine-tuned ResNet-18 features on 1,000 PASCAL VOC test images")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_map_tool(python: str, dry_run: bool = False) -> None:
    map_dir = DETECTION_DIR / "mAP"
    if map_dir.exists():
        return
    run_process(
        "detection-map-setup",
        ["git", "clone", "https://github.com/Cartucho/mAP.git", str(map_dir)],
        ROOT,
        dry_run=dry_run,
    )
    if not dry_run:
        (map_dir / "input" / "ground-truth").mkdir(parents=True, exist_ok=True)
        (map_dir / "input" / "detection-results").mkdir(parents=True, exist_ok=True)


def expand_experiments(requested: list[str]) -> list[str]:
    aliases = {
        "all": ALL_STEPS,
        "q1": Q1_STEPS,
        "q2": Q2_STEPS,
        "detection": DETECTION_STEPS,
    }
    expanded = []
    for item in requested:
        for step in aliases.get(item, (item,)):
            if step not in expanded:
                expanded.append(step)
    return expanded


def run_step(step: str, python: str, dry_run: bool) -> None:
    if step == "q1-no-aug":
        run_process(
            step,
            [python, "q1_q2_classification/train_q1.py"],
            ROOT,
            env_updates={
                "HW1_DISABLE_AUGMENTATION": "1",
                "HW1_LOG_DIR": str(RUNS_DIR / step),
            },
            dry_run=dry_run,
        )
    elif step == "q1-aug":
        run_process(
            step,
            [python, "q1_q2_classification/train_q1.py"],
            ROOT,
            env_updates={
                "HW1_DISABLE_AUGMENTATION": "0",
                "HW1_LOG_DIR": str(RUNS_DIR / step),
            },
            dry_run=dry_run,
        )
    elif step == "q2-train":
        run_process(
            step,
            [python, "q1_q2_classification/train_q2.py"],
            ROOT,
            env_updates={
                "HW1_DISABLE_AUGMENTATION": "0",
                "HW1_LOG_DIR": str(RUNS_DIR / step),
                "HW1_CHECKPOINT_PATH": str(Q2_CHECKPOINT),
            },
            dry_run=dry_run,
        )
    elif step == "q2-tsne":
        create_q2_tsne(ARTIFACTS_DIR / "q2_tsne.png", dry_run=dry_run)
    elif step == "detection-tests":
        run_process(step, [python, "-m", "test_object_detection"], DETECTION_DIR, dry_run=dry_run)
    elif step == "detection-gt":
        run_process(step, [python, "train.py", "--visualize_gt"], DETECTION_DIR, dry_run=dry_run)
    elif step == "detection-overfit":
        run_process(step, [python, "train.py", "--overfit"], DETECTION_DIR, dry_run=dry_run)
    elif step == "detection-full":
        run_process(step, [python, "train.py"], DETECTION_DIR, dry_run=dry_run)
    elif step == "detection-demo":
        run_process(
            step,
            [python, "train.py", "--inference", "--test_inference"],
            DETECTION_DIR,
            dry_run=dry_run,
        )
    elif step == "detection-map":
        ensure_map_tool(python, dry_run=dry_run)
        run_process(step, [python, "train.py", "--inference"], DETECTION_DIR, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown experiment: {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experiments and generate the artifacts required by HW1."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=("all", "q1", "q2", "detection", *ALL_STEPS),
        help="Experiment groups or individual steps to run in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later independent experiments after a failed command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = expand_experiments(args.experiments)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    failures = []
    for step in steps:
        try:
            run_step(step, sys.executable, args.dry_run)
        except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
            failures.append((step, error))
            print(f"[{step}] FAILED: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed experiments:", file=sys.stderr)
        for step, error in failures:
            print(f"  - {step}: {error}", file=sys.stderr)
        return 1
    print("\nAll selected experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
