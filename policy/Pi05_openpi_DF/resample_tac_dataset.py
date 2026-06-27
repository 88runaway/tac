#!/usr/bin/env python3
"""
从 tac_all 按指定任务+数量随机采样，生成新的 per-task 数据集目录。

采样策略：为每个任务随机选取指定数量的 episode，通过符号链接引用原始
parquet / video 文件，只重写 meta/*.jsonl 和 meta/info.json，**不复制视频**。

Usage:
    # 方式一：命令行逐任务指定（task=N 格式）
    python policy/Pi05_openpi_DF/resample_tac_dataset.py \
        --tasks lift_can=30 pull_out_key=30 insert_hole=30 \
        --output_dir /data/zjb/data/UniVTAC/tac_subset_30

    # 方式二：JSON 配置（与 multitask_config.json 格式兼容）
    python policy/Pi05_openpi_DF/resample_tac_dataset.py \\
        --config policy/Pi05_openpi_DF/multitask_config.json \\
        --output_dir /data/zjb/data/UniVTAC/tac_subset

    # 调试用 dry-run（只打印，不写文件）
    python policy/Pi05_openpi_DF/resample_tac_dataset.py \\
        --tasks lift_can=5 insert_HDMI=5 --dry_run

训练时使用重组数据集（--task all 已支持 --multitask_data_dir）：
    python policy/Pi05_openpi_DF/train_df.py \\
        --task all \\
        --multitask_data_dir /data/zjb/data/UniVTAC/tac_subset_30 \\
        ...
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

# ─── 默认路径 ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
UNIVTAC_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SOURCE_DIR = Path("/data/zjb/data/UniVTAC/tac_all")


# ─── 核心函数 ─────────────────────────────────────────────────────────────────

def resample_task(
    task_name: str,
    src_task_dir: Path,
    dst_task_dir: Path,
    n_episodes: int,
    seed: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """从单个任务目录中随机采样 n_episodes 个 episode，写到 dst_task_dir。

    文件策略：
      - data/chunk-000/episode_XXXXXX.parquet → 重写，将 episode_index / index 列
        重映射为连续值（0, 1, 2, ...），文件名也用新索引命名。
        lerobot 的 episode_data_index 用连续下标索引，原始非连续索引会导致
        IndexError: index N out of bounds for size M。
      - videos/chunk-000/<key>/episode_XXXXXX.mp4  → 符号链接，用新索引命名。
      - meta/*.jsonl / info.json → 重新生成，episode_index 均为新连续值。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # ── 读取源 meta ────────────────────────────────────────────────────────────
    src_meta = src_task_dir / "meta"
    with open(src_meta / "info.json") as f:
        src_info = json.load(f)

    all_episodes = []
    with open(src_meta / "episodes.jsonl") as f:
        for line in f:
            all_episodes.append(json.loads(line.strip()))

    all_stats = {}
    stats_path = src_meta / "episodes_stats.jsonl"
    if stats_path.exists():
        with open(stats_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                all_stats[rec["episode_index"]] = rec

    with open(src_meta / "tasks.jsonl") as f:
        tasks_lines = [l.strip() for l in f if l.strip()]

    total_available = len(all_episodes)
    if n_episodes > total_available:
        print(f"  [WARNING] {task_name}: 请求 {n_episodes} 但只有 {total_available} 个 episode，"
              f"使用全部 {total_available} 个。")
        n_episodes = total_available

    # ── 随机采样 ───────────────────────────────────────────────────────────────
    rng = random.Random(seed)
    selected = sorted(rng.sample(all_episodes, n_episodes), key=lambda e: e["episode_index"])

    total_frames = sum(e["length"] for e in selected)

    print(f"\n  [{task_name}] {total_available} → {n_episodes} episodes, {total_frames} frames")
    if dry_run:
        orig_indices = [e["episode_index"] for e in selected]
        print(f"    original episode indices: {orig_indices}")
        return True

    # ── 创建目标目录 ───────────────────────────────────────────────────────────
    if dst_task_dir.exists():
        if overwrite:
            shutil.rmtree(dst_task_dir)
        else:
            print(f"    已存在：{dst_task_dir}（加 --overwrite 覆盖）")
            return True

    (dst_task_dir / "meta").mkdir(parents=True)

    chunks_size = src_info.get("chunks_size", 1000)
    src_data_root = src_task_dir / "data"
    dst_data_root = dst_task_dir / "data"
    src_vid_root = src_task_dir / "videos"
    dst_vid_root = dst_task_dir / "videos"

    # ── 重写 parquet 文件（重映射 episode_index / index）─────────────────────
    # lerobot 用 episode_data_index["from"][ep_idx] 按连续下标查找，
    # 因此新数据集的 episode_index 必须是 0, 1, 2, ...
    global_frame_offset = 0
    for new_ep_idx, ep in enumerate(selected):
        orig_ep_idx = ep["episode_index"]
        orig_chunk = orig_ep_idx // chunks_size
        src_pq = src_data_root / f"chunk-{orig_chunk:03d}" / f"episode_{orig_ep_idx:06d}.parquet"
        if not src_pq.exists():
            print(f"  [WARNING] missing parquet: {src_pq}, skipping episode.")
            continue

        table = pq.read_table(src_pq)
        df = table.to_pandas()

        # 重映射列
        n_rows = len(df)
        df["episode_index"] = new_ep_idx
        df["index"] = list(range(global_frame_offset, global_frame_offset + n_rows))
        global_frame_offset += n_rows

        new_chunk = new_ep_idx // chunks_size
        dst_chunk_dir = dst_data_root / f"chunk-{new_chunk:03d}"
        dst_chunk_dir.mkdir(parents=True, exist_ok=True)
        dst_pq = dst_chunk_dir / f"episode_{new_ep_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, schema=table.schema), dst_pq)

    # ── 符号链接：视频文件（用新连续索引命名）────────────────────────────────
    if src_vid_root.exists():
        for new_ep_idx, ep in enumerate(selected):
            orig_ep_idx = ep["episode_index"]
            orig_chunk = orig_ep_idx // chunks_size
            new_chunk = new_ep_idx // chunks_size
            src_fname = f"episode_{orig_ep_idx:06d}.mp4"
            dst_fname = f"episode_{new_ep_idx:06d}.mp4"
            for key_dir in src_vid_root.glob(f"chunk-{orig_chunk:03d}/*/"):
                src_file = key_dir / src_fname
                dst_key_dir = dst_vid_root / f"chunk-{new_chunk:03d}" / key_dir.name
                dst_key_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_key_dir / dst_fname
                if src_file.exists():
                    os.symlink(src_file.resolve(), dst_file)

    # ── 写 meta/info.json ──────────────────────────────────────────────────────
    new_info = dict(src_info)
    new_info["total_episodes"] = n_episodes
    new_info["total_frames"] = total_frames
    video_keys = [k for k, v in src_info["features"].items() if v.get("dtype") == "video"]
    new_info["total_videos"] = n_episodes * len(video_keys)
    with open(dst_task_dir / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    # ── 写 meta/episodes.jsonl（新连续 episode_index）────────────────────────
    with open(dst_task_dir / "meta" / "episodes.jsonl", "w") as f:
        for new_ep_idx, ep in enumerate(selected):
            new_ep = dict(ep)
            new_ep["episode_index"] = new_ep_idx
            f.write(json.dumps(new_ep) + "\n")

    # ── 写 meta/episodes_stats.jsonl（新连续 episode_index）──────────────────
    if all_stats:
        with open(dst_task_dir / "meta" / "episodes_stats.jsonl", "w") as f:
            for new_ep_idx, ep in enumerate(selected):
                orig_idx = ep["episode_index"]
                if orig_idx in all_stats:
                    rec = dict(all_stats[orig_idx])
                    rec["episode_index"] = new_ep_idx
                    f.write(json.dumps(rec) + "\n")

    # ── 写 meta/tasks.jsonl ────────────────────────────────────────────────────
    with open(dst_task_dir / "meta" / "tasks.jsonl", "w") as f:
        for line in tasks_lines:
            f.write(line + "\n")

    return True


def resample_dataset(
    task_specs: dict[str, int],
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
    overwrite: bool = False,
    dry_run: bool = False,
):
    """对多个任务批量采样，生成新的 per-task 数据集目录。

    Args:
        task_specs: {task_name: n_episodes}
        source_dir: tac_all 根目录（per-task 子目录结构）
        output_dir: 输出目录（同样是 per-task 结构，供 --multitask_data_dir 使用）
        seed:        随机种子（保证可复现）
        overwrite:   覆盖已有目录
        dry_run:     仅打印，不写文件
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  TAC Dataset Resampler")
    print(f"  Source:  {source_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Tasks:   {task_specs}")
    print(f"  Seed:    {seed}")
    print(f"  Dry run: {dry_run}")
    print(f"{sep}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for task_name, n_eps in task_specs.items():
        src_task_dir = source_dir / task_name
        if not src_task_dir.is_dir():
            print(f"\n  [ERROR] {task_name}: 源目录不存在 {src_task_dir}，跳过。")
            continue
        dst_task_dir = output_dir / task_name
        if resample_task(task_name, src_task_dir, dst_task_dir, n_eps, seed, overwrite, dry_run):
            ok += 1

    total_eps = sum(task_specs.values())
    print(f"\n{sep}")
    print(f"  Done: {ok}/{len(task_specs)} tasks, {total_eps} total episodes requested")
    print(f"  Output: {output_dir}")
    if not dry_run:
        print(f"\n  训练命令示例:")
        print(f"    python policy/Pi05_openpi_DF/train_df.py \\")
        print(f"        --task all \\")
        print(f"        --multitask_data_dir {output_dir} \\")
        print(f"        --train_config policy/Pi05_openpi_DF/config/train_df.yaml \\")
        print(f"        --warm_start_ckpt /data/zjb/ckpts/pi05_all_128_20k/params")
    print(f"{sep}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 tac_all 随机采样指定数量 episode，生成新的多任务数据集目录。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 任务规格：二选一
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--tasks", nargs="+", metavar="TASK=N",
        help="任务采样规格，格式 task_name=n_episodes，例如：lift_can=30 insert_hole=30",
    )
    grp.add_argument(
        "--config", type=str, metavar="JSON",
        help=(
            "multitask_config.json 路径。自动读取 tasks 字段中的 num_episodes，"
            "可通过 --override_n 整体覆盖每个任务的采样数量。"
        ),
    )

    parser.add_argument(
        "--source_dir", type=str, default=str(DEFAULT_SOURCE_DIR),
        help=f"源数据集根目录（默认 {DEFAULT_SOURCE_DIR}）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出数据集根目录（per-task 子目录，供 --multitask_data_dir 使用）",
    )
    parser.add_argument(
        "--override_n", type=int, default=None,
        help="（与 --config 配合）整体覆盖每个任务的采样数量",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（保证可复现，默认 42）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已存在的任务输出目录",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="仅打印采样信息，不实际写入文件",
    )

    args = parser.parse_args()

    # ── 解析任务规格 ───────────────────────────────────────────────────────────
    task_specs: dict[str, int] = {}

    if args.tasks:
        for spec in args.tasks:
            if "=" not in spec:
                parser.error(f"--tasks 格式错误：'{spec}'，应为 task_name=N")
            name, n_str = spec.split("=", 1)
            task_specs[name.strip()] = int(n_str.strip())
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        for task_name, task_cfg in cfg.get("tasks", {}).items():
            n = args.override_n if args.override_n is not None else task_cfg.get("num_episodes", 30)
            task_specs[task_name] = n

    resample_dataset(
        task_specs=task_specs,
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
