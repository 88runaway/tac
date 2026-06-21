from collections.abc import Iterator, Sequence
import concurrent.futures
import io
import logging
import multiprocessing
import os
import typing
from pathlib import Path
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

# ── Compatibility patch for lerobot + newer HF datasets library ──────────────
# lerobot uses `torch.stack(hf_dataset["column"])` which fails when the `datasets`
# library (>=4.x) returns a Column object instead of a list of tensors.
# Fix: monkey-patch torch.stack to handle Column objects transparently.
_orig_torch_stack = torch.stack

def _compat_torch_stack(tensors, *args, **kwargs):
    if type(tensors).__name__ == "Column":
        tensors = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in tensors]
    return _orig_torch_stack(tensors, *args, **kwargs)

torch.stack = _compat_torch_stack

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _build_tactile_mmap_cache(
    dataset_dir: str,
    tactile_keys: list[str],
    num_threads: int = 32,
) -> tuple[dict[str, str], dict[tuple[int, int], int], dict[int, list[int]]]:
    """Pre-decode all tactile PNG frames from parquet into memory-mapped .npy files.

    The decoded float32 arrays are saved to ``{dataset_dir}/.tactile_cache/{key}.npy``.
    Workers open them with ``np.load(path, mmap_mode='r')`` — all workers share
    the same OS page cache, so physical RAM usage is ~14 GB total (not per-worker).

    Returns:
        cache_paths: {key: str} paths to the .npy mmap files.
        frame_lookup: {(episode_idx, frame_in_episode) -> cache_pos}
        episode_bounds: {episode_idx -> [min_frame, max_frame]}
    """
    import glob
    import pyarrow.parquet as pq
    from PIL import Image

    cache_dir = Path(dataset_dir) / ".tactile_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openpi")

    parquet_files = sorted(glob.glob(str(Path(dataset_dir) / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"[TactileCache] No parquet files found in {dataset_dir}/data")

    # First pass: read metadata + raw bytes from all parquet files
    all_rows: list[dict] = []
    for pf in parquet_files:
        tbl = pq.read_table(pf)
        for i in range(len(tbl)):
            row: dict = {
                "global_idx": int(tbl["index"][i].as_py()),
                "episode_idx": int(tbl["episode_index"][i].as_py()),
                "frame_idx": int(tbl["frame_index"][i].as_py()),
            }
            for k in tactile_keys:
                entry = tbl[k][i].as_py()
                row[k] = entry["bytes"] if isinstance(entry, dict) else entry
            all_rows.append(row)

    all_rows.sort(key=lambda r: r["global_idx"])
    n = len(all_rows)

    # Build lookup tables
    frame_lookup: dict[tuple[int, int], int] = {}
    episode_bounds: dict[int, list[int]] = {}
    for pos, r in enumerate(all_rows):
        frame_lookup[(r["episode_idx"], r["frame_idx"])] = pos
        e, f = r["episode_idx"], r["frame_idx"]
        if e not in episode_bounds:
            episode_bounds[e] = [f, f]
        else:
            if f < episode_bounds[e][0]:
                episode_bounds[e][0] = f
            if f > episode_bounds[e][1]:
                episode_bounds[e][1] = f

    # Probe image shape
    probe_bytes = all_rows[0][tactile_keys[0]]
    probe_img = np.array(Image.open(io.BytesIO(probe_bytes)).convert("RGB"))
    H, W = probe_img.shape[:2]

    # Check if cache files already exist with correct shape
    cache_paths: dict[str, str] = {}
    all_cached = True
    for k in tactile_keys:
        safe_name = k.replace(".", "_").replace("/", "_")
        npy_path = cache_dir / f"{safe_name}.npy"
        cache_paths[k] = str(npy_path)
        if npy_path.exists():
            try:
                probe = np.load(str(npy_path), mmap_mode="r")
                if probe.shape == (n, H, W, 3) and probe.dtype == np.float32:
                    del probe
                    continue
                del probe
            except Exception:
                pass
        all_cached = False

    if all_cached:
        logger.info(f"[TactileCache] Reusing existing mmap cache at {cache_dir} ({n} frames)")
        return cache_paths, frame_lookup, episode_bounds

    # Decode and write to mmap files
    logger.info(
        f"[TactileCache] Building mmap cache: {n} frames × {len(tactile_keys)} keys "
        f"→ {cache_dir} (using {num_threads} threads) …"
    )
    # Create mmap-backed output arrays
    mmaps: dict[str, np.ndarray] = {}
    for k in tactile_keys:
        npy_path = cache_paths[k]
        arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(n, H, W, 3))
        mmaps[k] = arr

    def _decode_row(args: tuple[int, dict]) -> None:
        pos, row = args
        for k in tactile_keys:
            img = Image.open(io.BytesIO(row[k])).convert("RGB")
            mmaps[k][pos] = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(_decode_row, enumerate(all_rows)))

    # Flush to disk
    for arr in mmaps.values():
        arr.flush()
    del mmaps

    total_gb = n * H * W * 3 * 4 * len(tactile_keys) / 1e9
    logger.info(f"[TactileCache] Done. Cache size on disk: {total_gb:.2f} GB")
    return cache_paths, frame_lookup, episode_bounds


class TactilePreloadedDataset(Dataset):
    """Wraps a LeRobotDataset, replacing live PNG decoding for tactile images with
    a memory-mapped numpy cache.

    On first ``__getitem__`` call (inside each worker process), the ``.npy`` files
    are opened in read-only mmap mode.  All workers share the same OS page cache
    pages, so physical RAM usage is ~14 GB total regardless of ``num_workers``.

    The mmap files are built once in the main process (or reused if they already
    exist from a previous run) and live at ``{dataset_dir}/.tactile_cache/``.
    """

    def __init__(
        self,
        base_dataset,
        dataset_dir: str,
        tactile_keys: list[str],
        frame_offsets: list[int],
        num_decode_threads: int = 32,
    ):
        self._base = base_dataset
        self._tactile_keys = tactile_keys
        self._frame_offsets = frame_offsets
        # Build / reuse mmap cache in main process
        self._cache_paths, self._frame_lookup, self._ep_bounds = _build_tactile_mmap_cache(
            dataset_dir, tactile_keys, num_decode_threads
        )
        # Lazy: each worker opens its own mmap handle on first __getitem__
        self._mmaps: dict[str, np.ndarray] | None = None

    def _open_mmaps(self) -> None:
        self._mmaps = {
            k: np.load(path, mmap_mode="r") for k, path in self._cache_paths.items()
        }

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index) -> dict:
        if self._mmaps is None:
            self._open_mmaps()

        item = self._base[index]

        ep = int(item["episode_index"])
        frame = int(item["frame_index"])
        lo, hi = self._ep_bounds[ep]

        for k in self._tactile_keys:
            frames_list = []
            for off in self._frame_offsets:
                f = max(lo, min(hi, frame + off))
                pos = self._frame_lookup[(ep, f)]
                frames_list.append(np.array(self._mmaps[k][pos]))
            item[k] = np.stack(frames_list, axis=0)  # (num_blocks, H, W, 3)

        return item


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    root = getattr(data_config, "root", None)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    delta_ts = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    if data_config.extra_delta_timestamps:
        delta_ts.update(data_config.extra_delta_timestamps)

    # Detect tactile keys with multi-frame delta_timestamps (> 1 timestamp each)
    # and replace their loading with a pre-decoded RAM cache to avoid per-sample
    # PNG decode overhead in every DataLoader worker.
    tactile_preload_keys = []
    tactile_frame_offsets_map: dict[str, list[int]] = {}
    if data_config.extra_delta_timestamps:
        fps = dataset_meta.fps
        for k, dts in data_config.extra_delta_timestamps.items():
            if len(dts) > 1 and "tactile" in k:
                tactile_preload_keys.append(k)
                tactile_frame_offsets_map[k] = [round(dt * fps) for dt in dts]

    if tactile_preload_keys:
        # Create base dataset WITHOUT tactile keys in delta_timestamps so that
        # LeRobot only decodes 1 frame per finger (current frame) instead of
        # num_blocks. The TactilePreloadedDataset wrapper replaces that with the
        # full multi-frame stack from the in-memory cache.
        base_delta_ts = {k: v for k, v in delta_ts.items() if k not in tactile_preload_keys}
        base_dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=root,
            delta_timestamps=base_delta_ts,
            video_backend="pyav",
        )
        if data_config.prompt_from_task:
            base_dataset = TransformedDataset(
                base_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        # All tactile keys share the same frame offsets (same block_size / fps)
        frame_offsets = tactile_frame_offsets_map[tactile_preload_keys[0]]
        dataset_dir = root if root is not None else repo_id
        dataset = TactilePreloadedDataset(
            base_dataset,
            dataset_dir=str(dataset_dir),
            tactile_keys=tactile_preload_keys,
            frame_offsets=frame_offsets,
        )
        return dataset

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        prefetch_factor=getattr(config, "prefetch_factor", None),
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        prefetch_factor: Number of batches loaded in advance by each worker. None uses the
            PyTorch default (2). Values of 4–8 help when each sample is large (e.g. tactile).
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            prefetch_factor: Number of batches loaded in advance per worker. None uses the
                PyTorch default (2). For large samples (e.g. with tactile images), 4–8 helps.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def sharding(self) -> jax.sharding.Sharding | None:
        return self._sharding

    def iter_numpy(self):
        """Yield raw numpy batches without JAX conversion.

        Safe to call from a non-JAX thread because it never touches JAX.
        """
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield batch

    def __iter__(self):
        for batch in self.iter_numpy():
            if self._sharding is not None:
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
            else:
                yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def torch_data_loader(self) -> TorchDataLoader:
        """Access the underlying TorchDataLoader for numpy-level prefetching."""
        assert isinstance(self._data_loader, TorchDataLoader)
        return self._data_loader

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
