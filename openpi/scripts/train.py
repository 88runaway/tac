import csv
import dataclasses
import functools
import logging
import platform
import queue
import threading
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


class _NumpyPrefetchIterator:
    """Prefetch numpy batches in a background thread without touching JAX.

    The PyTorch DataLoader worker interaction (which is the slow part) runs in
    the background thread.  The background thread never calls any JAX API, so
    it is immune to the JAX global-interpreter-lock contention that crippled
    the old ``_PrefetchIterator`` (which converted to JAX arrays in the
    background thread and got serialised with ``jax.device_get``).

    The caller is responsible for converting the returned numpy batch to JAX
    arrays in the main thread (fast DMA, overlaps with GPU compute).
    """

    def __init__(self, numpy_iter, buf_size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=buf_size)
        self._stop = False

        def _worker():
            try:
                while not self._stop:
                    try:
                        item = next(numpy_iter)
                    except StopIteration:
                        break
                    self._q.put(item)
            except Exception as exc:
                self._q.put(exc)
            finally:
                self._q.put(None)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __iter__(self):
        return self


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        result = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(result, tuple):
            chunked_loss, aux = result
        else:
            chunked_loss, aux = result, {}
        return jnp.mean(chunked_loss), aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    _is_kernel = nnx.All(
        nnx.Param,
        nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
        lambda _, x: x.value.ndim > 1,
    )
    kernel_params = nnx.state(model, _is_kernel)

    # Separate action expert vs tac expert kernel params.
    # tac expert params: paths containing "tac_expert" or ending with "_2" suffix (3rd Gemma expert).
    _is_tac_kernel = nnx.All(
        _is_kernel,
        nnx_utils.PathRegex(".*(tac_expert|_2/|_2$).*"),
    )
    _is_action_kernel = nnx.All(
        _is_kernel,
        nnx.Not(nnx_utils.PathRegex(".*(tac_expert|_2/|_2$).*")),
    )
    action_kernel_params = nnx.state(model, _is_action_kernel)
    tac_kernel_params = nnx.state(model, _is_tac_kernel)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "act_param_norm": optax.global_norm(action_kernel_params),
        "tac_param_norm": optax.global_norm(tac_kernel_params),
        **{k: v for k, v in aux.items()},
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    # Split data pipeline: numpy prefetch (background, no JAX) → JAX convert (main thread).
    _torch_dl = data_loader.torch_data_loader
    _dl_sharding = _torch_dl.sharding

    def _numpy_to_model_batch(np_batch):
        jax_batch = jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(_dl_sharding, x),
            np_batch,
        )
        return _model.Observation.from_dict(jax_batch), jax_batch["actions"]

    numpy_prefetch = _NumpyPrefetchIterator(_torch_dl.iter_numpy(), buf_size=8)
    batch = _numpy_to_model_batch(next(numpy_prefetch))
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    try:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    except Exception as _e:
        logging.warning(f"wandb image logging failed (non-fatal): {_e}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # ── txt loss logger ────────────────────────────────────────────────────────
    import pathlib as _pathlib
    _loss_log_path = _pathlib.Path(str(config.checkpoint_dir)) / "loss_log.csv"
    _write_header = not _loss_log_path.exists() or _loss_log_path.stat().st_size == 0
    _loss_log_file = _loss_log_path.open("a", newline="", buffering=1)
    _loss_csv = csv.writer(_loss_log_file)
    if _write_header:
        _loss_csv.writerow(["step", "loss", "action_loss", "tac_loss", "grad_norm", "act_param_norm", "tac_param_norm"])

    # Accumulate info on-device to avoid costly jax.device_get sync that stalls
    # the GPU pipeline.  Only sync the small running-mean scalars at log time.
    _acc_info: dict[str, Any] | None = None
    _acc_count = 0

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        if _acc_info is None:
            _acc_info = jax.tree.map(lambda x: x, info)
        else:
            _acc_info = jax.tree.map(lambda a, b: a + b, _acc_info, info)
        _acc_count += 1

        # Get pre-fetched numpy batch (instant — already in queue) and convert
        # to JAX arrays in the main thread.  DMA overlaps with GPU compute.
        batch = _numpy_to_model_batch(next(numpy_prefetch))

        # Pace: wait for the current GPU step to finish before dispatching
        # the next one. This prevents the async dispatch queue from growing
        # unboundedly, which would cause long device_get stalls at log time.
        jax.block_until_ready(train_state)

        if step % config.log_interval == 0:
            # GPU is already synced by block_until_ready above, so
            # device_get returns almost instantly (just copies scalars).
            reduced_info = jax.device_get(
                jax.tree.map(lambda x: x / _acc_count, _acc_info)
            )
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            try:
                wandb.log(reduced_info, step=step)
            except Exception as _e:
                logging.warning(f"wandb.log failed (non-fatal): {_e}")
            _loss_csv.writerow([
                step,
                reduced_info.get("loss", ""),
                reduced_info.get("action_loss", ""),
                reduced_info.get("tac_loss", ""),
                reduced_info.get("grad_norm", ""),
                reduced_info.get("act_param_norm", ""),
                reduced_info.get("tac_param_norm", ""),
            ])
            _acc_info = None
            _acc_count = 0

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    _loss_log_file.close()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
