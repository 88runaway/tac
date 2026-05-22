"""
Random-touch data collection task for PCA training.

The robot opens its gripper and explores the workspace with random joint-space
perturbations so that the tactile sensors contact the target object in a variety
of ways.  No task goal is defined; an episode is considered successful when a
minimum number of contact frames have been accumulated.

This mirrors the "random touch" data collection described in the RDP paper,
which is used to train a PCA model that covers a wide range of tactile
deformation modes rather than only those seen in demonstration trajectories.

Usage (via collect_random_touch.sh):
    python scripts/collect_contact.py random_touch random_touch
"""

from ._base_task import *
import numpy as np
import torch


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1, 0.15, 0.15),
                rot=(-0.354, -0.354, -0.612, -0.612),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5,
                focus_distance=1.0,
                horizontal_aperture=3.6,
                clipping_range=(0.1, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]

    # Maximum frames to save per episode (prevents runaway episodes)
    max_save_frames: int = 600

    # Minimum contact frames required to consider an episode worth saving
    min_contact_frames: int = 20

    # Tactile depth threshold (mm) below which a frame is counted as "in contact"
    # gsmini gel resting depth is ~28.5 mm; values below indicate contact
    contact_depth_threshold: float = 28.2

    # Joint-space perturbation scale (radians) applied each step
    joint_noise_scale: float = 0.04

    # Steps between each random perturbation (allows sim to settle)
    steps_per_action: int = 3

    # Maximum number of perturbation steps per episode
    max_action_steps: int = 200

    # Gripper opening fraction (0=closed, 1=fully open); use a small opening
    # so the fingertips are close together and likely to contact the object
    gripper_open_fraction: float = 0.35

    # Whether to keep_contact check (abort episode if contact is lost mid-episode)
    keep_contact: bool = False

    use_adaptive_grasp: bool = False


class Task(BaseTask):
    """
    Random-touch task: place object on stand, move gripper to a neighbourhood
    above the object with a small opening, then perturb joint angles randomly
    to generate diverse tactile contacts.
    """

    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._contact_frame_count: int = 0

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def create_actors(self):
        stand_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0])
        prism_pose = stand_pose.add_bias([0, 0, 0.06])

        self.stand = self._actor_manager.add_from_usd_file(
            name="stand",
            asset_path="Stand.usd",
            pose=stand_pose,
            density=1e5,
        )
        self.prism_name = os.environ.get("PRISM_NAME", "Hemisphere")
        self.prism = self._actor_manager.add_from_usd_file(
            name="prism",
            asset_path=f"Bar_{self.prism_name}.usd",
            pose=prism_pose,
            density=1e5,
        )

    # ------------------------------------------------------------------
    # Pre-move: open gripper and descend to just above the object
    # ------------------------------------------------------------------
    def pre_move(self):
        self.delay(10)

        # Open gripper to the configured fraction
        self.move(self.atom.open_gripper(self.cfg.gripper_open_fraction))

        # Move to a position slightly above the object with gripper oriented
        # downward so both fingerpads face the prism surface
        approach_height = 0.081 + 0.005 * self.rng.uniform(-1, 1)
        target_pose = self.prism.get_pose().add_bias([0.0, 0.0, approach_height])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0],
        )
        cid = self.prism.register_point(cpose, type="contact")
        # Move close but do NOT close the gripper
        self.move(
            self.atom.grasp_actor(
                self.prism, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
            )
        )
        self._contact_frame_count = 0

    # ------------------------------------------------------------------
    # Main episode: random joint perturbations
    # ------------------------------------------------------------------
    def _play_once(self):
        cfg: TaskCfg = self.cfg
        contact_frames = 0

        # Capture the "home" joint configuration reached after pre_move.
        # Move to sim device once so all in-loop arithmetic stays on GPU.
        dev = self.device  # e.g. "cuda:0"
        home_qpos = self._robot_manager.get_qpos()[0, :7].to(dev)

        for _ in range(cfg.max_action_steps):
            if not self.plan_success:
                break

            # Sample a small random perturbation around the home pose so the
            # gripper stays near the object and doesn't wander too far
            noise = torch.from_numpy(
                self.rng.uniform(
                    -cfg.joint_noise_scale,
                    cfg.joint_noise_scale,
                    size=7,
                ).astype(np.float32)
            ).to(dev)
            gripper_pos = cfg.gripper_open_fraction  # keep opening fixed
            action = torch.cat(
                [home_qpos + noise, torch.tensor([gripper_pos], device=dev)],
                dim=0,
            )

            # Apply action and step simulation
            for _ in range(cfg.steps_per_action):
                self._robot_manager.set_arm(action[:-1], force=True)
                self._robot_manager.set_gripper(action[-1], force=True)
                self._step(is_save=True)
                if self.save_count >= cfg.max_save_frames:
                    self.plan_success = False
                    break

            # Count contact frames based on tactile depth
            depth = self._tactile_manager.get_min_depth()  # tensor([left, right])
            if (depth < cfg.contact_depth_threshold).any():
                contact_frames += 1

        self._contact_frame_count = contact_frames
        self.metadata["contact_frames"] = contact_frames
        self.metadata["prism"] = self.prism_name

    # ------------------------------------------------------------------
    # Success: enough contact frames were observed
    # ------------------------------------------------------------------
    def check_success(self):
        return self._contact_frame_count >= self.cfg.min_contact_frames
