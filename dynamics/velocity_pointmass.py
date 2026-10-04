import warnings

import torch
from torch import Tensor
from omegaconf import DictConfig

from diffaero.dynamics.base_dynamics import BaseDynamics
from diffaero.utils.randomizer import build_randomizer


class VelocityPointMassModel(BaseDynamics):
    """Point mass driven by velocity commands (PX4 offboard velocity loop).

    Re-implementation of the pre-reorg pmv dynamics (the original was lost
    with the 2026-07 checkout; the schema below matches the surviving
    checkpoints' hydra configs plus the keys superfly's DiffAeroVelPolicy
    reads at deploy time).

    State: [p(3), v(3)]. Action: velocity command in the yaw-local frame
    (action_frame="local") or world frame, 2-dim [vx, vy] when planar else
    3-dim. The achieved velocity tracks the command through a first-order
    lag, alpha = 1 - exp(-lmbda * dt) per step -- byte-matching the lag
    DiffAeroVelPolicy._apply_velocity_lag applies before handing the
    setpoint to PX4. Planar policies never command z: v_z is identically 0
    and altitude stays where the episode started.

    Attitude is level (zero roll/pitch) with yaw slewing toward the
    velocity EMA at max_yaw_rate deg/s, held below yaw_hold_speed --
    matching DiffAeroVelPolicy.slew_yaw_ned_cmd, so the training camera
    pose reproduces the deployed one.

    action_space="vx_vz_yawrate" (non-holonomic): the action is
    [vx, vz, yaw_rate] in the yaw-local frame -- forward and up velocity
    plus a yaw-rate command [rad/s]; there is no lateral velocity command,
    so the forward camera always faces the commanded motion. The state
    grows to [p(3), v(3), yaw, yaw_rate]. The yaw rate tracks its command
    through its own first-order lag (lmbda_yaw); yaw integrates it with
    gradient, so the velocity/position losses reach the yaw-rate action.
    The velocity command is rotated by the yaw at command time -- what the
    deploy bridge does with the measured heading before sending a world-
    frame setpoint to PX4 -- and then lags in the world frame as above.

    plant="px4_fit" (vx_vz_yawrate only) replaces that first-order lag with
    the velocity loop refitted to PX4 flying the real Starling 2 Max USD in
    Isaac (superfly_expert_sampler wt/v8-chunk scripts/fit_px4_plant.py, 40
    trials; constants in its sim_episode.py PX4_*): the command is delayed
    delay_steps env steps, its horizontal part scaled by gain_xy, then
    a_cmd = kv (v_sp - v), clamped like the sampler's clamp_command
    (|a_xy| <= a_xy_max, |a + g| in [thrust_min, thrust_max]), reaches the
    vehicle through a first-order acceleration lag acc_lag_s, integrated at
    n_substeps per env step. The yaw-rate command shares the delay. The
    acceleration is state (self._acc), carried with gradient. This model IS
    PX4's response, so the deploy bridge sends setpoints with no software lag.

    solver_type / n_substeps are accepted for config-schema compatibility
    but unused: the model integrates in closed form (exact first-order lag
    + trapezoidal position update) once per step.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.type = "pointmass_vel"
        assert self.n_agents == 1, "VelocityPointMassModel supports single-agent envs only."
        self.action_frame: str = cfg.action_frame
        assert self.action_frame in ["world", "local"], \
            f"Invalid action frame: {self.action_frame}. Must be 'world' or 'local'."
        assert bool(cfg.action_is_velocity), \
            "velocity_pointmass configs must set action_is_velocity: true (deploy contract)."
        self.planar: bool = bool(cfg.get("planar", False))
        self.action_space: str = str(cfg.get("action_space", "xy" if self.planar else "xyz"))
        assert self.action_space in ["xyz", "xy", "vx_vz_yawrate"], \
            f"Invalid action_space: {self.action_space}. Must be 'xyz', 'xy' or 'vx_vz_yawrate'."
        assert (self.action_space == "xy") == self.planar, \
            "planar: true is the same thing as action_space: xy"
        self.yaw_rate_action: bool = self.action_space == "vx_vz_yawrate"
        if self.yaw_rate_action:
            assert self.action_frame == "local", "vx_vz_yawrate commands are yaw-local by definition"
        self.state_dim = 8 if self.yaw_rate_action else 6
        self.action_dim = 2 if self.planar else 3

        self._state = torch.zeros(self.n_envs, self.state_dim, device=device)
        self._vel_ema = torch.zeros(self.n_envs, 3, device=device)
        self._acc = torch.zeros(self.n_envs, 3, device=device)
        self._yaw = torch.zeros(self.n_envs, device=device)

        self.align_yaw_with_target_direction: bool = cfg.align_yaw_with_target_direction
        self.align_yaw_with_vel_ema: bool = cfg.align_yaw_with_vel_ema

        self.vel_ema_factor = build_randomizer(cfg.vel_ema_factor, [self.n_envs, 1], device=device)
        self.lmbda = build_randomizer(cfg.lmbda, [self.n_envs, 1], device=device)
        self.max_vel_xy = build_randomizer(cfg.max_vel.xy, [self.n_envs], device=device)
        self.max_vel_z = build_randomizer(cfg.max_vel.z, [self.n_envs], device=device)
        self.max_yaw_rate = build_randomizer(cfg.max_yaw_rate, [self.n_envs], device=device)
        self.yaw_hold_speed: float = float(cfg.yaw_hold_speed)
        if self.yaw_rate_action:
            self.max_vel_x = build_randomizer(cfg.max_vel.x, [self.n_envs], device=device)
            self.reverse_vel_x: float = float(cfg.reverse_vel_x)
            self.lmbda_yaw = build_randomizer(cfg.lmbda_yaw, [self.n_envs], device=device)
            # Episodes start goal-facing (deploy YAW phase) plus this much
            # uniform heading error, so the policy learns to turn toward the goal.
            self.init_yaw_jitter_deg: float = float(cfg.get("init_yaw_jitter_deg", 0.))
        self.plant: str = str(cfg.get("plant", "first_order"))
        assert self.plant in ["first_order", "px4_fit"], f"Invalid plant: {self.plant}"
        if self.plant == "px4_fit":
            assert self.yaw_rate_action, "plant=px4_fit is implemented for action_space=vx_vz_yawrate"
            pc = cfg.px4_fit
            self.px4_kv = build_randomizer(pc.kv, [self.n_envs, 1], device=device)
            self.px4_gain_xy = build_randomizer(pc.gain_xy, [self.n_envs, 1], device=device)
            self.px4_acc_lag = build_randomizer(pc.acc_lag_s, [self.n_envs, 1], device=device)
            self.px4_delay = build_randomizer(pc.delay_steps, [self.n_envs], device=device)
            self.px4_max_delay = int(round(float(pc.delay_steps.max if pc.delay_steps.enabled
                                                 else pc.delay_steps.default)))
            self.px4_a_xy_max = float(pc.a_xy_max)
            self.px4_thrust_min = float(pc.thrust_min)
            self.px4_thrust_max = float(pc.thrust_max)
            self.n_substeps = max(1, int(cfg.n_substeps))
            # [max_delay + 1, n_envs, 4]: newest first; rows are [v_cmd_world(3), yaw_rate_cmd]
            self._cmd_buf = torch.zeros(self.px4_max_delay + 1, self.n_envs, 4, device=device)

    @property
    def min_action(self) -> Tensor:
        if self.yaw_rate_action:
            return torch.stack([
                torch.full_like(self.max_vel_x.value, -self.reverse_vel_x),
                -self.max_vel_z.value,
                -torch.deg2rad(self.max_yaw_rate.value)], dim=-1)
        if self.planar:
            return torch.stack([-self.max_vel_xy.value, -self.max_vel_xy.value], dim=-1)
        return torch.stack(
            [-self.max_vel_xy.value, -self.max_vel_xy.value, -self.max_vel_z.value], dim=-1)

    @property
    def max_action(self) -> Tensor:
        if self.yaw_rate_action:
            return torch.stack([
                self.max_vel_x.value,
                self.max_vel_z.value,
                torch.deg2rad(self.max_yaw_rate.value)], dim=-1)
        if self.planar:
            return torch.stack([self.max_vel_xy.value, self.max_vel_xy.value], dim=-1)
        return torch.stack(
            [self.max_vel_xy.value, self.max_vel_xy.value, self.max_vel_z.value], dim=-1)

    def detach(self):
        super().detach()
        self._vel_ema.detach_()
        self._acc.detach_()
        if self.plant == "px4_fit":
            self._cmd_buf = self._cmd_buf.detach()

    @property
    def yaw(self) -> Tensor:
        return self._state[..., 6].detach() if self.yaw_rate_action else self._yaw

    @property
    def yaw_rate(self) -> Tensor:
        """Achieved yaw rate [rad/s] (zeros unless action_space=vx_vz_yawrate)."""
        if self.yaw_rate_action:
            return self._state[..., 7].detach()
        return torch.zeros_like(self._yaw)

    @property
    def q(self) -> Tensor:
        half = 0.5 * self.yaw
        zero = torch.zeros_like(half)
        return torch.stack([zero, zero, torch.sin(half), torch.cos(half)], dim=-1)

    @property
    def _q(self) -> Tensor:
        warnings.warn("Quaternion with gradient is not supported in velocity point mass model. "
                      "Returning detached version instead.")
        return self.q

    @property
    def w(self) -> Tensor:
        if self.yaw_rate_action:
            zero = torch.zeros_like(self.yaw_rate)
            return torch.stack([zero, zero, self.yaw_rate], dim=-1)
        warnings.warn("Access of angular velocity in velocity point mass model is not supported. "
                      "Returning zero tensor instead.")
        return torch.zeros_like(self.p)

    @property
    def _w(self) -> Tensor:
        return self.w

    @property
    def _p(self) -> Tensor: return self._state[..., 0:3]
    @property
    def _v(self) -> Tensor: return self._state[..., 3:6]
    @property
    def _a(self) -> Tensor: return self._acc

    def set_yaw(self, env_idx: Tensor, yaw: Tensor) -> None:
        """Point the (level) body frame at the given world yaw for env_idx.

        The env calls this on reset so each episode starts facing its
        target, reproducing the deploy YAW phase that hands over to the
        policy already goal-facing."""
        if self.yaw_rate_action:
            # out of place: _state may still be referenced by the autograd graph
            state = self._state.clone()
            state[env_idx, 6] = yaw
            self._state = state
        else:
            self._yaw[env_idx] = yaw

    def reset_idx(self, env_idx: Tensor) -> None:
        mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        mask[env_idx] = True
        mask3 = mask.unsqueeze(-1).expand_as(self._vel_ema)
        self._vel_ema = torch.where(mask3, 0., self._vel_ema)
        self._acc = torch.where(mask3, 0., self._acc)
        self._yaw = torch.where(mask, 0., self._yaw)
        if self.plant == "px4_fit":
            # hover command in flight before the handover
            self._cmd_buf = torch.where(mask[None, :, None], 0., self._cmd_buf)

    def step(self, U: Tensor) -> None:
        if self.yaw_rate_action:
            return self._step_yaw_rate(U)
        if self.planar:
            U3 = torch.cat([U, torch.zeros_like(U[..., :1])], dim=-1)
        else:
            U3 = U
        if self.action_frame == "local":
            v_cmd = torch.matmul(self.Rz, U3.unsqueeze(-1)).squeeze(-1)
        else:
            v_cmd = U3

        v = self._v
        alpha = 1.0 - torch.exp(-self.lmbda.value * self.dt)
        v_next = torch.lerp(v, v_cmd, alpha)
        if self.planar:
            v_next = torch.cat([v_next[..., :2], torch.zeros_like(v_next[..., 2:3])], dim=-1)
        p_next = self._p + self.dt * 0.5 * (v + v_next)
        next_state = torch.cat([p_next, v_next], dim=-1)

        self._state = self.grad_decay(next_state)
        self._acc = (v_next - v) / self.dt
        # vel_ema keeps its gradient: the env's vel_loss differentiates
        # through it (same as PointMassModelBase.update_state).
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)
        with torch.no_grad():
            self._slew_yaw()

    def _step_yaw_rate(self, U: Tensor) -> None:
        """vx_vz_yawrate step: U = [vx, vz, yaw_rate_cmd] in the yaw-local frame."""
        vx, vz, r_cmd = U.unbind(dim=-1)
        yaw, r = self._state[..., 6], self._state[..., 7]
        # rotate by the (differentiable) heading at command time
        v_cmd = torch.stack([vx * torch.cos(yaw), vx * torch.sin(yaw), vz], dim=-1)
        if self.plant == "px4_fit":
            return self._step_px4_fit(v_cmd, r_cmd, yaw, r)

        v = self._v
        alpha = 1.0 - torch.exp(-self.lmbda.value * self.dt)
        v_next = torch.lerp(v, v_cmd, alpha)
        p_next = self._p + self.dt * 0.5 * (v + v_next)

        alpha_r = 1.0 - torch.exp(-self.lmbda_yaw.value * self.dt)
        r_next = torch.lerp(r, r_cmd, alpha_r)
        yaw_next = yaw + self.dt * 0.5 * (r + r_next)
        # wrap without breaking the gradient (atan2 of sin/cos is smooth)
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        next_state = torch.cat([p_next, v_next, yaw_next.unsqueeze(-1), r_next.unsqueeze(-1)], dim=-1)
        self._state = self.grad_decay(next_state)
        self._acc = (v_next - v) / self.dt
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)

    def _step_px4_fit(self, v_cmd: Tensor, r_cmd: Tensor, yaw: Tensor, r: Tensor) -> None:
        cmd = torch.cat([v_cmd, r_cmd.unsqueeze(-1)], dim=-1)
        self._cmd_buf = torch.cat([cmd.unsqueeze(0), self._cmd_buf[:-1]], dim=0)
        k = self.px4_delay.value.round().long().clamp(0, self.px4_max_delay)
        delayed = self._cmd_buf[k, torch.arange(self.n_envs, device=self.device)]  # [n_envs, 4]
        g_xy = self.px4_gain_xy.value
        v_sp = torch.cat([delayed[:, :2] * g_xy, delayed[:, 2:3]], dim=-1)
        r_sp = delayed[:, 3]

        h = self.dt / self.n_substeps
        beta = 1.0 - torch.exp(-h / self.px4_acc_lag.value)
        g_vec = self._G_vec.expand_as(v_sp)
        p, v, a = self._p, self._v, self._acc
        for _ in range(self.n_substeps):
            a_cmd = self.px4_kv.value * (v_sp - v)
            n_xy = a_cmd[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            a_cmd = torch.cat([a_cmd[:, :2] * (self.px4_a_xy_max / n_xy).clamp(max=1.), a_cmd[:, 2:3]], dim=-1)
            f = a_cmd - g_vec  # specific thrust (g_vec points down)
            nf = f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            f = f * (nf.clamp(self.px4_thrust_min, self.px4_thrust_max) / nf)
            a_cmd = f + g_vec
            a = a + beta * (a_cmd - a)
            v_new = v + a * h
            p = p + 0.5 * h * (v + v_new)
            v = v_new

        alpha_r = 1.0 - torch.exp(-self.lmbda_yaw.value * self.dt)
        r_next = torch.lerp(r, r_sp, alpha_r)
        yaw_next = yaw + self.dt * 0.5 * (r + r_next)
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        next_state = torch.cat([p, v, yaw_next.unsqueeze(-1), r_next.unsqueeze(-1)], dim=-1)
        self._state = self.grad_decay(next_state)
        self._acc = a
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)

    @torch.no_grad()
    def _slew_yaw(self) -> None:
        """Rate-limited yaw toward the velocity EMA (deploy slew_yaw_ned_cmd)."""
        speed_xy = self._vel_ema[..., :2].norm(dim=-1)
        desired = torch.atan2(self._vel_ema[..., 1], self._vel_ema[..., 0])
        err = desired - self._yaw
        err = torch.atan2(torch.sin(err), torch.cos(err))
        max_step = torch.deg2rad(self.max_yaw_rate.value) * self.dt
        step = torch.clamp(err, -max_step, max_step)
        self._yaw = torch.where(speed_xy >= self.yaw_hold_speed, self._yaw + step, self._yaw)
