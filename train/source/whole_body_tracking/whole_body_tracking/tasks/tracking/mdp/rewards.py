from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MultiMotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def vr_position_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_pos = command.robot_anchor_pos_w
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_pos_rel = command.robot_vr_3point_pos_w - robot_anchor_pos[:, None, :]
    vr_pos_rel = quat_apply(robot_anchor_quat_inv, vr_pos_rel)
    ref_anchor_pos = command.anchor_pos_w
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_pos_rel_ref = command.vr_3point_body_pos_w - ref_anchor_pos[:, None, :]
    vr_pos_rel_ref = quat_apply(ref_anchor_quat_inv, vr_pos_rel_ref)

    error = torch.sum(
        torch.square(vr_pos_rel_ref - vr_pos_rel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def vr_orientation_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_quat_rel = quat_mul(robot_anchor_quat_inv, command.vr_3point_body_quat_w)
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_quat_rel_ref = quat_mul(ref_anchor_quat_inv, command.vr_3point_body_quat_w)

    error = (
        quat_error_magnitude(vr_quat_rel_ref, vr_quat_rel)
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)
