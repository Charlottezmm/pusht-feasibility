"""Pure protocol core for the PushT three-path probe-adjust wrapper.

This module intentionally does not construct the environment, load a policy, or accept the true
simulator setting. A runtime adapter may provide ordinary policy observations and privileged
measurement records, but the action-decision boundary remains explicit here.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


FIXED = "fixed"
PROBE_NO_ADJUST = "probe-no-adjust"
PROBE_ADJUST = "probe-adjust"
PATHS = (FIXED, PROBE_NO_ADJUST, PROBE_ADJUST)
PRIVILEGED_MEASUREMENT_SOURCE = "simulator_info_block_pose"
DEFAULT_ADJUSTED_SCALE = 1.25
IDENTITY_SCALE = 1.0
NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ProbePlan:
    direction: np.ndarray
    target: np.ndarray
    actions: np.ndarray
    sha256: str
    direction_valid: bool


@dataclass(frozen=True)
class ProbeMeasurement:
    signed_response: float
    path_length: float
    contact_steps: int
    valid: bool
    invalid_reasons: tuple[str, ...]
    measurement_source: str


@dataclass(frozen=True)
class ActionDecision:
    path: str
    base_target: np.ndarray
    executed_target: np.ndarray
    scale: float
    response_used: bool
    clamped: bool


def _validate_path(path):
    if path not in PATHS:
        raise ValueError(f"Unknown wrapper path: {path}")
    return path


def _vector2(value, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError(f"{label} must have shape (2,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values.")
    return vector


def _bounds(action_low, action_high):
    low = _vector2(action_low, "action_low")
    high = _vector2(action_high, "action_high")
    if np.any(low > high):
        raise ValueError("action_low must not exceed action_high.")
    return low, high


def _positions(block_positions):
    positions = np.asarray(block_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (2,) or len(positions) == 0:
        raise ValueError(
            f"block_positions must have shape (n, 2) with n >= 1, got {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("block_positions must contain only finite values.")
    return positions


def probe_array_sha256(actions):
    array = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"probe actions must have shape (n, 2), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("probe actions must contain only finite values.")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_probe_plan(
    agent_position,
    block_position,
    action_low,
    action_high,
    target_offset,
    probe_length,
):
    agent = _vector2(agent_position, "agent_position")
    block = _vector2(block_position, "block_position")
    low, high = _bounds(action_low, action_high)
    if not np.isfinite(target_offset) or target_offset < 0:
        raise ValueError("target_offset must be finite and non-negative.")
    if not isinstance(probe_length, int) or isinstance(probe_length, bool) or probe_length <= 0:
        raise ValueError("probe_length must be a positive integer.")

    displacement = block - agent
    distance = float(np.linalg.norm(displacement))
    direction_valid = distance > NUMERICAL_TOLERANCE
    if direction_valid:
        direction = displacement / distance
        target = np.clip(block + float(target_offset) * direction, low, high)
    else:
        direction = np.zeros(2, dtype=np.float64)
        target = np.clip(block, low, high)
    actions = np.repeat(target[None, :], probe_length, axis=0)
    return ProbePlan(
        direction=direction,
        target=target,
        actions=actions,
        sha256=probe_array_sha256(actions),
        direction_valid=direction_valid,
    )


def signed_response(block_before, block_after, probe_direction):
    before = _vector2(block_before, "block_before")
    after = _vector2(block_after, "block_after")
    direction = _vector2(probe_direction, "probe_direction")
    norm = float(np.linalg.norm(direction))
    if norm <= NUMERICAL_TOLERANCE:
        raise ValueError("probe_direction must have non-zero finite norm.")
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("probe_direction must be unit-normalized.")
    return float(np.dot(after - before, direction))


def trajectory_path_length(block_positions):
    positions = _positions(block_positions)
    if len(positions) == 1:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def measure_probe_response(
    block_positions,
    probe_direction,
    direction_valid,
    contact_steps,
    measurement_source,
):
    positions = _positions(block_positions)
    direction = _vector2(probe_direction, "probe_direction")
    if not isinstance(direction_valid, bool):
        raise ValueError("direction_valid must be a bool.")
    if not isinstance(contact_steps, int) or isinstance(contact_steps, bool) or contact_steps < 0:
        raise ValueError("contact_steps must be a non-negative integer.")
    if not isinstance(measurement_source, str) or not measurement_source:
        raise ValueError("measurement_source must be a non-empty string.")

    invalid_reasons = []
    if not direction_valid:
        invalid_reasons.append("invalid_probe_direction")
        response = 0.0
    else:
        response = signed_response(positions[0], positions[-1], direction)
    if contact_steps == 0:
        invalid_reasons.append("no_probe_contact")
    if measurement_source != PRIVILEGED_MEASUREMENT_SOURCE:
        invalid_reasons.append("unexpected_measurement_source")

    return ProbeMeasurement(
        signed_response=response,
        path_length=trajectory_path_length(positions),
        contact_steps=contact_steps,
        valid=not invalid_reasons,
        invalid_reasons=tuple(invalid_reasons),
        measurement_source=measurement_source,
    )


def select_scale(
    path,
    measurement,
    tau,
    adjusted_scale=DEFAULT_ADJUSTED_SCALE,
):
    _validate_path(path)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")
    if not np.isfinite(adjusted_scale) or adjusted_scale <= 0:
        raise ValueError("adjusted_scale must be finite and positive.")
    if path != PROBE_ADJUST:
        return IDENTITY_SCALE
    if not isinstance(measurement, ProbeMeasurement):
        raise ValueError("probe-adjust requires a ProbeMeasurement.")
    if not measurement.valid:
        return IDENTITY_SCALE
    if measurement.signed_response < tau:
        return float(adjusted_scale)
    return IDENTITY_SCALE


def adjust_target(
    agent_position,
    base_target,
    scale,
    action_low,
    action_high,
):
    agent = _vector2(agent_position, "agent_position")
    base = _vector2(base_target, "base_target")
    low, high = _bounds(action_low, action_high)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    return np.clip(agent + float(scale) * (base - agent), low, high)


def decide_executed_action(
    path,
    agent_position,
    base_target,
    action_low,
    action_high,
    measurement,
    tau,
    adjusted_scale=DEFAULT_ADJUSTED_SCALE,
):
    _validate_path(path)
    base = _vector2(base_target, "base_target")
    scale = select_scale(path, measurement, tau, adjusted_scale)
    executed = adjust_target(
        agent_position=agent_position,
        base_target=base,
        scale=scale,
        action_low=action_low,
        action_high=action_high,
    )
    response_used = bool(path == PROBE_ADJUST and measurement is not None and measurement.valid)
    return ActionDecision(
        path=path,
        base_target=base,
        executed_target=executed,
        scale=scale,
        response_used=response_used,
        clamped=not np.array_equal(executed, _vector2(agent_position, "agent_position") + scale * (base - _vector2(agent_position, "agent_position"))),
    )


def phase_budget(path, total_steps, probe_start, probe_length):
    _validate_path(path)
    values = {
        "total_steps": total_steps,
        "probe_start": probe_start,
        "probe_length": probe_length,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
        raise ValueError("total_steps, probe_start, and probe_length must be integers.")
    if total_steps <= 0 or probe_start < 0 or probe_length <= 0:
        raise ValueError("Budget values must be positive, with probe_start >= 0.")
    if probe_start + probe_length > total_steps:
        raise ValueError("Probe window must fit inside the total budget.")

    is_fixed = path == FIXED
    return {
        "prefix_steps": probe_start,
        "probe_steps": 0 if is_fixed else probe_length,
        "continuation_steps": probe_length if is_fixed else 0,
        "post_window_steps": total_steps - probe_start - probe_length,
    }


def new_episode_record(path):
    _validate_path(path)
    return {
        "path": path,
        "controller_visible": {
            "ordinary_observation": None,
            "base_target": None,
        },
        "action_decision": {
            "scale": IDENTITY_SCALE,
            "response_used": False,
            "executed_target": None,
        },
        "evaluator_only": {
            "probe": None,
            "response": None,
            "metrics": None,
            "hashes": None,
            "hidden_setting": None,
        },
        "stop_reason": None,
    }
