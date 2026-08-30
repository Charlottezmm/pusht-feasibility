# PushT three-path debug-only logging smoke

Date: 2026-08-30

Status: **runtime/interface/logging Pass; effectiveness not evaluated**

## Question and claim boundary

This run asks whether the frozen three-path wrapper can execute against the real PushT environment
and Diffusion Policy while preserving the declared pairing and logging invariants. It does **not** ask
whether `probe-adjust` improves final coverage.

The run uses one seed and an artificial `debug_tau=1000` solely to force the valid-probe adjustment
branch. It is excluded from calibration, formal evaluation, effectiveness claims, and adaptation
claims.

## Frozen debug configuration

- Environment: `gym_pusht/PushT-v0`
- Observation: `pixels_agent_pos`
- Simulator damping: `0.0`
- Environment seed: `1000`
- Policy seed: `101000`
- Paths: `fixed`, `probe-no-adjust`, `probe-adjust`
- Total step budget per path: `300`
- Probe start / length: `20 / 5`
- Probe target offset: `20.0`
- Debug threshold: `tau=1000.0`
- Adjusted scale: `1.25`
- Device: CPU
- Protocol SHA-256: `c389226f824539b2c387233c2b05d8c331145b01bcd73a076754e6ef77ab4502`
- Wrapper SHA-256: `38664873ba9bc311e3c64ced403313f735a287b21c25ad10628877a2bd178577`
- Adapter SHA-256: `a8ae89bda53ca8ca25a55a8a1fe3959d8b01d8bec6be27cd20ce60c02ec8d797`

Command:

```bash
<controller-python> \
  experiments/three_path_logging_smoke.py \
  --checkpoint-dir <diffusion-pusht-checkpoint> \
  --compat-config-dir runs/2026-08-29-controller-gate/compat-config \
  --device cpu \
  --damping 0.0 \
  --seed 1000 \
  --max-steps 300 \
  --probe-start 20 \
  --probe-length 5 \
  --target-offset 20 \
  --debug-tau 1000 \
  --debug-only \
  --output-dir runs/2026-08-30-three-path-debug-d0-seed1000-v0.2
```

Raw output directory (gitignored):
`runs/2026-08-30-three-path-debug-d0-seed1000-v0.2/`

## Observed results

| Path | Steps | Stop reason | Max coverage | Final coverage | Success | Response used | Scale 1.25 steps |
|---|---:|---|---:|---:|:---:|---:|---:|
| `fixed` | 143 | `terminated_success` | 0.954689 | 0.954689 | true | 0 | 0 |
| `probe-no-adjust` | 300 | `environment_truncated` | 0.943641 | 0.943641 | false | 0 | 0 |
| `probe-adjust` | 300 | `environment_truncated` | 0.597246 | 0.098403 | false | 275 | 275 |

Both probe paths recorded the same valid simulator-instrumented measurement:

- measurement source: `simulator_info_block_pose`
- contact steps: `4`
- signed response: `66.10180630435096`
- path length: `68.4489619415971`

## Pairing and log validation

The independent validator returned `passed=true`, with every declared check true:

- same seed, simulator setting, budget, initial state, policy prefix, and pre-probe state;
- same probe action hash, post-probe state, and probe measurement for both probe paths;
- complete step logs, valid actions, and finite recorded values;
- only `probe-adjust` consumed the response and executed the `1.25` branch;
- every record remained marked `debug_only=true`.

Representative equality readback:

- initial state: `45c029130d8476cf8f8d07468b49bbbf195b36749ce31530f8d5d258a63c879e`
- shared prefix: `24933b1d3fb972629dc3bbcf200f4f959cba1852605e357034094e42d43c850b`
- pre-probe state: `1826e82015b2383e24a2be7b2a74e13b7cba5bda7f4a7ce84499b46154a3475e`
- probe action: `24bf2b64eb47d87973fcb9be70849e0794b36497f818d233e6d2d9e682dd6e37`
- post-probe state: `3b6b8a945739e0ee2a0ac84ca2b88080adee04a2c593ce3027a40e35239a3c3c`
- probe measurement: `46cf7d61f84577502efbb07e155e73300c8bbbc93477f86201b55af59ea69d0f`

## Interpretation

The runtime/interface/logging smoke passes: all three paths ran against real PushT, the two probe
paths were paired through the measurement point, and response use was isolated to `probe-adjust`.
For this single debug seed, the observed final coverage was lower under `probe-adjust`; that fact is
reported but must not be generalized. The run cannot establish that the frozen `1.25` rule is
effective or ineffective, and it does not demonstrate adaptation.

## Next decision

Use separate calibration seeds to obtain and freeze `tau` without consulting evaluation outcomes.
Then run the predeclared paired multi-seed evaluation of `probe-adjust` versus `probe-no-adjust`, using
the mean paired final-coverage difference as the primary effect estimate. Preserve invalid probes,
failures, and zero-coverage episodes according to the frozen protocol rather than tuning around them.
