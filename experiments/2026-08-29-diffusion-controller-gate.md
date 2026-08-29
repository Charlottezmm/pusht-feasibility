# Diffusion Policy controller gate and paired damping diagnostic

- Date: 2026-08-29
- Status: `PASS` for reproducible controller execution/evidence
- Nominal strong entry gate: `PASS` at `damping=0.0`
- Dynamics-shift strong entry gate: `FAIL` at `damping=1.0`
- Claim boundary: three simulator seeds; no adaptation, robustness, mechanism, real-friction, or real-robot claim

## Question and predeclared gates

Can the pinned `lerobot/diffusion_pusht` checkpoint produce auditable, non-floor PushT task
performance locally, and how does the same controller behave when the simulator condition changes
from its official `damping=0.0` setting to the project's earlier `damping=1.0` setting?

The execution/evidence gate required three complete episodes with valid, reviewable logs and
non-floor task-performance variation. The stronger engineering entry gate required at least one
success or median maximum coverage `>=0.5`. These are bounded engineering gates, not statistical
population claims.

Metric roles were frozen before the main run:

- maximum coverage: controller gate metric and best overlap reached during the episode;
- final coverage: endpoint diagnostic;
- success: raw coverage `>0.95` before the shared 300-step budget is exhausted.

## Frozen identity and environment

- LeRobot source commit: `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`
- Checkpoint repository: `lerobot/diffusion_pusht`
- Checkpoint revision: `84a7c23178445c6bbf7e1a884ff497017910f653`
- Checkpoint weights SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`
- Compatibility-config SHA-256: `188568e0cb5c28188bf3ea411d88fbb8e9287843840bad606ea0edb4db297c11`
- Evaluator SHA-256 for the reported runs: `1f6505994628e4a90a9d56949a532ee34113ad151d3fcae6b1942bed27efe693`
- Repository base commit: `be854397c441c2d2ad5c4d684f655adf2de169c6`
- Python: `3.11.1`
- Device for reported episodes: CPU, float32, no AMP
- Environment: `gym_pusht/PushT-v0`, `pixels_agent_pos`, 300-step budget
- Environment seeds: `1000,1001,1002`
- Policy seed rule: `100000 + environment_seed`
- Key packages: LeRobot 0.1.0, torch 2.6.0, torchvision 0.21.0, diffusers
  0.32.2, huggingface-hub 0.28.1, gymnasium 0.29.1, gym-pusht 0.1.5,
  pymunk 6.11.0, safetensors 0.5.2

The full installed freeze and dependency-resolution reports are retained under the ignored raw
evidence directory `runs/2026-08-29-controller-gate/`. The old LeRobot commit's retired `pyav`
distribution name was replaced by `av==14.1.0`, and `draccus==0.9.3` was pinned. No policy code or
checkpoint weights were modified. The checkpoint config compatibility copy removes only the later
runtime fields `device` and `use_amp`; strict safetensor loading remains enabled.

## Device blocker and recorded fallback

The MPS path did not complete an episode:

1. The first attempt exposed an evaluator loader defect: the concrete `DiffusionConfig` decoder was
   incorrectly used for a registry config containing `type`. The evaluator was patched to decode
   through `PreTrainedConfig` and assert `DiffusionConfig`.
2. The second attempt showed that safetensors 0.5.2 could not open the file directly on MPS. Weight
   loading was changed to strict CPU staging followed by an explicit move to the inference device.
3. The third MPS attempt entered a rollout but produced action `[nan, nan]` at step 17. The first 16
   actions and logged metrics were finite. `PYTORCH_ENABLE_MPS_FALLBACK=0`; the incomplete episode
   was retained and was not counted as controller performance evidence.

CPU was therefore declared as a new configuration. It restarted from environment and policy reset,
used separate output directories, and did not continue or aggregate the partial MPS trajectory.

## Results

The official-condition CPU smoke at seed 1000 completed in 143 steps with success and maximum/final
coverage `0.9546893997`. After the project-reference `damping=1.0` run showed non-floor but weak task
performance, seeds 1001 and 1002 were added at `damping=0.0` as a declared paired diagnostic.

| Seed | Maximum coverage, `d=0` | Final coverage, `d=0` | Success, `d=0` | Maximum coverage, `d=1` | Final coverage, `d=1` | Success, `d=1` | Paired max difference, `d0-d1` |
|---:|---:|---:|:---:|---:|---:|:---:|---:|
| 1000 | 0.954689 | 0.954689 | true | 0.434978 | 0.000000 | false | +0.519711 |
| 1001 | 0.952370 | 0.952370 | true | 0.000068 | 0.000000 | false | +0.952302 |
| 1002 | 0.978649 | 0.978649 | true | 0.243673 | 0.170761 | false | +0.734976 |

Aggregate readback:

- `damping=0.0`: median maximum coverage `0.954689`; success `3/3`;
- `damping=1.0`: median maximum coverage `0.243673`; success `0/3`;
- mean paired maximum-coverage difference: `+0.735663`;
- all three paired maximum-coverage differences were positive.

For every pair, the initial compact state, environment seed, policy seed, CPU device, first action,
checkpoint/config/evaluator hashes, and 300-step budget matched. Every counted step log had the
expected row count and SHA-256, all actions and metrics were finite, and all actions were inside the
declared action space.

## Interpretation and decision

The execution/evidence gate passed because the three `damping=1.0` episodes were complete,
reviewable, and contained non-floor task-performance variation. Their negative success result is
valid performance evidence, not a reason to repeat the run. The strong entry gate failed at that
setting: there were no successes and median maximum coverage was below 0.5.

The paired diagnostic supports a narrower and more useful protocol decision: for this checkpoint,
`damping=0.0` is the controller nominal condition and `damping=1.0` is a declared dynamics-shift
condition. The nominal strong entry gate passed. This also weakens the explanation that 300 steps
were generally insufficient, because all three nominal episodes succeeded within 143 steps.

The data do not identify the mechanism of degradation at `damping=1.0`; overshoot or another
closed-loop mismatch remains a hypothesis. Pymunk `Space.damping` controls simulator velocity
retention/decay and is not verified real friction. Three paired seeds do not establish population
performance, robustness, or adaptation.

## Next step

Create a new protocol version before further evaluation:

1. freeze `damping=0.0` as nominal and `damping=1.0` as the dynamics-shift diagnostic;
2. freeze the shared probe, response feature, adjustment rule, information boundary, budgets, and
   hashes before seeing three-path results;
3. implement and unit-test the `fixed`, `probe-no-adjust`, and `probe-adjust` wrapper paths;
4. run a one-seed logging smoke only after schema, pairing, clamp, NaN, and hidden-setting leakage
   checks pass.

Ordinary closed-loop action divergence is not adaptation evidence. The future primary causal
comparison must be `probe-adjust` versus `probe-no-adjust`, which execute the identical probe and
differ only in whether the declared response feature enters the frozen adjustment rule.

## Reproduction interface

The evaluator is `experiments/diffusion_controller_evaluator.py`. A portable invocation is:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 "$PUSHT_CONTROLLER_PYTHON" \
  experiments/diffusion_controller_evaluator.py \
  --checkpoint-dir "$PUSHT_CHECKPOINT_DIR" \
  --compat-config-dir "$PUSHT_COMPAT_CONFIG_DIR" \
  --device cpu \
  --damping 0.0 \
  --seeds 1000,1001,1002 \
  --max-steps 300 \
  --output-dir "$PUSHT_OUTPUT_DIR"
```

The output directory must not already exist. Each run writes `config.json`, `episodes.jsonl`,
`summary.json`, and per-step JSONL logs; failures additionally write `failure.json` without
rewriting earlier attempts.
