# Probe-response threshold calibration

- Date: 2026-08-31 (Asia/Singapore)
- Stage / gate: Gate A before formal three-path evaluation
- Status: `PASS`
- Repository base commit at runtime: `3c73d1c340fef63c9c5be0521bb1116c5694f51f`
- Raw output: `runs/2026-08-31-probe-response-calibration-v0.1`
- Claim boundary: threshold-selection evidence only; no controller-effectiveness claim

## Question and frozen rule

What single threshold follows from the predeclared calibration manifest and can be loaded unchanged
by the later formal evaluator?

The manifest fixed environment seeds `100..109`, evaluator-only settings
`damping={0.0,1.0}`, CPU execution, a 20-step controller prefix, and a 5-step shared probe. Invalid
probe measurements would be retained but excluded from the pooled median. At least five valid
measurements were required:

```text
tau = median(all valid signed response x values)
```

The calibration runner did not accept a threshold, did not execute a response-conditioned
adjustment, and did not use final task coverage to select `tau`.

## Frozen identity and environment

- Checkpoint revision: `84a7c23178445c6bbf7e1a884ff497017910f653`
- Checkpoint weights SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`
- Compatibility-config SHA-256: `188568e0cb5c28188bf3ea411d88fbb8e9287843840bad606ea0edb4db297c11`
- Protocol SHA-256: `c389226f824539b2c387233c2b05d8c331145b01bcd73a076754e6ef77ab4502`
- Wrapper SHA-256: `38664873ba9bc311e3c64ced403313f735a287b21c25ad10628877a2bd178577`
- Calibration runner SHA-256: `3e49bf180ef1a33db732e120dd25318b92dc189da5bf31fb3c52640e7c0fcc91`
- macOS `26.6.2` (`arm64`); Python `3.11.1`; CPU, no AMP
- Environment: `gym_pusht/PushT-v0`, observation type `pixels_agent_pos`
- Key packages: LeRobot 0.1.0, torch 2.6.0, torchvision 0.21.0, diffusers 0.32.2,
  gymnasium 0.29.1, gym-pusht 0.1.5, numpy 2.1.3, pymunk 6.11.0
- Dependency readback after the run: `pip check` reported no broken requirements

## Procedure

The output directory did not exist at preflight. Set the controller Python and pinned checkpoint
locations for the local machine, then reproduce the frozen arguments with:

```bash
PUSHT_CONTROLLER_PYTHON="${PUSHT_CONTROLLER_PYTHON:?set controller Python}" \
PUSHT_CHECKPOINT_DIR="${PUSHT_CHECKPOINT_DIR:?set pinned checkpoint directory}" \
"$PUSHT_CONTROLLER_PYTHON" \
  experiments/probe_response_calibration.py \
  --checkpoint-dir "$PUSHT_CHECKPOINT_DIR" \
  --compat-config-dir runs/2026-08-29-controller-gate/compat-config \
  --device cpu \
  --output-dir runs/2026-08-31-probe-response-calibration-v0.1
```

The output directory is write-once; reproduction must use a new directory rather than overwrite the
reported evidence. The ignored raw config retains the exact machine-local executable and artifact
locations used for this run.

## Raw observations

| Seed | damping | valid | signed response `x` | path length | contact steps |
|---:|---:|:---:|---:|---:|---:|
| 100 | 0.0 | true | 64.648836 | 67.855015 | 4 |
| 101 | 0.0 | true | 83.442702 | 91.874187 | 5 |
| 102 | 0.0 | true | 1.995713 | 20.374390 | 5 |
| 103 | 0.0 | true | 58.051560 | 67.664391 | 5 |
| 104 | 0.0 | true | 68.391938 | 70.381112 | 5 |
| 105 | 0.0 | true | 58.747817 | 69.037247 | 5 |
| 106 | 0.0 | true | 55.785569 | 66.663408 | 5 |
| 107 | 0.0 | true | 78.913736 | 90.131155 | 5 |
| 108 | 0.0 | true | 58.717262 | 68.179572 | 5 |
| 109 | 0.0 | true | 17.895981 | 27.672901 | 5 |
| 100 | 1.0 | true | 122.658083 | 138.209675 | 3 |
| 101 | 1.0 | true | 169.381149 | 183.104695 | 2 |
| 102 | 1.0 | true | -10.560470 | 43.736043 | 3 |
| 103 | 1.0 | true | 157.333540 | 162.871485 | 2 |
| 104 | 1.0 | true | 100.885080 | 117.816223 | 4 |
| 105 | 1.0 | true | 146.322986 | 153.502848 | 2 |
| 106 | 1.0 | true | 132.983928 | 144.164133 | 4 |
| 107 | 1.0 | true | 180.467920 | 199.538441 | 2 |
| 108 | 1.0 | true | 145.574981 | 152.160290 | 3 |
| 109 | 1.0 | true | 27.156312 | 64.520810 | 2 |

All 20 planned attempts were present and valid: `10/10/0` attempted/valid/invalid in each setting.
The negative signed response remained valid; sign was not used as an exclusion rule.

## Independent checks and result

- Manifest and observed identities matched exactly; no evaluation seed appeared.
- All 20 step logs contained 25 rows, for 500 finite rows total.
- Configured damping matched simulator `space_damping` in every attempt.
- Step-log, attempts, manifest, summary, checkpoint, config, protocol, wrapper, and runner identities
  were readable and hash-consistent.
- Independent median recomputation from the 20 raw valid values produced
  `tau=73.6528373524437`, exactly matching the persisted summary.

Gate A is `PASS`: the threshold artifact is complete, auditable, and met the frozen minimum-valid
criterion. This Pass means the formal evaluator may load this exact artifact. It does not mean
`probe-adjust` improves final coverage, that `tau` is physically real or optimal, or that the probe
identifies friction or damping.
