# Experiment: <short title>

## Metadata

- Date:
- Stage / gate:
- Status: designed | running | pass | patch | repeat | stopped
- Base commit, dirty paths and source-file SHA-256 snapshot:
- Raw output directory (new, write-once):
- Data role: debug | calibration | evaluation | exploratory reuse
- Current source establishing this question (raw config/manifest + record):

## Question

What single question can this experiment answer?

## Expected outcome and rationale

What do you expect, and why?

## Variables and controls

- Independent variable:
- Dependent metric / observable:
- Constants and controls:
- Primary baseline and auxiliary reference, with rationale:
- Independent unit / repeated observations / pairing key:
- Uncertainty summary at independent-unit level:
- Calibration/evaluation separation and prior exposure:
- Seeds (environment, policy and other RNGs):
- Known confounders:

## Success and stopping criteria

- Pass requires:
- Patch when:
- Repeat when:
- Stop when:
- Maximum wall time / device / threads / episodes / steps / spend:
- Pilot necessity and smallest measurement-path check:

## Environment

- OS / architecture:
- Python:
- Packages:
- Environment ID and arguments:
- Source version / commit, dirty-source hashes and upstream revision:
- Weights / compatibility config / calibration artifact hashes:
- Actual interpreter and complete dependency snapshot:
- Preflight output and known drift from historical environment:

## Procedure

1.
2.
3.

Exact command:

```bash
# Mark proposed/not executed until this exact command has current execution evidence.
```

Command evidence: exact argv, timestamp, exit status, stdout/stderr path, source snapshot and
persisted readback. Link the report; distinguish import/help checks from an actual episode run.

## Failure and exclusion ledger

| Attempt / pair | Planned role | Status | Failure class / reason | Inclusion decision | Replacement run |
|---|---|---|---|---|---|

Retain timeouts, setup errors, invalid probes and incomplete pairs. Invalid probes use the frozen
fallback and remain in the primary evaluation; calibration invalid-x exclusions follow its own
protocol. Do not exclude zero/negative outcomes. Record “none” only after manifest reconciliation.

## Raw observations

Record outputs before interpreting them.

## Checks

- [ ] Inputs and configuration match the design.
- [ ] Outputs have expected shapes, types, and ranges.
- [ ] A clean-process rerun was attempted.
- [ ] Failures and warnings were preserved.
- [ ] No private or oversized artifact is staged for Git.

## Results

Add tables or figures with metric definitions and units where applicable.

## Interpretation

What does the evidence support?

## Limitations

What remains unknown or could provide another explanation?

## Understanding check

Explain the interaction loop, variables, metric, seed behavior, and reproduction steps in your own
words.

## Gate decision

Run status:
Measurement validity (checks and unchecked limits):
Scientific hypothesis (supported / unsupported / inconclusive):
Understanding status (do not infer from automated tests):

Pass | Patch | Repeat | Stop

Reason and next action:
