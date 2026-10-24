# G1 walk differentiable-physics SHAC baseline

**Status: experimental and not yet stable.** The latest formal run completed 25
updates and failed during update 26 with `Nan grad in qpos or dofs_vel found at
step 91`. There is no completed mature PPO reference, formal SHAC checkpoint or
sample/time-efficiency result. The [handoff](docs/SHAC_HANDOFF.md) and
[evidence archive](artifacts/shac/2026-10-24/README.md) preserve the exact state
before deleting the original server directories. Paths under the old server's
workspace/data-disk directories below are historical; use the handoff for a fresh clone.

This branch implements pathwise actor-critic through Genesis rigid-body physics.
The PPO runner and SHAC use the same G1 walk robot, all 19 reward terms and their
aggregation, observation/action definitions, PD controller timing, randomization,
MLP architectures and initial network weights. This is a walking baseline; it
does not implement an AMP discriminator or a learned dynamics model.

The engine is pinned to Genesis v1.4.5 commit
`1b6dfa927f3836d49cf80df7a024116499e68d15`. `prepare_shac_engine.py` applies a
reviewable patch only inside this checkout's `.deps/genesis`. It adds exact
link-velocity VJPs and a net-contact-force output adjoint using the existing
implicit constraint solve. Source hashes are checked before each run. The
original project and its installed engine are preserved.

The patch also corrects the analytic plane-cylinder geometry reverse. A cylinder's
radial support slides with its orientation; treating its local witness as constant
omits that derivative. The original multi-contact forward is retained. Each accepted
contact records its discrete detection ID, and the reverse rebuilds the support,
inertial-frame perturbation axes, rotations, depth correction and smooth refinement.
Its local geometry VJP uses Torch autograd in batches of at most 2048 seeded
contacts, then accumulates into the rigid solver's existing adjoints. This avoids
Quadrants' large reverse-kernel launch stack at 4096 environments. Only occupied
contacts with exactly nonzero output seeds are gathered; no finite differences or
extra physics transitions are used during training. Nonfinite inputs or gradients
raise before consuming that chunk's seeds.

Both algorithms share GJK contacts, disabled box-box detection, pyramidal cones,
no hibernation/noslip, 100 constraint iterations, tolerance `1e-8` and line-search
tolerance `1e-6`. Independent `differentiable_contact=True` selects AD's forward
contact generation for PPO without allocating an AD tape. The shared buffers
allow 64 collision pairs and 128 resolved contacts per environment; overflows
raise immediately before reward/reset/optimization. These numerical settings support the contact adjoint; they
should be recorded when comparing against an older stock PPO run. Zero-copy
setters are disabled because v1.4.5 mutates cached Torch views across grad modes.
The isolated PPO also corrects timeout bootstrap to gamma times value, gives
physical falls precedence, and logs raw episode rewards.

SHAC uses reparameterized Gaussian actions, a frozen target critic with state
derivatives, a 24-step discounted actor objective, TD(lambda) critic targets,
target retention 0.4 and clipped actor/critic gradients. Resets wait until the
backward pass. Post-terminal physics transitions are counted even though their
rewards and critic samples are masked. Full solver checkpoints restore contact
caches and warm starts after differentiation. Actor learning rate is fixed;
PPO keeps its native KL adaptation. Both start at the same configured rate and
share the entropy schedule.

Snapshots also preserve pending tracked setters: Genesis clears the gradient
tape on reset but keeps queued targets. Backward must retain the final pending
push for the next window. Continuous external-wrench randomization is rejected
until its replay is implemented; the original G1 walk configuration leaves it
disabled. Velocity pushes and periodic PD randomization are preserved.

Torch's unused CUDA cache is released before the full solver checkpoint as well
as before reverse and restore. Quadrants needs device scratch even when exporting
the checkpoint to CPU NumPy arrays. Snapshot memory counters record device free
bytes and Torch reserved bytes before and after that flush. The formal 128-env
evaluator stays resident; a pilot with the default 16-env debug evaluator does
not establish that the formal memory footprint is safe.

PD substeps request the original core state without allocating unused link
velocity, angular velocity, contact force and origin-offset outputs. The joint
position/velocity read nodes and their gradients remain tracked. Observation
and reward reads keep all outputs and their VJPs. This reduces live tape memory
without reducing the number of environments or the differentiation horizon.

## Validation and execution

Run on `ssh limh24`, from `/home/limh24/workspace/Genesis-Humanoid-shac`
(branch `codex/genesis-shac`). The original checkout is preserved:

```bash
# On limh24, keep simulation caches and temporary files on the data disk.
source /mnt/big_disk/limh24/genesis-shac-20261023/runtime_environment.sh
uv sync --frozen --python 3.12
.venv/bin/python examples/prepare_shac_engine.py
.venv/bin/python -m pytest -q tests/test_shac_returns.py tests/test_ppo_timeout.py tests/test_walk_quality.py tests/test_walk_evidence.py tests/test_walk_comparison.py tests/test_genesis_scene_config.py tests/test_leggedrobot_randomization.py
.venv/bin/python examples/validate_shac_gradients.py --warmup-controls 8 --epsilon 0.06
.venv/bin/python examples/validate_walk_equivalence.py
```

The full acceptance artifact also requires contact-basis VJP checks and actual
stochastic actor updates. Formal runs fail closed unless their engine/source
hashes, robot assets, dependency versions, GPU identity, precision and all
required validation flags match. Float64 diagnostics cannot authorize float32
training. Failed diagnostics must remain
available; finite-difference checks at contact transitions require care about
active sets and float32 solver residuals.

Gradient fixtures are diagnostics, separate from the fixed policy-quality evaluator.
The float32 check uses eight nominal warmup controls and a `0.06` parameter
perturbation to retain the same contact branches while resolving the finite-difference
signal. It checks all nine horizon/output pairs, repeated identical inputs and
contact detection IDs, and records solver residuals and cylinder radial norms.
The older 24-control fixture and smaller perturbations remain as failed diagnostics;
they are not silently turned into passes. Contact-switching or near-degenerate
geometry is not guaranteed to have a smooth finite-difference limit.

```bash
# Actor/update and capacity proof with the complete formal evaluator, debugging only.
.venv/bin/python examples/run_walk_benchmark.py --algorithm shac --debug --full-evaluation \
  --num-envs 4096 --iterations 10 --output logs/shac/actor_debug
.venv/bin/python examples/assemble_walk_gate.py \
  --gradients logs/shac/gradient_evidence.json \
  --equivalence logs/shac/equivalence_gate.json \
  --actor-run logs/shac/actor_debug

# Debug only: this cannot establish sample efficiency.
.venv/bin/python examples/run_serial_walk_comparison.py --debug \
  --num-envs 4096 --iterations 2 --output logs/shac/debug_pair

# Formal paired seed: PPO finishes before SHAC starts.
.venv/bin/python examples/run_serial_walk_comparison.py \
  --num-envs 4096 --iterations 50000 --seed 0 --output logs/shac/formal_seed0
```

Every child acquires a GPU lock and checks existing compute processes before
initializing CUDA. The serial parent never initializes CUDA. It stops on a
failure and does not kill unrelated processes.

## Quality and accounting

The reference is the median of the final five scheduled evaluations of the
4096-environment, 50,000-iteration PPO run. Fixed command profiles and a separate
20-second, 128-environment evaluator measure survival, command tracking error,
return and foot slip. Three consecutive evaluations must meet the predeclared
matched-quality rule. Absolute minimum quality prevents a weak PPO reference
from lowering the target. The complete rule is saved in each run's config.

SwanLab project `g1-walk-shac-vs-ppo` provides `quality_by_steps/*` against actual
new training transitions and `quality_by_time/*` against wall time including
startup, compilation, evaluation and logging. Evaluation transitions and active
training time are separate counters. Forward physics substeps are separate from
control transitions; backward replay consumes computation/time, not new data.
Debug runs have explicit debug names/tags and cannot enter formal comparisons.
`--debug --full-evaluation` also uses the formal evaluation interval and
50,000-iteration entropy schedule, preserving the initial formal trajectory
while bounding only the debug run's duration.

`compare_walk_runs.py` refuses pilots, mismatched task/engine/source/runtime
identities, changed acceptance code/protocol or unfinished runs.
It reports whether SHAC actually attained the same quality with fewer steps and
less time. An implementation or a gradient check alone is not evidence of an
efficiency improvement.

After both formal runs, comparison also saves PNG/SVG figures against transitions
and total elapsed time and publishes them as a directly viewable SwanLab comparison
run. The local conclusion survives a cloud upload failure. Missing, duplicated or
nonfinite evaluation records cannot establish three consecutive quality confirmations.
