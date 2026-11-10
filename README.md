# VVenC-RL: PPO-QP for Object Detection

## Agent v4.2 learning repair (2026-11-10)

The current `agent-v4-dev` requires a **new v4.2 run**, not a migration of the
10-episode v4.1 pilot. It adds masked CTU counterfactual baselines, observed
final-clip score credit, paired vanilla Train outcomes, action-conditioned
local TP/FP/FN auxiliary supervision and bounded auxiliary gradients.
Greedy Validation stays primary; two fixed-seed sampled probes expose the
sampling/argmax gap without selecting a lucky seed. All-zero stream drift
now fails on every paired result. Native codec source/binary is unchanged.

Read the [v4.2 learning contract and limitations](docs/AGENT_V42_LEARNING_REPAIR.md)
and [run instructions](docs/AGENT_V4_RUN.md). The Kaggle notebook includes a
Train-only native reproducibility/headroom diagnostic before the new pilot.
These repairs are not a claim of measured VVenC quality improvement.

## Agent v4.1 development history (2026-11-09)

Branch `agent-v4-dev` adds an isolated QP agent: categorical CTU actions,
causal encoder/history features, an optional source-luma CNN, train-only
car-presence supervision, independent critic, aligned clip-return objective,
two-GPU collection, paired Validation and checksummed resume/evidence exports.

Read [design and research](docs/AGENT_V4_DESIGN.md) and
[running/pilot instructions](docs/AGENT_V4_RUN.md). Start a **new** v4 run;
EP650/v3 checkpoints and the ongoing EP650 Test evaluation use their original
source. No v4 quality gain is claimed before real codec training/Validation.
The sections below describe legacy work, not v4 training results.

The [pre-pilot audit](docs/AGENT_V4_AUDIT_20261109.md) records additional
fixes for non-finite data, final-step KL overshoot, baseline decision order,
runtime source identity and resume archive integrity. These checks are not
evidence of a measured v4 task-quality improvement.

## Complete-line bridge repair and v3 EP471 recovery (2026-11-07)

The v3 run collected EP473 but stopped at committed EP471: the other worker's
bridge parsed a partially appended STATE line (28 fields instead of 46) during
EP472. A growing regular file can return a nonempty, unfinished `readline()`.
The bridge now buffers fragments until newline; malformed complete records and
idle timeouts still fail. Policy sampling, PPO, reward, codecs and schedule are
unchanged. Fragmented/full-line process tests preserve identical trajectories.

`--allow-bridge-io-upgrade` explicitly accepts only the verified original v3
EP471 checkpoint `17aced9b0449e3d27344b826b10bbf5dbde491a46397660e73d10444ddc82543`.
It checks a clean source diff restricted to this IO repair, backs up progress,
records original/new source configs, and preserves checkpoint bytes, Adam/RNG
state and all history. The first new update loads the original collecting
checkpoint config; subsequent checkpoints use the repaired source provenance.
Portable resume also works if interrupted before that first new update.
Do not edit pinned files in place or bypass the config/hash checks.

For this recovery, resume 471 with a bounded `--max-episodes 180`, which ends at
650 (179 additional episodes), then performs the normal seven-video/five-rate
validation. Recollect both EP472 and EP473; never reuse the uncommitted half-pair.
Keep three epochs/975 total. This IO repair does not establish improved learning
or detection quality; select the final checkpoint using paired Validation.

**Current branch: fresh-run v3 after the EP467 audit.** Training at this HEAD
uses `spatial-ctu-independent-ln-v3` and cannot resume the EP68/325/467 v2
weights. Preserve those exports and use their original pinned source
`e0b586c46f61e2ebbf131e4eb83f98817ff6e374` for legacy training/inference.
See [Fresh-run v3](#fresh-run-v3-after-the-ep467-audit) below for current settings.
Older stage-by-stage instructions describe the original experiment, not a
checkpoint migration to v3. `ppo_agent_legacy.py` is for old offline diagnostics.

Research implementation of PPO-Clip rate control on top of
[VVenC v1.14.0](https://github.com/fraunhoferhhi/vvenc), the Fraunhofer H.266/VVC
encoder. The current agent adjusts spatial delta-QP maps to preserve frozen
YOLOv5n car detections while following a target bitrate. Partition and MTS are
future extensions; the current implementation learns QP only.

This is the single project guide. Kaggle provides compute: check out a pinned
commit, build the codecs, attach the locked inputs and invoke the source CLI.
Training code must not be embedded or rewritten inside a notebook.

## Fresh-run v3 after the EP467 audit

The original run reached467episodes /234PPO updates. Complete EP325 validation
gained0.292 F1 percentage points while using3.31% more bitrate. All its
deterministic actions were-2. Later probes showed EP325/467 byte-identical to
fixed-2 on four sampled clip/rate pairs. On1024common probe states, fusion
tanh saturation was99.66% atEP68 and100% atEP325/467; EP467 value std was about
5.1e-5. These observations identify failure modes, not a proven single cause.
195/200later updates hit KL stopping, also stopping critic fitting in v2.
Large value loss alone does not prove critic gradients dominate the actor.
Frame/POC, AU accounting, actual weight changes and checkpoint history passed.

### Current changes

- Independent actor/critic parameters. The actor retains the local CNN,
  4x3CTU head, xy coordinates and8global encoder statistics; LayerNorm/SiLU
  replace the saturated fusion tanh. The critic consumes all36spatial+8global
  features through two normalized128-unit hidden layers and an unbounded
  scalar output. No downstream detector enters inference.
- Orthogonal initialization, actor output gain0.01 and critic gain1.0. Fresh
  deterministic actions start at zero; the codec gate verifies equality with
  vanilla and fixed-zero bitstreams.
- Keep Gaussian latent ->2*tanh ->round and the correct latent likelihood/
  Jacobian for PPO ratios. Exploration uses exact entropy of the actual five
  integer choices per cell, computed by Normal CDF at
  atanh(-.75,-.25,.25,.75), summed over12conditionally independent cells.
  Log Gaussian/quantized entropy separately. Latent PPO was not mathematically
  invalid; the old Gaussian entropy proxy did not measure integer exploration.
- Actor LR1e-4, critic LR3e-4, Adam eps1e-5; independent gradient clipping0.5.
  One serialized Adam with disjoint parameter groups preserves exact resume.
  `optimizer_steps` counts actor updates, `critic_optimizer_steps` counts value
  updates. KL stopping freezes actor gradients, including Adam momentum, while
  the critic finishes planned minibatches. Sampled KL is not a strict bound or
  a rollback. Keep gamma.99, GAE lambda.95, clip.2, epochs5, minibatch64,
  targetKL.03, valuecoef.5, entropycoef.01.
- Interleave all65Train videos before repeating a video at its next rate;
  each epoch still has exactly65x5=325pairs. Preserve window selection, split
  locks, QPA/profile, output-POC auditing, episode GAE boundaries, two frozen
  collectors and atomic progress/bundles. Sanitize inherited VVENC_RL settings.
- Reward stays frame-level pseudo-reference F1 (documented empty-reference
  rule) minus.1*absolute log cumulative rate error in policy decision order.
  Validation now reuses the exact frame reward function, true AU bits and
  each encode's policy decision order. It records mean training-style reward
  and discounted return using gamma from the checkpoint. Select checkpoints
  by `macro_ppo_discounted_return`, balanced over the35Val pairs. Clip F1,
  actual bitrate and the old final-rate score remain separate reported metrics.
  Deterministic Val differs from stochastic training; matching reward/gamma
  does not prove equal expected returns or improved rate-distortion performance.
  Assess actual-rate RD curves and fixed-map controls, not F1 alone.
- Original diagnostic replays explicitly use `ppo_agent_legacy.py`; new
  trainer/bridge reject old architecture checkpoints. Never disable source,
  config or hash checks to resume EP467 in v3.

### Research basis and limitations

[RL-RC-DoT, Gadot et al., CVPR2025](https://arxiv.org/html/2501.12216v2)
uses PPO, Gaussian block actions and a self-supervised task/rate objective,
plus a local reward-prediction auxiliary head. Our VVC/QPA, F1, encoder-order
and CTU-state adaptations differ from its x264/precision setup. The auxiliary
head is not implemented here. Do not claim an exact reproduction or transfer
its published gains to VVenC.

[Andrychowicz et al., ICLR2021](https://arxiv.org/html/2006.05990v1), sections3.2/3.3,
motivates independent value networks, initialization and normalization.
Its continuous-control results do not prove our LayerNorm/SiLU or quantized
entropy choices for VVC: these are hypotheses motivated by measured failures.
[PPO](https://arxiv.org/abs/1707.06347) and
[GAE](https://arxiv.org/abs/1506.02438) provide the retained objectives.
[Implementation Matters](https://arxiv.org/abs/2005.12729) motivates measuring
implementation choices. No downstream-task gain for v3 is established yet.

### Fresh-run checks and procedure

Run `python -m unittest discover -s scripts/rl -p 'test_*.py'` and then the
source CLI `scripts/rl/restart_preflight.py` with `--repo`, `--encoder`,
`--decoder` and a NEW `--output` directory. The real codec gate uses32synthetic
frames, requires the decoder output-POC auditor and performs no YOLO/optimizer
step. It verifies vanilla/zero/fresh deterministic PPO bitstream equality,
stochastic action transport, inference parity and actual decode mapping.
CPU tests cover probability/entropy gradients, synthetic state-dependent value
learning after actor KL stopping, independent parameters, old-checkpoint
rejection and exact seeded resume. Synthetic tests are not task-performance
evidence.

Local verification initially passed79 CPU tests; objective-alignment changes
add regressions for decision-order/discounted rewards, portable Val resume and
checkpoint selection when clip score and training return disagree. A real32-frame, four-encode native
Windows codec/bridge preflight passed; deterministic/stochastic bridge
log-probability/value errors were zero and both output-POC audits passed.
Windows binary hashes differ from the pinned Linux/Kaggle binaries. The
notebook repeats the gate using the canonical Kaggle codecs before collecting
real rewards. No new GPU task training or v3 quality gain is claimed here.

Use the pinned v3 notebook and a new run ID. The requested plan is three
epochs:975episodes, all65Train/7Val videos, five rates and256frames.
Collect10episodes first, resume the SAME run for90more to inspect100,
then continue the975episode schedule. Full35-pair validation occurs at
325/650/975. Session limits
count additional episodes. Do not change source, epochs, seed or PPO settings
on resume. A successful bounded session is not complete training/validation.
Monitor critic variation/explained variance against nonconstant returns,
integer histograms/entropy, actor/value step counts, gradients and rate/task
outcomes. A uniform map alone is not an automatic failure: compare controls.
Select/freeze by discounted Val return, then run100Test videos at the SAME full-clip protocol as
the completed baseline. Never replace full Test clips with256frames or tune
on Test. Three epochs with one seed is not a convergence guarantee. Source
pins before objective alignment, including c51ebc9, retain their original
selection rule; do not silently resume their runs with a changed source/config.

## Source layout and original experiment

| Location | Responsibility |
| --- | --- |
| `source/Lib/EncoderLib/RLObservation.h` | Extract encoder observations before applying the current action |
| `source/Lib/EncoderLib/RLQpAction.h`, `EncCu.cpp` | Read delta-QP maps and apply them in the encoder |
| `scripts/rl/ppo_agent.py` | Spatial actor, critic, PPO-Clip and GAE |
| `scripts/rl/ppo_policy_bridge.py` | Exchange observations/actions with the encoder |
| `scripts/rl/real_reward_pilot.py` | Encode, decode, run frozen detection and collect real rewards |
| `scripts/rl/reward_join.py`, `reward_metrics.py` | Align data by video/POC and compute detection/rate rewards |
| `scripts/rl/full_train_dual_gpu.py` | Current two-collector training entry point |
| `scripts/rl/training_session.py` | Resume, archive restore and missing epoch validation |
| `scripts/rl/full_validate.py`, `validation_session.py` | Paired validation and portable video-rate resume |
| `scripts/rl/benchmark_cpu_threads.py` | Fixed-policy two-worker 2/2 vs 4/4 thread benchmark |
| `scripts/rl/export_training_results.py` | Analysis archive, separate from resume state |
| `scripts/rl/test_*.py` | CPU unit and integration tests |

The spatial observation has shape `[3, 3, 4]`; the global vector has eight
features. The actor preserves the 3x4 grid through its convolutional features,
combines local features with global context and coordinates, and produces 12
Gaussian latent actions. These are tanh-bounded and rounded to integer delta-QP
values in `[-2, 2]`. PPO stores the sampled continuous actions and their log
probabilities; it does not reconstruct probabilities from rounded QP values.
The critic predicts a scalar state value for learning.

The current geometry is fixed to 480x320 frames at 30 fps. Each sampled video
window starts a new encoder episode. Frozen YOLOv5n predictions on source
frames are a consistency reference, not human ground truth; reported detection
preservation is not human-labeled mAP. Beta=0.1 is a provisional starting value,
not a validated optimum. Compare actual bitrate, not just target bitrate, when
reporting compression/detection improvements.

## Build VVenC

From this repository root, with CMake and a supported C++ compiler installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2
```

The default static Release encoder is `bin/release-static/vvencapp`.
Build VVdeC v3.2.0 separately and pass its executable with `--decoder`.
See the upstream [build guide](https://github.com/fraunhoferhhi/vvenc/wiki/Build)
and [usage guide](https://github.com/fraunhoferhhi/vvenc/wiki/Usage) for other
platforms and general encoder options.

## Experiment and scope

- PPO-Clip, spatial CTU actor v2, shared actor/critic features; QP only.
- Unchanged observations, 12 delta-QP values in [-2, 2], frozen YOLOv5n reward.
- 65 Train videos x 5 rates x 3 epochs = 975 episodes, 256 frames per episode.
- Two collectors use the same frozen checkpoint, one YOLO GPU each. Only after
  both finish and policy parity passes is there one combined PPO update on CPU.
- GAE is cut at each episode boundary. An epoch has 162 pairs and one singleton.
- Paired deterministic Validation uses 7 Val videos x 5 rates at every epoch end.
  The 100-video Test set is not used for training or checkpoint selection.

This revision uses the nonnegative k3 KL stop estimator and session protocol v3. Start
a **new official experiment**; keep the older 22-episode benchmark as historical
evidence. Its archive is intentionally rejected for continuation by this runner.
Freeze this source commit for every session of the new experiment.

## Preconditions

Use the same Kaggle dependencies and locked inputs as the tested pilot: Python,
PyTorch/torchvision with CUDA, two visible T4 GPUs, ffmpeg/ffprobe, VVenC built at
`bin/release-static/vvencapp`, and the locked VVdeC v3.2.0 executable. The dataset
manifest and `baseline_config.lock.json` must be available under `/kaggle/input`.
The runner checks the locked detector commit and weights before launching workers.
Build with the CMake commands above; no source edits are needed
on Kaggle. CPU-only unit tests can run without the videos, CUDA or codec binaries.

## Launch from Kaggle or a shell

After cloning and checking out the chosen exact commit, from the repository root:

```bash
python -m unittest discover -s scripts/rl -p 'test_*.py'

python -u scripts/rl/full_train_dual_gpu.py \
  --repo "$PWD" \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/ppo_qp_official \
  --epochs 3 --frames 256 --seed 20261103 --beta 0.1 \
  --run-validation --validation-frames 256 --max-episodes 2 \
  --encoder-threads 4 --decoder-threads 4 --export-analysis
```

Use `--max-episodes 100` for bounded sessions after checking the smoke output.
It is an upper bound: a session may end one episode earlier to preserve the same
pair grouping as an uninterrupted run. `0` means finish the remaining schedule.
No wall-clock timeout is imposed by this flag; allow time for epoch Validation.
Do not run two trainer parents against the same output directory simultaneously.

## Resume

In the same filesystem, invoke the same command with the same output folder.
To resume on a fresh Kaggle session, attach `PPO_QP_FULL_TRAIN_LATEST.zip`, use a
fresh output directory, and add:

```bash
--resume-archive /kaggle/input/YOUR_CHECKPOINT/PPO_QP_FULL_TRAIN_LATEST.zip
```

The source restores weights, optimizer, progress, epoch-end checkpoints and
pair-level validation progress. The source commit, runner checksum, model/update versions,
dataset lock and training/validation configuration must match. Random policy
sampling and PPO minibatch permutations use deterministic per-episode/per-update
seeds, including on resume. Saved RNG states are supplementary; recovery does
not depend on a notebook restoring them.

Progress is committed only after fresh collection and successful checkpoint
creation. Interrupted episode/checkpoint attempts are retained under their old
names; retries use a new name. No partial rollout is reused for an update.
On startup, any completed epoch with missing Validation is processed before new
collection. Each completed video-rate pair is committed atomically and the compact
ZIP is refreshed before the next pair. Resume skips completed pairs and reruns an
interrupted pair from the beginning. Only videos with pending rates need new
reference YUV/detections. An incomplete legacy directory without pair progress is
retained and retried under a new name. Checkpoint, protocol, codec binary hashes
and thread counts must match; completed matching summaries are reused.

## Evidence and diagnostics

Two archives have different purposes:

- `PPO_QP_FULL_TRAIN_LATEST.zip`: latest model/optimizer, epoch-end and best
  checkpoints, training progress/metrics, validation progress and summaries.
  Refreshed atomically after each PPO update and each completed validation pair.
  It contains no accumulated episode logs, trajectories, detections or YUV files.
- `PPO_QP_ANALYSIS.zip`: JSON/JSONL/CSV/log/text evidence currently present in
  the run directory. Export once at a session boundary with `--export-analysis`,
  or run `python scripts/rl/export_training_results.py --run-dir RUN_DIRECTORY`.
  It excludes model weights, action exchange files, YUVs and bitstreams.

Download **both archives after each session**. Compact restore does not recreate
historical detailed logs; retain previous sessions' analysis archives separately.
Training progress still carries cumulative episode/update metrics. Large YUVs
and bitstreams follow the existing cleanup/keep options.

PPO checks KL before the next optimizer step. With
`x = new_logp.detach() - old_logp`, `approx_kl` is
`mean(expm1(x) - x)` (k3, clamped at zero for floating-point roundoff).
`approx_kl_signed` retains `mean(-x)` (k1) for comparison. Stop diagnostics include
`max_minibatch_kl`, `stop_kl` and `stop_kl_signed`. The probability ratio used by
the PPO loss is **not detached**. Policy/value loss, entropy proxy, clip fraction
and optimizer-step counts are also logged.

The k3 estimate is nonnegative per sample but remains a sampled stopping
heuristic, not an exact trust-region bound or a guarantee of lower variance in
every regime. The default target KL remains 0.03; inspect both estimators and
optimizer-step counts in a new pilot rather than raising the threshold just to
suppress stopping. A zero delta-QP map is a valid action and is accepted.

Timings distinguish preprocessing, encoding/policy exchange, decoding/alignment,
detector loading, reference detection, decoded detection and reward joining.
Profile these before changing worker count, CPU threads or model batching.

## CPU thread benchmark

`--encoder-threads` and `--decoder-threads` default to 4/4 and propagate to every
collector and paired validation encode/decode. Counts are part of the frozen
experiment configuration; changing them requires a new run. The historical
baseline used 4/4, so 2/2 is a benchmark candidate, not a new accepted anchor.

After obtaining a smoke checkpoint, compare two concurrent collectors with the
same frozen policy, video windows and paired seeds, without PPO updates:

```bash
python -u scripts/rl/benchmark_cpu_threads.py \
  --repo "$PWD" \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --checkpoint /kaggle/working/ppo_qp_official/checkpoints/CHOSEN_CHECKPOINT.pt \
  --output /kaggle/working/thread_benchmark \
  --train-indices 0 1 --frames 256 --bitrate-kbps 120 --repeats 3
```

The benchmark alternates profile order and records pair wall time, phase timings,
actual bitrate, detection preservation and action histograms in
`CPU_THREADS_BENCHMARK.json`. Compare the medians and the encoding outcomes:
fewer threads are not necessarily faster, and changing threads may change the
bitstream. This small benchmark does not establish speed across the full dataset.

## Verification boundary

CPU tests cover PPO updates, KL stopping without an extra optimizer step,
both KL estimators and their Gaussian reference, actor gradients, thread forwarding,
two-collector policy identity, GAE boundaries, pair-level interrupted Validation,
portable restore without raw files, separate resume/analysis archives, progress integrity, epoch/session batching, and worker failure cleanup.
They do not establish real-video performance or a speedup on two T4s. Before a
long run, perform a two-episode smoke, restore it in a new session, and run paired
Validation on the real environment. Encoder QP/lambda behavior is unchanged in
this revision and still requires real-encoder zero/positive/negative-action checks,
including QP/lambda consistency under QPA. Baseline-lock content integrity also
needs hardening before the long official run; matching the declared lock hash
alone does not prove that its contents are unchanged.

## Upstream attribution and license

This project extends VVenC; the original encoder is developed by Fraunhofer HHI
and the [VVenC authors](AUTHORS). See [LICENSE.txt](LICENSE.txt) for the terms of
use. Third-party notices and licenses remain inside `thirdparty/`.

When referencing VVenC in academic work, use the upstream citation:

```bibtex
@InProceedings{VVenC,
  author    = {Wieckowski, Adam and Brandenburg, Jens and Hinz, Tobias and Bartnik, Christian and George, Valeri and Hege, Gabriel and Helmrich, Christian and Henkel, Anastasia and Lehmann, Christian and Stoffers, Christian and Zupancic, Ivan and Bross, Benjamin and Marpe, Detlev},
  booktitle = {Proc. IEEE International Conference on Multimedia Expo Workshops (ICMEW)},
  date      = {2021},
  title     = {VVenC: An Open And Optimized VVC Encoder Implementation},
  doi       = {10.1109/ICMEW53276.2021.9455944},
  pages     = {1-2},
}
```

**Copyright (c) 2019-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVenC Authors.**

**All rights reserved.**

**VVenC® is a registered trademark of the Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.**


## Alignment repair and official EP58 recovery (2026-11-05)

The old three-sample luma-similarity abort was a false positive on the official
200 kbps episode 60. A locally compiled, output-instrumented VVdeC v3.2.0
confirmed output POC 0..255 for both archived episodes 59 and 60. Its decoded
YUV matched each archived canonical YUV byte-for-byte. Episode 60's globally
averaged MSE for shifts -1/0/+1 was 44.732694/34.259238/43.690946, despite two
of the three old samples crossing the heuristic threshold. This is alignment
evidence, not a completed PPO update or a detection/bitrate improvement claim.

Training now requires a separate POC auditor beside the canonical decoder.
The canonical encoder/decoder and detector settings remain locked. The auditor
records POC immediately after each progressive frame is written, including
frames released during decoder flush. Every training and validation decode
must have output indices/POCs 0..N-1 and identical canonical/auditor YUV bytes.
The training collector also checks luma over all interior frames; a strong
systematic shift still fails. Static/tied frames cannot independently prove a
shift. Logs, mapping JSONL and hashes are retained, and temporary audit YUV is
removed. This adds a second CPU decode per bitstream; measure session runtime.

Build the auditor from the clean upstream VVdeC v3.2.0 checkout used to build
the canonical decoder (commit 649f0b2fafee977c998d7e4d674f8b88d952e3a3):

```bash
python scripts/rl/build_vvdec_audit.py \
  --source /kaggle/working/vvdec_v3_2_0 \
  --work /kaggle/working/vvdec_poc_audit_build \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp
python -m unittest discover -s scripts/rl -p 'test_*.py'
```

Use a new --work directory; the builder refuses overwrites. It produces
vvdecapp_poc_audit beside vvdecapp and records its build provenance. It does
not replace the canonical decoder. Pin this repository's repaired commit
when building/running; do not modify source in notebook cells.

The explicit --allow-alignment-upgrade flag accepts only the verified official
58-episode/29-update ec6 checkpoint (SHA256
96affafcbb6e495dbe4d6f94d046900acdf08b22f40640eb8ded47f241b6a115).
It checks the original progress and checkpoint/config before writing anything,
requires a clean committed source tree, checks the source diff against an
alignment-only allowlist and exact collector/runner/validation transformations,
and permits only source-commit and runner-hash config changes. PPO, reward,
action, encoder source, thread counts, schedule and hyperparameters are not
migrated. The original checkpoint bytes stay unchanged. Original progress is
saved as progress_before_alignment_upgrade.json; the committed progress records
both configs, boundary and source-diff hash in alignment_upgrade. The old
checkpoint loads with its original config; new PPO checkpoints use the repaired
source config. Later source changes require a separately reviewed upgrade.

After restoring the verified recovery bundle's INNER
PPO_QP_FULL_TRAIN_LATEST.zip to a known input path, run only the 58->60 smoke:

```bash
python -u scripts/rl/full_train_dual_gpu.py \
  --repo /kaggle/working/vvenc_rl_source \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/vvc_ppo_qp_official \
  --resume-archive /kaggle/working/official_resume_input.zip \
  --allow-alignment-upgrade \
  --epochs 3 --frames 256 --seed 20261103 --beta 0.1 \
  --run-validation --validation-frames 256 \
  --encoder-threads 4 --decoder-threads 4 \
  --max-episodes 2 --export-analysis
```

The output directory must be empty for archive restore. Same-runtime resume
omits --resume-archive. Both episodes 59 and 60 are collected afresh from the
unchanged checkpoint 58 before update 30. Require completed=60, updates=30,
valid mapping/byte parity, successful YOLO/reward, and a committed new checkpoint
before longer sessions. Save the new rolling resume ZIP and analysis archive.
No epoch validation is expected before 325. Official full training/test remain
unfinished; a fresh GPU smoke after this repair is still required.

The old official notebook v1.1 pins ec6 and requires exact manifest/config/hash
matches. It cannot directly import this source upgrade. Its orchestration and
manifest export must be updated to call the audited upgrade and carry its
provenance; do not change hashes or disable checks to make it accept a bundle.
A source CLI smoke above does not by itself create a notebook-compatible
OFFICIAL_SESSION_EXPORT manifest. Keep original archives and export provenance.


## Isolated EP467 policy diagnostics

`scripts/rl/diagnose_policy.py` diagnoses saturation without advancing official
training. Its source commit is separate from the official e0b586c training pin.
The original PPO, reward, bridge, encoder and training scripts are unchanged.
Do not migrate official checkpoints merely to use this diagnostic addition.
Build/reuse the canonical encoder and inference bridge from a clean worktree at
e0b586c46f61e2ebbf131e4eb83f98817ff6e374, retaining the existing codec hashes.
Run the new diagnostic CLI from its separately pinned tools checkout/worktree.

Required inputs are the verified 68-episode EXPORT_2 and 467-episode EXPORT_3,
each as an intact official export ZIP or a Kaggle-expanded directory containing
its manifest. The reader supports nested ZIPs and expanded nested folders.
It does not execute anything from an archive and loads tensors weights-only.

```bash
python /kaggle/working/vvenc_diagnostic_tools/scripts/rl/diagnose_policy.py \
  --mode offline --export2 /kaggle/input/phase-3 \
  --export3 /kaggle/input/phase-4 \
  --output /kaggle/working/vvc_ppo_ep467_diagnostics
```

Offline output includes exact post-quantization Gaussian action probabilities,
conditional discrete entropy, continuous Gaussian entropy, deterministic action
maps, and actual PPOTrainer.update replays for update35 (checkpoint68) and
update164 (checkpoint325), with policy/value gradient contributions and per-step
clipping/KL observations. Updated replay models/optimizers are discarded. Input
checkpoints are checked before/after and never overwritten or exported as new
training checkpoints. Replays are diagnostic, not additional official updates.

Recorded EP325 validation actions are verified against checkpoint325. EP68/467
actions on those stored states are labeled offline comparisons, not fresh
validation trajectories. Piecewise-linear comparisons of F1 at equal bitrate
use only the overlapping actual-rate range per video; they are not BD-rate or
held-out Test results. CPU integration on the real exports reproduced optimizer
step counts 4/6 for updates35/164 with original checkpoint hashes unchanged.

For small fresh probes, attach the same BDD100K and baseline lock inputs and
build the separate output-POC decoder auditor as documented above. At least one
CUDA GPU is needed for frozen YOLO. Default probes use validation indices0,1,
256frames, rates80,120: 32 encodes total (vanilla, fixed -2/-1/0/+1/+2,
deterministic checkpoint325 and checkpoint467). No Train/Test videos are encoded
in this stage and no optimizer is run in the online stage.

```bash
python /kaggle/working/vvenc_diagnostic_tools/scripts/rl/diagnose_policy.py \
  --mode online --output /kaggle/working/vvc_ppo_ep467_diagnostics \
  --repo /kaggle/working/vvenc_rl_source \
  --encoder /kaggle/working/vvenc_rl_source/bin/release-static/vvencapp \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --val-indices 0,1 --rates 80,120 --frames 256 --threads 4
```

Each completed method is committed atomically to probe_progress.json and can
resume with identical settings. Re-runs preserve incomplete attempts under fresh
names. The diagnostic compares both exact training-style mean frame/cumulative
rate rewards and aggregate validation scores, preserving policy decision order.
It requires zero-map/vanilla bitstream equality. For a policy issuing all -2,
it requires equality with the fixed -2 bitstream before interpreting scores.
Large decoded YUV is removed after each decode; logs, hashes, action maps,
detections and small bitstreams are kept. Report exports must exclude temporary
YUV and be named DIAGNOSTIC, not OFFICIAL_SESSION or training resume archives.
