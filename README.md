# VVenC-RL: PPO-QP for Object Detection

Research implementation of PPO-Clip rate control on top of
[VVenC v1.14.0](https://github.com/fraunhoferhhi/vvenc), the Fraunhofer H.266/VVC
encoder. The current agent adjusts spatial delta-QP maps to preserve frozen
YOLOv5n car detections while following a target bitrate. Partition and MTS are
future extensions; the current implementation learns QP only.

This is the single project guide. Kaggle provides compute: check out a pinned
commit, build the codecs, attach the locked inputs and invoke the source CLI.
Training code must not be embedded or rewritten inside a notebook.

## Source layout and agent

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
