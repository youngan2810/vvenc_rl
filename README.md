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
| `scripts/rl/full_validate.py` | Paired vanilla/PPO validation without learning |
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

This revision changes KL stop ordering and versions the session protocol. Start
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
  --run-validation --validation-frames 256 --max-episodes 2
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
available evidence. The source commit, runner checksum, model/update versions,
dataset lock and training/validation configuration must match. Random policy
sampling and PPO minibatch permutations use deterministic per-episode/per-update
seeds, including on resume. Saved RNG states are supplementary; recovery does
not depend on a notebook restoring them.

Progress is committed only after fresh collection and successful checkpoint
creation. Interrupted episode/checkpoint attempts are retained under their old
names; retries use a new name. No partial rollout is reused for an update.
On startup, any completed epoch with missing Validation is processed before new
collection. A completed matching validation summary is reused; incomplete Val
is rerun in a new directory. This recovery is at epoch level, not at each Val pair.

## Evidence and diagnostics

The compact ZIP is a resume bundle, not the complete experiment archive. It saves:

- latest model and optimizer; epoch-end checkpoints; best validation checkpoint;
- progress with checkpoint chain, policy-parity errors, reward components,
  detection counts, action histogram and phase timings;
- episode summaries and access-unit bit counts; parent/collector logs;
- detailed rollouts, detections and bridge logs for the latest PPO update;
- available validation summaries.

All episode details remain in the live run directory. Export that directory
separately if every historical trajectory/detection file is needed. Large YUVs
and bitstreams are cleaned by default; existing keep flags control that behavior.

PPO logs separate policy loss, value loss, Gaussian entropy proxy, clip fraction,
average KL, maximum minibatch KL and the KL that triggered stopping. KL is checked
before taking another optimizer step, using the existing signed sampled estimator;
it is a stopping heuristic, not an exact trust-region bound. A negative average
estimate alone is not a failure. Never raise target KL merely to disable stopping.

Timings distinguish preprocessing, encoding/policy exchange, decoding/alignment,
detector loading, reference detection, decoded detection and reward joining.
Profile these before changing worker count, CPU threads or model batching.

## Verification boundary

CPU tests cover PPO updates, KL stopping without an extra optimizer step,
two-collector policy identity, GAE boundaries, interrupted Validation, portable
restore, progress integrity, epoch/session batching, and worker failure cleanup.
They do not establish real-video performance or a speedup on two T4s. Before a
long run, perform a two-episode smoke, restore it in a new session, and run paired
Validation on the real environment. Encoder QP/lambda behavior is unchanged in
this revision and still requires the planned real-encoder zero-action check.

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
