# PPO-QP Full Training v2 — research development

Branch full-training-dev is derived from the validated rl-agent-dev commit 5a76b04.
master and rl-agent-dev are preserved unchanged.

## Architecture

- PPO-Clip + GAE, single policy, five VVenC rate targets: 50, 80, 120,
  160 and 200 kbps. Only Delta-QP is learned. Partition/MTS stay vanilla.
- Actor v2 retains 3x4 CTU spatial positions, with a per-cell CNN head
  conditioned on global context and 2-D position. Critic uses global
  pooling. Pilot checkpoint Iteration 5 is NOT compatible with v2.
- Encoder Global State slot 1 is causal decision-order progress, rather
  than display-order POC divided by total frames. POC is still used as
  the identity key for Action, AU bitstream and decoded YOLO detections.
- Source MOV is converted to 480x320 YUV420 at 30 fps; windows are
  selected by frame index after fps normalization. A new window is a
  new episode with a fresh GOP / rate-control state.

## Kaggle requirements

Add Input BDD100K from robikscube, VCM_BDD100K_LOCK_v1 and
VVenC_Pilot_v1_2_QA_LOCKED_SNAPSHOT; enable GPU and Internet.
Build VVenC vvencapp and VVdeC v3.2.0 first.

Run 29 Python unit tests:
    python -m unittest discover -s scripts/rl -p 'test_*.py' -v

Check the VVenC live action hook using the existing 16-frame
run_online_smoke.py script before starting GPU training.

## Full training

The deterministic schedule contains 65 unique locked Train videos x five
rates per epoch = 325 on-policy episodes per epoch. It uses a seeded
window start per Train-video/epoch, each with fresh PPO sampling. By
default a window has 256 frames and PPO performs 5 epochs / minibatch64.

Example limited run (actual PPO updates, not a fake dry run):
    python scripts/rl/full_train.py --repo /kaggle/working/vvenc_rl \
      --decoder /kaggle/working/vvdec/bin/release-static/vvdecapp \
      --output /kaggle/working/ppo_qp_full_training \
      --epochs 3 --frames 256 --beta 0.1 --seed 20261103 \
      --max-episodes 2 --run-validation

Resume using the same output directory and arguments. Change only
--max-episodes to 0 to process the remaining complete schedule.
After each 325-episode epoch, --run-validation triggers paired
Vanilla/PPO evaluation on seven Val videos x five targets.

Download PPO_QP_FULL_TRAIN_LATEST.zip after each session. It contains
the latest checkpoint (checkpoint/latest.pt), progress.json and
optional best-validation report. On a new Kaggle Session restore
the checkpoint under the relative filename recorded in progress.json.
Use the accompanying Kaggle notebook for automated restoration.

Separate no-update validation:
    python scripts/rl/full_validate.py --repo /kaggle/working/vvenc_rl \
      --decoder /kaggle/working/vvdec/bin/release-static/vvdecapp \
      --checkpoint /path/to/latest.pt \
      --output /kaggle/working/validation_v2 \
      --frames 256

Use 7 Val videos and all five rate targets for meaningful
model selection. Never train or tune on the 100 locked Test videos.

## Scientific and operational limitations

- Beta=0.1 is the inherited pilot starting value; it has NOT been
  selected as optimal on Validation. GOP startup can distort the
  cumulative bitrate penalty, requiring sensitivity/ablation analysis.
- Each episode is a sampled window, not an entire 40-second clip.
- Spatial mapping remains fixed 480x320 / 4x3 CTUs / Delta-QP +-2.
- State features have explicit fixed numerical scaling, but no
  Train-set estimated mean/std normalization in this version.
- Frozen YOLOv5n supplies car pseudo-labels from source frames.
  F1 is detection-preservation F1, NOT human-labeled mAP.
- Validation compares the same target settings and reports ACTUAL
  bitrates; equal targets need not mean exactly equal realized rates.
- File-based PPO Python/C++ IPC has overhead, which needs profiling.
- If a Kaggle run is interrupted inside an episode, do not reuse a
  partially collected rollout. Quarantine the incomplete episode
  directory (and any orphan next checkpoint) before resume.
- Full Training performance gains remain unverified until real GPU
  runs and paired Validation.

Verified locally on modified source: 29/29 Python tests PASS, C++
vvencapp Release build PASS, zero-action bit-exact to Vanilla on
a 16-frame synthetic 480x320 clip, PPO produces nonzero Delta-QP and
a different bitstream, both decode with VVdeC. This is not an
end-to-end full GPU training result.
