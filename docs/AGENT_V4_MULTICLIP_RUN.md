# Run the new multi-clip pilot

**Current user-requested default:** [exactly one epoch /325 episodes](AGENT_V4_EPOCH1_RUN.md).
The notebook now sets `NEW_EPOCHS=1` and cap325, with a one-epoch resume guard
and explicit completion status. The smoke10 instructions below are historical
and explain the earlier pilot configuration; do not use their cap/epoch plan
for the updated one-epoch launcher.

The [follow-up source review](AGENT_V4_SOURCE_REVIEW.md) additionally requires
actual completed-episode metadata in checkpoints. Standalone Validation now
reports the candidate's true episode/update count, automatically enables the
declared post-training quick controls, and returns2 when pairs remain pending.

Use `notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb`. The standalone Downloads
copy has the same filename and content. Import the updated notebook into a
fresh Kaggle session, T4x2, Internet ON, with the same three inputs:
BDD100K Train00 MOV dataset, `bdd100k.zip` locked manifest and `snapshot.zip`
baseline protocol lock. Do not mount an old model/results archive for a new run.

## First run:10 episodes

```python
MODE = 'new'
MAX_ADDITIONAL_EPISODES = 10
RESUME_INPUT = ''
SOURCE_COMMIT = ''  # resolve current agent-v4-dev, record its exact SHA
NEW_ROLLOUTS_PER_UPDATE = 4
NEW_PPO_MINIBATCH = 64
NEW_ACTOR_LR = 1e-4
NEW_VAL_EVERY_EPISODES = 20
NEW_SPATIAL_CONTROLS = True
RUN_HEADROOM_DIAGNOSTIC = True
SESSION_HOURS = 11.75
```

Run All. The notebook runs unit tests, builds the original golden codecs,
checks hashes, synthetic preflight (including nonzero exact replay), Train0
headroom and initial quick Val before PPO. Every native encode must confirm
four threads/MaxParallelFrames1. Canonical build paths must stay unchanged.

Working output: `/kaggle/working/vvc_agent_v4_multiclip_run`.
Setup logs: `/kaggle/working/v4_multiclip_setup_logs`.
It does not use or overwrite `/kaggle/working/vvc_agent_v42_fpp1_run`.

If time permits,10 episodes produce3 updates:4 clips,4 clips,2 clips,
2560 Train frames. Each four-clip update uses two collector waves on GPU0/1,
all from the same checkpoint. Epoch boundaries and session caps may create
smaller final batches. A failed second wave cannot commit a partial update.

Quick Val runs at EP0 and EP10 for this smoke test. Each is four greedy pairs
plus eight fixed-seed sampled diagnostics. At EP10, the four greedy pairs also
receive two measured fixed-map controls each. These add cost but no PPO data.
Budget/native gates may stop the session before10; always inspect completed,
updates, pending Validation and return code rather than assuming cap reached.

Console should show `V4_FROZEN_BATCH_WAVE 1/2`, `2/2`, then a single
`V4_PPO_UPDATE_PASS`. Inspect gradient RMS, within-frame credit/probability
variation, mixed-delta frame fraction, sampled change from initialization,
and greedy-minus-frame-replay score. More spatial variation is not itself
a success criterion; measured task/rate outcomes decide.

## Evidence and continuation

Download `VVC_AGENT_V4_KAGGLE_SESSION.zip`, containing committed resume/results,
setup/preflight/headroom logs and session status. Setup failures also generate
this evidence archive before progress exists, except for hard VM termination.
Keep the old283a16e pilot50 archive separately as a frozen control.

After review, continuation from a genuinely completed10 to total50 uses:

```python
MODE = 'resume'
MAX_ADDITIONAL_EPISODES = 40
RESUME_INPUT = '/kaggle/input/YOUR_DATASET/VVC_AGENT_V4_RESUME.zip'
SOURCE_COMMIT = ''  # inherited from this new experiment archive
```

An extracted resume directory with `bundle_manifest.json` is also supported.
Do not assume10 completed if the first session stopped early. The cap is
ADDITIONAL episodes. All model/PPO/rollout/Validation settings come from the
verified archive, regardless of NEW_* values. A different source, reward or
batch config requires a separate new experiment, never an in-place migration.

Validation follows absolute completed-episode milestones20/40 plus the
session endpoint; full Val is35 greedy pairs at325/650/975. Pilot caps are not
part of the frozen experiment, so splitting a run at10 can add a partial-batch
update compared with one uninterrupted50 run. Do not compare runs by update
count alone or demand bit-exact training equivalence across different caps.

## Direct CLI and controls

```bash
python -u scripts/rl/agent_v4_train.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/vvc_agent_v4_multiclip_run \
  --preset content-task --epochs 3 --seed 20261109 --frames 256 \
  --validation-frames 256 --workers 2 --rollouts-per-update 4 \
  --ppo-minibatch 64 --actor-lr .0001 --val-every-episodes 20 \
  --validation-spatial-controls --reward clip-delta --rate-mode symmetric \
  --beta .1 --rate-tolerance 0 --credit-assignment counterfactual-final \
  --auxiliary .1 --auxiliary-mode task-outcome --max-episodes 10 --max-hours 11.5
```

Notebook budgets include setup; direct driver `--max-hours` begins at driver
startup. A fair two-clip ablation is a separate output/run with
`--rollouts-per-update 2 --ppo-minibatch 32`, otherwise identical new-source
settings. This is a declared ablation, not permission to automatically train it.

Standalone full35 Validation of a new-revision candidate uses its exact source
pin with `agent_v4_validate.py`. Optional `--spatial-controls` adds the two
fixed controls to all35 greedy pairs; `--quick` uses four greedy/eight sampled.
Controls and sampled seeds never select best. Old EP50 from283a16e requires
its original source/launcher for frozen evaluation, not this revision's loader.

This pilot is not a guarantee of learning improvement. Read the
[research decisions](AGENT_V4_MULTICLIP_RESEARCH.md) before interpreting it.
