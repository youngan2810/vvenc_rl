# Run exactly one epoch of the reviewed multi-clip agent

User-requested default,2026-11-10: replace the10-episode smoke cap with a new
one-epoch experiment. This launcher change does not alter the reviewed learning
algorithm, reward, learning rates, FPP1 codec or dataset split.

## Defaults

```python
MODE = 'new'
MAX_ADDITIONAL_EPISODES = 325
RESUME_INPUT = ''
SOURCE_COMMIT = ''
NEW_EPOCHS, NEW_SEED = 1, 20261109
```

All other defaults remain: two GPUs, four frozen-policy clips/update,
minibatch64, quick Val every20 completed episodes, spatial controls enabled,
headroom/native gates enabled and a11.75-hour session budget including setup.
Use the updated same-named `VVenC_Agent_v4_Pilot_Kaggle.ipynb` from Downloads,
T4x2/Internet ON, and the same three inputs. Start with a fresh Working run;
do not resume the old pilot50 or an older three-epoch experiment into this plan.

## What completes an epoch

-65 locked Train videos ×5 target rates =325 complete video-rate episodes.
-256 frames/episode =83200 training frame transitions, not83200 unique source
  frames. The same video's selected window is revisited at different rates.
-With the default cap and milestones,81 four-clip updates plus one single-clip
  tail produce82 updates. Other manually chosen intermediate caps may add
  partial-batch updates; update count alone is not the comparison budget.
-Quick Val at0/20/40/.../320; full35 greedy pairs at325. Native preflight,
  headroom and initial Val must pass before any PPO update.
-The final cell prints `EPOCH1_STATUS COMPLETE` only when325 episodes and the
  full35-pair Validation are complete and no Validation remains pending.
  Training return code0 alone does not establish epoch completion.

The schedule has **one epoch total**, so it cannot silently start epoch2 even
if the additional-episode cap is left at325 during resume.

## Multiple sessions

This is a total training target, not a promise that setup,325 paired episodes
and all Validation/controls fit one11.75-hour session. At each session end,
download the entire `VVC_AGENT_V4_KAGGLE_SESSION.zip`. For continuation, add
that complete ZIP as a Kaggle input; do not extract just the compact RESUME:

```python
MODE = 'resume'
MAX_ADDITIONAL_EPISODES = 325
RESUME_INPUT = '/kaggle/input/YOUR_DATASET/VVC_AGENT_V4_KAGGLE_SESSION.zip'
SOURCE_COMMIT = ''
```

The archive must belong to this reviewed one-epoch experiment. Configuration
and exact source pin are inherited. The driver clamps the cap to the remaining
one-epoch schedule, so resuming fromEP80 cannot train beyondEP325. If325 Train
episodes are done but full Val is pending, resume finishes Val before stopping.
The notebook verifies the outer manifest, both inner manifests, matching
progress, fresh exports and committed-evidence coverage before checking out
the pinned source. Old launchers/compact-only archives are not accepted by
this cumulative-evidence launcher. Original CLI compact resume remains available
for recovery, but does not claim to restore historical raw evidence.

## Evidence exported

The complete session contains initialization and every Validation checkpoint,
latest/best/pending, optimizer/RNG state, cumulative raw episode/Validation
evidence and per-session environment/native-gate/setup logs. Resume restores
both RESUME and RESULTS into the run, so the next RESULTS remains cumulative.
No previous ZIP is recursively embedded inside the next ZIP.

`RESULTS/reports/` contains `updates.csv`, `pairs.csv`,
`validation_history.csv`, checkpoint and case indexes, `completeness.json`,
PPO/Validation PNG plots and a latest-greedy mean actor delta-QP heatmap.
CSV precision/recall are computed from saved TP/FP/FN. Quick and full Validation
are plotted separately; sampled diagnostics retain their fixed seeds.
Plots are derived and optional if Matplotlib is unavailable; the report records
plotting status, while raw evidence export continues.

The last notebook cell prints **EPOCH1_STATUS** and **ARTIFACT_STATUS** separately.
Inspect `EXPORTS`: both resume and results must be `fresh`, with no export error.
Only a complete, fresh cumulative session is accepted for notebook continuation.
Keep downloaded session backups even though later healthy exports are cumulative.
Atomic local ZIP exports cannot ensure that a VM killed before persistent
Kaggle output is saved will preserve its currently writing batch.

See [export design and verification](AGENT_V4_EVIDENCE_EXPORTS.md).

Keep learning claims separate: read greedy/sampled progress, measured spatial
controls, actual bitrate and full Val deltas. More training provides more
observations; it does not guarantee useful CTU allocation or a codec gain.
See [research decisions](AGENT_V4_MULTICLIP_RESEARCH.md) and the
[source review](AGENT_V4_SOURCE_REVIEW.md).
