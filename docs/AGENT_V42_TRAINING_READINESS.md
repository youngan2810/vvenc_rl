# Training readiness repairs (2026-11-10)

These changes repair numerical targets and interruption/evidence handling on
top of the FPP1 runtime. The categorical policy architecture, CTU action space,
learning objective, clip-level reward, optimizer settings and pilot50 budget
remain the same. Native source and golden binaries remain unchanged.

## Measured final-score targets

Previously, the trainer reconstructed final clip scores by reducing float32
prefix-score differences, then subtracted a float32 vanilla score. A CPU
regression reproduces a spurious delta of `7.748603820800781e-07` even when the
measured clip and vanilla scores are identical. Advantage normalization can
amplify small residuals; they must not masquerade as an observed improvement.
This is a numerical training defect, separate from the native drift failure
that stopped the earlier run before PPO.

Production final-score training now supplies the measured `metrics.score`
from each complete collected clip. The trainer checks agreement with the
telescoping rewards and a constant score per episode, subtracts the measured
paired vanilla score in float64, then casts the delta to model dtype. Identical
measured scores cancel exactly. The experiment records
`measured-float64-delta-before-cast-v1`. Legacy joint-GAE ablations keep their
existing float32 discounted targets. Model-only synthetic callers may still
test reconstructed returns without a collected measured score.

## Validation commits

Pending quick/full Validation now commits in the same `progress.json` update
as its checkpoint and PPO update. An interruption immediately after that commit
therefore resumes Validation before collecting more Train data. Startup also
recovers missed periodic quick Validation from an older same-source progress
snapshot. Initial quick, epoch-end full, and pilot-end quick scheduling retain
their original selection rules; sampled diagnostics never select best.
Detector asset initialization is inside the driver's evidence-export `finally`.

## Failure evidence

The generated notebook embeds standard-library bootstrap helpers from
`agent_v4/notebook_support.py`, allowing export before source cloning/Torch
setup succeeds. Setup, build/native gates and training cells preserve the
original exception and write `failure_traceback.log` plus an atomic
`VVC_AGENT_V4_KAGGLE_SESSION.zip`. Available setup/preflight/headroom evidence
is included without raw YUV or checkpoint files. When training started,
committed resume/results are included if available; the status records missing
progress, return code and export errors. A setup failure does not attach stale
training bundles from Working. Interrupted launcher subprocesses are terminated.
Hard VM termination still cannot guarantee a final ZIP.

## Run

Use the existing `VVenC_Agent_v4_Pilot_Kaggle.ipynb` filename, updated from this
source. Fresh session, T4x2, Internet ON, same three dataset inputs, `MODE='new'`,
`MAX_ADDITIONAL_EPISODES=50`, empty `RESUME_INPUT` and `SOURCE_COMMIT`.
Do not pin an older source or resume a previous experiment after source changes.
FPP1 paired vanilla and all native/zero/POC/source gates remain mandatory.

CPU regression results are recorded in `AGENT_V42_READINESS_CHECKS.json`.
They do not establish native all-clip determinism or real RL quality gain.
The broader Kaggle gates and actual50-episode pilot remain to be run.
