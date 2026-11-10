# Follow-up source review after multi-clip implementation

Date:2026-11-10. Reviewed parent: `ec3a418795eaf9ede85ece13e75b23899e764936`.
This is a correctness/performance follow-up, not another algorithm revision.
New10, four clips/update, minibatch64, reward, learning rates, architecture,
categorical semantics and FPP1 native settings are retained.

## Confirmed findings and fixes

### Trained standalone Validation candidates were labeled EP0

The standalone evaluator created a validation-only progress with an empty
`completed` list. The shared staging helper interpreted its length as the
training episode count. A trained candidate was therefore reported at EP0,
got the initial zero gate and did not automatically enable post-training quick
spatial controls. Training-driver Validation already had the correct count;
the defect affected standalone evaluation and provenance.

The old helper was reproduced with a candidate at iteration3/EP10: it reported
EP0. Checkpoints now record `completed_episodes` and production experiments
declare `checkpoint_metadata_version=completed-episodes-v1`. Resume checks it
against committed progress. Standalone Validation carries the actual count
and iteration into pending jobs, paths, configuration and reports. Three
updates cannot be assumed to mean12 episodes: the smoke cap produces4+4+2.
Unknown counts are rejected rather than guessed. The launcher rejects resume
from older experiments lacking the metadata contract before source checkout.

### Incomplete standalone Validation returned success

The evaluator ignored `validate_pending()` returning false at its time reserve.
It now emits `V4_STANDALONE_VALIDATION_INCOMPLETE`, returns exit code2, and
keeps pending progress/results for resume. Completed evaluation returns0.
Frozen/pending checkpoint SHA is also checked before launching any collector.

### Redundant actor computation in every PPO minibatch

The update computed actor features, then called the joint model again just
to obtain critic value/legacy presence logits. The second call reran the actor
CNN/attention graph unnecessarily. `Agent.value()` now evaluates only the
independent critic; auxiliary presence reuses the first actor tokens.
No parameter topology, objective, optimizer configuration or action rule changes.
Tests enforce that critic-only backward reaches no actor parameters and the
update does not call the joint actor/critic forward beyond parity checking.

## Review coverage and evidence

The review traced complete-episode reward joining and measured float64 deltas,
POC identity, own-action masking and pre-fit critic freezing, two-wave policy
immutability, failed-wave commits, GPU worker isolation, source/checkpoint
identity, compact resume/export, episode/epoch/session boundaries and the
exclusion of Validation/control data from PPO and checkpoint selection.
The known sparse-credit and sampling/argmax limitations remain empirical
learning questions rather than a claim that code tests prove task quality.

`AGENT_V4_SOURCE_REVIEW_EQUIVALENCE.json` records comparison with the saved
pre-review model implementation: all three architecture presets, both auxiliary
modes, fixed data/actions and identical update RNG. These are small synthetic
optimizer checks, not real codec training or a population-return guarantee.
The episode-label reproduction is also recorded there.

Final regression counts, log/notebook hashes and limits are in
`AGENT_V4_SOURCE_REVIEW_CHECKS.json`. Native live/fixed replay and actual GPU
training still require the Kaggle smoke run. Native C++, golden binaries and
the v3 branch remain preserved. Use the updated same-named pilot notebook;
do not migrate older source-pinned experiments into the reviewed source.
