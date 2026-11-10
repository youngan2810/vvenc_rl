# Multi-clip training revision: evidence, research and decisions

Date:2026-11-10. Base source: `283a16ef47bfb7f175ad43b496a01d2e52313e8a`.
This is an experimental training revision, not an empirically established optimum.
The actor/critic architecture, categorical action space, reward and FPP1 codec
are retained. Real training of this revision has not run locally.

## Evidence motivating the change

The user's pilot archive SHA256 is
`f2c14556af18e1ae84a43ece2ae76f16af04a1280c488cc11348ac3f2f2f8995`.
It completed50 Train episodes/25 updates/12800 frames at the base source.
The independent local audit checked2591 manifest files,98 paired results,
50176 online/vanilla detection frames,219 streams/AU/header sets and213 saved
decode mappings. The four Validation vanilla streams stayed stable across
checkpoints and probes. These are archive checks, not new detector/native runs.

| Observation | Interpretation used here |
|---|---|
| Greedy quick-Val delta F1:0,+0.264,+0.878,+1.105 percentage points at EP0/20/40/50 | Positive pilot signal on two Val videos/two rates; not full Validation |
| EP50 actual bitrate +1.383% pooled over four greedy pairs | F1 gain is not an equal-actual-rate result |
| 962/1024 frames use one delta across all12 CTUs | Spatial benefit needs a simpler comparator |
| Sampled diagnostic mean delta F1 +1.324pp initially, +0.449pp at EP50 | Do not infer expected-policy improvement from greedy alone |
| 14/25 updates backtrack, final KL remains below0.03 | Trust guard works; larger/diverse batches deserve a controlled test |
| Each update contains512 frame transitions but only two terminal clip outcomes | Distinguish transition count from independent outcome coverage |

## Primary sources and what is actually adopted

- [PPO, Schulman et al.](https://arxiv.org/pdf/1707.06347): alternate policy
  sampling with optimization on a frozen collected batch. We preserve exact
  categorical action likelihoods, the existing per-head clipped extension and
  joint KL guard. The paper does not establish that4 clips or minibatch64 is
  optimal for this VVenC environment, nor prove our extension improves returns.
- [Andrychowicz et al., on-policy implementation study](https://arxiv.org/html/2006.05990v1):
  collection size, minibatch handling and learning rate affect results. Their
  continuous-control settings motivate an ablation, not a universal parameter
  recipe. We vary the collection/minibatch setup and leave learning rate1e-4,
  gamma1 and the other optimizer settings unchanged for this first test.
- [COMA, Foerster et al.](https://arxiv.org/html/1705.08926v3): a central critic
  can form an own-action-independent counterfactual baseline. We retain masking
  before action mixing and freeze it before fitting the current batch. Our
  actor target remains observed final clip delta minus the frozen baseline;
  unexecuted Q estimates never become invented codec rewards. The study's
  advantage recomputation advice is not copied in a way that would leak this
  batch's fitted outcome into its own baseline. This pass-wise recommendation
  comes from the on-policy implementation study, not from the COMA paper.
- [RL-RC-DoT, Gadot et al.](https://arxiv.org/html/2501.12216v2): block-level
  downstream auxiliary prediction motivates local representation supervision.
  Its x264 block precision, Gaussian parameterization and published results
  are not reproduced here. V4 retains measured, globally matched and
  area-conserved local TP/FP/FN targets; those are not causal CTU sensitivities.
  No per-CTU F1 is substituted for global clip F1. Paper conclusions are not
  evidence that the present12-CTU VVenC controller has learned spatial coding.

## Implementation

### Collection separated from GPU concurrency

`workers=2` remains the number of simultaneous GPU collectors.
`rollouts_per_update=4` is a separate setting: collect two waves of two clips
using the same immutable checkpoint, then perform one PPO update. Checkpoint
SHA is checked before/after each wave and again before optimization.
No old rollout replay, partial-wave PPO update or mixtures of policy versions.
Log actual clip count, unique videos, episode boundaries, waves and policy SHA.

Default minibatch changes32→64. With256 frames/clip and four PPO epochs,
a full new batch has1024 transitions and at most64 critic minibatch steps;
the old two-clip/32 configuration also had at most64 steps per update.
The new scheme performs fewer updates per equal episode budget. Both episode
count and optimizer/interaction counts must be reported in comparisons.
An ablation is available with rollouts2/minibatch32 under the same new source.

Batch size is cut at session cap, each325-episode epoch boundary and each
20-episode quick-Val milestone. Validation timing is episode-based, independent
of the number of optimizer updates. Full35 greedy pairs remain the only best
checkpoint selector. Pending Val commits atomically with the checkpoint.

Each wave uses the existing collector process cleanup. Budget exhaustion after
a wave abandons the whole update and keeps the last committed weights; future
sessions recollect the attempts. Reserve includes the number of waves.

### Diagnostics for spatial credit and representation

Before critic fitting, report advantage/baseline variation across CTUs within
each frame, variation of frame means, Q-bin spread and within-frame policy JS.
Report raw gradient RMS by source-image, local-state, global/history, spatial
fusion, policy and auxiliary parameter groups before clipping/Adam. These are
dependency/optimization diagnostics, not causal feature importance or a reward.

Each measured rollout also reports mixed-delta frame fraction and per-CTU
intervention statistics. Equal deltas do not imply equal absolute native QPs.
Sampled quick-Val changes relative to the initial fixed-seed probes are explicit
in the summary, so a greedy gain cannot silently stand in for sampling progress.

### Measured controls, never extra policy training

Post-training quick greedy pairs additionally encode/decode/detect:

1. Uniform−1, a declared fixed intervention.
2. Frame-uniform open-loop replay: replace each observed greedy action map by
   its modal CTU delta for that POC. Ties prefer smaller absolute delta, then
   lower integer. Detections/GT are not used to construct the maps.

The replay is not a separately trained frame-only policy: it holds the derived
future maps fixed even if encoder states change. Neither control is an
equal-actual-bitrate comparison; all actual rates and deltas are archived.
Both are explicitly non-trainable and selection-ineligible. They never enter
PPO, auxiliary minibatches, sampled seed selection or best checkpoint choice.
Full Val adds controls only when explicitly requested in the standalone CLI.
They add8 fixed encode/decode/detection cases per post-training quick Val.

Native preflight now checks sampled-live versus exact fixed-map replay bytes,
in addition to existing zero/nonzero/POC/parity gates. Identical replay maps
must reproduce the stream; identical streams must reproduce task metrics.
This requires confirmation on Kaggle; no new native binary/hash is accepted.

## Version, protocol and limits

Update contract: `frozen-multiclip-ctu-ppo-outcome-v4`.
Collection contract: `frozen-multiclip-waves-v1`. Model/state/schema remain
v4.2 architecture / causal-state1 / `vvenc-agent-v4/2`.
Old update-contract checkpoints, including pilot EP50, cannot migrate.
Keep the base pilot checkpoint/archive as a frozen scientific control.

No C++ changes, Gaussian rounding, forced nonzero, enforced spatial variation,
action centering after sampling, fabricated local bit labels or Test tuning.
Dataset/detector/reward/seed/FPP1 and golden hashes remain the same.
There is no claim yet that this revision learns better or fixes spatial
allocation. Pilot10 is an operational smoke test (three updates4+4+2), not a
learning-quality verdict. Only after review should it continue toward50 and
full Validation, with learning claims requiring measured controls/rate curves.

See [run instructions](AGENT_V4_MULTICLIP_RUN.md) and
`AGENT_V4_MULTICLIP_CHECKS.json` for local verification and its scope.
