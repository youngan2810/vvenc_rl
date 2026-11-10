# Cumulative evidence for the one-epoch Kaggle launcher

2026-11-10. Addresses the three independently reproduced export gaps at
`cc341e3`: missing initial/quick checkpoints, raw-history loss across compact
resume, and stale RESULTS suppressing fresh failure evidence.

## Contract

`experiment.artifact_contract = cumulative-evidence-v1` identifies new runs.
No actor/critic, PPO objective, reward, sampling, optimizer, native codec,
dataset split or evaluation/checkpoint-selection protocol changes in this patch.
The source commit and experiment metadata necessarily change for a fresh run.

RESUME now retains initialization, all completed Validation candidates,
latest, best and pending checkpoints, with existing checksummed atomic writes.
RESULTS retains raw non-YUV/non-IPC evidence and derived reports. Each complete
session ZIP has a manifest over payloads/status and distinguishes fresh, stale
and missing exports. Resume and results exports run independently. Any export
error also includes current raw failure evidence; an old archive is explicitly
marked stale and rejected for normal notebook continuation.

The notebook accepts the entire prior `VVC_AGENT_V4_KAGGLE_SESSION.zip` for
resume. Before restoring, it verifies all outer and inner file hashes, safe and
unique names, equality of overlapping files, status/progress counts, source
identity and completeness. Restoration writes no files until these checks pass.
Both compact state and raw evidence are restored; subsequent RESULTS is
cumulative. Checkpoints from quick milestones remain available for later
matched evaluations. There is no nested chain of old session ZIPs.

Session-specific environments, native gates, headroom and setup logs are copied
into `session_history/<session-id>/` before export and survive continuation.
The restored parent archive hash is recorded for provenance.

## Reports

- `updates.csv`: all recorded PPO metrics and cumulative frame transitions.
- `pairs.csv`: committed Train and completed Val greedy/sampled pairs and
  measured controls; task counts, derived precision/recall, actual rate and
  paired vanilla/deltas. Interrupted/uncommitted attempts stay in raw evidence
  but are excluded from these performance tables.
- `validation_history.csv`: recorded quick/full aggregates and seeded probes.
- `checkpoint_index.json/csv`, `case_index.json`: pointers for replay/audit.
- `completeness.json`: required committed raw artifacts, rollout checksum/POC
  coverage, validation pair counts, checkpoint checksums and plotting status.
- `ppo_learning.png`, `validation_learning.png`, `delta_qp_heatmap.png`:
  derived diagnostics when Matplotlib is available. Heatmap is actor delta-QP,
  not final applied encoder QP. It averages the latest completed greedy Val
  protocol; quick and full curves are separate measurement series.

Coverage is a storage check, not an extra codec-quality gate. Existing native,
pair identity and detector/POC gates still establish the individual results.
YUV/IPC files are intentionally excluded; source MOV and configuration can
regenerate YUV, and selected integer actions are retained in rollout JSONL.
No mandatory Test use, extra encoding, new detector, artificial diversity
reward or inflated statistical confidence is introduced by reporting.

## Limits and research basis

Checkpoint/evaluation logging follows the separation described by
[SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
and [PPO logger documentation](https://stable-baselines3.readthedocs.io/en/master/common/logger.html).
Model and optimizer state follow [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
Keeping individual cases supports future uncertainty analysis as motivated by
[Agarwal et al.](https://arxiv.org/abs/2108.13264); fixed sampled-evaluation seeds
are not independent training runs. Rate-task comparisons and spatial map
inspection are motivated by [RL-RC-DoT](https://arxiv.org/html/2501.12216v2).

One epoch remains 325 Train episodes; full35 is required for EPOCH1 complete.
Full35 does not automatically acquire sampled/spatial controls; those remain
quick-Val diagnostics. Logging does not supply missing training seeds,
equal-actual-rate experiments, human-GT mAP or a learned spatial-allocation gain.
Cumulative ZIPs grow with collected evidence; keep disk/output capacity for
both uncompressed Working and atomic archive replacement. A hard VM kill
cannot execute Python finally or persist an unfinished file automatically.

Verification results are recorded in `AGENT_V4_EVIDENCE_EXPORT_CHECKS.json`.
Tests use synthetic archives and mocked collectors, not a new Linux/GPU codec run.
