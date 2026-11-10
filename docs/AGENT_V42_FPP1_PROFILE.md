# V4.2: explicit single-frame parallelism runtime (2026-11-10)

The new runtime is `golden-vvenc-fpp1/1`: encoder threads4,
`MaxParallelFrames=1`, decoder threads4. This is a declared protocol derivative.
It changes encoding results even though native source and golden binaries stay
unchanged. The existing baseline lock remains an input provenance check; it is
not a claim that the historical 500-pair outputs use this new runtime.

## Evidence and choice

The original failed initial Validation had zero PPO updates. On Val video
`00b04b12-a7d7eb85`, 256 frames at120kbps, even vanilla encoding produced two
bitstreams under the old automatic frame-parallel setting. The RateCtrl pointer
candidate still produced both variants in the27-case A/B run and was rejected
as a drift fix. No candidate C++ change or candidate binary is adopted.

The user's subsequent `VVC_V42_NATIVE_PROFILES.zip` has SHA256
`4d4ac3fabbe0285900337d8d59491a0c2759d650b243e0dc626c03eaf4c60fcd`.
An independent archive audit checked CRC, all2457 manifest entries, all27
bitstream digests, byte lengths, AU-bit sums and POC coverage, zero trajectories,
fixed maps and effective native threading headers. Each group contains three
vanilla, three live-zero and three fixed-zero encodes with two concurrent jobs.

| Runtime | Identical repeats | Bytes | Actual kbps |
|---|---:|---:|---:|
| 4 threads / FPP1 |9/9|125307|117.4753125|
| 4 threads / FPP0 |9/9|124603|116.8153125|
| 1 thread / FPP1 |9/9|125307|117.4753125|

Within each group there is also one state snapshot hash. The two FPP1 groups
share stream SHA256
`388f0ce758d2071fcde2bfcb3527ec87a4460820e35a7e447526ec72b88417bc`.
We choose4threads/FPP1 to retain four encoder threads while explicitly limiting
frame parallelism. These data support a runtime workaround, not a confirmed
root-cause line, a speed ranking, or an all-clip determinism guarantee.
No decoder, detector, or optimizer ran in this diagnostic. Binary/source hashes
are recorded by its runner; the ZIP omits those artifacts for independent rehash.

## Implementation and gates

`agent_v4.codec.encoder_command` is shared by live policy, fixed map and vanilla.
Every command specifies
`--additional GOPSize=32:LookAhead=1:MaxParallelFrames=1`; every successful
encode must report `Threads: 4 (parallel frames: 1)` in its native header.
Experiment/checkpoint metadata records the complete runtime and explicitly
declares historical baseline incompatibility. Collectors and standalone
Validation reject missing or different runtime metadata. Resume config equality
and the launcher also reject an older pre-FPP1 experiment, including iter0.

Golden encoder SHA remains
`856cee2e9f0e419423aff85c55fe8d01cfbd9b18e6847bd145bcde940080c2cb`;
canonical source remains `e0b586c46f61e2ebbf131e4eb83f98817ff6e374`.
No native hash whitelist, C++ edit, v3 edit, or weakening of zero identity gates.

The same named pilot notebook now defaults to a **new50-episode FPP1 run**.
Output is `/kaggle/working/vvc_agent_v42_fpp1_run`; setup logs use
`v42_fpp1_setup_logs`. Keep the existing dataset/split/detector inputs.
Before PPO it runs synthetic native preflight, Train0 repeated-vanilla/zero
headroom at50/120kbps, and initial quick Val covering Val0/1 at50/120kbps
(four greedy pairs with explicit zero gates plus eight sampled diagnostics).
Any mismatch stops the run; later all-zero paired results retain the same gate.

This broader Kaggle gate and subsequent real training are still pending.
Each Train/Val result compares against freshly encoded vanilla at this exact
runtime. Do not combine the old500-pair baseline with these deltas or directly
rank old v3 outputs against this new run; a future v3 comparison requires
re-evaluation under a common declared protocol. Test remains excluded from tuning.
The v4.2 learning algorithm is unchanged by this runtime repair.
