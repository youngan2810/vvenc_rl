"""Reuse audited codec settings, POC transport, AU accounting and decoder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re

from au_rate_audit import read_au_bits
from real_reward_pilot import run_cmd, W, H, FPS

PROFILE_VERSION = 'golden-vvenc-fpp1/1'


def runtime_profile():
    """A declared derivative of the original lock, not the old 500-pair anchor."""
    return {'version':PROFILE_VERSION,'encoder_threads':4,'decoder_threads':4,
            'max_parallel_frames':1,'gop_size':32,'lookahead':1,'qpa':1,
            'preset':'medium','passes':1,'internal_bitdepth':10,'mtprofile':-1,
            'paired_anchor':'fresh-vanilla-same-profile',
            'historical_500_pair_baseline_compatible':False}


def require_runtime_profile(experiment):
    if (experiment.get('codec_profile') != runtime_profile() or
            experiment.get('encoder_threads') != 4 or experiment.get('decoder_threads') != 4):
        raise ValueError('Codec runtime profile differs; start a NEW fpp1 experiment, do not migrate old progress')


def encoder_command(encoder, source, stream, frames, rate, threads=4):
    if threads != runtime_profile()['encoder_threads']:
        raise ValueError('The fpp1 experiment requires four encoder threads')
    return [str(encoder),'-i',str(source),'-s',f'{W}x{H}','-r',str(FPS),'-f',str(frames),
        '-c','yuv420','--preset','medium','--internal-bitdepth','10','--bitrate',f'{rate}k',
        '--passes','1','--qpa','1','--intraperiod','32','--refreshtype','cra','--sdr','sdr_709',
        '--threads',str(threads),'--mtprofile','-1','--additional',
        'GOPSize=32:LookAhead=1:MaxParallelFrames=1','-o',str(stream)]


def verify_effective_profile(log_path):
    text=log_path.read_text(errors='replace')
    match=re.search(r'Threads\s*:\s*(\d+)\s+\(parallel frames:\s*(\d+)\)',text)
    if match is None or tuple(map(int,match.groups())) != (4,1):
        raise ValueError('Native log does not confirm encoder threads4 / parallel frames1: '+str(log_path))


def vanilla_encode(encoder, source, output, frames, bitrate_kbps, encoder_threads=4):
    """V4 paired anchor uses EXACTLY the same explicit runtime profile as RL."""
    output.mkdir(parents=True,exist_ok=False)
    stream=output/'vanilla.266'
    env={k:v for k,v in os.environ.items() if not k.startswith('VVENC_RL_')}
    env.update(VVENC_RL_STATE_FILE=str(output/'states.txt'),VVENC_RL_AU_BITS_FILE=str(output/'au_bits.csv'))
    run_cmd(encoder_command(encoder,source,stream,frames,bitrate_kbps,encoder_threads),
            output/'encoder.log',env=env,timeout=1800)
    verify_effective_profile(output/'encoder.log')
    return stream


def require_zero_identity(stream, vanilla, rows, metrics, baseline_metrics, output):
    """Fail closed for all-zero pairs; tolerate only floating summation noise."""
    import math
    from real_reward_pilot import sha256_file
    from full_train import atomic_json
    if not all(a == 0 for r in rows for a in r['action']):
        return False
    if sha256_file(stream) != sha256_file(vanilla):
        atomic_json(output/'zero_identity_failure.json',{
            'gate':'ZERO_ACTION_STREAM_MISMATCH','online_sha256':sha256_file(stream),
            'vanilla_sha256':sha256_file(vanilla),'quality_claim':False})
        raise ValueError('All-zero action bitstream differs from vanilla; diagnose native/IPC')
    if (set(metrics) != set(baseline_metrics) or any(
            not math.isclose(v,baseline_metrics[k],rel_tol=0,abs_tol=1e-12)
            for k,v in metrics.items())):
        atomic_json(output/'zero_metric_failure.json',{
            'gate':'IDENTICAL_STREAM_METRIC_MISMATCH','metrics':metrics,
            'baseline':baseline_metrics,'quality_claim':False})
        raise ValueError('Identical zero stream has different task metrics; detector reproducibility failure')
    return True


def encode_fixed_map(encoder, source, out, frames, rate, maps, threads=4):
    """Declared intervention/replay, never a trainable PPO rollout."""
    if len(maps) != frames or any(len(m) != 12 or any(type(a) != int or not -2 <= a <= 2 for a in m) for m in maps):
        raise ValueError('Fixed map requires legal [frames,12] integer actions')
    out.mkdir(parents=True,exist_ok=False)
    action_map = out/'fixed_map.txt'
    action_map.write_text(''.join(str(p)+' '+' '.join(map(str,m))+'\n' for p,m in enumerate(maps)),encoding='ascii')
    env = {k:v for k,v in os.environ.items() if not k.startswith('VVENC_RL_')}
    env.update(VVENC_RL_QP_MAP=str(action_map),VVENC_RL_STATE_FILE=str(out/'states.txt'),
               VVENC_RL_AU_BITS_FILE=str(out/'au_bits.csv'))
    stream = out/'fixed.266'
    run_cmd(encoder_command(encoder,source,stream,frames,rate,threads),out/'encoder.log',env=env,timeout=1800)
    verify_effective_profile(out/'encoder.log')
    return stream,read_au_bits(out/'au_bits.csv',stream,frames)


def encode(repo, encoder, source, out, frames, checkpoint, rate, seed=7,
           deterministic=False, threads=4, zero=False, evaluation=False):
    out.mkdir(parents=True, exist_ok=False)
    action_dir, states, trajectory, ready = (out/"actions", out/"states.txt",
                                            out/"trajectory.jsonl", out/"ready.txt")
    cmd = [sys.executable, "-u", str(repo/"scripts/rl/agent_v4_bridge.py"),
           "--state-file", str(states), "--action-dir", str(action_dir),
           "--trajectory-file", str(trajectory), "--frames", str(frames),
           "--checkpoint", str(checkpoint), "--source-yuv", str(source),
           "--seed", str(seed), "--ready-file", str(ready)]
    if deterministic:
        cmd.append("--deterministic")
    if zero:
        cmd.append("--zero-action")
    if evaluation:
        cmd.append("--evaluation")
    with (out/"bridge.log").open("w", encoding="utf-8") as log:
        bridge = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+60
            while not ready.exists() and bridge.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            if not ready.exists():
                raise RuntimeError("v4 bridge did not become ready; inspect " + str(out/"bridge.log"))
            env = {k:v for k,v in os.environ.items() if not k.startswith("VVENC_RL_")}
            env.update(VVENC_RL_ACTION_DIR=str(action_dir), VVENC_RL_STATE_FILE=str(states),
                       VVENC_RL_AU_BITS_FILE=str(out/"au_bits.csv"))
            stream = out/"ppo.266"
            run_cmd(encoder_command(encoder,source,stream,frames,rate,threads),
                out/"encoder.log", env=env, timeout=1800)
            verify_effective_profile(out/'encoder.log')
            if bridge.wait(timeout=60):
                raise RuntimeError("v4 bridge failed: " + (out/"bridge.log").read_text(errors="replace")[-2000:])
            rows = [json.loads(s) for s in trajectory.read_text().splitlines() if s]
            if len(rows) != frames or {r["poc"] for r in rows} != set(range(frames)):
                raise ValueError("Incomplete policy trajectory")
            return stream, rows, read_au_bits(out/"au_bits.csv", stream, frames)
        finally:
            if bridge.poll() is None:
                bridge.kill()
                bridge.wait()

