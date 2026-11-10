#!/usr/bin/env python3
"""Real32-frame codec/bridge gate, synthetic source, no YOLO/training/data split."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agent_v4.checkpoint import save
from agent_v4.codec import encode, encode_fixed_map, vanilla_encode, runtime_profile
from agent_v4.model import Agent, ModelConfig, PPOConfig, Trainer
from agent_v4.reward import RewardConfig
from agent_v4.state import SourceFrames, stack_states
from au_rate_audit import read_au_bits
from decoder_alignment import verify_decode_mapping, require_auditor
from full_train import atomic_json
from real_reward_pilot import run_cmd, sha256_file


def run(repo, encoder, decoder, output):
    require_auditor(decoder)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    frames, rate = 32,120
    source = output/"synthetic.yuv"
    yy,xx = np.indices((320,480))
    with source.open("wb") as f:
        for t in range(frames):
            f.write((16+(xx//3+yy//2+t*3)%220).astype(np.uint8).tobytes())
            f.write(np.full((160,240),128,np.uint8).tobytes()*2)
    torch.manual_seed(20261109)
    model = Agent().eval()
    trainer = Trainer(model)
    checkpoint = output/"fresh_v4.pt"
    save(checkpoint,model,trainer,{"model_config":asdict(model.config),
        "ppo_config":asdict(trainer.config),"reward":asdict(RewardConfig())},0)
    original_sha = sha256_file(checkpoint)
    vanilla = vanilla_encode(encoder,source,output/"vanilla",frames,rate,4)
    nonzero,_ = encode_fixed_map(encoder,source,output/"fixed_minus2",frames,rate,[[-2]*12 for _ in range(frames)],4)
    if sha256_file(vanilla) == sha256_file(nonzero):
        raise ValueError("Nonzero intervention did not change this synthetic bitstream")
    results = {}
    for name,deterministic,zero in (("zero",True,True),("initial_mode",True,False),("sampled",False,False)):
        out = output/name
        stream,rows,aus = encode(repo,encoder,source,out,frames,checkpoint,rate,42,deterministic,4,zero)
        if deterministic and sha256_file(stream) != sha256_file(vanilla):
            raise ValueError("Initial/zero v4 mode must preserve vanilla bytes in this gate")
        for row in rows:
            fields = (out/"actions"/f'{row["poc"]}.txt').read_text().split()
            if list(map(int,fields)) != [row["poc"]]+row["action"]:
                raise ValueError("Action file != recorded policy action")
        errors = None
        if not deterministic:
            replay,_=encode_fixed_map(encoder,source,output/'sampled_fixed_replay',frames,rate,
                [next(r['action'] for r in rows if r['poc']==p) for p in range(frames)],4)
            if sha256_file(replay)!=sha256_file(stream):
                raise ValueError('Sampled live actions must reproduce the fixed-map stream before spatial controls')
            with SourceFrames(source,frames) as source_frames:
                states = stack_states(rows,source_frames,True)
            errors = trainer.parity(states,torch.tensor([r["action"] for r in rows]),
                torch.tensor([r["log_prob"] for r in rows]),torch.tensor([r["value"] for r in rows]))
        decoded = out/"decoded10.yuv"
        run_cmd([str(decoder),"-b",str(stream),"-o",str(decoded),"--threads","4"],out/"decode.log",timeout=600)
        mapping = verify_decode_mapping(stream,decoded,decoder,frames,4)
        decoded.unlink()
        results[name] = {"stream_sha256":sha256_file(stream),"parity_errors":errors,
            "action_histogram":dict(Counter(str(a) for r in rows for a in r["action"])),
            "au_bits":aus[-1].cumulative_bits,"alignment":mapping}
    if sha256_file(checkpoint) != original_sha:
        raise ValueError("Preflight changed checkpoint")
    summary = {"gate":"AGENT_V4_NATIVE_PREFLIGHT_PASS","frames":frames,
        'codec_profile':runtime_profile(),
        "encoder_sha256":sha256_file(encoder),"decoder_sha256":sha256_file(decoder),
        "checkpoint_sha256":original_sha,"cases":results,
        "quality_claim":False,"ppo_updates":0}
    summary['sampled_live_fixed_replay_byte_identity']=True
    atomic_json(output/"preflight.json",summary)
    print(summary["gate"],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","encoder","decoder","output"):
        p.add_argument("--"+name,type=Path,required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()

