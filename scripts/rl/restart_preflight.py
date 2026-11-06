"""Real codec/bridge gate before a NEW v3 run. No YOLO or PPO optimization.

Four encodes of a deterministic synthetic 32-frame clip use the official codec
settings. Require fresh deterministic PPO == zero-map == vanilla bitstreams;
verify stochastic action transport, CPU log-probability/value parity and both
PPO streams' actual decoder output-POC/byte parity. Not a task-quality test.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import os

import numpy as np
import torch

from ppo_agent import MODEL_VERSION,PPOActorCritic
from full_train import atomic_json
from real_reward_pilot import encode_with_ppo,run_cmd,sha256_file
from diagnose_policy import fixed_encode
from decoder_alignment import require_auditor,verify_decode_mapping


def run(repo,encoder,decoder,out):
    if out.exists():raise FileExistsError('Preflight needs a new output directory')
    require_auditor(decoder)
    out.mkdir(parents=True)
    for key in list(os.environ):
        if key.startswith('VVENC_RL_'):os.environ.pop(key,None)
    torch.set_num_threads(2)
    frames=32;rate=120
    reference=out/'synthetic_8bit.yuv'
    yy,xx=np.indices((320,480))
    with reference.open('wb') as stream:
        for t in range(frames):
            luma=(16+(xx//3+yy//2+t*3)%220).astype(np.uint8)
            luma[120:180,(t*7)%400:(t*7)%400+64]=200
            stream.write(luma.tobytes())
            stream.write(np.full((160,240),128,np.uint8).tobytes()*2)
    torch.manual_seed(20261103)
    model=PPOActorCritic().eval()
    checkpoint=out/'fresh_initial.pt'
    torch.save(dict(model_version=MODEL_VERSION,model_state=model.state_dict()),checkpoint)
    before=sha256_file(checkpoint)
    vanilla=fixed_encode(encoder,reference,out/'vanilla',frames,rate,None,4)
    zero=fixed_encode(encoder,reference,out/'fixed_zero',frames,rate,0,4)
    assert sha256_file(vanilla)==sha256_file(zero),'Zero map changed vanilla bitstream'
    results={}
    for name,deterministic in [('deterministic',True),('stochastic',False)]:
        case=out/name
        stream,rows,au=encode_with_ppo(repo,encoder,reference,case,frames,
            policy_checkpoint=checkpoint,policy_seed=42,bitrate_kbps=rate,
            deterministic=deterministic,encoder_threads=4)
        histogram=Counter(v for row in rows for v in row['discrete_action'])
        errors=[]
        for row in rows:
            sp=torch.tensor([row['spatial']]);gl=torch.tensor([row['global']])
            raw=torch.tensor([row['raw_action']])
            with torch.no_grad():lp,_,value=model.evaluate(sp,gl,raw)
            assert torch.equal((2*raw.tanh()).round().int()[0],torch.tensor(row['discrete_action']))
            errors.append((abs(float(lp)-row['log_prob']),abs(float(value)-row['value'])))
        assert max(max(pair) for pair in errors)<3e-4,'Bridge/current model mismatch'
        if deterministic:
            assert histogram=={0:frames*12},'Fresh deterministic policy is not zero-map'
            assert sha256_file(stream)==sha256_file(vanilla),'Fresh deterministic PPO differs from vanilla'
        else:
            assert len(histogram)>1,'Stochastic bridge did not explore integer actions'
        decoded=case/'decoded_10bit.yuv'
        run_cmd([str(decoder),'-b',str(stream),'-o',str(decoded),'--threads','4'],case/'decoder.log',timeout=600)
        mapping=verify_decode_mapping(stream,decoded,decoder,frames,4)
        results[name]=dict(action_histogram=dict(histogram),stream_sha256=sha256_file(stream),
            max_logprob_error=max(p[0] for p in errors),max_value_error=max(p[1] for p in errors),
            decoder_mapping=mapping)
        decoded.unlink()
    assert sha256_file(checkpoint)==before
    reference.unlink()
    report=dict(schema='vvenc-restart-preflight/1',model_version=MODEL_VERSION,frames=frames,
        encoder_sha256=sha256_file(encoder),decoder_sha256=sha256_file(decoder),
        zero_map_and_fresh_deterministic_equal_vanilla=True,checkpoint_bytes_unchanged=True,
        no_optimizer_update=True,scope='Synthetic codec/bridge correctness only, not task performance',results=results)
    atomic_json(out/'restart_preflight.json',report)
    print('RESTART_CODEC_BRIDGE_PREFLIGHT_PASS',out,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--encoder',type=Path,required=True)
    p.add_argument('--decoder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.repo,a.encoder,a.decoder,a.output)
