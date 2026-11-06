"""Isolated EP467 diagnostics. Never advances or rewrites an official run.

Offline: verified input exports, quantized-action entropy, matched-rate curves,
and instrumented replays of the ACTUAL PPOTrainer.update on disposable models.
Online: small validation-only fixed-QP sweep using the canonical codec/reward.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest import mock
import zipfile

import numpy as np
import torch

from full_train import atomic_json, load_trainer
from ppo_agent import PPOActorCritic, compute_gae, estimate_kl
from real_reward_pilot import sha256_file

OFFICIAL_COMMIT = 'e0b586c46f61e2ebbf131e4eb83f98817ff6e374'
LOCK_SHA = '54aa575cba547e4452b5ab03422c491cd5f103db5f66f4e0e80b1a2268315213'
ENC_SHA = '856cee2e9f0e419423aff85c55fe8d01cfbd9b18e6847bd145bcde940080c2cb'
DEC_SHA = '1553776eb512e02335bb4925993a6b86fecb459f06cb51045539866091d38a73'
CHECKPOINTS = {68: 'd37d7335d103b947c0e850ad62a25426877a440e7a24226cddeca2d028b676c2',
               325: '771bd008069c1135b773b0e884dc42f1d2b4cd9da58762466baa8f3831e8423f',
               467: '89fbf46276c6e8139107b9c3aa951f79d70edcd58acca48c5a191ef4490ac8fe'}
MANIFEST = 'VVC_PPO_OFFICIAL_RUN_MANIFEST.json'
sha = lambda b: hashlib.sha256(b).hexdigest()


class View:
    """Read ZIP or Kaggle-expanded directory without bulk extraction."""
    def __init__(self, value):
        self.path = Path(value) if not isinstance(value, bytes) else None
        self.zip = None if self.path is not None and self.path.is_dir() else zipfile.ZipFile(io.BytesIO(value) if isinstance(value, bytes) else value)

    def read(self, name):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe member')
        if self.zip is not None:
            return self.zip.read(name)
        target = (self.path/relative).resolve()
        if not target.is_relative_to(self.path.resolve()): raise ValueError('Escaping member')
        return target.read_bytes()

    def inner(self, name):
        if self.zip is None and not (self.path/name).is_file():
            return View(self.path/Path(name).with_suffix(''))
        return View(self.read(name))

    def names(self):
        return self.zip.namelist() if self.zip else [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]

    def json(self, name): return json.loads(self.read(name))
    def rows(self, name): return [json.loads(x) for x in self.read(name).decode().splitlines() if x.strip()]
    def crc(self):
        if self.zip is not None and self.zip.testzip() is not None: raise ValueError('CRC failed')
    def close(self):
        if self.zip: self.zip.close()


def finite(x):
    if isinstance(x, torch.Tensor): return bool(torch.isfinite(x).all())
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def verified_bundle(path, count):
    outer = View(path); outer.crc()
    manifest = outer.json(MANIFEST)
    if (manifest.get('completed_episodes') != count or manifest.get('source_commit') != OFFICIAL_COMMIT
            or manifest.get('baseline_lock_sha256') != LOCK_SHA
            or manifest.get('encoder_sha256') != ENC_SHA or manifest.get('decoder_sha256') != DEC_SHA):
        raise ValueError('Unexpected export origin/count/codec')
    resume = outer.inner('PPO_QP_FULL_TRAIN_LATEST.zip'); resume.crc()
    progress = resume.json('progress.json')
    weights = resume.read('checkpoint/latest.pt')
    if (sha(weights) != CHECKPOINTS[count] or progress['latest_sha256'] != CHECKPOINTS[count]
            or manifest['latest_checkpoint_sha256'] != CHECKPOINTS[count]
            or progress['training_config'] != manifest['config']
            or len(progress['completed']) != count or len(progress['updates']) != manifest['ppo_updates']):
        raise ValueError('Checkpoint/progress/manifest mismatch')
    state = torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True)
    if not finite(state) or state['training_config'] != progress['training_config']:
        raise ValueError('Invalid checkpoint tensors/config')
    return outer, resume, progress, weights


def action_distribution(mean, std):
    """Exact probabilities after 2*tanh(z) and integer rounding (ties have zero mass)."""
    bounds = torch.atanh(torch.tensor([-.75, -.25, .25, .75], dtype=mean.dtype, device=mean.device))
    cdf = torch.distributions.Normal(mean[..., None], std[..., None]).cdf(bounds)
    p = torch.cat((cdf[..., :1], cdf[..., 1:]-cdf[..., :-1], 1-cdf[..., -1:]), dim=-1).clamp_min(0)
    return p / p.sum(dim=-1, keepdim=True)


def action_report(state, rows, compare_logged=False):
    model = PPOActorCritic(); model.load_state_dict(state['model_state']); model.eval()
    h = Counter(); sums = Counter(); low = float('inf'); high = -float('inf'); max_error = 0.; mismatch = 0
    with torch.no_grad():
        for offset in range(0, len(rows), 256):
            batch = rows[offset:offset+256]
            sp = torch.tensor([r['spatial'] for r in batch], dtype=torch.float32)
            gl = torch.tensor([r['global'] for r in batch], dtype=torch.float32)
            d, values = model(sp, gl); raw = d.mean; bounded = 2*raw.tanh(); action = bounded.round().clamp(-2, 2).int()
            p = action_distribution(raw, d.stddev)
            h.update(action.flatten().tolist()); low = min(low, raw.min().item()); high = max(high, raw.max().item())
            sums['frames'] += len(batch)
            sums['discrete_entropy_sum'] += float((-(p*p.clamp_min(1e-30).log()).sum(-1)).sum())
            sums['gaussian_entropy_sum'] += float(d.entropy().sum())
            sums['minus2_probability_sum'] += float(p[..., 0].sum())
            sums['within_frame_mean_std_sum'] += float(raw.std(dim=1, unbiased=False).sum())
            sums['uniform_maps'] += int((action == action[:, :1]).all(dim=1).sum())
            if compare_logged:
                max_error = max(max_error, (raw-torch.tensor([r['raw_action'] for r in batch])).abs().max().item())
                mismatch += int((action != torch.tensor([r['discrete_action'] for r in batch])).sum())
    n = sums['frames']; total = n*12
    result = dict(frames=n, deterministic_histogram=dict(h), raw_mean_min=low, raw_mean_max=high,
                  uniform_map_fraction=sums['uniform_maps']/n,
                  gaussian_entropy_per_frame=sums['gaussian_entropy_sum']/n,
                  quantized_conditional_entropy_per_frame=sums['discrete_entropy_sum']/n,
                  mean_probability_minus2=sums['minus2_probability_sum']/total,
                  raw_spatial_std_per_frame=sums['within_frame_mean_std_sum']/n,
                  scope='Offline inference on archived states; not a fresh encoder rollout')
    if compare_logged:
        result.update(max_logged_mean_error=max_error, logged_action_mismatches=mismatch)
        if mismatch or max_error > 1e-4: raise ValueError('Archived validation policy parity failed')
    return result


def gradient_norm(grads):
    return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


def replay_update(checkpoint, config, rows, seed, expected_metrics):
    """Actual PPO update on a private model/Adam, with observational hooks only."""
    from verify_real_ppo_update import validate_unchanged_collecting_policy
    before_hash = sha256_file(checkpoint)
    _, model, trainer = load_trainer(checkpoint, config, LOCK_SHA)
    rollout=[];parities=[];begin=0
    for i,row in enumerate(rows):
        if row['done']:
            episode,checked=validate_unchanged_collecting_policy(model,rows[begin:i+1])
            rollout.extend(episode);parities.append(checked);begin=i+1
    if begin!=len(rows):raise ValueError('Unterminated collecting episode')
    parity=dict(episodes=len(parities),max_old_logp_error=max(p['max_old_logp_error'] for p in parities),
                max_old_value_error=max(p['max_old_value_error'] for p in parities))
    sp = torch.stack([t.spatial for t in rollout]); gl = torch.stack([t.global_ for t in rollout])
    raw = torch.stack([t.raw_action for t in rollout]); oldlog = torch.tensor([t.log_prob for t in rollout])
    rewards = torch.tensor([t.reward for t in rollout]); values = torch.tensor([t.value for t in rollout])
    nxt = torch.tensor([t.next_value for t in rollout]); dones = torch.tensor([t.done for t in rollout])
    adv, ret = compute_gae(rewards, values, nxt, dones, trainer.gamma, trainer.gae_lambda)
    start = 0
    for end, done in enumerate(dones.tolist()):
        if done:
            a, r = compute_gae(rewards[start:end+1], values[start:end+1], nxt[start:end+1], dones[start:end+1], trainer.gamma, trainer.gae_lambda)
            if not torch.allclose(a, adv[start:end+1]) or not torch.allclose(r, ret[start:end+1]):
                raise ValueError('GAE crosses episode boundary')
            start = end+1
    if start != len(rows): raise ValueError('Unterminated rollout')
    advantage = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    index = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:trainer.minibatch_size]
    logp, entropy, v = model.evaluate(sp[index], gl[index], raw[index])
    ratio = (logp-oldlog[index]).exp()
    pi_loss = -torch.minimum(ratio*advantage[index],ratio.clamp(1-trainer.clip_eps,1+trainer.clip_eps)*advantage[index]).mean()
    value_loss = trainer.value_coef*torch.nn.functional.mse_loss(v,ret[index])
    entropy_loss = -trainer.entropy_coef*entropy.mean()
    shared = [p for name,p in model.named_parameters() if name.startswith(('spatial_encoder.','global_encoder.','fusion.'))]
    pg = torch.autograd.grad(pi_loss,shared,retain_graph=True,allow_unused=True)
    vg = torch.autograd.grad(value_loss,shared,retain_graph=True,allow_unused=True)
    eg = torch.autograd.grad(entropy_loss,shared,allow_unused=True)
    pn,vn = gradient_norm(pg),gradient_norm(vg)
    dot = sum(float((a*b).sum()) for a,b in zip(pg,vg) if a is not None and b is not None)
    first = dict(policy_shared_gradient_norm=pn,weighted_value_shared_gradient_norm=vn,
                 entropy_shared_gradient_norm=gradient_norm(eg),
                 policy_value_gradient_cosine=dot/(pn*vn) if pn*vn else None,
                 value_to_policy_gradient_ratio=vn/pn if pn else None)
    del pg,vg,eg,pi_loss,value_loss,entropy_loss
    trace=[]; original_step=trainer.optimizer.step; original_clip=torch.nn.utils.clip_grad_norm_
    pending={}
    def clip(parameters, *args, **kwargs):
        parameters=list(parameters)
        pending['pre_clip_norm']=gradient_norm([p.grad for p in parameters])
        result=original_clip(parameters,*args,**kwargs)
        pending['post_clip_norm']=gradient_norm([p.grad for p in parameters])
        return result
    def step(*args,**kwargs):
        result=original_step(*args,**kwargs)
        with torch.no_grad():
            lp,_,_=model.evaluate(sp,gl,raw);k3,k1=estimate_kl(lp-oldlog)
            d,_=model(sp,gl);actions=(2*d.mean.tanh()).round().clamp(-2,2)
        trace.append(dict(step=len(trace)+1,**pending,whole_rollout_kl_k3=float(k3),whole_rollout_kl_k1=float(k1),
                          deterministic_minus2_fraction=float((actions==-2).float().mean())))
        return result
    trainer.optimizer.step=step
    torch.manual_seed(seed)
    with mock.patch('torch.nn.utils.clip_grad_norm_',side_effect=clip):
        metrics=trainer.update(rollout)
    if len(trace)!=int(metrics['optimizer_steps']) or sha256_file(checkpoint)!=before_hash:
        raise ValueError('Replay integrity failed')
    variance=float(ret.var(unbiased=False));explained=1-float((ret-values).var(unbiased=False))/variance if variance else None
    return dict(scope='Diagnostic replay only; updated model/optimizer are discarded, never saved for resume',
                seed=seed,parity=parity,first_minibatch_gradients=first,
                advantage_mean=float(adv.mean()),advantage_std=float(adv.std(unbiased=False)),
                returns_mean=float(ret.mean()),returns_std=float(ret.std(unbiased=False)),
                critic_explained_variance=explained,metrics=metrics,archived_metrics=expected_metrics,
                optimizer_step_count_matches_archive=metrics['optimizer_steps']==expected_metrics['optimizer_steps'],
                steps=trace,checkpoint_bytes_unchanged=True)


def save_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def offline(export2,export3,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new diagnostic output directory')
    out.mkdir(parents=True)
    b2,r2,p2,w2=verified_bundle(export2,68);b3,r3,p3,w3=verified_bundle(export3,467)
    a=b3.inner('PPO_QP_ANALYSIS.zip');a.crc()
    try:
        if p3['completed'][:68]!=p2['completed'] or p3['updates'][:34]!=p2['updates']:
            raise ValueError('Exports do not share official history')
        if p3!=a.json('progress.json'):raise ValueError('Stale analysis archive')
        inputs=out/'inputs';inputs.mkdir()
        for n,b in [(68,w2),(325,r3.read('checkpoints/ckpt_iter_000163.pt')),(467,w3)]:
            if sha(b)!=CHECKPOINTS[n]:raise ValueError('Checkpoint hash mismatch')
            (inputs/f'ep{n}.pt').write_bytes(b)
        atomic_json(out/'source_config.json',p3['training_config'])
        val=a.json('latest_validation.json')
        if val['checkpoint_sha256']!=CHECKPOINTS[325] or len(val['pairs'])!=35:
            raise ValueError('Validation identity mismatch')
        states=[]
        for pair in val['pairs']:
            states.extend(a.rows(f"validation_epoch_001/val_{pair['val_index']:02d}/rate_{pair['target_kbps']}/ppo/trajectory.jsonl"))
        reports={}
        for n in (68,325,467):
            state=torch.load(inputs/f'ep{n}.pt',map_location='cpu',weights_only=True)
            reports[str(n)]=action_report(state,states,compare_logged=n==325)
        replays={}
        for episode,iteration in [(68,35),(325,164)]:
            update=p3['updates'][iteration-1]
            if update['collector_checkpoint_sha256']!=CHECKPOINTS[episode]:raise ValueError('Replay collecting policy mismatch')
            records=[]
            for step in update['episode_steps']:
                ep=p3['completed'][step-1];name=ep['episode_dir']+'/joined_reward_ppo.jsonl'
                data=a.read(name)
                if sha(data)!=ep['rollout_sha256']:raise ValueError('Rollout hash mismatch')
                records.extend([json.loads(x) for x in data.decode().splitlines()])
            replays[str(iteration)]=replay_update(inputs/f'ep{episode}.pt',p3['training_config'],records,
                p3['training_config']['seed']+200000+episode,update['ppo_metrics'])
        rows=[];matched=[]
        for video in sorted({p['video_id'] for p in val['pairs']}):
            pairs=sorted([p for p in val['pairs'] if p['video_id']==video],key=lambda p:p['target_kbps'])
            x0=np.array([p['vanilla']['actual_kbps'] for p in pairs]);y0=np.array([p['vanilla']['f1'] for p in pairs])
            x1=np.array([p['ppo']['actual_kbps'] for p in pairs]);y1=np.array([p['ppo']['f1'] for p in pairs])
            if not (np.diff(x0)>0).all() or not (np.diff(x1)>0).all():raise ValueError('Non-monotonic rate points; do not interpolate')
            low=max(x0.min(),x1.min());high=min(x0.max(),x1.max())
            grid=np.linspace(low,high,101)
            delta=np.interp(grid,x1,y1)-np.interp(grid,x0,y0)
            matched.append(dict(video_id=video,common_low_kbps=float(low),common_high_kbps=float(high),
                mean_delta_f1_linear_interp=float(delta.mean()),scope='Diagnostic interpolation within overlap; not BD-rate or test result'))
            for p in pairs:rows.append(dict(video_id=video,target_kbps=p['target_kbps'],vanilla_kbps=p['vanilla']['actual_kbps'],
                ppo_kbps=p['ppo']['actual_kbps'],vanilla_f1=p['vanilla']['f1'],ppo_f1=p['ppo']['f1'],
                delta_score=p['ppo']['score']-p['vanilla']['score']))
        save_csv(out/'validation_points.csv',rows);save_csv(out/'matched_rate_diagnostic.csv',matched)
        report=dict(schema='vvenc-policy-diagnostic/1',official_source_commit=OFFICIAL_COMMIT,
            official_training_not_advanced=True,checkpoint_sha256=CHECKPOINTS,
            offline_action_reports=reports,ppo_replays=replays,matched_rate_diagnostic=matched,
            validation_reward_note='Training: mean per-frame detection term minus cumulative-rate penalty in policy order. Validation: aggregate clip F1 minus final-rate penalty. Their rankings can differ.',
            original_export2=str(export2),original_export3=str(export3))
        atomic_json(out/'offline_report.json',report)
        plot_validation(out,rows)
        print('OFFLINE_DIAGNOSTICS_PASS',out)
    finally:
        for v in (a,r2,r3,b2,b3):v.close()


def plot_validation(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for ax,video in zip(axes.flat,sorted({r['video_id'] for r in rows})):
        selected=[r for r in rows if r['video_id']==video]
        for method,color in [('vanilla','#2463A6'),('ppo','#D66D19')]:
            ax.plot([r[method+'_kbps'] for r in selected],[r[method+'_f1'] for r in selected],'-o',color=color,label=method)
        ax.set_title(video,fontsize=9);ax.set_xlabel('Actual kbps');ax.set_ylabel('Detection agreement F1');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');axes.flat[0].legend()
    fig.suptitle('EP325 validation: actual bitrate vs detection preservation (not test results)')
    fig.savefig(out/'validation_rate_f1.png',dpi=160);plt.close(fig)


def clean_env():
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('VVENC_RL_'):env.pop(key,None)
    return env


def fixed_encode(encoder,reference,out,frames,rate,delta,threads):
    """Same canonical settings; only fixed delta map and observational logs vary."""
    from real_reward_pilot import run_cmd
    if delta not in (None,-2,-1,0,1,2):raise ValueError('Invalid fixed delta-QP')
    out.mkdir(parents=True,exist_ok=False)
    env=clean_env()
    if delta is not None:
        action=out/'fixed_map.txt'
        action.write_text(''.join(str(poc)+' '+' '.join([str(delta)]*12)+'\n' for poc in range(frames)),encoding='ascii')
        env['VVENC_RL_QP_MAP']=str(action)
    env['VVENC_RL_STATE_FILE']=str(out/'states.txt')
    env['VVENC_RL_AU_BITS_FILE']=str(out/'au_bits.csv')
    stream=out/'fixed.266'
    cmd=[str(encoder),'-i',str(reference),'-s','480x320','-r','30','-f',str(frames),'-c','yuv420',
         '--preset','medium','--internal-bitdepth','10','--bitrate',f'{rate}k','--passes','1','--qpa','1',
         '--intraperiod','32','--refreshtype','cra','--sdr','sdr_709','--threads',str(threads),'--mtprofile','-1',
         '--additional','GOPSize=32:LookAhead=1','-o',str(stream)]
    run_cmd(cmd,out/'encoder.log',env=env,timeout=1200)
    return stream


def probe_metrics(decoder,detector,stream,reference_detections,video,frames,rate,out,threads):
    from real_reward_pilot import run_cmd,convert_10bit_to_8bit,detect_all
    from decoder_alignment import verify_decode_mapping
    from reward_join import read_detections
    from reward_metrics import calculate_frame_reward,match_one_to_one
    from au_rate_audit import read_au_bits
    decoded=out/'decoded_10bit.yuv';rec8=out/'decoded_8bit.yuv'
    run_cmd([str(decoder),'-b',str(stream),'-o',str(decoded),'--threads',str(threads)],out/'decoder.log',timeout=600)
    convert_10bit_to_8bit(decoded,rec8,frames)
    mapping=verify_decode_mapping(stream,decoded,decoder,frames,threads)
    detfile=out/'decoded_detections.jsonl';detect_all(detector,rec8,frames,video,detfile)
    ref=read_detections(reference_detections,video,set(range(frames)))
    rec=read_detections(detfile,video,set(range(frames)))
    au=read_au_bits(out/'au_bits.csv',stream,frames);bits={a.poc:a.au_bits for a in au}
    order=[int(s.split()[1]) for s in (out/'states.txt').read_text().splitlines() if s.startswith('STATE ')]
    if len(order)!=frames or set(order)!=set(range(frames)):raise ValueError('Missing decision order')
    cumulative=0;rewards=[];terms=[];penalties=[]
    for step,poc in enumerate(order,1):
        cumulative+=bits[poc]
        reward=calculate_frame_reward(video_id=video,poc=poc,encoded_step=step,pseudo_gt=ref[poc],reconstructed=rec[poc],
            actual_frame_bits=bits[poc],cumulative_bits=cumulative,frame_rate=30,target_bps=rate*1000,beta=.1)
        rewards.append(reward.reward);terms.append(reward.detection_term);penalties.append(reward.bitrate_penalty)
    counts=Counter()
    for poc in range(frames):
        c=match_one_to_one(ref[poc],rec[poc],.5);counts.update(tp=c.tp,fp=c.fp,fn=c.fn)
    den=2*counts['tp']+counts['fp']+counts['fn'];f1=2*counts['tp']/den if den else 0.
    actual=stream.stat().st_size*8*30/frames/1000
    result=dict(tp=counts['tp'],fp=counts['fp'],fn=counts['fn'],f1=f1,actual_kbps=actual,
        validation_score=f1-.1*abs(math.log(actual/rate)),training_style_reward=float(np.mean(rewards)),
        mean_frame_detection_term=float(np.mean(terms)),mean_cumulative_rate_penalty=float(np.mean(penalties)),
        stream_sha256=sha256_file(stream),decoded_sha256=mapping['decoded_sha256'],decision_order_sha256=sha(json.dumps(order).encode()))
    decoded.unlink();rec8.unlink()
    return result


def probe_plan(indices_text,rates_text):
    indices=[int(x) for x in indices_text.split(',')];rates=[int(x) for x in rates_text.split(',')]
    if len(set(indices))!=len(indices) or not all(0<=i<7 for i in indices):raise ValueError('Validation indices only')
    if len(set(rates))!=len(rates) or not set(rates)<=set([50,80,120,160,200]):raise ValueError('Invalid target rates')
    return indices,rates


def online(args):
    from full_train_dual_gpu import initialize_detector_assets
    from real_reward_pilot import (read_lock_manifest,read_baseline_config,dataset_source,preprocess_mov,
        load_frozen_yolov5,detect_all,encode_with_ppo)
    from training_session import new_attempt
    from decoder_alignment import require_auditor
    indices,rates=probe_plan(args.val_indices,args.rates)
    commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=OFFICIAL_COMMIT or subprocess.check_output(['git','-C',str(args.repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Canonical encoder/bridge checkout must be clean at the official commit')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for frozen detector')
    if sha256_file(args.encoder)!=ENC_SHA or sha256_file(args.decoder)!=DEC_SHA:raise ValueError('Canonical codec hash changed')
    auditor=require_auditor(args.decoder)
    out=args.output
    report=json.loads((out/'offline_report.json').read_text())
    before={n:sha256_file(out/f'inputs/ep{n}.pt') for n in (68,325,467)}
    if any(before[n]!=CHECKPOINTS[n] for n in before):raise ValueError('Diagnostic checkpoint changed')
    splits,records=read_lock_manifest(Path('/kaggle/input'));lock=read_baseline_config(Path('/kaggle/input'))
    if lock['lock_sha256']!=LOCK_SHA:raise ValueError('Baseline lock mismatch')
    initialize_detector_assets(args.repo,lock);detector,provenance=load_frozen_yolov5(args.repo,lock)
    cfg=dict(frames=args.frames,indices=indices,rates=rates,threads=args.threads,checkpoint_sha256=before,
             encoder_sha256=ENC_SHA,decoder_sha256=DEC_SHA,auditor_sha256=sha256_file(auditor),official_source_commit=OFFICIAL_COMMIT,
             frozen_detector=provenance)
    progress_path=out/'probe_progress.json'
    if progress_path.exists():
        progress=json.loads(progress_path.read_text())
        if progress['config']!=json.loads(json.dumps(cfg)):raise ValueError('Probe resume config differs')
    else:progress=dict(config=cfg,completed=[]);atomic_json(progress_path,progress)
    methods=[('vanilla',None),('fixed_m2',-2),('fixed_m1',-1),('fixed_0',0),('fixed_p1',1),('fixed_p2',2),('ppo325',None),('ppo467',None)]
    for index in indices:
        video=splits['val'][index]
        if video in splits['train'] or video in splits['test']:raise ValueError('Split leakage')
        sample=out/f'probes/val_{index:02d}';sample.mkdir(parents=True,exist_ok=True)
        ref=sample/'reference.yuv';detections=sample/'reference.detections.jsonl'
        if not ref.is_file():preprocess_mov(dataset_source(records,video),ref,args.frames,sample/'source_ffmpeg.log',start_frame=0)
        if ref.stat().st_size!=args.frames*480*320*3//2:raise ValueError('Reference frame count mismatch')
        if not detections.is_file():detect_all(detector,ref,args.frames,video,detections)
        for rate in rates:
            for method,delta in methods:
                if any(r['val_index']==index and r['target_kbps']==rate and r['method']==method for r in progress['completed']):continue
                case=new_attempt(sample/f'rate_{rate}_{method}');began=time.monotonic()
                if method.startswith('ppo'):
                    checkpoint=out/f'inputs/ep{method[3:]}.pt'
                    stream,trajectory,_=encode_with_ppo(args.repo,args.encoder,ref,case,args.frames,policy_checkpoint=checkpoint,
                        policy_seed=42,bitrate_kbps=rate,deterministic=True,encoder_threads=args.threads)
                    actions=Counter(v for r in trajectory for v in r['discrete_action'])
                else:
                    stream=fixed_encode(args.encoder,ref,case,args.frames,rate,delta,args.threads)
                    actions=Counter({delta if delta is not None else 0:args.frames*12})
                metrics=probe_metrics(args.decoder,detector,stream,detections,video,args.frames,rate,case,args.threads)
                result=dict(val_index=index,video_id=video,target_kbps=rate,method=method,**metrics,
                    action_histogram=dict(actions),case_dir=case.relative_to(out).as_posix(),elapsed_seconds=time.monotonic()-began)
                progress['completed'].append(result);atomic_json(progress_path,progress)
                save_csv(out/'probe_results.csv',[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in progress['completed']])
                # Keep hashes/maps/logs/bitstreams; remove large input YUV only after all cases for this video.
                print('DIAGNOSTIC_PROBE_PASS',index,rate,method,metrics['f1'],metrics['actual_kbps'],flush=True)
    comparisons=[]
    for index in indices:
        for rate in rates:
            group={r['method']:r for r in progress['completed'] if r['val_index']==index and r['target_kbps']==rate}
            if group['vanilla']['stream_sha256']!=group['fixed_0']['stream_sha256']:
                raise ValueError('Zero fixed map differs from vanilla; inspect encoder boundary before interpreting gains')
            for name in ('ppo325','ppo467'):
                policy=group[name]
                all_minus2=policy['action_histogram']=={'-2':args.frames*12} or policy['action_histogram']=={-2:args.frames*12}
                same=policy['stream_sha256']==group['fixed_m2']['stream_sha256']
                comparisons.append(dict(val_index=index,target_kbps=rate,policy=name,all_actions_minus2=all_minus2,
                    bitstream_identical_to_fixed_minus2=same,
                    training_reward_winner=max(group,key=lambda k:group[k]['training_style_reward']),
                    validation_score_winner=max(group,key=lambda k:group[k]['validation_score'])))
                if all_minus2 and not same:raise ValueError('Same action maps differ in bitstream: audit settings/order')
    if any(sha256_file(out/f'inputs/ep{n}.pt')!=before[n] for n in before):raise ValueError('Input checkpoint modified')
    atomic_json(out/'online_report.json',dict(schema='vvenc-policy-probe/1',config=cfg,comparisons=comparisons,
        completed_cases=len(progress['completed']),expected_cases=len(indices)*len(rates)*len(methods),
        official_training_not_advanced=True,checkpoint_bytes_unchanged=True,
        scope='Small validation diagnosis only; no test videos or official training updates'))
    for ref in (out/'probes').glob('val_*/reference.yuv'):ref.unlink()
    print('ONLINE_DIAGNOSTICS_PASS',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['offline','online'],required=True)
    p.add_argument('--export2',type=Path);p.add_argument('--export3',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',type=Path);p.add_argument('--encoder',type=Path);p.add_argument('--decoder',type=Path)
    p.add_argument('--frames',type=int,default=256);p.add_argument('--val-indices',default='0,1');p.add_argument('--rates',default='80,120')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(2)
    for key in list(os.environ):
        if key.startswith('VVENC_RL_'):os.environ.pop(key,None)
    if not 32<=args.frames<=512 or args.threads<1:p.error('Invalid frame/thread count')
    if args.mode=='offline':
        if not args.export2 or not args.export3:p.error('Both official exports are required')
        offline(args.export2,args.export3,args.output)
    else:
        if not all((args.repo,args.encoder,args.decoder)):p.error('Codec/repo paths required')
        online(args)


if __name__=='__main__':main()
