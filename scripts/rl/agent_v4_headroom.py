#!/usr/bin/env python3
"""Train-only real-clip zero/replay/headroom diagnostics; NO optimizer updates.

Teacher-guided maps are privileged diagnostic probes, not deployable policies,
causal labels for unexecuted actions, or mathematical upper bounds.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import math
import subprocess
import time

import torch

from agent_v4.checkpoint import save
from agent_v4.codec import encode, encode_fixed_map, require_zero_identity, vanilla_encode, runtime_profile
from agent_v4.collect import decode_detect
from agent_v4.model import Agent, PPOConfig, Trainer
from agent_v4.reward import RewardConfig, occupancy, native_decision_rows, join
from agent_v4.runtime import verify_source
from agent_v4.session import export_results
from agent_v4.train import ENCODER_SHA, DECODER_SHA, cleanup_raw
from au_rate_audit import read_au_bits
from decoder_alignment import require_auditor
from full_train import atomic_json
from full_train_dual_gpu import initialize_detector_assets
from real_reward_pilot import (read_lock_manifest,read_baseline_config,dataset_source,
                              preprocess_mov,load_frozen_yolov5,detect_all,sha256_file)
from reward_join import read_detections


def run(a):
    if (not 32 <= a.frames <= 512 or not 2 <= a.repeats <= 5 or
            not math.isfinite(a.max_hours) or a.max_hours <= 0 or len(set(a.rates)) != len(a.rates)):
        raise ValueError('Invalid diagnostic frames/repeats/budget')
    if not torch.cuda.is_available():
        raise ValueError('Headroom diagnostic needs the frozen GPU detector')
    encoder = a.repo/'bin/release-static/vvencapp'
    if sha256_file(encoder) != ENCODER_SHA or sha256_file(a.decoder) != DECODER_SHA:
        raise ValueError('Pinned codec SHA mismatch')
    require_auditor(a.decoder)
    commit = subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    verify_source(a.repo,{'source_commit':commit})
    splits,records = read_lock_manifest(a.input_root)
    if not 0 <= a.index < len(splits['train']):
        raise ValueError('Only locked Train indices are accepted')
    vid = splits['train'][a.index]
    if vid in splits['val'] or vid in splits['test']:
        raise ValueError('Split violation')
    lock = read_baseline_config(a.input_root)
    initialize_detector_assets(a.repo,lock)
    a.output.mkdir(parents=True,exist_ok=False)
    began = time.monotonic();deadline = began+a.max_hours*3600
    report = {'purpose':'train-only-native-headroom','source_commit':commit,'video_id':vid,
              'split':'train','index':a.index,'frames':a.frames,'repeats':a.repeats,
              'encoder_sha256':sha256_file(encoder),'decoder_sha256':sha256_file(a.decoder),
              'codec_profile':runtime_profile(),
              'results':[],'ppo_updates':0,'privileged_teacher_maps':True,
              'quality_claim':False}
    try:
        source = a.output/'reference_8bit.yuv'
        report['preprocessing'] = preprocess_mov(dataset_source(records,vid),source,a.frames,
                                                a.output/'source_ffmpeg.log',start_frame=0)
        report['source_sha256'] = sha256_file(source)
        detector,provenance = load_frozen_yolov5(a.repo,lock)
        report['detector'] = provenance
        detection_file = a.output/'source_detections.jsonl'
        detect_all(detector,source,a.frames,vid,detection_file)
        source_boxes = read_detections(detection_file,vid,set(range(a.frames)))
        roi = [[-1 if x > 0 else 1 for x in occupancy(source_boxes[p])] for p in range(a.frames)]
        maps = {'fixed_zero':[[0]*12 for _ in range(a.frames)],
                'uniform_minus1':[[-1]*12 for _ in range(a.frames)],
                'uniform_plus1':[[1]*12 for _ in range(a.frames)],'teacher_roi':roi}
        torch.set_num_threads(2);torch.manual_seed(20261109)
        model=Agent();trainer=Trainer(model,PPOConfig())
        ckpt=a.output/'diagnostic_only.pt'
        save(ckpt,model,trainer,{'model_config':asdict(model.config),'ppo_config':asdict(trainer.config),
                               'reward':asdict(RewardConfig())},0)
        ckpt_sha=sha256_file(ckpt)
        for rate in a.rates:
            base_stream=base_metrics=None
            cases=[f'vanilla{r}' for r in range(a.repeats)]+list(maps)+['live_zero']
            for case in cases:
                if time.monotonic()+600 >= deadline:
                    raise TimeoutError('Headroom budget reserve reached; no training started')
                out=a.output/f'rate{rate}'/case
                if case.startswith('vanilla'):
                    stream=vanilla_encode(encoder,source,out,a.frames,rate,4)
                    aus=read_au_bits(out/'au_bits.csv',stream,a.frames)
                    decision_rows=native_decision_rows(out/'states.txt',a.frames)
                elif case=='live_zero':
                    stream,decision_rows,aus=encode(a.repo,encoder,source,out,a.frames,ckpt,rate,
                        deterministic=True,zero=True)
                else:
                    stream,aus=encode_fixed_map(encoder,source,out,a.frames,rate,maps[case],4)
                    decision_rows=native_decision_rows(out/'states.txt',a.frames)
                    for row in decision_rows:row['action']=maps[case][row['poc']]
                decoded,mapping=decode_detect(a.decoder,detector,stream,out,a.frames,vid,4)
                joined,metrics=join(decision_rows,aus,source_boxes,decoded,rate*1000,RewardConfig(),vid)
                metrics['selection_score']=metrics['reward_sum']
                if case=='vanilla0':
                    base_stream,base_metrics=stream,metrics
                if case.startswith('vanilla') or case in ('fixed_zero','live_zero'):
                    require_zero_identity(stream,base_stream,[{'action':[0]*12}],metrics,base_metrics,out)
                row={'rate':rate,'case':case,'stream_sha256':sha256_file(stream),'alignment':mapping,
                     'stream_changed':sha256_file(stream)!=sha256_file(base_stream),
                     'metrics':metrics,'delta_score':metrics['score']-base_metrics['score'],
                     'delta_f1':metrics['f1']-base_metrics['f1'],
                     'delta_kbps':metrics['actual_kbps']-base_metrics['actual_kbps']}
                atomic_json(out/'summary.json',row)
                report['results'].append(row)
                atomic_json(a.output/'headroom.json',report)
                print('V4_HEADROOM_CASE '+str(rate)+' '+case+' delta_score='+str(row['delta_score']),flush=True)
        if sha256_file(ckpt)!=ckpt_sha or sha256_file(source)!=report['source_sha256']:
            raise ValueError('Diagnostic source/checkpoint mutated')
        report.update(native_reproducibility='PASS',
            observed_probe_gain=any(r['delta_score']>1e-4 for r in report['results']),
            gate='AGENT_V42_HEADROOM_DIAGNOSTIC_PASS')
    except BaseException as exc:
        report.update(gate='AGENT_V42_HEADROOM_DIAGNOSTIC_FAIL',failure=str(exc))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began
        atomic_json(a.output/'headroom.json',report)
        cleanup_raw(a.output)
        print('V4_HEADROOM_RESULTS='+str(export_results(a.output)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','decoder','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--rates',type=int,nargs='+',choices=(50,80,120,160,200),default=[50,120])
    p.add_argument('--frames',type=int,default=256)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--max-hours',type=float,default=1.5)
    run(p.parse_args())


if __name__=='__main__':main()
