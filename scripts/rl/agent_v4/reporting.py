"""Derived tables and export coverage; never training input or selection rules."""
import csv
import json
import math
import os
from pathlib import Path

from full_train import atomic_json
from real_reward_pilot import sha256_file
from training_session import contained_path


def flatten(value,prefix=''):
    out={}
    for k,v in value.items():
        name=prefix+k
        if isinstance(v,dict):out.update(flatten(v,name+'.'))
        else:out[name]=json.dumps(v) if isinstance(v,(list,tuple)) else v
    return out


def table(path,rows):
    fields=sorted({k for r in rows for k in r}) or ['no_completed_records']
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)


def pair_row(summary,metadata,method='policy'):
    m=summary.get('metrics',{})
    tp,fp,fn=(m.get(k) for k in ('tp','fp','fn'))
    row={**metadata,'method':method,**{k:summary.get(k) for k in
        ('video_id','split','index','rate','frames','start_frame','seed','deterministic','policy_sha256','elapsed_seconds')},
        **flatten(m,'metrics.'),**flatten(summary.get('vanilla',{}),'vanilla.'),
        **{k:summary.get(k) for k in ('delta_f1','delta_kbps','delta_score')},
        **flatten(summary.get('action_diagnostics',{}),'action.')}
    if tp is not None and fp is not None:row['precision']=tp/(tp+fp) if tp+fp else 0.
    if tp is not None and fn is not None:row['recall']=tp/(tp+fn) if tp+fn else 0.
    return row


def plots(out,p,updates,heatmap=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:return {'available':False,'reason':str(exc)}
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,key in zip(axes.flat,('policy_loss','value_loss','entropy','exact_final_kl')):
        pairs=[(r['frame_transitions'],r.get('metrics.'+key)) for r in updates]
        pairs=[(x,y) for x,y in pairs if isinstance(y,(int,float)) and math.isfinite(y)]
        if pairs:ax.plot(*zip(*pairs),marker='.',markersize=3)
        ax.set(xlabel='Training frame transitions',ylabel=key);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'ppo_learning.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    frames=p.get('experiment',{}).get('frames',256)
    for ax,key in zip(axes,('macro_delta_f1','macro_delta_kbps')):
        for kind in ('quick','full'):
            vv=[v for v in p.get('validations',[]) if v['kind']==kind and key in v]
            if vv:ax.plot([v['episodes']*frames for v in vv],[v[key] for v in vv],marker='o',label=kind+' greedy')
        if key=='macro_delta_f1':
            seeds=sorted({d['seed'] for v in p.get('validations',[]) for d in v.get('stochastic_diagnostics',[])})
            for seed in seeds:
                xy=[(v['episodes']*frames,d[key]) for v in p.get('validations',[]) for d in v.get('stochastic_diagnostics',[]) if d['seed']==seed and key in d]
                if xy:ax.plot(*zip(*xy),linestyle='--',label='sampled seed '+str(seed))
        ax.axhline(0,color='gray',linewidth=.7);ax.set(xlabel='Training frame transitions',ylabel=key);ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'validation_learning.png',dpi=140);plt.close(fig)
    files=['ppo_learning.png','validation_learning.png']
    if heatmap is not None:
        fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(heatmap,vmin=-2,vmax=2,cmap='coolwarm')
        for y in range(3):
            for x in range(4):ax.text(x,y,f'{heatmap[y][x]:.3f}',ha='center',va='center')
        ax.set(title='Latest greedy Val: mean actor delta-QP',xlabel='CTU column',ylabel='CTU row')
        fig.colorbar(im,ax=ax,label='Actor delta, not final encoder QP')
        fig.tight_layout();fig.savefig(out/'delta_qp_heatmap.png',dpi=140);plt.close(fig)
        files.append('delta_qp_heatmap.png')
    return {'available':True,'files':files}


def write_reports(root):
    root=Path(root);p=json.loads((root/'progress.json').read_text())
    out=root/'reports';out.mkdir(exist_ok=True)
    missing=[];checked=set();cases=[];pairs=[];checkpoints=[];heatmaps=[]
    all_summaries={}
    for path in sorted(root.rglob('summary.json')):
        if 'reports' in path.parts:continue
        try:s=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if 'rollout_sha256' in s:all_summaries.setdefault(s['rollout_sha256'],[]).append((path.parent,s))

    def require(path):
        name=path.relative_to(root).as_posix();checked.add(name)
        if not path.is_file():missing.append({'path':name,'reason':'missing committed artifact'})

    def case(folder,s,meta):
        folder=contained_path(root,folder.relative_to(root).as_posix())
        for name in ('summary.json','rollout.jsonl','source_detections.jsonl','source_ffmpeg.log',
                     'online/ppo.266','online/states.txt','online/au_bits.csv','online/encoder.log',
                     'online/decoder.log','online/decoded_detections.jsonl','online/trajectory.jsonl'):
            require(folder/name)
        if 'vanilla' in s:
            for name in ('vanilla.266','states.txt','au_bits.csv','encoder.log','decoder.log','decoded_detections.jsonl'):
                require(folder/'vanilla'/name)
        for control in s.get('spatial_controls',{}):
            for name in ('summary.json','fixed.266','fixed_map.txt','states.txt','au_bits.csv','encoder.log','decoder.log','decoded_detections.jsonl'):
                require(folder/'controls'/control/name)
        rollout=folder/'rollout.jsonl'
        if rollout.is_file():
            if sha256_file(rollout)!=s.get('rollout_sha256'):
                missing.append({'path':rollout.relative_to(root).as_posix(),'reason':'rollout checksum mismatch'})
            else:
                try:
                    n=0;seen=set()
                    with rollout.open() as f:
                        for line in f:
                            r=json.loads(line);n+=1;seen.add(r['poc'])
                    if n!=s['frames'] or seen!=set(range(s['frames'])):raise ValueError('Incomplete POC set')
                except (ValueError,KeyError) as exc:
                    missing.append({'path':rollout.relative_to(root).as_posix(),'reason':str(exc)})
        cases.append({'path':folder.relative_to(root).as_posix(),**meta})
        hm=s.get('action_diagnostics',{}).get('per_ctu_mean_delta_qp')
        if s.get('deterministic') and hm is not None:heatmaps.append((meta.get('episode',0),meta.get('phase'),hm))
        pairs.append(pair_row(s,meta))
        for name,c in s.get('spatial_controls',{}).items():
            cs={**s,'metrics':c['metrics'],'action_diagnostics':{},
                'delta_f1':c.get('delta_f1_vs_vanilla'),'delta_kbps':c.get('delta_kbps_vs_vanilla'),
                'delta_score':c.get('delta_score_vs_vanilla')}
            pairs.append(pair_row(cs,{**meta,'selection_eligible':False},name))

    for e in p.get('completed',[]):
        folder=contained_path(root,e['episode']);require(folder/'summary.json')
        if (folder/'summary.json').is_file():
            s=json.loads((folder/'summary.json').read_text())
            if s.get('rollout_sha256')!=e.get('rollout_sha256'):
                missing.append({'path':e['episode'],'reason':'summary differs from committed rollout'})
            case(folder,s,{'phase':'train','episode':e.get('step'),'plan':json.dumps(e.get('plan',{}))})
    for state_file in sorted(root.glob('validation_*/progress.json')):
        state=json.loads(state_file.read_text());cfg=state.get('config',{})
        recorded=next((v for v in p.get('validations',[]) if v['episodes']==cfg.get('episodes') and v['kind']==cfg.get('kind')),None)
        if recorded:
            greedy=sum(bool(s.get('deterministic')) for s in state.get('completed',[]))
            sampled=len(state.get('completed',[]))-greedy
            if greedy!=recorded['pairs'] or sampled!=recorded.get('diagnostic_pairs',0):
                missing.append({'path':state_file.relative_to(root).as_posix(),'reason':'Validation pair count differs from committed metrics'})
        for s in state.get('completed',[]):
            candidates=[(f,t) for f,t in all_summaries.get(s.get('rollout_sha256'),[]) if f.parent==state_file.parent and t==s]
            if not candidates:
                missing.append({'path':state_file.parent.relative_to(root).as_posix(),
                    'reason':'missing raw Validation pair','video_id':s.get('video_id'),'rate':s.get('rate'),'seed':s.get('seed')})
            else:case(*candidates[0],{'phase':cfg.get('kind'),'episode':cfg.get('episodes'),
                'checkpoint_iteration':cfg.get('checkpoint_iteration'),'selection_eligible':bool(s.get('deterministic')) and cfg.get('kind')=='full'})
    for v in p.get('validations',[]):require(root/f"validation_{v['episodes']:06d}_{v['kind']}"/'progress.json')
    expected={p['latest_checkpoint']:p['latest_sha256']}
    for v in p.get('validations',[]):expected[v['checkpoint']]=v['sha256']
    pending=p.get('pending_validation')
    if pending:expected[pending['checkpoint']]=pending['sha256']
    if p.get('best_checkpoint'):expected[p['best_checkpoint']]=p['best_sha256']
    initial=root/'checkpoints/iter_000000.pt'
    if initial.exists():expected.setdefault('checkpoints/iter_000000.pt',sha256_file(initial))
    for name,h in expected.items():
        path=contained_path(root,name);require(path)
        if path.is_file() and sha256_file(path)!=h:missing.append({'path':name,'reason':'checkpoint checksum mismatch'})
        checkpoints.append({'path':name,'sha256':h,'latest':name==p['latest_checkpoint'],
            'best':name==p.get('best_checkpoint'),'validation_episodes':[v['episodes'] for v in p.get('validations',[]) if v['checkpoint']==name]})
    updates=[];frames=0
    for u in p.get('updates',[]):
        frames+=u.get('transitions',0)
        updates.append({**flatten(u),'frame_transitions':frames,'completed_episodes':max(u.get('steps') or [0])})
    table(out/'updates.csv',updates)
    table(out/'pairs.csv',pairs)
    table(out/'validation_history.csv',[flatten(v) for v in p.get('validations',[])])
    table(out/'checkpoint_index.csv',[flatten(v) for v in checkpoints])
    atomic_json(out/'checkpoint_index.json',checkpoints)
    atomic_json(out/'case_index.json',cases)
    # Plotting is derived and optional; failures must not suppress raw evidence.
    latest=max(((ep,phase) for ep,phase,hm in heatmaps),default=None)
    hm=[m for ep,phase,m in heatmaps if (ep,phase)==latest]
    heatmap=[[sum(m[y][x] for m in hm)/len(hm) for x in range(4)] for y in range(3)] if hm else None
    try:plot_status=plots(out,p,updates,heatmap)
    except Exception as exc:plot_status={'available':False,'reason':str(exc)}
    complete={'artifact_contract':'cumulative-evidence-v1','committed_evidence_complete':not missing,
        'completed_episodes':len(p.get('completed',[])),'updates':len(p.get('updates',[])),
        'training_frame_transitions':frames,'committed_pairs':len(cases),'required_paths_checked':len(checked),
        'missing_or_invalid':missing,'plots':plot_status,
        'raw_yuv_excluded_by_design':True,'uncommitted_attempts_excluded_from_tables':True,
        'note':'Artifact coverage is separate from epoch completion and quality claims.'}
    atomic_json(out/'completeness.json',complete)
    return complete
