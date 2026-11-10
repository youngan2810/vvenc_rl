"""Measured action/learning diagnostics, never extra actor rewards or observations."""
import collections
import numpy as np
import torch


def action_diagnostics(rows):
    actions=np.asarray([r['action'] for r in rows])
    if (actions.ndim!=2 or actions.shape[1]!=12 or len(actions)==0 or
            not np.issubdtype(actions.dtype,np.integer) or not np.isin(actions,[-2,-1,0,1,2]).all()):
        raise ValueError('Expected complete legal CTU actions')
    result={'frames':len(actions),'nonzero_ctu_fraction':float((actions!=0).mean()),
        'mean_delta_qp':float(actions.mean()),
        'mixed_delta_frame_fraction':float((np.ptp(actions,axis=1)>0).mean()),
        'per_ctu_mean_delta_qp':actions.mean(0).reshape(3,4).tolist(),
        'distinct_frame_maps':len({tuple(a) for a in actions})}
    if all('probabilities' in r for r in rows):
        probs=np.asarray([r['probabilities'] for r in rows],dtype=np.float64)
        if (probs.shape!=(len(rows),12,5) or not np.isfinite(probs).all() or
                (probs<0).any() or not np.allclose(probs.sum(-1),1,atol=2e-6,rtol=0)):
            raise ValueError('Invalid categorical probabilities')
        avg=probs.mean(1,keepdims=True)
        kl=np.where(probs>0,probs*(np.log(probs.clip(1e-30))-np.log(avg.clip(1e-30))),0)
        result['within_frame_policy_js']=float(kl.sum(-1).mean())
    return result


def fixed_control_maps(rows,frames):
    """Open-loop modal-frame replay, NOT a separately trained frame controller.

    Modal CTU delta per POC; ties prefer smaller absolute delta, then lower
    integer. No GT, detection or quality metric is consulted. Actual bitrate
    need not match either the policy or the uniform-minus1 intervention.
    """
    action_diagnostics(rows)
    if len(rows)!=frames or {r['poc'] for r in rows}!=set(range(frames)):
        raise ValueError('Control maps require every unique POC')
    replay=[None]*frames
    for r in rows:
        counts=collections.Counter(r['action']);most=max(counts.values())
        mode=min((a for a,n in counts.items() if n==most),key=lambda a:(abs(a),a))
        replay[r['poc']]=[int(mode)]*12
    return {'uniform_minus1':[[-1]*12 for _ in range(frames)],'frame_uniform_replay':replay}


def spatial_credit_diagnostics(advantage,baseline,probs,q=None):
    """Within-frame variation of FROZEN pre-fit credit; not a causal proof."""
    result={'advantage_within_frame_std':float(advantage.std(1,unbiased=False).mean()),
        'advantage_frame_mean_std':float(advantage.mean(1).std(unbiased=False)),
        'baseline_within_frame_std':float(baseline.std(1,unbiased=False).mean())}
    mean=probs.mean(1,keepdim=True)
    result['policy_within_frame_js']=float((probs*(probs.clamp_min(1e-30).log()-mean.clamp_min(1e-30).log())).sum(-1).mean())
    if q is not None:result['predicted_action_q_std']=float(q.std(-1,unbiased=False).mean())
    return result


def gradient_group_indices(model,parameters):
    lookup={id(p):i for i,p in enumerate(parameters)}
    groups={}
    for name,p in model.named_parameters():
        if id(p) not in lookup:continue
        group=('source_image' if name.startswith('actor.image.') else
            'local_state' if name.startswith('actor.local.') else
            'global_history' if name.startswith('actor.context.') else
            'spatial_fusion' if name.startswith('actor.') else
            'policy_head' if name.startswith('action_head.') else
            'auxiliary_heads' if name.startswith(('outcome_head.','occupancy_head.')) else 'legacy_actor')
        groups.setdefault(group,[]).append(lookup[id(p)])
    return groups


def gradient_rms(groups,parameters,**gradients):
    """Raw parameter RMS, before global clipping/Adam; not feature importance."""
    result={}
    for group,indices in groups.items():
        size=sum(parameters[i].numel() for i in indices)
        result[group]={name:float(torch.stack([g[i].square().sum() for i in indices if g[i] is not None]).sum().sqrt()/size**.5)
                       if any(g[i] is not None for i in indices) else 0.
                       for name,g in gradients.items()}
    return result
