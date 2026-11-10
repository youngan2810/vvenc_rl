"""Contract tests for the redesigned agent; no synthetic result is a codec gain."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from agent_v4 import SCHEMA, STATE_VERSION
from agent_v4.checkpoint import load, save
from agent_v4.model import Agent, ModelConfig, PPOConfig, Trainer, advantages
from agent_v4.reward import RewardConfig, score, occupancy, join, native_decision_rows
from agent_v4.session import export_resume, restore, export_results
from agent_v4.state import StateBuilder, SourceFrames, FRAME_BYTES, geometry, stack_states
from au_rate_audit import AUBits
from reward_metrics import Detection, DetectionCounts

torch.set_num_threads(1)


def line(poc, step, frames=32, activity=.02):
    values = [.6,step/frames,.01*step,1-.01*step,.6,.65,float(step == 0),.5]
    values += [activity]*12+[.01]*12+[.65]*12
    return "STATE " + str(poc) + " " + " ".join(str(x) for x in values) + "\n"


def batch(n=8):
    b = StateBuilder(n)
    rows = []
    for i in range(n):
        poc, s = b.observe(line(i, i, n, .01+i*.002))
        rows.append({"poc":poc, **{k:v.tolist() for k,v in s.items()}})
        b.commit_action([i%5-2]*12)
    return stack_states(rows, None, False)


def experiment(mc, pc):
    return {"model_config":asdict(mc), "ppo_config":asdict(pc), "reward":asdict(RewardConfig())}


class StateContracts(unittest.TestCase):
    def test_content_mmap_is_released_before_file_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"source.yuv"
            p.write_bytes(bytes([100])*FRAME_BYTES)
            with SourceFrames(p,1) as source:
                tensor = source.thumbnail(0)
            p.unlink()  # Windows catches an unreleased memory map here.
            self.assertEqual(float(tensor.float().mean()),100.)
            with self.assertRaisesRegex(ValueError,"closed"):
                source.thumbnail(0)

    def test_missing_source_cannot_silently_disable_content(self):
        with self.assertRaisesRegex(ValueError,"actual source"):
            stack_states([],None,True)

    def test_causal_history_and_reordered_poc(self):
        b = StateBuilder(3)
        _, s0 = b.observe(line(0,0,3))
        self.assertEqual(float(s0["history"].sum()), 0)
        b.commit_action([-2]*12)
        _, s1 = b.observe(line(2,1,3))
        self.assertEqual(float(s1["history"][-1,-1]), 1)
        self.assertTrue(torch.equal(s1["local"][3], -torch.ones(3,4)))
        b.commit_action([2]*12)
        _, s2 = b.observe(line(1,2,3))
        self.assertTrue(torch.equal(s2["local"][3], torch.ones(3,4)))
        self.assertEqual(float(s2["history"][-2,-1]), 1)

    def test_no_double_observation_or_fractional_action(self):
        b = StateBuilder(2)
        b.observe(line(0,0,2))
        with self.assertRaises(RuntimeError):
            b.observe(line(1,1,2))
        with self.assertRaises(ValueError):
            b.commit_action([.1]*12)
        b.commit_action([0]*12)
        with self.assertRaises(ValueError):
            b.observe(line(0,1,2))

    def test_clock_mismatch_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            StateBuilder(32).observe(line(0,1))
        with self.assertRaises(ValueError):
            StateBuilder(32).observe(line(0,0,activity=float("nan")))

    def test_current_poc_only_and_edge_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.yuv"
            raw = np.zeros((3,FRAME_BYTES), np.uint8)
            raw[1,:480*320] = 100
            raw[2,:480*320] = 240
            path.write_bytes(raw.tobytes())
            frames = SourceFrames(path, 3)
            self.assertEqual(float(frames.thumbnail(1).float().mean()), 100)
            with self.assertRaises(ValueError):
                frames.thumbnail(3)
            del frames
        g = geometry()
        self.assertAlmostEqual(float(g[3,2]), .75)
        self.assertAlmostEqual(float(g[11,2]), .375)
        self.assertAlmostEqual(float(g[:,2].sum())*128*128, 480*320)

    def test_ablation_ignores_new_history(self):
        b = StateBuilder(2, rich=False)
        b.observe(line(0,0,2))
        b.commit_action([-2]*12)
        _, s = b.observe(line(1,1,2))
        self.assertEqual(float(s["history"].abs().sum()), 0)
        self.assertEqual(float(s["local"][3:].abs().sum()), 0)


class RewardContracts(unittest.TestCase):
    def test_baseline_uses_own_native_order_for_legacy_return(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"states.txt"
            path.write_text("".join(line(p,i,3) for i,p in enumerate([2,1,0])))
            rows = native_decision_rows(path,3)
            self.assertEqual([r["poc"] for r in rows],[2,1,0])
            box = Detection(10,10,40,40)
            src = {i:[box] for i in range(3)}
            dec = {0:[box],1:[],2:[]}
            aus = [AUBits(i+1,i,100,800,0) for i in range(3)]
            config = RewardConfig(mode="legacy-frame")
            own,_ = join(rows,aus,src,dec,24000,config,"val")
            wrong,_ = join([{"poc":i} for i in range(3)],aus,src,dec,24000,config,"val")
            discounted = lambda rr:sum(.99**i*r["reward"] for i,r in enumerate(rr))
            self.assertNotAlmostEqual(discounted(own),discounted(wrong),places=6)

    def test_clip_delta_policy_gradient_matches_terminal_objective(self):
        # Enumerate a tiny causal two-decision environment, including a
        # reference effect from action1 onto picture2. This checks objective
        # credit assignment, rather than only the telescoping identity.
        theta = torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        probability = theta.sigmoid()
        expected_score = theta*0
        score_gradient = theta*0
        config = RewardConfig()
        for a1 in (0,1):
            for a2 in (0,1):
                p1 = probability if a1 else 1-probability
                p2 = probability if a2 else 1-probability
                c1 = DetectionCounts(a1,0,1-a1)
                tp2 = int(a1 or a2)
                counts = DetectionCounts(a1+tp2,0,2-a1-tp2)
                bits1 = 800+400*a1
                prefix = score(c1,bits1,1,30000,config)
                final = score(counts,bits1+800+400*a2,2,30000,config)
                expected_score = expected_score+p1*p2*final
                # Return_t = final - prefix_(t-1), gamma=lambda=1.
                score_gradient = score_gradient+(p1*p2).detach()*((a1-probability)*final+
                                      (a2-probability)*(final-prefix))
        derivative = torch.autograd.grad(expected_score,theta)[0]
        self.assertAlmostEqual(float(score_gradient.detach()),float(derivative),places=12)

    def test_telescoping_invariant_to_poc_order(self):
        box = Detection(10,10,40,40)
        src = {0:[box],1:[],2:[box]}
        dec = {0:[box],1:[box],2:[]}
        aus = [AUBits(i+1,p,10*(p+1),80*(p+1),0) for i,p in enumerate([2,0,1])]
        c = RewardConfig()
        r1, m1 = join([{"poc":p} for p in [0,2,1]], aus, src, dec, 120000, c, "train")
        r2, m2 = join([{"poc":p} for p in [2,1,0]], aus, src, dec, 120000, c, "train")
        self.assertAlmostEqual(sum(x["reward"] for x in r1), m1["score"])
        self.assertAlmostEqual(m1["score"], m2["score"])
        self.assertEqual(r1[0]["au_bits"], 80)
        self.assertEqual(sum(r["done"] for r in r1), 1)

    def test_cap_and_symmetric_are_distinct(self):
        counts = DetectionCounts(1,0,0)
        sym = score(counts, 1000, 30, 2000, RewardConfig())
        cap = score(counts, 1000, 30, 2000, RewardConfig(rate_mode="cap"))
        self.assertLess(sym, cap)
        self.assertEqual(cap, 1)
        with self.assertRaises(ValueError):
            RewardConfig(mode="legacy-frame", rate_mode="cap")

    def test_empty_clip_and_invalid_poc(self):
        self.assertEqual(score(DetectionCounts(0,0,0),1000,30,1000,RewardConfig()),0)
        self.assertLess(score(DetectionCounts(0,2,0),1000,30,1000,RewardConfig()),0)
        with self.assertRaises(ValueError):
            join([{"poc":0},{"poc":0}],[],{}, {}, 1000, RewardConfig(), "x")

    def test_overlapping_teacher_boxes_are_union_not_sum(self):
        b = Detection(384,256,480,320)
        areas = occupancy([b,b])
        self.assertEqual(areas[-1],1)
        self.assertEqual(sum(areas),1)
        half = occupancy([Detection(384,256,432,320)])
        self.assertEqual(half[-1],.5)


class PolicyContracts(unittest.TestCase):
    def test_default_content_ppo_matches_single_frame_bridge_and_updates_both_cnns(self):
        torch.manual_seed(71)
        model = Agent().eval()
        trainer = Trainer(model,PPOConfig(epochs=1,minibatch=2))
        state = batch(4)
        state["image"].random_(16,236)
        actions,lps,values = [],[],[]
        with torch.no_grad():
            for i in range(4):
                a,lp,v,_ = model.sample({k:t[i:i+1] for k,t in state.items()})
                actions.append(a);lps.append(lp);values.append(v)
        actor_before = model.actor.image[0].weight.detach().clone()
        critic_before = model.critic.image[0].weight.detach().clone()
        model.train()
        metrics = trainer.update(state,torch.cat(actions),torch.cat(lps),torch.cat(values),
            torch.tensor([.2,-.1,.4,-.3]),torch.tensor([False,True,False,True]),
            torch.tensor([[.1,0]*6,[0,.2]*6]*2),task_outcomes=torch.rand(4,12,3))
        self.assertLess(max(metrics["collector_parity"]),2e-4)
        self.assertLessEqual(metrics["exact_final_kl"],trainer.config.target_kl)
        self.assertFalse(torch.equal(actor_before,model.actor.image[0].weight))
        self.assertFalse(torch.equal(critic_before,model.critic.image[0].weight))

    def test_nan_policy_data_and_critic_are_rejected_before_update(self):
        model = Agent(ModelConfig(content=False))
        trainer = Trainer(model)
        s = batch(2)
        with torch.no_grad():
            a,lp,v,_ = model.sample(s)
        with self.assertRaisesRegex(ValueError,"old_logp"):
            trainer.parity(s,a,lp*float("nan"),v)
        with self.assertRaisesRegex(ValueError,"old_values"):
            trainer.parity(s,a,lp,v*float("nan"))
        with torch.no_grad():
            model.value_head[-1].weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            trainer.parity(s,a,lp,v)
        self.assertEqual(len(trainer.actor_optimizer.state),0)

    def test_last_actor_step_cannot_bypass_full_batch_kl_guard(self):
        torch.manual_seed(42)
        model = Agent(ModelConfig(content=False))
        trainer = Trainer(model,PPOConfig(epochs=1,minibatch=8,actor_lr=.3,target_kl=.001))
        s = batch(8)
        with torch.no_grad():
            a,lp,v,_ = model.sample(s)
        metrics = trainer.update(s,a,lp,v,torch.linspace(-1,1,8),
            torch.tensor([False]*7+[True]),torch.ones(8,12),task_outcomes=torch.rand(8,12,3))
        self.assertGreater(metrics["proposed_final_kl"],trainer.config.target_kl)
        self.assertLessEqual(metrics["exact_final_kl"],trainer.config.target_kl)
        self.assertLess(metrics["accepted_actor_scale"],1)
        self.assertTrue(metrics["actor_optimizer_reset"])
        self.assertGreater(len(trainer.critic_optimizer.state),0)
        # No stale full-step Adam moments survive a partial accepted step.
        self.assertEqual(len(trainer.actor_optimizer.state),0)

    def test_action_only_control_preserves_legacy_backbone(self):
        from ppo_agent import PPOActorCritic
        legacy = PPOActorCritic()
        control = Agent(ModelConfig(False,False,False))
        for name in ("spatial_encoder","spatial_norm","global_encoder","fusion","critic_encoder","critic"):
            self.assertEqual(str(getattr(legacy,name)), str(getattr(control.control,name)))
        s = batch(2)
        self.assertEqual(control(s)[0].probs.shape,(2,12,5))
        self.assertEqual(len(set(map(id,control.actor_parameters())) & set(map(id,control.critic_parameters()))),0)

    def test_actor_only_inference_has_identical_actions_and_skips_critic(self):
        for config in (ModelConfig(),ModelConfig(False,False,False)):
            m = Agent(config).eval()
            s = batch(2)
            expected = m.sample(s,True)
            critic = m.control.critic if hasattr(m,"control") else m.critic
            with patch.object(critic,"forward",side_effect=RuntimeError("critic must not run")):
                actual = m.sample(s,True,actor_only=True)
            self.assertTrue(torch.equal(expected[0],actual[0]))
            self.assertTrue(torch.equal(expected[1],actual[1]))

    def test_categorical_matches_integer_probabilities(self):
        model = Agent(ModelConfig(content=False))
        s = batch()
        action, lp, value, probs = model.sample(s)
        self.assertTrue(torch.all((action >= -2) & (action <= 2)))
        manual = probs.gather(-1,(action+2).unsqueeze(-1)).squeeze(-1).log().sum(-1)
        self.assertTrue(torch.allclose(lp,manual,atol=1e-5))
        deterministic = model.sample(s, True)[0]
        self.assertTrue(torch.equal(deterministic,probs.argmax(-1)-2))
        self.assertEqual(len(set(map(id,model.actor_parameters())) & set(map(id,model.critic_parameters()))),0)

    def test_content_and_current_features_receive_aux_gradient(self):
        model = Agent()
        s = batch(2)
        s["image"].random_(16,236)
        _, _, aux = model(s)
        aux.square().mean().backward()
        self.assertGreater(float(model.actor.image[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.actor.local[0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.critic.local[0].weight.grad)

    def test_complete_episode_returns_no_cross_boundary(self):
        r = torch.tensor([1.,2.,10.,20.])
        d = torch.tensor([False,True,False,True])
        _, returns = advantages(r,torch.zeros(4),d)
        self.assertTrue(torch.equal(returns,torch.tensor([3.,2.,30.,20.])))
        with self.assertRaises(ValueError):
            advantages(r,torch.zeros(4),torch.zeros(4,dtype=torch.bool))

    def test_real_optimizer_parity_aux_and_reload(self):
        torch.manual_seed(8)
        model = Agent(ModelConfig(content=False))
        trainer = Trainer(model,PPOConfig(epochs=2,minibatch=4))
        s = batch()
        with torch.no_grad():
            act, lp, value, _ = model.sample(s)
        before_actor = model.action_head.weight.detach().clone()
        before_critic = model.value_head[-1].weight.detach().clone()
        rewards = torch.arange(8).float()/10
        dones = torch.zeros(8,dtype=torch.bool)
        dones[3] = dones[-1] = True
        metrics = trainer.update(s,act,lp,value,rewards,dones,torch.rand(8,12),task_outcomes=torch.rand(8,12,3))
        self.assertGreater(metrics["actor_steps"],0)
        self.assertFalse(torch.equal(before_actor,model.action_head.weight))
        self.assertFalse(torch.equal(before_critic,model.value_head[-1].weight))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"v4.pt"
            exp = experiment(model.config, trainer.config)
            save(path,model,trainer,exp,1)
            _, restored, tr = load(path,exp,train=True)
            self.assertTrue(torch.equal(model.sample(s,True)[0],restored.sample(s,True)[0]))
            self.assertTrue(torch.equal(model(s)[1],restored(s)[1]))
            self.assertEqual(len(tr.actor_optimizer.state),len(trainer.actor_optimizer.state))
            with self.assertRaises(ValueError):
                load(path,{**exp,"changed":True})

    def test_off_policy_and_legacy_checkpoints_rejected(self):
        model = Agent(ModelConfig(content=False))
        tr = Trainer(model)
        s = batch(2)
        with torch.no_grad():
            a,lp,v,_ = model.sample(s)
        with self.assertRaises(ValueError):
            tr.parity(s,a,lp+1,v)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"old.pt"
            torch.save({"model_version":"spatial-ctu-independent-ln-v3"},path)
            with self.assertRaises(ValueError):
                load(path)


class ExportAndIPCContracts(unittest.TestCase):
    def test_resume_roundtrip_and_corrupt_or_unsafe_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"run"
            (root/"checkpoints").mkdir(parents=True)
            (root/"checkpoints/latest.pt").write_bytes(b"checkpoint")
            p = {"latest_checkpoint":"checkpoints/latest.pt", "validations":[],
                 "latest_sha256":hashlib.sha256(b"checkpoint").hexdigest()}
            (root/"progress.json").write_text(json.dumps(p))
            archive = export_resume(root,p)
            restored = Path(tmp)/"restored"
            restore(archive,restored)
            self.assertEqual((restored/"checkpoints/latest.pt").read_bytes(),b"checkpoint")
            unsafe = Path(tmp)/"unsafe.zip"
            with zipfile.ZipFile(unsafe,"w") as z:
                z.writestr("../escape",b"x")
                z.writestr("bundle_manifest.json",json.dumps({"schema":SCHEMA,"files":{"../escape":"bad"}}))
            with self.assertRaises(ValueError):
                restore(unsafe,Path(tmp)/"noescape")
            corrupt = Path(tmp)/"corrupt.zip"
            with zipfile.ZipFile(corrupt,"w") as z:
                z.writestr("data",b"bad")
                z.writestr("bundle_manifest.json",json.dumps({"schema":SCHEMA,"files":{"data":"wrong"}}))
            with self.assertRaises(ValueError):
                restore(corrupt,Path(tmp)/"nocorrupt")
            self.assertFalse((Path(tmp)/"nocorrupt").exists())

    def test_corrupt_checkpoint_cannot_replace_last_good_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root/"latest.pt"
            ckpt.write_bytes(b"original")
            p = {"latest_checkpoint":"latest.pt","latest_sha256":hashlib.sha256(b"original").hexdigest(),
                 "validations":[]}
            (root/"progress.json").write_text(json.dumps(p))
            archive = export_resume(root,p)
            before = archive.read_bytes()
            ckpt.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                export_resume(root,p)
            self.assertEqual(archive.read_bytes(),before)
            ckpt.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError,"committed"):
                export_resume(root,{**p,"uncommitted_change":True})
            self.assertEqual(archive.read_bytes(),before)

    def test_collectors_reject_wrong_or_dirty_source(self):
        from agent_v4.runtime import verify_source
        with patch("agent_v4.runtime.subprocess.check_output",return_value="wrong"):
            with self.assertRaisesRegex(ValueError,"source_commit"):
                verify_source(Path("repo"),{"source_commit":"expected"})
        with patch("agent_v4.runtime.subprocess.check_output",side_effect=["expected"," M model.py"]):
            with self.assertRaisesRegex(ValueError,"clean"):
                verify_source(Path("repo"),{"source_commit":"expected"})
        with patch("agent_v4.runtime.subprocess.check_output",side_effect=["expected",""]):
            verify_source(Path("repo"),{"source_commit":"expected"})

    def test_evidence_export_keeps_detections_bits_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.yuv","source_detections.jsonl","stream.266","alignment.json","trajectory.jsonl"):
                (root/name).write_bytes(b"x")
            result = export_results(root)
            with zipfile.ZipFile(result) as z:
                self.assertNotIn("source.yuv",z.namelist())
                for name in ("source_detections.jsonl","stream.266","alignment.json","trajectory.jsonl"):
                    self.assertIn(name,z.namelist())

    def test_bridge_handles_partial_lines_and_reordering_with_exact_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Agent(ModelConfig(content=False))
            trainer = Trainer(model)
            checkpoint = root/"ckpt.pt"
            save(checkpoint,model,trainer,experiment(model.config,trainer.config),0)
            source = root/"source.yuv"
            source.write_bytes(bytes(FRAME_BYTES*3))
            state,trajectory,ready,actions = root/"states",root/"trajectory",root/"ready",root/"actions"
            script = Path(__file__).with_name("agent_v4_bridge.py")
            with (root/"bridge.log").open("w") as log:
                proc = subprocess.Popen([sys.executable,str(script),"--state-file",str(state),
                    "--action-dir",str(actions),"--trajectory-file",str(trajectory),
                    "--checkpoint",str(checkpoint),"--source-yuv",str(source),
                    "--ready-file",str(ready),"--frames","3","--seed","9","--idle-timeout","10"],
                    stdout=log,stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+15
                    while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(ready.exists())
                    with state.open("w") as f:
                        for i,poc in enumerate((0,2,1)):
                            text = line(poc,i,3)
                            f.write(text[:20]); f.flush()
                            time.sleep(.04)
                            self.assertFalse((actions/f"{poc}.txt").exists())
                            f.write(text[20:]); f.flush()
                            while not (actions/f"{poc}.txt").exists() and proc.poll() is None and time.monotonic() < deadline:
                                time.sleep(.01)
                            self.assertTrue((actions/f"{poc}.txt").exists())
                    self.assertEqual(proc.wait(timeout=15),0)
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
            rows = [json.loads(s) for s in trajectory.read_text().splitlines()]
            self.assertEqual([r["poc"] for r in rows],[0,2,1])
            s = stack_states(rows,None,False)
            tr = Trainer(model)
            tr.parity(s,torch.tensor([r["action"] for r in rows]),
                torch.tensor([r["log_prob"] for r in rows]),torch.tensor([r["value"] for r in rows]))


class CoordinatorContracts(unittest.TestCase):
    """Real v4 orchestration/optimizer/export, with synthetic codec fixtures.

    These fixtures test transactional training, not codec quality or CUDA.
    """
    def test_two_collectors_commit_and_portable_resume(self):
        import agent_v4.train as train
        from real_reward_pilot import sha256_file
        calls = []
        commits = []
        original_commit = train.commit_progress
        def capture_commit(root,progress):
            original_commit(root,progress)
            commits.append(json.loads((root/'progress.json').read_text()))

        def fake_jobs(jobs, repo, decoder, checkpoint, input_root, timeout):
            ckpt, model, _ = load(checkpoint)
            for job in jobs:
                out, frames = job["output"], job["frames"]
                out.mkdir(parents=True)
                source = out/"reference_8bit.yuv"
                source.write_bytes(bytes(FRAME_BYTES*frames))
                builder = StateBuilder(frames, model.config.rich_state)
                rows = []
                torch.manual_seed(job["seed"])
                for i in range(frames):
                    poc, state = builder.observe(line(i,i,frames))
                    s = {k:v[None] for k,v in state.items()}
                    s["image"] = torch.zeros(1,1,160,240,dtype=torch.uint8)
                    with torch.no_grad():
                        action,lp,value,_ = model.sample(s,job["split"] == "val" and not job.get('stochastic_validation'))
                    row = {"poc":poc, **{k:v.tolist() for k,v in state.items()},
                        "action":action[0].tolist(), "log_prob":float(lp[0]), "value":float(value[0]),
                        "reward":float(action.float().mean())*.01, "done":i == frames-1,
                        "trainable":job["split"] == "train", "teacher_occupancy":[.1 if i%2 else 0.]*12,
                        'task_outcomes':[[.1,.2,.3]]*12}
                    rows.append(row)
                    builder.commit_action(row["action"])
                (out/"rollout.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                score = ((.95 if job.get('stochastic_validation') else .5) if job['split']=='val'
                         else sum(r['reward'] for r in rows))
                summary = {"split":job["split"],"index":job["index"],"video_id":f'{job["split"]}{job["index"]}',
                    "frames":frames,"rate":job["rate"],"start_frame":job["start"],"seed":job["seed"],
                    "deterministic":job["split"] == "val" and not job.get('stochastic_validation',False), "policy_sha256":sha256_file(checkpoint),
                    "experiment":ckpt["experiment"],"source_sha256":sha256_file(source),
                    "rollout_sha256":sha256_file(out/"rollout.jsonl"),
                    "metrics":{"score":score,"selection_score":score,"f1":.5,"actual_kbps":job["rate"]},
                    "action_histogram":{"0":frames*12},"delta_score":score-.5,"delta_f1":0.,"delta_kbps":0.,
                    'vanilla':{'score':.5}}
                (out/"summary.json").write_text(json.dumps(summary))
            calls.append((len(jobs),str(checkpoint),[j["split"] for j in jobs]))

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            repo = root/"repo"
            enc = repo/"bin/release-static/vvencapp"
            enc.parent.mkdir(parents=True)
            enc.write_bytes(b"encoder-fixture")
            decoder = root/"vvdecapp"
            decoder.write_bytes(b"decoder-fixture")
            (root/"vvdecapp_poc_audit").write_bytes(b"auditor-fixture")
            a = Namespace(repo=repo,decoder=decoder,output=root/"run",input_root=root/"input",
                resume_archive=None,preset="content-task",reward=None,rate_mode="symmetric",
                rate_tolerance=0.,beta=.1,auxiliary=0.,frames=32,validation_frames=32,epochs=1,
                max_episodes=2,seed=19,workers=2,val_every_updates=25,worker_timeout=60,max_hours=10.)
            splits = {"train":[f"train{i}" for i in range(65)],"val":[f"val{i}" for i in range(7)],
                      "test":[f"test{i}" for i in range(100)]}
            for target,value in [("torch.cuda.is_available",lambda:True),("torch.cuda.device_count",lambda:2),
                ("agent_v4.train.ENCODER_SHA",sha256_file(enc)),("agent_v4.train.DECODER_SHA",sha256_file(decoder)),
                ("agent_v4.train.read_lock_manifest",lambda x:(splits,{})),
                ("agent_v4.train.read_baseline_config",lambda x:{"lock_sha256":"lock"}),
                ("agent_v4.train.initialize_detector_assets",lambda *x:None),
                ("agent_v4.train.dataset_source",lambda *x:Path("train.mov")),
                ("agent_v4.train.video_frame_capacity",lambda x:1000),
                ("agent_v4.train.subprocess.check_output",lambda cmd,**kw:"a"*40 if "rev-parse" in cmd else ""),
                ("agent_v4.train.run_jobs",fake_jobs),
                ("agent_v4.train.commit_progress",capture_commit)]:
                stack.enter_context(patch(target,value))
            train.run(a)
            first = json.loads((a.output/"progress.json").read_text())
            self.assertEqual(len(first["completed"]),2)
            self.assertEqual(len(first["updates"]),1)
            self.assertEqual(first["updates"][0]["transitions"],64)
            first_update_commit = next(p for p in commits if len(p['updates'])==1)
            self.assertEqual(first_update_commit['pending_validation']['episodes'],2)
            self.assertEqual(first_update_commit['pending_validation']['kind'],'quick')
            self.assertTrue(any(n == 2 and split == ["train","train"] for n,_,split in calls))
            self.assertNotIn("best_checkpoint",first)  # quick probes cannot select best
            self.assertEqual(first['validations'][0]['mean_score'],.5)
            self.assertEqual(first['validations'][0]['stochastic_diagnostics'][0]['mean_score'],.95)
            self.assertEqual(first['validations'][0]['pairs'],4)
            self.assertEqual(first['validations'][0]['diagnostic_pairs'],8)
            a.resume_archive = a.output/"VVC_AGENT_V4_RESUME.zip"
            a.output = root/"restored"
            train.run(a)
            second = json.loads((a.output/"progress.json").read_text())
            self.assertEqual(len(second["completed"]),4)
            self.assertEqual(len(second["updates"]),2)
            self.assertEqual(second["updates"][1]["collecting_policy_sha256"],first["latest_sha256"])
            self.assertTrue((a.output/"VVC_AGENT_V4_RESULTS.zip").is_file())
            a.resume_archive = None
            with patch("agent_v4.train.run_jobs",side_effect=RuntimeError("one collector failed")):
                with self.assertRaisesRegex(RuntimeError,"one collector failed"):
                    train.run(a)
            failed = json.loads((a.output/"progress.json").read_text())
            self.assertEqual(len(failed["completed"]),4)
            self.assertEqual(failed["latest_sha256"],second["latest_sha256"])
            self.assertEqual(len(failed["updates"]),2)
            with patch('agent_v4.train.initialize_detector_assets',side_effect=RuntimeError('asset setup failed')):
                with self.assertRaisesRegex(RuntimeError,'asset setup failed'):
                    train.run(a)
            self.assertTrue((a.output/'VVC_AGENT_V4_RESULTS.zip').is_file())
            bad = dict(second)
            bad["completed"] = list(second["completed"])
            bad["completed"][0] = {**bad["completed"][0],"step":9}
            with self.assertRaises(ValueError):
                train.valid_progress(a.output,bad,bad["experiment"],train.create_schedule(a.seed,a.epochs))


if __name__ == "__main__":
    unittest.main()

