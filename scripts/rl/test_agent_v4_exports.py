"""Portable evidence across two sessions, milestone retention and failure exports."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from agent_v4.session import export_resume,export_results,restore
from agent_v4.notebook_support import export_session,read_session_progress,verify_zip
from agent_v4.reporting import write_reports


def sha(data):return hashlib.sha256(data).hexdigest()


class PortableEvidence(unittest.TestCase):
    def fixture(self,base):
        run=base/'run';(run/'checkpoints').mkdir(parents=True)
        exp={'source_commit':'fixture-source','artifact_contract':'cumulative-evidence-v1','frames':2}
        cps={f'checkpoints/iter_{i:06d}.pt':f'weights{i}'.encode() for i in range(3)}
        for n,b in cps.items():(run/n).write_bytes(b)
        def pair(folder,split,cp,greedy):
            folder.mkdir(parents=True)
            data=''.join(json.dumps({'poc':i,'action':[0]*12})+'\n' for i in range(2)).encode()
            s={'split':split,'index':0,'rate':50,'video_id':'video','seed':9,'frames':2,
               'start_frame':0,'deterministic':greedy,'policy_sha256':sha(cps[cp]),'experiment':exp,
               'rollout_sha256':sha(data),'source_sha256':'source',
               'metrics':{'tp':2,'fp':1,'fn':1,'f1':2/3,'score':.6,'actual_kbps':51},
               'vanilla':{'tp':1,'fp':1,'fn':2,'f1':.4,'score':.3,'actual_kbps':50},
               'delta_f1':2/3-.4,'delta_score':.3,'delta_kbps':1,
               'action_diagnostics':{'per_ctu_mean_delta_qp':[[0]*4]*3}}
            files=['source_detections.jsonl','source_ffmpeg.log','online/ppo.266','online/states.txt',
                   'online/au_bits.csv','online/encoder.log','online/decoder.log','online/decoded_detections.jsonl',
                   'online/trajectory.jsonl','vanilla/vanilla.266','vanilla/states.txt','vanilla/au_bits.csv',
                   'vanilla/encoder.log','vanilla/decoder.log','vanilla/decoded_detections.jsonl']
            for n in files:
                f=folder/n;f.parent.mkdir(exist_ok=True,parents=True);f.write_bytes(b'evidence')
            (folder/'rollout.jsonl').write_bytes(data)
            (folder/'summary.json').write_text(json.dumps(s))
            return s
        completed=[]
        for i in range(2):
            folder=run/f'episodes/episode_{i+1:06d}'
            s=pair(folder,'train',f'checkpoints/iter_{i:06d}.pt',False)
            completed.append({'step':i+1,'episode':folder.relative_to(run).as_posix(),
                              'rollout_sha256':s['rollout_sha256'],'metrics':s['metrics']})
        vals=[]
        for i in (0,1):
            out=run/f'validation_{i:06d}_quick'
            s=pair(out/'val00_r50','val',f'checkpoints/iter_{i:06d}.pt',True)
            cfg={'kind':'quick','episodes':i,'checkpoint_iteration':i}
            (out/'progress.json').write_text(json.dumps({'config':cfg,'completed':[s]}))
            vals.append({'kind':'quick','episodes':i,'checkpoint':f'checkpoints/iter_{i:06d}.pt',
                         'sha256':sha(cps[f'checkpoints/iter_{i:06d}.pt']),'pairs':1,
                         'macro_delta_f1':s['delta_f1'],'macro_delta_kbps':1})
        p={'experiment':exp,'purpose':'training','completed':completed,'updates':[
            {'iteration':i+1,'steps':[i+1],'transitions':2,'metrics':{'exact_final_kl':.01}} for i in range(2)],
           'validations':vals,'latest_checkpoint':'checkpoints/iter_000002.pt',
           'latest_sha256':sha(cps['checkpoints/iter_000002.pt'])}
        (run/'progress.json').write_text(json.dumps(p))
        logs=base/'logs';logs.mkdir();(logs/'environment.json').write_text('{"fixture":true}')
        ctx=dict(WORK=base,RUN=run,LOGS=logs,TRAIN_STARTED=True,SOURCE_COMMIT=exp['source_commit'],TRAIN_RC=0)
        return run,p,ctx

    def export(self,ctx):
        with contextlib.redirect_stdout(io.StringIO()):return export_session(ctx)

    def test_all_milestones_are_exported_and_corruption_preserves_last_good(self):
        with tempfile.TemporaryDirectory() as td:
            run,p,ctx=self.fixture(Path(td));z=export_resume(run,p);before=z.read_bytes()
            with zipfile.ZipFile(z) as a:
                self.assertTrue(all(f'checkpoints/iter_{i:06d}.pt' in a.namelist() for i in range(3)))
            (run/'checkpoints/iter_000001.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):export_resume(run,p)
            self.assertEqual(before,z.read_bytes())

    def test_full_session_roundtrip_preserves_old_raw_evidence_and_environment(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,p,ctx=self.fixture(base);outer=self.export(ctx)
            self.assertEqual(read_session_progress(outer),p)
            restored=base/'restored';restore(outer,restored)
            for name in ('episodes/episode_000001/rollout.jsonl','episodes/episode_000001/online/ppo.266',
                         'validation_000000_quick/val00_r50/summary.json','checkpoints/iter_000000.pt'):
                self.assertEqual((restored/name).read_bytes(),(run/name).read_bytes())
            self.assertEqual(len(list(restored.glob('session_history/*/setup_logs/environment.json'))),1)
            ctx2={**ctx,'RUN':restored};ctx2.pop('_EVIDENCE_SESSION_ID',None)
            second=self.export(ctx2);self.assertEqual(read_session_progress(second),p)
            third=base/'third';restore(second,third)
            self.assertEqual(len(list(third.glob('session_history/*/setup_logs/environment.json'))),2)
            self.assertTrue((third/'episodes/episode_000001/online/ppo.266').is_file())

    def test_resume_failure_still_exports_new_results_and_labels_old_resume(self):
        with tempfile.TemporaryDirectory() as td:
            run,p,ctx=self.fixture(Path(td));self.export(ctx)
            (run/'new_failure.log').write_text('new error evidence')
            (run/p['latest_checkpoint']).write_bytes(b'corrupt')
            outer=self.export(ctx)
            with zipfile.ZipFile(outer) as z:
                status=json.loads(z.read('session_status.json'))
                self.assertEqual(status['exports'],{'resume':'stale','results':'fresh'})
                self.assertIn('resume:',status['export_error'])
                self.assertIn('training_failure/new_failure.log',z.namelist())
                with zipfile.ZipFile(io.BytesIO(z.read('VVC_AGENT_V4_RESULTS.zip'))) as q:
                    self.assertIn('new_failure.log',q.namelist())
            with self.assertRaisesRegex(ValueError,'stale'):read_session_progress(outer)

    def test_results_failure_exports_current_raw_fallback_and_rejects_resume(self):
        with tempfile.TemporaryDirectory() as td:
            run,p,ctx=self.fixture(Path(td));self.export(ctx)
            (run/'new_failure.log').write_text('must survive')
            with patch('agent_v4.session.export_results',side_effect=OSError('disk fixture')):
                outer=self.export(ctx)
            with zipfile.ZipFile(outer) as z:
                self.assertEqual(json.loads(z.read('session_status.json'))['exports']['results'],'stale')
                self.assertIn('training_failure/new_failure.log',z.namelist())
            with self.assertRaisesRegex(ValueError,'stale'):read_session_progress(outer)

    def test_tables_recompute_counts_and_flag_missing_committed_files(self):
        with tempfile.TemporaryDirectory() as td:
            run,p,ctx=self.fixture(Path(td));r=write_reports(run)
            self.assertTrue(r['committed_evidence_complete']);self.assertEqual(r['training_frame_transitions'],4)
            with (run/'reports/pairs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4);self.assertAlmostEqual(float(rows[0]['precision']),2/3)
            (run/'episodes/episode_000001/online/ppo.266').unlink()
            r=write_reports(run);self.assertFalse(r['committed_evidence_complete'])
            self.assertTrue(any(x['path'].endswith('ppo.266') for x in r['missing_or_invalid']))

    def test_manifest_corruption_duplicate_and_unsafe_paths_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,p,ctx=self.fixture(base);outer=self.export(ctx)
            bad=base/'bad.zip'
            with zipfile.ZipFile(outer) as src,zipfile.ZipFile(bad,'w') as dst:
                for n in src.namelist():dst.writestr(n,b'changed' if n=='session_status.json' else src.read(n))
            dest=base/'not_created'
            with self.assertRaisesRegex(ValueError,'checksum'):restore(bad,dest)
            self.assertFalse(dest.exists())
            for name in ('../escape','/absolute','a\\b'):
                with zipfile.ZipFile(bad,'w') as z:
                    z.writestr(name,b'x');z.writestr('results_manifest.json',json.dumps({'schema':'vvenc-agent-v4/2','files':{name:sha(b'x')}}))
                with zipfile.ZipFile(bad) as z:
                    with self.assertRaises(ValueError):verify_zip(z,'results_manifest.json')

    def test_missing_committed_data_is_reported_and_not_accepted_as_full_resume(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,p,ctx=self.fixture(base)
            (run/'episodes/episode_000001/online/ppo.266').unlink()
            outer=self.export(ctx)
            with self.assertRaisesRegex(ValueError,'missing committed raw evidence'):
                restore(outer,base/'must_not_exist')
            self.assertFalse((base/'must_not_exist').exists())

    def test_partial_pending_validation_survives_cumulative_restore(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,p,ctx=self.fixture(base)
            v=p['validations'].pop()
            p['pending_validation']={**v,'output':'validation_000001_quick','checkpoint_iteration':1}
            (run/'progress.json').write_text(json.dumps(p))
            outer=self.export(ctx);dest=base/'pending'
            restore(outer,dest)
            self.assertEqual(json.loads((dest/'progress.json').read_text())['pending_validation'],p['pending_validation'])
            self.assertTrue((dest/'validation_000001_quick/val00_r50/online/ppo.266').is_file())

    def test_inner_resume_results_disagreement_rejected_before_any_restore_write(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run,p,ctx=self.fixture(base);outer=self.export(ctx)
            with zipfile.ZipFile(outer) as z:contents={n:z.read(n) for n in z.namelist() if n!='session_manifest.json'}
            modified=io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(contents['VVC_AGENT_V4_RESULTS.zip'])) as z:
                entries={n:z.read(n) for n in z.namelist() if n!='results_manifest.json'}
            different=json.loads(entries['progress.json']);different['extra']='different snapshot'
            entries['progress.json']=json.dumps(different).encode()
            with zipfile.ZipFile(modified,'w') as z:
                for n,b in entries.items():z.writestr(n,b)
                z.writestr('results_manifest.json',json.dumps({'schema':'vvenc-agent-v4/2','files':{n:sha(b) for n,b in entries.items()}}))
            contents['VVC_AGENT_V4_RESULTS.zip']=modified.getvalue()
            bad=base/'disagree.zip'
            with zipfile.ZipFile(bad,'w') as z:
                for n,b in contents.items():z.writestr(n,b)
                z.writestr('session_manifest.json',json.dumps({'schema':'vvenc-agent-v4/2','files':{n:sha(b) for n,b in contents.items()}}))
            dest=base/'must_not_exist'
            with self.assertRaisesRegex(ValueError,'disagree'):restore(bad,dest)
            self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
