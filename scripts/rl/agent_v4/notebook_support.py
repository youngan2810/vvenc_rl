"""Standard-library bootstrap helpers embedded in the generated Kaggle launcher."""
import json
import os
from pathlib import Path
import signal
import subprocess
import traceback
import zipfile
import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import PurePosixPath
import uuid


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_zip(z,manifest_name):
    """Verify names and every byte before restore; no executable serialization."""
    names=z.namelist()
    if len(names)!=len(set(names)) or manifest_name not in names:
        raise ValueError('Duplicate entries or missing '+manifest_name)
    manifest=json.loads(z.read(manifest_name))
    if manifest.get('schema')!='vvenc-agent-v4/2' or set(names)!=set(manifest['files'])|{manifest_name}:
        raise ValueError('Archive entries differ from manifest')
    for name,digest in manifest['files'].items():
        p=PurePosixPath(name)
        if not name or p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or str(p)!=name:
            raise ValueError('Unsafe archive path: '+name)
        h=hashlib.sha256()
        with z.open(name) as src:
            for block in iter(lambda:src.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError('Archive checksum mismatch: '+name)
    return manifest


@contextmanager
def verified_session_parts(archive):
    """Full session is the portable continuation unit, including raw history."""
    with tempfile.TemporaryDirectory(prefix='v4_session_restore_') as td:
        with zipfile.ZipFile(archive) as outer:
            verify_zip(outer,'session_manifest.json')
            status=json.loads(outer.read('session_status.json'))
            if any(status.get('exports',{}).get(k)!='fresh' for k in ('resume','results')):
                raise ValueError('Session has stale/missing exports; inspect session_status.json')
            paths=[]
            for name in ('VVC_AGENT_V4_RESUME.zip','VVC_AGENT_V4_RESULTS.zip'):
                dest=Path(td)/name
                with outer.open(name) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
                paths.append(dest)
        with zipfile.ZipFile(paths[0]) as rz,zipfile.ZipFile(paths[1]) as ez:
            rm=verify_zip(rz,'bundle_manifest.json');em=verify_zip(ez,'results_manifest.json')
            for name in set(rm['files']) & set(em['files']):
                if rm['files'][name]!=em['files'][name]:raise ValueError('Resume/results disagree: '+name)
            p=json.loads(rz.read('progress.json'))
            if (status['completed']!=len(p.get('completed',[])) or status['updates']!=len(p.get('updates',[])) or
                    status.get('source_commit')!=p['experiment']['source_commit'] or
                    status.get('progress_sha256')!=hashlib.sha256(rz.read('progress.json')).hexdigest()):
                raise ValueError('Session status differs from progress')
            inventory=json.loads(ez.read('reports/completeness.json'))
            if not inventory['committed_evidence_complete']:
                raise ValueError('Session is missing committed raw evidence; inspect completeness report')
        yield paths[0],paths[1],status


def read_session_progress(archive):
    with verified_session_parts(archive) as (resume,results,status):
        with zipfile.ZipFile(resume) as z:return json.loads(z.read('progress.json'))


def capture_session_history(context,run,progress):
    """Keep previous session environments/gates when results are restored."""
    sid=context.setdefault('_EVIDENCE_SESSION_ID',uuid.uuid4().hex)
    history=run/'session_history'/sid
    history.mkdir(parents=True,exist_ok=True)
    for label,key in [('setup_logs','LOGS'),('native_preflight','PREFLIGHT'),('train_headroom','HEADROOM')]:
        src_root=context.get(key)
        if src_root is None:continue
        for src in sorted(Path(src_root).rglob('*')):
            if (not src.is_file() or src.is_symlink() or src.suffix in ('.yuv','.pt','.zip','.tmp') or
                    '__pycache__' in src.parts or 'actions' in src.parts):continue
            dst=history/label/src.relative_to(src_root)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    (history/'session.json').write_text(json.dumps({
        'source_commit':context.get('SOURCE_COMMIT'),'stage':context.get('SESSION_STAGE'),
        'completed':len(progress.get('completed',[])) if progress else None,
        'updates':len(progress.get('updates',[])) if progress else None,
        'training_return_code':context.get('TRAIN_RC')}),encoding='utf-8')


def stop_launcher_process(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == 'posix':os.killpg(proc.pid,signal.SIGTERM)
        else:proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'posix':os.killpg(proc.pid,signal.SIGKILL)
            else:proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def export_session(context):
    """Export even when setup failed before any training progress existed."""
    work,run,logs = (Path(context[k]) for k in ('WORK','RUN','LOGS'))
    progress = None
    errors = []
    exports = {'resume':'missing','results':'missing'}
    if context.get('TRAIN_STARTED') and (run/'progress.json').exists():
        try:
            progress = json.loads((run/'progress.json').read_text())
        except Exception as exc:
            errors.append('progress: '+str(exc))
        try:capture_session_history(context,run,progress)
        except Exception as exc:errors.append('session history: '+str(exc))
        # Independent attempts: a bad checkpoint must not suppress new evidence.
        for kind,name in [('resume','VVC_AGENT_V4_RESUME.zip'),('results','VVC_AGENT_V4_RESULTS.zip')]:
            exports[kind]='stale' if (run/name).exists() else 'missing'
            try:
                from agent_v4.session import export_resume,export_results
                if kind=='resume':
                    if progress is None:raise ValueError('No readable committed progress')
                    export_resume(run,progress)
                else:export_results(run)
                with zipfile.ZipFile(run/name) as check:
                    verify_zip(check,'bundle_manifest.json' if kind=='resume' else 'results_manifest.json')
                exports[kind]='fresh'
            except Exception as exc:errors.append(kind+': '+str(exc))
    dest = work/'VVC_AGENT_V4_KAGGLE_SESSION.zip'
    tmp = dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        included = set()
        manifest={}
        def add(path,name):
            z.write(path,name);manifest[name]=file_digest(path)
        if context.get('TRAIN_STARTED'):
            for name in ('VVC_AGENT_V4_RESUME.zip','VVC_AGENT_V4_RESULTS.zip'):
                if (run/name).is_file():
                    add(run/name,name);included.add(name)
        roots = [('setup_logs',logs),('native_preflight',context.get('PREFLIGHT')),
                 ('train_headroom',context.get('HEADROOM'))]
        # Preserve raw failure evidence if the driver's results export failed.
        if context.get('TRAIN_STARTED') and (errors or 'VVC_AGENT_V4_RESULTS.zip' not in included):
            roots.append(('training_failure',run))
        for prefix,root in roots:
            if root is None:continue
            root = Path(root)
            for path in sorted(root.rglob('*')):
                if (not path.is_file() or path.is_symlink() or
                        path.suffix in ('.yuv','.pt','.zip','.tmp') or '__pycache__' in path.parts):
                    continue
                add(path,prefix+'/'+path.relative_to(root).as_posix())
        status = {'stage':context.get('SESSION_STAGE'),'training_started':bool(context.get('TRAIN_STARTED')),
                  'training_return_code':context.get('TRAIN_RC'),'source_commit':context.get('SOURCE_COMMIT'),
                  'completed':len(progress['completed']) if progress else (None if context.get('TRAIN_STARTED') else 0),
                  'updates':len(progress['updates']) if progress else (None if context.get('TRAIN_STARTED') else 0),
                  'pending_validation':progress.get('pending_validation') if progress else None,
                  'progress_sha256':file_digest(run/'progress.json') if (run/'progress.json').is_file() else None,
                  'exports':exports,'export_error':'; '.join(errors) or None,
                  'artifact_contract':'cumulative-evidence-v1'}
        data=json.dumps(status).encode()
        z.writestr('session_status.json',data);manifest['session_status.json']=hashlib.sha256(data).hexdigest()
        z.writestr('session_manifest.json',json.dumps({'schema':'vvenc-agent-v4/2','files':manifest}))
    os.replace(tmp,dest)
    print('V4_SESSION_EVIDENCE',dest,flush=True)
    print('V4_EXPORT_STATUS',json.dumps(exports),'ERROR',status['export_error'],flush=True)
    return dest


def record_session_failure(context):
    """Keep the original exception; evidence failure must not replace it."""
    try:
        logs=Path(context['LOGS']);logs.mkdir(parents=True,exist_ok=True)
        (logs/'failure_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        dest=export_session(context)
        try:
            from IPython.display import display,FileLink
            display(FileLink(str(dest)))
        except ImportError:
            pass
    except Exception as exc:
        print('V4_EVIDENCE_EXPORT_FAILED',str(exc),flush=True)
