"""Standard-library bootstrap helpers embedded in the generated Kaggle launcher."""
import json
import os
from pathlib import Path
import signal
import subprocess
import traceback
import zipfile


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
    error = None
    if context.get('TRAIN_STARTED') and (run/'progress.json').exists():
        try:
            progress = json.loads((run/'progress.json').read_text())
            from agent_v4.session import export_resume,export_results
            export_resume(run,progress)
            export_results(run)
        except Exception as exc:
            error = str(exc)
    dest = work/'VVC_AGENT_V4_KAGGLE_SESSION.zip'
    tmp = dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        included = set()
        if context.get('TRAIN_STARTED'):
            for name in ('VVC_AGENT_V4_RESUME.zip','VVC_AGENT_V4_RESULTS.zip'):
                if (run/name).is_file():
                    z.write(run/name,name);included.add(name)
        roots = [('setup_logs',logs),('native_preflight',context.get('PREFLIGHT')),
                 ('train_headroom',context.get('HEADROOM'))]
        # Preserve raw failure evidence if the driver's results export failed.
        if context.get('TRAIN_STARTED') and 'VVC_AGENT_V4_RESULTS.zip' not in included:
            roots.append(('training_failure',run))
        for prefix,root in roots:
            if root is None:continue
            root = Path(root)
            for path in sorted(root.rglob('*')):
                if (not path.is_file() or path.is_symlink() or
                        path.suffix in ('.yuv','.pt','.zip','.tmp') or '__pycache__' in path.parts):
                    continue
                z.write(path,prefix+'/'+path.relative_to(root).as_posix())
        status = {'stage':context.get('SESSION_STAGE'),'training_started':bool(context.get('TRAIN_STARTED')),
                  'training_return_code':context.get('TRAIN_RC'),'source_commit':context.get('SOURCE_COMMIT'),
                  'completed':len(progress['completed']) if progress else (None if context.get('TRAIN_STARTED') else 0),
                  'updates':len(progress['updates']) if progress else (None if context.get('TRAIN_STARTED') else 0),
                  'pending_validation':progress.get('pending_validation') if progress else None,
                  'export_error':error}
        z.writestr('session_status.json',json.dumps(status))
    os.replace(tmp,dest)
    print('V4_SESSION_EVIDENCE',dest,flush=True)
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
