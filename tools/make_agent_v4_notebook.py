"""Generate a thin Kaggle launcher; algorithm and recovery stay source-owned."""
from pathlib import Path
import json
import textwrap
import hashlib


def cell(kind, text):
    c = {"cell_type":kind,"metadata":{},"source":textwrap.dedent(text).strip()+"\n",
         "id":hashlib.sha256((kind+text).encode()).hexdigest()[:12]}
    if kind == "code":
        c.update(execution_count=None,outputs=[])
    return c


cells = [cell("markdown", """
# VVenC Agent v4 — new pilot / strict resume

Đây là run **v4 mới**, không resume EP650. T4×2, Internet ON.
Input: locked65/7/100 manifest + Baseline protocol lock + BDD100K Train00 MOV.
Lần đầu: `MODE='new'`,10 episode. Chỉ chạy dài sau khi xem paired Val.
Resume v4: đổi MODE và đặt RESUME_INPUT là đường dẫn ZIP hoặc thư mục
đã giải nén của **VVC_AGENT_V4_RESUME**; cấu hình/commit được kế thừa tự động.
Không dùng EXPORT_5 hay bundle Test EP650.
"""), cell("code", """
MODE = 'new'  # 'new' hoặc 'resume'
MAX_ADDITIONAL_EPISODES = 10
RESUME_INPUT = ''  # Ví dụ /kaggle/input/datasets/owner/v4-session/VVC_AGENT_V4_RESUME
SOURCE_COMMIT = ''  # New: trống=resolve branch rồi ghi SHA; resume: lấy từ archive
SESSION_HOURS = 11.75  # Tổng budget từ đầu notebook, gồm setup; driver dành thời gian export
NEW_PRESET = 'content-task'
NEW_EPOCHS, NEW_SEED = 3, 20261109
assert MODE in ('new','resume') and MAX_ADDITIONAL_EPISODES >= 0
assert MODE == 'resume' or not RESUME_INPUT, 'Có resume input: chọn MODE=resume'
"""), cell("code", r'''
from pathlib import Path, PurePosixPath
import os,sys,json,time,subprocess,hashlib,zipfile,shutil,platform
SESSION_START = time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
for key in list(os.environ):
    if key.startswith('VVENC_RL_'):os.environ.pop(key)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==2, 'Chọn GPU T4 x2 và restart Session'
assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))
WORK=Path('/kaggle/working'); INPUT=Path('/kaggle/input')
REPO=WORK/'vvenc_rl_v4'; CODEC=WORK/'vvenc_canonical_v4'
VVDEC=WORK/'vvdec_v3_2_0'; RUN=WORK/'vvc_agent_v4_run'
LOGS=WORK/'v4_setup_logs';LOGS.mkdir(exist_ok=True)
CANONICAL='e0b586c46f61e2ebbf131e4eb83f98817ff6e374'
ENC=REPO/'bin/release-static/vvencapp'; DEC=VVDEC/'bin/release-static/vvdecapp'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd,cwd=None,check=True):
    path=LOGS/f'{len(list(LOGS.glob("*.log"))):03d}.log'
    print('$',' '.join(map(str,cmd)),flush=True)
    with path.open('w') as log:
        p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        last=time.monotonic()
        while p.poll() is None:
            time.sleep(2)
            if time.monotonic()-last>40:
                print('RUNNING',path.name,flush=True);last=time.monotonic()
        rc=p.returncode
    print(path.read_text(errors='replace')[-2200:],flush=True)
    if check and rc:raise RuntimeError(f'Failed rc={rc}; inspect {path}')
    return rc
RESUME_ARCHIVE=None
if MODE=='resume':
    assert RESUME_INPUT,'Đặt RESUME_INPUT là ZIP hoặc thư mục v4 resume trong Input'
    src=Path(RESUME_INPUT)
    RESUME_ARCHIVE=WORK/'v4_verified_input.zip'
    if src.is_dir():
        manifest=json.loads((src/'bundle_manifest.json').read_text())
        assert manifest['schema']=='vvenc-agent-v4/1'
        with zipfile.ZipFile(RESUME_ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
            for name,digest in manifest['files'].items():
                rel=PurePosixPath(name)
                assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name
                path=src/name
                assert sha(path)==digest, name
                z.write(path,name)
            z.write(src/'bundle_manifest.json','bundle_manifest.json')
    else:
        assert src.is_file(),src
        shutil.copy2(src,RESUME_ARCHIVE)
    with zipfile.ZipFile(RESUME_ARCHIVE) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        assert manifest['schema']=='vvenc-agent-v4/1'
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(manifest['files'])|{'bundle_manifest.json'}
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        prior=json.loads(z.read('progress.json'))
    assert prior['purpose']=='training' and prior['schema']=='vvenc-agent-v4/1'
    EXP=prior['experiment'];SOURCE_COMMIT=EXP['source_commit']
    print('RESUME_V4_EPISODES_BEFORE',len(prior['completed']))
else:
    EXP=None
    assert not (RUN/'progress.json').exists(),'Có run trong Working: dùng resume, không new'
    print('FRESH_V4_RUN — không nạp checkpoint v3/EP650')
for path,branch,commit in ((REPO,'agent-v4-dev',SOURCE_COMMIT),(CODEC,'full-training-dev',CANONICAL)):
    if not (path/'.git').exists():
        assert not path.exists(),path
        run(['git','clone','--branch',branch,'--depth','60','https://github.com/youngan2810/vvenc_rl.git',path])
    if path==REPO and MODE=='new' and not commit:
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
        run(['git','-C',path,'fetch','origin','agent-v4-dev'])
        run(['git','-C',path,'checkout','--detach','FETCH_HEAD'])
    if commit:
        if subprocess.run(['git','-C',str(path),'cat-file','-e',commit+'^{commit}'],capture_output=True).returncode:
            run(['git','-C',path,'fetch','origin',commit])
        run(['git','-C',path,'checkout','--detach',commit])
    assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
SOURCE_COMMIT=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
# An interactive kernel may still hold imports from a previous checkout.
# Training runs in fresh subprocesses; notebook-side export must match too.
import importlib
root_modules={p.stem for p in (REPO/'scripts/rl').glob('*.py')}
for name in list(sys.modules):
    if name=='agent_v4' or name.startswith('agent_v4.') or name in root_modules:
        del sys.modules[name]
importlib.invalidate_caches()
sys.path.insert(0,str(REPO/'scripts/rl'))
from real_reward_pilot import read_lock_manifest,read_baseline_config
splits,records=read_lock_manifest(INPUT);locked=read_baseline_config(INPUT)
print('V4_SOURCE_PIN',SOURCE_COMMIT,'SPLITS',{k:len(v) for k,v in splits.items()})
'''), cell("markdown", """
## Test và build codec giữ nguyên profile
Build từ canonical checkout riêng. Hash khác thì dừng, không bỏ kiểm tra.
Preflight v4 chạy trên clip synthetic, chưa chứng minh chất lượng task.
"""), cell("code", """
run([sys.executable,'-m','unittest','discover','-s','scripts/rl','-p','test_*.py'],cwd=REPO)
run(['cmake','-S',CODEC,'-B',CODEC/'build','-DCMAKE_BUILD_TYPE=Release'])
run(['cmake','--build',CODEC/'build','--config','Release','-j','2'])
ENC.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(CODEC/'bin/release-static/vvencapp',ENC)
if not (VVDEC/'.git').exists():
    assert not VVDEC.exists()
    run(['git','clone','--branch','v3.2.0','--depth','1','https://github.com/fraunhoferhhi/vvdec.git',VVDEC])
run(['cmake','-S',VVDEC,'-B',VVDEC/'build','-DCMAKE_BUILD_TYPE=Release'])
run(['cmake','--build',VVDEC/'build','--config','Release','-j','2'])
from agent_v4.train import ENCODER_SHA,DECODER_SHA
assert sha(ENC)==ENCODER_SHA,(sha(ENC),ENCODER_SHA)
assert sha(DEC)==DECODER_SHA,(sha(DEC),DECODER_SHA)
AUDITOR=DEC.with_name(DEC.name+'_poc_audit')
if not AUDITOR.exists():
    run([sys.executable,REPO/'scripts/rl/build_vvdec_audit.py','--source',VVDEC,
         '--work',WORK/'vvdec_v4_poc_audit','--decoder',DEC])
from training_session import new_attempt
PREFLIGHT=new_attempt(WORK/'v4_native_preflight')
run([sys.executable,REPO/'scripts/rl/agent_v4_preflight.py','--repo',REPO,'--encoder',ENC,
     '--decoder',DEC,'--output',PREFLIGHT])
environment={'source_commit':SOURCE_COMMIT,'python':sys.version,'torch':torch.__version__,
 'platform':platform.platform(),'encoder_sha256':sha(ENC),'decoder_sha256':sha(DEC),
 'auditor_sha256':sha(AUDITOR),'baseline_lock_sha256':locked['lock_sha256'],
 'gpu':[torch.cuda.get_device_name(i) for i in range(2)]}
(LOGS/'environment.json').write_text(json.dumps(environment,indent=2))
with (LOGS/'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,stderr=subprocess.STDOUT,check=True)
"""), cell("markdown", """
## Training và paired Val
Mặc định chỉ10 episode mới, không tự chạy975. Resume kế thừa nguyên cấu hình,
chỉ đổi số episode bổ sung/thời gian. Driver xuất resume sau mỗi PPO commit;
khi lỗi thường vẫn xuất evidence trong finally. Không cam kết batch đang ghi
sẽ còn nguyên nếu VM hard-kill. Theo dõi progress thật ở cell cuối.
"""), cell("code", """
remaining=SESSION_HOURS*3600-(time.monotonic()-SESSION_START)
assert remaining>2400,'Setup đã dùng gần hết budget; không khởi chạy batch mới'
cfg=EXP or {'preset':NEW_PRESET,'epochs':NEW_EPOCHS,'seed':NEW_SEED,'frames':256,
 'validation_frames':256,'workers':2,'val_every_updates':25,
 'reward':{'mode':'legacy-frame' if NEW_PRESET=='categorical-control' else 'clip-delta',
           'rate_mode':'symmetric','beta':.1,'tolerance':0},
 'ppo_config':{'auxiliary':.1 if NEW_PRESET=='content-task' else 0.}}
cmd=[sys.executable,'-u',REPO/'scripts/rl/agent_v4_train.py','--repo',REPO,'--decoder',DEC,
 '--output',RUN,'--max-episodes',MAX_ADDITIONAL_EPISODES,'--max-hours',remaining/3600]
for option in ('preset','epochs','seed','frames','validation_frames','workers','val_every_updates'):
    cmd+=['--'+option.replace('_','-'),cfg[option]]
for option,key in (('reward','mode'),('rate-mode','rate_mode'),('beta','beta'),('rate-tolerance','tolerance')):
    cmd+=['--'+option,cfg['reward'][key]]
cmd+=['--auxiliary',cfg['ppo_config']['auxiliary']]
if RESUME_ARCHIVE is not None and not (RUN/'progress.json').exists():
    cmd+=['--resume-archive',RESUME_ARCHIVE]
TRAIN_RC=run(cmd,cwd=REPO,check=False)
print('TRAIN_PROCESS_RETURN_CODE',TRAIN_RC)
"""), cell("code", """
from IPython.display import display,FileLink
from agent_v4.session import export_resume,export_results
progress_path=RUN/'progress.json'
assert progress_path.exists(),'Chưa có training progress; xem setup/training log'
progress=json.loads(progress_path.read_text())
export_resume(RUN,progress);export_results(RUN)
print('COMPLETED',len(progress['completed']),'UPDATES',len(progress['updates']),
      'PENDING_VAL',progress.get('pending_validation'),'BEST_FULL_VAL',progress.get('best_checkpoint'))
if progress['validations']:print('LATEST_VAL',progress['validations'][-1])
session=WORK/'VVC_AGENT_V4_KAGGLE_SESSION.zip'
with zipfile.ZipFile(session,'w',zipfile.ZIP_STORED) as z:
    for name in ('VVC_AGENT_V4_RESUME.zip','VVC_AGENT_V4_RESULTS.zip'):
        z.write(RUN/name,name)
    for path in LOGS.rglob('*'):
        if path.is_file():z.write(path,'setup_logs/'+path.relative_to(LOGS).as_posix())
    for path in PREFLIGHT.rglob('*'):
        if path.is_file() and path.suffix!='.yuv':z.write(path,'native_preflight/'+path.relative_to(PREFLIGHT).as_posix())
    z.writestr('session_status.json',json.dumps({'training_return_code':TRAIN_RC,
      'completed':len(progress['completed']),'source_commit':SOURCE_COMMIT}))
display(FileLink(str(session)))
display(FileLink(str(RUN/'VVC_AGENT_V4_RESUME.zip')))
display(FileLink(str(RUN/'VVC_AGENT_V4_RESULTS.zip')))
print('SESSION_EXPORTED — kiểm tra return code và Val trước khi quyết định chạy dài')
""")]

notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1]/"notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    for i,c in enumerate(cells):
        if c["cell_type"] == "code":compile(c["source"],f"cell_{i}","exec")
    print(dest)
