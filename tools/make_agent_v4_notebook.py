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
# VVenC Agent v4 — frozen multi-clip PPO / FPP1

Đây là run **v4 mới**, không resume EP650. T4×2, Internet ON.
Input: locked65/7/100 manifest + Baseline protocol lock + BDD100K Train00 MOV.
Lần đầu: `MODE='new'`,10 episode smoke test của bản multi-clip mới. Review trước khi tăng tiếp.
Resume v4: đổi MODE và đặt RESUME_INPUT là đường dẫn ZIP hoặc thư mục
đã giải nén của **VVC_AGENT_V4_RESUME**; cấu hình/commit được kế thừa tự động.
Không dùng EXPORT_5, bundle Test EP650 hoặc checkpoint pilot v4.1. V4.2 là
experiment mới: paired Train, critic phản thực, auxiliary task-outcome.
Quick Val =4 greedy pairs +8 sampled diagnostic pairs (2 seed cố định).
Pilot10 có3 PPO update (4+4+2 clip), chưa đủ kết luận learning gain.
Review xong mới resume thêm40 để đạt tổng50. Quick Val theo mốc episode20/40
và cuối session, không theo số update; full Val35 cặp mỗi325 episode.
Runtime mới: **4 encoder threads, MaxParallelFrames=1**, golden binary giữ nguyên.
Paired vanilla dùng cùng runtime mới. Không so trực tiếp với bảng baseline500 cũ.
Không resume checkpoint pilot50 cũ từ283a16e. Update contract và batching đã đổi.
Mỗi update gom4 clip dưới cùng checkpoint bằng2 lượt thu thập trên2 GPU.
Quick Val sau training có đối chứng uniform−1 và frame-uniform open-loop replay;
chúng không vào PPO, không chọn best và không phải đối chứng cùng actual bitrate.
Native preflight, Train headroom và initial paired Val phải PASS trước PPO.
Nếu setup/gate lỗi, notebook tự xuất `VVC_AGENT_V4_KAGGLE_SESSION.zip` cùng
traceback và log hiện có, kể cả khi chưa tạo training progress.
"""), cell("code", """
MODE = 'new'  # 'new' hoặc 'resume'
MAX_ADDITIONAL_EPISODES = 10
RESUME_INPUT = ''  # Ví dụ /kaggle/input/datasets/owner/v4-session/VVC_AGENT_V4_RESUME
SOURCE_COMMIT = ''  # New: trống=resolve branch rồi ghi SHA; resume: lấy từ archive
SESSION_HOURS = 11.75  # Tổng budget từ đầu notebook, gồm setup; driver dành thời gian export
NEW_PRESET = 'content-task'
NEW_AUXILIARY_MODE = 'task-outcome'  # 'presence' chỉ cho ablation run mới
NEW_ROLLOUTS_PER_UPDATE = 4  # Hai waves, vẫn chỉ dùng2 GPU
NEW_PPO_MINIBATCH = 64
NEW_ACTOR_LR = 1e-4  # Giữ nguyên để tách hiệu ứng batching
NEW_VAL_EVERY_EPISODES = 20
NEW_SPATIAL_CONTROLS = True
RUN_HEADROOM_DIAGNOSTIC = True  # Train-only, no PPO; chỉ chạy khi MODE=new
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
# The golden encoder was built at this absolute path. VVenC CHECK/THROW
# strings contain __FILE__, so changing the source directory can change SHA.
REPO=WORK/'vvenc_rl_v4'; CODEC=WORK/'vvenc_rl_source'
VVDEC=WORK/'vvdec_v3_2_0'; RUN=WORK/'vvc_agent_v4_multiclip_run'
LOGS=WORK/'v4_multiclip_setup_logs';LOGS.mkdir(exist_ok=True)
HEADROOM=None;PREFLIGHT=None;TRAIN_RC=None;TRAIN_STARTED=False
SESSION_STAGE='setup'
CANONICAL='e0b586c46f61e2ebbf131e4eb83f98817ff6e374'
ENC=REPO/'bin/release-static/vvencapp'; DEC=VVDEC/'bin/release-static/vvdecapp'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd,cwd=None,check=True):
    path=LOGS/f'{len(list(LOGS.glob("*.log"))):03d}.log'
    print('$',' '.join(map(str,cmd)),flush=True)
    return run_logged(cmd,path,cwd,check)

def run_logged(cmd,path,cwd,check):
    p=None
    with path.open('w') as log:
        p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            last=time.monotonic()
            offset=0;pending=''
            while p.poll() is None:
                time.sleep(2)
                with path.open(errors='replace') as reader:
                    reader.seek(offset);pending+=reader.read();offset=reader.tell()
                lines=pending.split('\n');pending=lines.pop()
                for line in lines:
                    if line.startswith(('V4_','AGENT_V4','Traceback','ERROR','ValueError','RuntimeError')):
                        print(line,flush=True)
                if time.monotonic()-last>40:
                    print('RUNNING',path.name,flush=True);last=time.monotonic()
            rc=p.returncode
        except BaseException:
            stop_launcher_process(p)
            raise
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
        assert manifest['schema']=='vvenc-agent-v4/2','V4.2 requires a NEW run; v4.1 pilot cannot migrate'
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
        assert manifest['schema']=='vvenc-agent-v4/2','Use the pinned old launcher for v4.1; do not migrate weights'
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(manifest['files'])|{'bundle_manifest.json'}
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        prior=json.loads(z.read('progress.json'))
    assert prior['purpose']=='training' and prior['schema']=='vvenc-agent-v4/2'
    EXP=prior['experiment'];SOURCE_COMMIT=EXP['source_commit']
    # Reject an older runtime BEFORE checking out its pinned source.
    expected_profile={'version':'golden-vvenc-fpp1/1','encoder_threads':4,'decoder_threads':4,
        'max_parallel_frames':1,'gop_size':32,'lookahead':1,'qpa':1,'preset':'medium',
        'passes':1,'internal_bitdepth':10,'mtprofile':-1,
        'paired_anchor':'fresh-vanilla-same-profile','historical_500_pair_baseline_compatible':False}
    assert EXP.get('codec_profile')==expected_profile, 'Runtime changed: start a NEW FPP1 run; old progress cannot migrate'
    assert EXP.get('encoder_threads')==EXP.get('decoder_threads')==4
    assert EXP.get('collection_version')=='frozen-multiclip-waves-v1', 'NEW multi-clip run required; keep the old pilot50 as control'
    assert EXP.get('update_version')=='frozen-multiclip-ctu-ppo-outcome-v4', 'Use the launcher pinned to this update contract'
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
## Test và build golden codec, runtime FPP1
Build từ canonical checkout riêng; binary/source giữ nguyên, runtime được khai báo mới.
Hash khác thì dừng, không bỏ kiểm tra.
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
from agent_v4.codec import runtime_profile,require_runtime_profile
if EXP is not None:require_runtime_profile(EXP)
print('V4_CODEC_PROFILE',runtime_profile(),flush=True)
print('ENCODER_SHA256',sha(ENC),'EXPECTED',ENCODER_SHA)
print('DECODER_SHA256',sha(DEC),'EXPECTED',DECODER_SHA)
assert sha(ENC)==ENCODER_SHA,(sha(ENC),ENCODER_SHA)
assert sha(DEC)==DECODER_SHA,(sha(DEC),DECODER_SHA)
AUDITOR=DEC.with_name(DEC.name+'_poc_audit')
if not AUDITOR.exists():
    run([sys.executable,REPO/'scripts/rl/build_vvdec_audit.py','--source',VVDEC,
         '--work',WORK/'vvdec_v4_poc_audit','--decoder',DEC])
from training_session import new_attempt
PREFLIGHT=new_attempt(WORK/'v4_multiclip_native_preflight')
run([sys.executable,REPO/'scripts/rl/agent_v4_preflight.py','--repo',REPO,'--encoder',ENC,
     '--decoder',DEC,'--output',PREFLIGHT])

if MODE=='new' and RUN_HEADROOM_DIAGNOSTIC:
    HEADROOM=new_attempt(WORK/'v4_multiclip_train_headroom')
    run([sys.executable,'-u',REPO/'scripts/rl/agent_v4_headroom.py','--repo',REPO,'--decoder',DEC,
         '--output',HEADROOM,'--index',0,'--rates',50,120,'--frames',256,'--repeats',2,
         '--max-hours',min(1.5,(SESSION_HOURS*3600-(time.monotonic()-SESSION_START))/3600)])
    print('HEADROOM',json.loads((HEADROOM/'headroom.json').read_text()),flush=True)
environment={'source_commit':SOURCE_COMMIT,'codec_profile':runtime_profile(),'python':sys.version,'torch':torch.__version__,
 'platform':platform.platform(),'encoder_sha256':sha(ENC),'decoder_sha256':sha(DEC),
 'auditor_sha256':sha(AUDITOR),'baseline_lock_sha256':locked['lock_sha256'],
 'gpu':[torch.cuda.get_device_name(i) for i in range(2)]}
(LOGS/'environment.json').write_text(json.dumps(environment,indent=2))
with (LOGS/'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,stderr=subprocess.STDOUT,check=True)
"""), cell("markdown", """
## Training và paired Val
Mặc định10 episode smoke mới, không tự chạy325/975. Resume kế thừa nguyên cấu hình,
chỉ đổi số episode bổ sung/thời gian. Driver xuất resume sau mỗi PPO commit;
khi lỗi thường vẫn xuất evidence trong finally. Không cam kết batch đang ghi
sẽ còn nguyên nếu VM hard-kill. Theo dõi progress thật ở cell cuối.
"""), cell("code", """
remaining=SESSION_HOURS*3600-(time.monotonic()-SESSION_START)
assert remaining>2400,'Setup đã dùng gần hết budget; không khởi chạy batch mới'
cfg=EXP or {'preset':NEW_PRESET,'epochs':NEW_EPOCHS,'seed':NEW_SEED,'frames':256,
 'validation_frames':256,'workers':2,'val_every_updates':25,
 'rollouts_per_update':NEW_ROLLOUTS_PER_UPDATE,'val_every_episodes':NEW_VAL_EVERY_EPISODES,
 'validation_spatial_controls':NEW_SPATIAL_CONTROLS,
 'reward':{'mode':'legacy-frame' if NEW_PRESET=='categorical-control' else 'clip-delta',
           'rate_mode':'symmetric','beta':.1,'tolerance':0},
 'ppo_config':{'minibatch':NEW_PPO_MINIBATCH,'actor_lr':NEW_ACTOR_LR,'auxiliary':.1 if NEW_PRESET=='content-task' else 0.,
               'auxiliary_mode':NEW_AUXILIARY_MODE,
               'credit_assignment':'joint-gae' if NEW_PRESET=='categorical-control' else 'counterfactual-final'}}
cmd=[sys.executable,'-u',REPO/'scripts/rl/agent_v4_train.py','--repo',REPO,'--decoder',DEC,
 '--output',RUN,'--max-episodes',MAX_ADDITIONAL_EPISODES,'--max-hours',remaining/3600]
for option in ('preset','epochs','seed','frames','validation_frames','workers','val_every_updates',
               'rollouts_per_update','val_every_episodes'):
    cmd+=['--'+option.replace('_','-'),cfg[option]]
for option,key in (('reward','mode'),('rate-mode','rate_mode'),('beta','beta'),('rate-tolerance','tolerance')):
    cmd+=['--'+option,cfg['reward'][key]]
cmd+=['--auxiliary',cfg['ppo_config']['auxiliary'],
      '--ppo-minibatch',cfg['ppo_config']['minibatch'],'--actor-lr',cfg['ppo_config']['actor_lr']]
cmd+=['--validation-spatial-controls' if cfg['validation_spatial_controls'] else '--no-validation-spatial-controls']
cmd+=['--auxiliary-mode',cfg['ppo_config']['auxiliary_mode'],
      '--credit-assignment',cfg['ppo_config']['credit_assignment']]
if RESUME_ARCHIVE is not None and not (RUN/'progress.json').exists():
    cmd+=['--resume-archive',RESUME_ARCHIVE]
TRAIN_STARTED=True
SESSION_STAGE='training'
TRAIN_RC=run(cmd,cwd=REPO,check=False)
print('TRAIN_PROCESS_RETURN_CODE',TRAIN_RC)
"""), cell("code", """
from IPython.display import display,FileLink
session=WORK/'VVC_AGENT_V4_KAGGLE_SESSION.zip'
if not session.exists():session=export_session(globals())
display(FileLink(str(session)))
for name in ('VVC_AGENT_V4_RESUME.zip','VVC_AGENT_V4_RESULTS.zip'):
    if (RUN/name).is_file():display(FileLink(str(RUN/name)))
print('SESSION_EXPORTED - inspect return code, completed, updates and Validation')
""")]


# Bootstrap must work even before cloning the project or importing Torch.
_support = (Path(__file__).resolve().parents[1]/'scripts/rl/agent_v4/notebook_support.py').read_text(encoding='utf-8')
_setup = cells[2]['source']
_cuda_start = _setup.index('import torch\n')
_cuda_end = _setup.index("CANONICAL=")
_bootstrap = _setup[:_cuda_start]+_setup[_setup.index("WORK=Path("):_cuda_end]+_support+'\n'
_body = _setup[_cuda_end:]
# Function definitions above the resume logic are bootstrap, too.
_resume = _body.index('RESUME_ARCHIVE=None\n')
_bootstrap += _body[:_resume]
_body = _setup[_cuda_start:_setup.index("WORK=Path(")]+_body[_resume:]

def guarded(source):
    return 'try:\n'+textwrap.indent(source,'    ')+'except BaseException:\n    record_session_failure(globals())\n    raise\n'

cells[2]['source'] = _bootstrap+guarded(_body)
cells[4]['source'] = guarded("SESSION_STAGE='build-and-native-gates'\n"+cells[4]['source'])
cells[6]['source'] = guarded(cells[6]['source'])+"SESSION_STAGE='training-ended'\nexport_session(globals())\n"

notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1]/"notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    for i,c in enumerate(cells):
        if c["cell_type"] == "code":compile(c["source"],f"cell_{i}","exec")
    print(dest)
