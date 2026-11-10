# Chạy Agent v4

**Hướng dẫn hiện hành:** [multi-clip pilot10](AGENT_V4_MULTICLIP_RUN.md).
Update contract mới gom4 clip/update bằng2 GPU, minibatch64, Validation theo
episode và có đối chứng spatial. Phải tạo experiment mới, giữ EP50 cũ làm
đối chứng. Phần bên dưới là lịch sử hướng dẫn v4.2 hai clip/update; không dùng
lệnh/cap/update-count cũ cho notebook hiện tại.

Runtime hiện tại là **FPP1 mới**:4 encoder threads, `MaxParallelFrames=1`,
4 decoder threads. Xem [bằng chứng và giới hạn](AGENT_V42_FPP1_PROFILE.md).
Phải tạo run mới cả khi run v4.2 cũ chỉ tới iter0; không resume archive trước FPP1.
RL và paired vanilla dùng cùng runtime, không dùng bảng500 baseline cũ làm anchor.
Đã bổ sung [sửa target số học và recovery](AGENT_V42_TRAINING_READINESS.md):
delta score đo trực tiếp, pending Val lưu cùng update, lỗi setup/gate cũng có
`VVC_AGENT_V4_KAGGLE_SESSION.zip` và traceback trước khi có training progress.

V4.2 bắt đầu **run mới**, schema `vvenc-agent-v4/2`. Không dùng pilot10 v4.1,
EXPORT_5/EP650 hoặc Test export làm resume. Source/weights/optimizer thay đổi;
checkpoint v4.1 bị từ chối. V3 và đánh giá EP650 giữ nguyên. Đọc
[learning repair](AGENT_V42_LEARNING_REPAIR.md) để biết các thay đổi và giới hạn.

Config notebook cho lần đầu:

```python
MODE = 'new'
MAX_ADDITIONAL_EPISODES = 50
RESUME_INPUT = ''
SOURCE_COMMIT = ''  # resolve agent-v4-dev rồi pin SHA thật
NEW_PRESET = 'content-task'
NEW_AUXILIARY_MODE = 'task-outcome'
NEW_VAL_EVERY_UPDATES = 10
RUN_HEADROOM_DIAGNOSTIC = True
SESSION_HOURS = 11.75
```

Working output mới là `/kaggle/working/vvc_agent_v42_fpp1_run`, setup logs ở
`/kaggle/working/v42_fpp1_setup_logs`. Wrapper đã stream các dòng V4_* ra màn hình
và vẫn giữ full log. Không sửa source/kernel của một job đang chạy.

Notebook có sẵn: [VVenC_Agent_v4_Pilot_Kaggle.ipynb](../notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb). Source của notebook được tạo bởi `tools/make_agent_v4_notebook.py`; notebook chỉ build/kiểm tra/gọi runner, không chứa bản sao thuật toán. Lần đầu giữ `MODE='new'`,50 episode. Lần resume đổi `MODE='resume'` và đặt `RESUME_INPUT`; notebook tự đọc các cấu hình còn lại từ bundle.

## Input và môi trường

Kaggle T4×2, Internet ON, cùng locked BDD100K Train00 MOV, dataset manifest65/7/100 và Baseline protocol lock như trước. Không cần input kết quả Test. Clone nhánh `agent-v4-dev`, pin lại commit thực tế và giữ sạch tracked files.

Codec giữ canonical source `e0b586c46f61e2ebbf131e4eb83f98817ff6e374`. Build encoder từ checkout canonical riêng, chép executable vào `REPO/bin/release-static/vvencapp`; tránh nhúng Git version mới vào binary. Build VVdeC3.2.0 và output-POC auditor bằng `scripts/rl/build_vvdec_audit.py`. Driver yêu cầu đúng hai binary SHA256 đã kiểm thử, và lưu cả hash auditor.

Encoder golden được build tại **`/kaggle/working/vvenc_rl_source`**, build directory `build`, Release với lệnh CMake đã khóa. Phải giữ đường dẫn này: macro CHECK/THROW của VVenC có `__FILE__`, nên source/build prefix có thể xuất hiện trong binary. Notebook v4 ban đầu dùng `vvenc_canonical_v4`, gây sai khác với điều kiện build golden. Đã khôi phục đường dẫn cũ; không whitelist hash khác chỉ từ một lần build. Compiler/flags cũng có thể ảnh hưởng hash, nên vẫn cần cả hai SHA assertions và native preflight.

Nếu phiên hiện tại đã dừng ở assertion encoder SHA, chạy [cell khôi phục đường dẫn build](../notebooks/V4_Encoder_Build_Path_Recovery_Cell.py) trước cell lỗi rồi chạy lại cell lỗi. Encoder sẽ được build ở thư mục canonical cũ; decoder đã build ở đúng đường dẫn được dùng lại. Không cần sửa tracked source của Agent hoặc thay expected SHA. Nếu hash vẫn khác, dừng tại assertion để kiểm compiler/build metadata trước khi training.

```bash
python -m unittest discover -s scripts/rl -p 'test_*.py'
python scripts/rl/agent_v4_preflight.py --repo /kaggle/working/vvenc_rl_v4 \
  --encoder /kaggle/working/vvenc_rl_v4/bin/release-static/vvencapp \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/v42_fpp1_native_preflight
```

Preflight không dùng YOLO hoặc dữ liệu Train/Test. Nếu không PASS thì kiểm tra log trước; không bỏ assert hash/POC.

## Pilot50 episode trên hai GPU

```bash
python -u scripts/rl/agent_v4_train.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/vvc_agent_v42_fpp1_run \
  --preset content-task --epochs 3 --frames 256 --workers 2 \
  --validation-frames 256 --seed 20261109 --val-every-updates 10 \
  --reward clip-delta --rate-mode symmetric --beta .1 --rate-tolerance 0 \
  --credit-assignment counterfactual-final --auxiliary-mode task-outcome --auxiliary .1 \
  --max-episodes 50 --max-hours 11.5
```

Hai GPU dùng cho hai frozen-YOLO collector độc lập. Encoder/decoder chạy CPU4 thread; actor bridge một CPU thread; PPO update trên CPU sau khi cả hai collector hoàn tất. Không phải VVenC encode hoặc PPO model chạy trên hai GPU.

Trước lần PPO đầu có quick paired Val:4 greedy pairs +8 sampled diagnostic
pairs. Mỗi greedy pair ban đầu có explicit zero gate; mọi all-zero paired
result đều phải byte-identical với vanilla. Với50 episode và2 collectors có
25 PPO update, tổng12.800 transitions. Quick Val ở0/20/40/50 episode, mỗi lần
4 greedy +8 sampled diagnostic pairs. Mỗi325 episode có full35 greedy-pair Val.
Sampled diagnostic không chọn best. Lịch vẫn975 episode; pilot không tự chạy hết.

50 episode là mốc quan sát xu hướng, chưa đủ kết luận thắng v3 EP650 đã train
với ngân sách lớn hơn. So delta greedy F1/score và actual bitrate với paired
vanilla trước; reward v3/v4 khác nhau nên không so trực tiếp raw return.
Session deadline/native gates vẫn có thể dừng trước50; luôn đọc completed
và pending Validation trong export. Không dùng MODE=resume với pilot v4.1.

Train cũng encode/decode paired vanilla cho từng clip để tạo target final
score delta, nên overhead lớn hơn v4.1. Notebook chạy Train-only headroom
diagnostic trước pilot (vanilla lặp, static/live-zero, uniform±1, teacher-ROI).
Nếu native gate fail thì dừng để xem evidence; không bỏ check. CLI riêng:

```bash
python -u scripts/rl/agent_v4_headroom.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/v42_fpp1_train_headroom \
  --index 0 --rates 50 120 --frames 256 --repeats 2 --max-hours 1.5
```

Diagnostic này không update weights. Nếu nó fail trước khi có training progress,
evidence riêng ở `v42_fpp1_train_headroom/VVC_AGENT_V4_RESULTS.zip` và `headroom.json`;
không thể xuất training session đầy đủ khi chưa có checkpoint training.

Trong log cần thấy `V4_PPO_UPDATE_PASS`, số completed tăng đúng, LP/value parity đạt, các loss hữu hạn và clip score thực tế. Xem cả sampled-vs-deterministic nonzero, delta F1 và actual kbps. Không đặt điều kiện bắt buộc agent phải chọn nhiều nonzero.

## Resume

Giữ hai file:

- `VVC_AGENT_V4_RESUME.zip`: latest weights, optimizer/RNG, progress, pending Val, best/full-Val candidates và checksums
- `VVC_AGENT_V4_RESULTS.zip`: trạng thái, trajectory, xác suất/action, source/decoded detections, AU bits, bitstreams, POC audits, logs và metrics; không chứa YUV lớn hoặc toàn bộ checkpoint trung gian

Các kết quả của phiên trước vẫn cần lưu riêng; `RESULTS.zip` của phiên resume mới không tự hợp nhất mọi bằng chứng cũ. Compact resume có lịch sử metrics nhưng không chứa toàn bộ trajectory cũ.

Thêm RESUME vào Kaggle Input, dùng output mới/rỗng và **giữ nguyên mọi tham số experiment**:

```bash
python -u scripts/rl/agent_v4_train.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/vvc_agent_v42_fpp1_run \
  --resume-archive /kaggle/input/YOUR_DATASET/VVC_AGENT_V4_RESUME.zip \
  --preset content-task --epochs 3 --frames 256 --workers 2 \
  --validation-frames 256 --seed 20261109 --val-every-updates 10 \
  --reward clip-delta --rate-mode symmetric --beta .1 --rate-tolerance 0 \
  --credit-assignment counterfactual-final --auxiliary-mode task-outcome --auxiliary .1 \
  --max-episodes 90 --max-hours 11.5
```

90 là số **bổ sung**;0 là chạy phần còn lại. Pin lại `experiment.source_commit` trong progress của archive trước khi resume. Code, reward, seed, frame/window, workers và codec đổi thì phải run mới. Chỉ session cap và thời gian được thay đổi.

Schema/update mới không migrate checkpoint từ `025a6df` hoặc các v4.1 trước.
Resume chỉ cho archive v4.2 của cùng experiment/source pin. Ví dụ90 bên trên
là minh họa cú pháp, không phải quyết định đã cho chạy tiếp90 episode.

Driver cũng resume trực tiếp khi output cũ có progress hợp lệ. Nếu Kaggle giải nén Input ZIP, notebook v4 có bước tái đóng gói sau khi kiểm hash từng file theo `bundle_manifest.json`; không cần giữ nguyên byte của ZIP ngoài như bundle Test EP650.

Progress chỉ commit sau khi cả batch có reward thật, LP/value parity PASS, PPO update và checkpoint đã ghi xong. Interrupted attempt có tên riêng và không được lấy làm rollout của policy tiếp theo. Pending Val phải hoàn tất trước khi collect tiếp.

Mốc `--max-hours` tính từ lúc chạy driver, không bao gồm build trước đó. Driver dừng cấp batch mới khi còn ít hơn worker-timeout+10phút để dành thời gian xuất file; có thể dừng sớm đáng kể. Notebook tính budget còn lại sau setup. VM bị hard-kill không bảo đảm giữ batch/ZIP đang ghi, chỉ bảo đảm bản resume atomic gần nhất. Luôn tải cả hai export khi kết thúc phiên.

## Full paired Val trước khi quyết định chạy dài

Có thể kiểm một checkpoint pilot ngay, không phải đợi325 episode:

```bash
python -u scripts/rl/agent_v4_validate.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --checkpoint /kaggle/working/vvc_agent_v42_fpp1_run/checkpoints/iter_000005.pt \
  --output /kaggle/working/v4_candidate_val --workers 2
```

Chạy lại đúng command và output để resume Val chưa hoàn tất; không update
weights. Full default là35 greedy pairs, frame count từ checkpoint. `--quick`
là4 greedy +8 sampled diagnostic pairs. Stand-alone không thay best trong run.

## Đối chứng

Tạo output khác cho từng run, cùng seed và ngân sách. Các lựa chọn:

| Thí nghiệm | Tham số |
|---|---|
| Chỉ categorical, backbone/state cũ | `--preset categorical-control --reward legacy-frame --auxiliary 0` |
| Encoder/history mới, reward cũ | `--preset encoder-history --reward legacy-frame --auxiliary 0` |
| Thêm content, chưa auxiliary | `--preset content-task --reward legacy-frame --auxiliary 0` |
| Thêm auxiliary | `--preset content-task --reward legacy-frame --auxiliary .1` |
| Default v4.2 | `--preset content-task --reward clip-delta --credit-assignment counterfactual-final --auxiliary-mode task-outcome --auxiliary .1` |
| Shared final-score baseline | `--preset content-task --reward clip-delta --credit-assignment shared-final --auxiliary .1` |
| Presence auxiliary ablation | `--preset content-task --reward clip-delta --auxiliary-mode presence --auxiliary .1` |

Không cần chạy dài tất cả ngay. Ưu tiên kiểm native gate, pilot và paired Val; tăng budget cho ứng viên có tín hiệu. Không mở Test để điều chỉnh beta/lr/architecture.
