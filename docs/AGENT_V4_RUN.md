# Chạy Agent v4

V4 bắt đầu từ đầu. Không dùng EXPORT_5/EP650 làm resume của v4. V3 và đánh giá đang chạy giữ nguyên.

Notebook có sẵn: [VVenC_Agent_v4_Pilot_Kaggle.ipynb](../notebooks/VVenC_Agent_v4_Pilot_Kaggle.ipynb). Source của notebook được tạo bởi `tools/make_agent_v4_notebook.py`; notebook chỉ build/kiểm tra/gọi runner, không chứa bản sao thuật toán. Lần đầu giữ `MODE='new'`,10 episode. Lần resume đổi `MODE='resume'` và đặt `RESUME_INPUT`; notebook tự đọc các cấu hình còn lại từ bundle.

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
  --output /kaggle/working/v4_native_preflight
```

Preflight không dùng YOLO hoặc dữ liệu Train/Test. Nếu không PASS thì kiểm tra log trước; không bỏ assert hash/POC.

## Pilot10 episode trên hai GPU

```bash
python -u scripts/rl/agent_v4_train.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --output /kaggle/working/vvc_agent_v4_run \
  --preset content-task --epochs 3 --frames 256 --workers 2 \
  --max-episodes 10 --max-hours 11.5
```

Hai GPU dùng cho hai frozen-YOLO collector độc lập. Encoder/decoder chạy CPU4 thread; actor bridge một CPU thread; PPO update trên CPU sau khi cả hai collector hoàn tất. Không phải VVenC encode hoặc PPO model chạy trên hai GPU.

Trước lần PPO đầu có quick paired Val với zero-action byte-identity gate. Sau10 Train episode có quick Val khác. Mỗi25 PPO update có quick Val; mỗi325 episode có full35-pair Val. Lịch vẫn65×5×3=975; pilot không tự động chạy hết.

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
  --output /kaggle/working/vvc_agent_v4_run \
  --resume-archive /kaggle/input/YOUR_DATASET/VVC_AGENT_V4_RESUME.zip \
  --preset content-task --epochs 3 --frames 256 --workers 2 \
  --max-episodes 90 --max-hours 11.5
```

90 là số **bổ sung**;0 là chạy phần còn lại. Pin lại `experiment.source_commit` trong progress của archive trước khi resume. Code, reward, seed, frame/window, workers và codec đổi thì phải run mới. Chỉ session cap và thời gian được thay đổi.

Review ngày2026-11-09 thêm `update_version=categorical-clip-aux-fullbatch-kl-v2`. Checkpoint tạo bởi bản v4 đầu `6a43f41` vẫn cần source cũ đã pin; không tự nạp vào trainer vừa sửa. Nếu chưa training v4 thì dùng notebook mới để bắt đầu pilot mới.

Driver cũng resume trực tiếp khi output cũ có progress hợp lệ. Nếu Kaggle giải nén Input ZIP, notebook v4 có bước tái đóng gói sau khi kiểm hash từng file theo `bundle_manifest.json`; không cần giữ nguyên byte của ZIP ngoài như bundle Test EP650.

Progress chỉ commit sau khi cả batch có reward thật, LP/value parity PASS, PPO update và checkpoint đã ghi xong. Interrupted attempt có tên riêng và không được lấy làm rollout của policy tiếp theo. Pending Val phải hoàn tất trước khi collect tiếp.

Mốc `--max-hours` tính từ lúc chạy driver, không bao gồm build trước đó. Driver dừng cấp batch mới khi còn ít hơn worker-timeout+10phút để dành thời gian xuất file; có thể dừng sớm đáng kể. Notebook tính budget còn lại sau setup. VM bị hard-kill không bảo đảm giữ batch/ZIP đang ghi, chỉ bảo đảm bản resume atomic gần nhất. Luôn tải cả hai export khi kết thúc phiên.

## Full paired Val trước khi quyết định chạy dài

Có thể kiểm một checkpoint pilot ngay, không phải đợi325 episode:

```bash
python -u scripts/rl/agent_v4_validate.py \
  --repo /kaggle/working/vvenc_rl_v4 \
  --decoder /kaggle/working/vvdec_v3_2_0/bin/release-static/vvdecapp \
  --checkpoint /kaggle/working/vvc_agent_v4_run/checkpoints/iter_000005.pt \
  --output /kaggle/working/v4_candidate_val --workers 2
```

Chạy lại đúng command và output để resume các cặp Val chưa hoàn tất; không update weights. Full default là7 video×5 rate, frame count lấy từ checkpoint. `--quick` chỉ4 cặp. Kết quả stand-alone không âm thầm thay best trong run training.

## Đối chứng

Tạo output khác cho từng run, cùng seed và ngân sách. Các lựa chọn:

| Thí nghiệm | Tham số |
|---|---|
| Chỉ categorical, backbone/state cũ | `--preset categorical-control --reward legacy-frame --auxiliary 0` |
| Encoder/history mới, reward cũ | `--preset encoder-history --reward legacy-frame --auxiliary 0` |
| Thêm content, chưa auxiliary | `--preset content-task --reward legacy-frame --auxiliary 0` |
| Thêm auxiliary | `--preset content-task --reward legacy-frame --auxiliary .1` |
| Objective clip đồng nhất với Val | `--preset content-task --reward clip-delta --auxiliary .1` |

Không cần chạy dài tất cả ngay. Ưu tiên kiểm native gate, pilot và paired Val; tăng budget cho ứng viên có tín hiệu. Không mở Test để điều chỉnh beta/lr/architecture.
