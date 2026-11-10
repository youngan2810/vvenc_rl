# Agent v4.2 — sửa cơ chế học sau pilot v4.1

**Cập nhật sau pilot50:** source283a16e đã chạy đủ50 episode/25 update.
Greedy quick Val tăng1,105 điểm F1, actual bitrate tăng1,383%; phần lớn delta
vẫn đồng nhất trong frame và sampled probe chưa tốt hơn khởi tạo. Revision
hiện hành: [frozen multi-clip PPO](AGENT_V4_MULTICLIP_RESEARCH.md), yêu cầu
new10 smoke; các hợp đồng dưới đây mô tả nền thuật toán v4.2 trước revision đó.

Trạng thái: đã triển khai và kiểm thử CPU. Run Kaggle v4.2 đầu dừng tại initial
Val do native zero-stream drift, trước mọi episode Train/PPO update. Chưa có
gain RL chất lượng thực nghiệm. Runtime hiện tại dùng experiment FPP1 mới;
xem [bằng chứng và gate còn chờ](AGENT_V42_FPP1_PROFILE.md).
V3 và source C++ không thay đổi.

## Vì sao cần sửa

Pilot v4.1 hoàn thành10 episode/5 PPO update nhưng greedy chọn0 ở mọi CTU.
Khoảng59% sampled nonzero không chứng minh policy triển khai tốt hơn.
Content/history cải thiện information budget, chưa sửa hết credit assignment:
một advantage frame dùng cho12 action, presence auxiliary không nhận action,
critic MC tail target chứa prefix task score không có trong input.

Một cặp Val256 frame còn có zero action nhưng stream khác vanilla. Native
preflight32 frame PASS không đủ bao phủ trường hợp này. V4.2 chặn kết quả đó,
không sửa raw metric hoặc whitelist golden SHA khác để tiếp tục.

## Hợp đồng learning mới

- Model `ctu-categorical-counterfactual-outcome-v4.2`
- Checkpoint/run schema `vvenc-agent-v4/2`
- Update `final-score-masked-ctu-ppo-outcome-aux-v3`
- State vẫn `causal-ctu-source-history/1`: actor dùng canonical current-source
  thumbnail, native state và history; không dùng current decoded detections
- Actor-only inference vẫn categorical argmax, không gọi critic/auxiliary

### Final-score target và paired Train

Default `clip-delta` giữ metric cuối clip:

    S = micro_F1 - 0.1 * abs(log(actual_bps/target_bps))

Reward vẫn được lưu theo prefix để audit, nhưng actor/critic default dùng
**toàn bộ final sum của từng episode**, không dùng tail S_T−S_prev. Mỗi Train
clip có paired vanilla trên cùng source/window/rate và detector. Target là
`D = S_policy − S_vanilla`, broadcast riêng trong từng clip, không qua boundary.
Baseline là một encode đối chứng độc lập với action-map được sample; không
trộn trajectory vanilla vào PPO và không replay PPO dữ liệu cũ.

Paired training tăng chi phí encode/decode/detector so với v4.1. Chưa triển
khai cache baseline qua phiên; mỗi paired encode/detection được giữ làm
evidence. Không so wall time hoặc interaction budget mà bỏ overhead này.

### Baseline phản thực theo CTU

Critic riêng tạo `Q_i(h,a_-i)[5]` cho12 CTU. One-hot của action i bị mask
**trước** khi mix action-map; state features của critic không chứa current
joint action. Output ứng với action thực được fit vào observed D; các bin
chưa thử không có nhãn codec thật. Q output được khởi tạo0 để không tạo
baseline tùy ý quá lớn trước lần fit đầu.

Trước bất kỳ critic fit nào của batch hiện tại:

    b_i = sum_k pi_old_i(k|h) * Q_frozen_i(h,a_-i)[k]
    A_i = D_observed - b_i

Q chỉ là baseline giảm variance, không thay D bằng Q(chosen). Với actor
factorized, baseline frozen không phụ thuộc action i có conditional expected
score-function gradient bằng0, dù critic chưa chính xác. Exact enumeration
test dùng cả một critic cố tình dự đoán sai để kiểm tra tính chất này.
Critic fit diễn ra sau khi A đã được freeze; không fit nhãn của batch rồi
quay lại tạo baseline cho chính batch đó.

PPO surrogate mới clip ratio **từng CTU**, cộng12 head rồi lấy mean batch.
Entropy là mean/CTU; exact KL vẫn là **joint sum12 head**, kiểm full rollout
sau bước cuối và backtrack/restore Adam moments nếu cần. Clip fraction giờ
là per-CTU trong chế độ này, không so trực tiếp với joint clip fraction cũ.
Đây là extension có đối chứng, không gọi là COMA nguyên bản hoặc PPO/TRPO
có bảo đảm return tăng. Common batch advantage normalization vẫn là một
thực hành xấp xỉ; test gradient lý thuyết dùng advantage chưa normalize.

Critic scalar cũng fit D để parity/diagnostics và ablation shared-final.
Mask invariance được test bằng đổi riêng action i: Q_i phải giữ nguyên.
Không có nhãn phản thực thật cho action chưa encode; calibration/headroom
vẫn cần codec thật.

### Auxiliary task-outcome và gradient budget

Source teacher presence vẫn lưu để ablation, nhưng default auxiliary mới
nhận actor tokens **và action-map thực**. Nó dự đoán log1p local TP/FP/FN
bằng SmoothL1. Head chỉ dùng khi training, không trở thành actor observation.

Matcher confidence-ordered chạy toàn frame. Mỗi TP/FN phân bổ theo diện tích
reference box giao CTU; FP theo reconstructed box. Mỗi event có tổng weight1,
nên tổng local counts bằng counts toàn frame, kể cả box băng qua nhiều ô và
CTU cạnh có kích thước khác. Không match riêng từng CTU gây double count.
Chưa có CTU bits native nên không bịa local bit labels từ AU tổng.

Task prediction là supervision theo outcome đã đo, chưa phải causal
sensitivity. `--auxiliary-mode presence` và `--auxiliary 0` là ablation rõ
ràng, mỗi thay đổi config tạo run mới.

Raw auxiliary gradient được project nếu ngược hướng primary PPO+entropy,
sau đó cap global L2 norm ở0.5×primary norm. Khi primary gradient bằng0 thì
không cho aux-only update được ghi như RL progress. Ghi norm, cosine,
conflict fraction và accepted scale. Đây là budget raw gradient, không phải
bảo đảm hướng bước Adam hoặc return; final joint KL guard vẫn áp dụng.

## Validation và tái lập native

Primary deployment rule là argmax, khóa trong experiment. Quick Val gồm:

-4 paired greedy cases:2 Val video × rates50/120
-8 paired sampled diagnostic cases: cùng4 case với seed20261117 và20261143

Sampled Val chạy actor-only, `trainable=False`, không vào optimizer, không
tranh best, không chọn seed tốt nhất. Full Val chỉ35 greedy paired cases;
chỉ full protocol đó được chọn best checkpoint. Mọi report giữ score/F1,
actual bitrate, greedy nonzero và diagnostic seed metrics tách biệt.
`NO_GREEDY_GAIN_ESTABLISHED` có nghĩa chưa có gain, không có nghĩa hội tụ.

Mỗi all-zero paired result phải byte-identical với vanilla; identical stream
phải khớp task metrics (chỉ tolerance floating sum1e-12). Mismatch lưu failure
JSON và dừng trước success/PPO commit. Lần initial quick Val còn chạy explicit
live-zero gate cho cả4 greedy pairs, thay vì chỉ cặp đầu.

Notebook mặc định có `agent_v4_headroom.py`: locked Train index0, frame0..255,
rates50/120. Mỗi rate chạy vanilla lặp2 lần, fixed-zero, live-zero, uniform±1
và teacher-ROI map. Zero/repeat mismatch dừng, giữ streams/AU/POC/detections.
Teacher maps là diagnostic đặc quyền, không phải actor policy hay upper bound.
Observed probe gain chỉ nói một map đã đo tốt hơn paired vanilla; không có
probe gain trên subset nhỏ cũng không chứng minh toàn bộ action space vô ích.
CLI này không update weights, không đọc Test, và có budget/export riêng.

Giữ source/binary canonical và golden SHA. V4.2 chưa sửa nguyên nhân native
stream drift vì archive chưa xác định nguyên nhân. Nếu gate fail, dùng fixed
replay so với online zero để khoanh vùng IPC/frame-parallel RC, không đổi
PPO/expected SHA để che lỗi. Nếu cần sửa C++ hoặc đổi profile thì phải tạo
protocol/experiment riêng và kiểm chứng lại baseline.

## Đối chứng và giới hạn

- Default: `--reward clip-delta --credit-assignment counterfactual-final`
- Tách baseline: `--credit-assignment shared-final`, vẫn paired final-score
- Tail/joint control: `--credit-assignment joint-gae --reward clip-delta`
- Legacy reward control: `--reward legacy-frame --credit-assignment joint-gae`
- Auxiliary ablation: task-outcome, presence, aux0; giữ budget khác tương đương

Train vẫn stochastic. Các sửa credit/outcome có thể giúp greedy policy học
action hữu ích nhưng không có bảo đảm argmax sẽ thắng. Nếu stochastic tăng
nhưng greedy không tăng, phải có Train-only measured policy-improvement/
distillation experiment riêng; chưa triển khai distillation trong bản này.
Không ép nonzero, không đổi decoder sau khi nhìn Val và không chọn trên Test.

Regression tests gồm complete-episode targets, exact policy-gradient
enumeration, own-action masking, outcome conservation, action-conditioned
aux/content gradients, projection budget, optimizer/KL/archive/resume và
hai-context bandit học greedy action khác0 đúng mục tiêu. Bài toán giả lập
không thay thế native/GPU validation và không chứng minh VVenC gain.

## Tài liệu nền

- [PPO](https://arxiv.org/abs/1707.06347), [GAE](https://arxiv.org/abs/1506.02438)
- [Counterfactual policy gradients](https://arxiv.org/abs/1705.08926)
- [RL-RC-DoT, block task-score auxiliary](https://arxiv.org/html/2501.12216v2)
