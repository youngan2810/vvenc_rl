# Rà soát Agent v4 trước pilot

Phạm vi: kiểm tra lại source v4 tại `6a43f41c0fea71d147213b4c8aef980ee7fe4f2a`, đối chiếu nền v3 `afaf5f0c463122067bd3ad6ba62f7913c799f900`. Các sửa chữa của lượt review nằm trong commit kế tiếp trên `agent-v4-dev`.

## Kết luận

Có thể gọi v4 là **bản cải tiến về thiết kế và triển khai**: thêm representation từ ảnh nguồn và lịch sử, categorical action khớp integer QP, auxiliary supervision và objective clip có contract rõ. Chưa có kết quả v4 để kết luận học nhanh hơn, tốt hơn v3, hoặc đã tối ưu. V4 vẫn là ứng viên cần pilot thực tế.

## Các vấn đề đã phát hiện và sửa

| Vấn đề ở bản đầu | Tác động có thể xảy ra | Sửa và kiểm chứng |
|---|---|---|
| Parity dùng `max(error, NaN)` mà không kiểm finite trước | NaN trong old log-prob/value có thể bị bỏ lọt tại cửa kiểm parity | Kiểm shape, dtype và finite của state/action/old-policy/reward/teacher trước optimizer; test chèn NaN vào log-prob và critic |
| Chỉ kiểm KL trước mỗi minibatch actor update | Bước actor/auxiliary cuối có thể vượt target mà `actor_early_stop` vẫn false | Đo lại joint categorical KL trên **toàn batch rollout** sau cập nhật. Backtrack actor nếu vượt target; phục hồi Adam moments trước update khi giảm bước; critic giữ cập nhật riêng |
| Export lấy hash mới của file mà không đối chiếu progress đã commit | Checkpoint hỏng có thể được đóng ZIP hợp lệ về CRC và ghi đè resume bundle tốt | Kiểm progress trên disk và SHA checkpoint latest/best/pending/full-Val trước khi ghi archive; test kiểm archive tốt giữ nguyên byte khi checkpoint bị sửa |
| `stack_states(...,use_content=True,source=None)` trả ảnh zero | Lỗi đường dữ liệu có thể âm thầm vô hiệu hóa content branch | Thiếu source của run content-enabled phải báo lỗi |
| Collector độc lập chưa kiểm source Git; standalone Val chưa từ chối dirty source | Trọng số đúng architecture có thể bị chạy với logic khác source đã pin | Collector và standalone Val kiểm HEAD + tracked source sạch; worker kiểm thêm hash split IDs |
| Baseline legacy reward dùng thứ tự quyết định của PPO | Nếu hai thứ tự khác nhau, discounted return của baseline bị tính theo clock khác | Đọc/kiểm native STATE của chính baseline, ghép AU theo POC và tích lũy theo thứ tự đó. Test dùng hai thứ tự khác nhau cho thấy sai khác discounted return |

Notebook cũng fetch branch mới khi bắt đầu run mới với SOURCE_COMMIT trống, tránh tái dùng một checkout sạch nhưng đã cũ trong Working. Các module Python đã import từ checkout trước được bỏ khỏi cache để helper export của notebook cũng đúng source mới. Resume vẫn pin commit từ archive, không tự nâng cấp thuật toán của run cũ.

Các lỗi được tái hiện bằng fixture/đầu vào cố tình bất thường. Ví dụ KL khoảng101.935 là **stress test actor_lr=.3**, gấp3000 lần mặc định1e−4, một minibatch cuối; không phải số đo training Kaggle mặc định. Không dùng các tái hiện này để kết luận nguyên nhân chất lượng v3 hay rằng training v4 đã collapse.

## Những điều được xác nhận về logic

- Actor không nhận teacher detections/reward hiện tại làm input; content chỉ đọc current source POC. History reset theo episode và lưu các quyết định trước theo encoding decision order
- Log-prob là tổng trên12 categorical head của đúng integer action gửi codec. Deterministic action là argmax; không có Gaussian-round ở v4
- Actor và critic không chia sẻ parameters. Auxiliary gradient đi vào actor representation, không vào critic
- Reward clip-delta với gamma=lambda=1 cộng về đúng clip score. Ngoài test telescoping, test enumeration một môi trường hai quyết định có reference effect xác nhận score-function gradient khớp gradient của expected terminal score, dưới giả thiết prefix không phụ thuộc action tương lai
- Done cắt GAE giữa các clip; collector dùng cùng immutable weights và parent kiểm LP/value parity trước update
- Resume kiểm optimizer/RNG/config/source/hash. Export kết quả và export resume có mục đích riêng; phải lưu cả bằng chứng các phiên trước

PPO clipping không áp đặt một hard KL constraint trong thuật toán gốc; xem [PPO paper](https://arxiv.org/abs/1707.06347) và [giải thích PPO-Clip](https://spinningup.openai.com/en/latest/algorithms/ppo.html). Guard bổ sung chỉ giới hạn **mean KL trên các state vừa thu thập**, không bảo đảm trên toàn phân phối dữ liệu hay bảo đảm return tăng. Nếu `accepted_actor_scale` thường rất nhỏ/0, cần xem lại learning rate/auxiliary qua một thí nghiệm mới, không coi đó là học tốt.

## Rủi ro học vẫn cần thực nghiệm

1. **Reward weight chưa được tối ưu:** beta.1, auxiliary.1 và positive weight3 là cấu hình khởi đầu. Đồng nhất reward với Val không bảo đảm tradeoff detection–bitrate tốt
2. **Auxiliary là car-box presence:** nó giúp học vị trí/nội dung có xe, chưa dạy được độ nhạy detection đối với từng ΔQP. Network cũng có thể học shortcut vị trí; cần ablation content/auxiliary với cùng budget
3. **Monte Carlo variance:** gamma=lambda=1 giữ credit theo objective clip nhưng variance có thể cao. Prefix reward là tín hiệu hậu xử lý; critic chỉ có causal policy state, không thấy prefix YOLO score
4. **Information còn giới hạn:** chưa lấy native motion/reference graph/pre-analysis mới, chưa có CU/TU state cho Partition/MTS. Canonical luma cũng khác buffer MCTF. Không mô tả v4 là đã khai thác toàn bộ encoder information
5. **Headroom/action và generalization:** ±2 có thể chưa tạo đủ tác động hữu ích; categorical vẫn có thể chọn toàn0. Clip256 frame không đại diện hoàn toàn cho full-video. Cần xem RD/F1 trên paired Val và chi phí inference thực tế

## Điều kiện chạy tiếp

Chạy native preflight v4 trên profile codec Kaggle đã pin, rồi pilot10 episode mới và paired Val. Xem cả F1, actual bitrate, entropy, sampled/deterministic action, reward/return variance, LP parity và accepted actor scale. Quick4 pairs là diagnostic, không đủ chốt chất lượng. Full35-pair Val và đối chứng cùng interaction budget cần có trước khi tuyên bố v4 vượt v3.

Các kiểm tra local là CPU contract/regression tests và notebook validation. Chưa chạy native v4 preflight hay v4 training T4×2 trong lượt review này. Kết quả cuối về bitrate/task quality còn mở.
