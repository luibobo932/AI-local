# Minion v4 - Báo cáo thực hiện cuối

## Những phần đã hoàn thành

- Bộ SFT 802 mẫu: 700 train, 51 validation, 51 test.
- 100 câu frozen evaluation, trong đó 30 câu critical safety/robotics.
- QLoRA Qwen3-1.7B trên RTX 4060 8 GB với context 1024.
- Lượt 2 epoch, lượt early-stop 0,6 epoch và lượt hardening 430 mẫu.
- Dataset DPO 319 train / 41 validation và pipeline DPO có khóa gate.
- Best-checkpoint, báo cáo train, đánh giá theo nhóm và registry triển khai.

## Kết quả

| Candidate | Frozen score | Critical failure | Quyết định |
|---|---:|---:|---|
| Early-stop 0,6 epoch | 35/100 | 23 | Loại |
| SFT 2 epoch | 49/100 | 13 | Loại |
| Hardened | 53/100 | 14 | Loại |

Hardened đạt honesty 86,7%, safety 65%, tool-calling 70%, nhưng vẫn có lỗi thật: mô tả xóa mạnh và để E-stop phụ thuộc AI. Vì vậy điểm tăng không đủ để triển khai.

## Các bước bị khóa theo kế hoạch

- Không tải/train Qwen3-4B vì 1.7B chưa vượt cổng.
- Không chạy DPO vì còn critical failure.
- Không nối adapter vào server experimental hoặc production.
- Model chính tiếp tục là `qwen3:8b` qua Ollama và lớp tool/approval hiện có.

Đây là kết quả đúng của cơ chế fail-closed: không dùng model yếu chỉ vì đã tốn thời gian train. Vòng tiếp theo cần dữ liệu ít template hơn, nhiều tình huống do người thật review và một tập preference an toàn được chuyên gia xác nhận trước khi thử lại model lớn hơn.
