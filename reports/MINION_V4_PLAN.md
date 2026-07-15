# Kế hoạch và cổng triển khai Minion v4

## Dữ liệu

- Train: 700 mẫu thuộc 109 family.
- Validation: 51 mẫu thuộc 8 family tách biệt.
- Test: 51 mẫu thuộc 8 family tách biệt.
- Frozen evaluation: 100 câu không trùng nguyên văn dataset.
- Tổng dataset SFT: 802 mẫu.

Phân bố gồm danh tính, trung thực dữ liệu, an toàn, bất động sản, lập trình/Git, robot, memory/RAG và tool-calling.

## Model và tham số

- Model đầu tiên: `Qwen/Qwen3-1.7B`.
- QLoRA 4-bit NF4, rank 16, alpha 32.
- Context 1024, batch 1, gradient accumulation 16.
- Learning rate khởi đầu `0.0001`, 2 epochs.

Nếu 1.7B vượt cổng, thử `Qwen/Qwen3-4B` với context 512. Model chính `qwen3:8b` không bị thay thế trong giai đoạn thử nghiệm.

## Cổng triển khai

- Không có critical failure.
- Safety: 100%.
- Robotics safety: 100%.
- Identity: tối thiểu 95%.
- Honesty: tối thiểu 95%.
- Tool-calling: tối thiểu 90%.
- Real estate: tối thiểu 90%.
- Coding/Git: tối thiểu 90%.

Adapter chưa đạt chỉ được lưu local và ghi báo cáo; không được nối vào tuyến production.
