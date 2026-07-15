# Minion SFT Seed v1

- Base model: `Qwen/Qwen3-0.6B`
- Dataset snapshot: 119 examples (`800c17e70eb2d742ce1b684394ee319a252b358e9d1979f27e52955ced113934`)
- QLoRA: 4-bit NF4, LoRA r=16, 3 epochs, context 512
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Train loss: 1.4700
- Eval loss: 2.2905
- Eval token accuracy: 0.6034
- Behavioral signal score: base 23.1%, adapter 46.2%

## Quyết định

Không triển khai v1 vào Minion chính. Adapter cải thiện hành vi tổng thể nhưng thất bại cổng an toàn ở một biến thể yêu cầu xóa dữ liệu mà không xác nhận. Dataset v2 bổ sung các mẫu đối kháng về xóa dữ liệu, ghi đè production, danh tính, kiểm chứng và an toàn robot.

Các file model trong `models/minion-sft-seed-v1/` chỉ lưu local; mã nguồn, hash dữ liệu và báo cáo được lưu trong Git để tái tạo.
