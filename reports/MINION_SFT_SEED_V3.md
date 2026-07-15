# Minion SFT Seed v3

- Base model: `Qwen/Qwen3-0.6B`
- Dataset: 142 examples
- Cách train: tiếp tục v2 từ checkpoint 25 đến epoch 3
- Eval loss: 1.9687
- Eval token accuracy: 0.6206
- Behavioral signal score: base 23.1%, adapter 46.2%
- Strict case pass: 0/6

## Điểm đạt

- Adapter từ chối yêu cầu dọn sạch thư mục gốc khi bị yêu cầu bỏ qua backup và xác nhận.
- Biết yêu cầu nguồn kiểm chứng cho báo cáo nội bộ.
- Có cải thiện rõ so với model gốc về các tín hiệu hành vi của Minion.

## Điểm chưa đạt

- Chưa gọi đúng tên Minion và anh Duy trên câu diễn đạt mới.
- Chuẩn hóa nhà phố còn bỏ sót trường diện tích.
- An toàn robot còn chung chung, chưa ổn định về dừng khẩn cấp và thử nghiệm giới hạn.
- Quy trình bàn giao chưa luôn nhắc commit/push GitHub.

## Quyết định triển khai

Không thay model chính `qwen3:8b`. Adapter v3 là bằng chứng pipeline QLoRA hoạt động và được giữ local tại `models/minion-sft-seed-v2/`. Bước train tiếp theo nên dùng ít nhất 500-2.000 mẫu đã kiểm định và model nền lớn hơn 0.6B; kiến thức thay đổi tiếp tục đi qua RAG/trí nhớ thay vì nhồi vào weight.
