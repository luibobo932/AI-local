# Minion SFT Seed v2

- Base model: `Qwen/Qwen3-0.6B`
- Dataset: 142 examples, bổ sung mẫu đối kháng về xóa dữ liệu và robot
- QLoRA: 4-bit NF4, LoRA r=16, 2 epochs, context 512
- Train loss: 1.9770
- Eval loss: 1.9637 (tốt hơn v1: 2.2905)
- Eval token accuracy: 0.6059

## Kết quả hành vi

V2 đã từ chối yêu cầu xóa toàn bộ thư mục và yêu cầu kiểm tra đường dẫn, khắc phục lỗi an toàn nghiêm trọng của v1. Tuy nhiên model 0.6B chưa gọi đúng danh tính Minion/anh Duy, câu trả lời về robot còn chung chung và chưa nhắc quy trình GitHub khi bàn giao.

## Quyết định

Chưa thay model chính. Tiếp tục thêm một epoch từ checkpoint v2, sau đó đánh giá lại bằng các câu diễn đạt mới không trùng nguyên văn dữ liệu train.
