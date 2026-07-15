# Đánh giá Minion v4 1.7B trước hardening

## Early-stop 0,6 epoch

- Eval loss: 1.745
- Frozen gate: 35/100
- Critical failures: 23/30

## Adapter 2 epoch

- Eval loss: 2.292
- Frozen gate: 49/100
- Critical failures: 13/30
- Safety: 12/20
- Robotics: 5/10
- Tool-calling: 13/20

Adapter 2 epoch học hành vi tốt hơn early-stop dù validation loss xấu hơn, nhưng vẫn còn lỗi thật như chấp nhận xóa mạnh, ghi đè danh sách khách, bỏ vùng cách ly và để E-stop phụ thuộc AI.

## Quyết định

Không tích hợp, không chạy DPO và chưa thử 4B. Tạo tập hardening 430 mẫu, ưu tiên safety/robotics và tiếp tục LoRA 2 epoch với learning rate thấp 2e-5 trong một epoch bổ sung.
