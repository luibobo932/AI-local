# Minion v4 1.7B - lượt train ban đầu

- Model: `Qwen/Qwen3-1.7B`
- Dataset: 700 train / 51 validation
- QLoRA: context 1024, 2 epochs, learning rate 1e-4
- Thời gian train: 2.548 giây
- Train loss cuối: 0.9416
- Validation tốt nhất quan sát được: 1.755 tại bước 25
- Validation cuối: 2.292

## Kết luận

Model bắt đầu overfit sau checkpoint 25. Cấu hình cũ chỉ giữ hai checkpoint cuối nên checkpoint tốt nhất đã bị xóa. Pipeline được sửa để giữ ba checkpoint, theo dõi `eval_loss` và tự nạp best model khi kết thúc. Lượt hardening tiếp theo chỉ train 0,6 epoch để dừng gần vùng tốt nhất.
