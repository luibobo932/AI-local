# Minion v5 — kết quả train local

## Quyết định

Không tích hợp bất kỳ adapter mới nào. Minion production tiếp tục dùng `qwen3:8b`.

## Các vòng đã chạy

| Candidate | Train | Đánh giá | Kết luận |
|---|---:|---:|---|
| `minion-v4-1.7b-dpo-repair` | 319 cặp DPO, 1 epoch | 46/100; 15 critical strict | Loại |
| `minion-v5-4b-pilot` | 700 mẫu, 0,4 epoch | 10/30 critical strict | Loại |
| `minion-v5-4b-hardened` | thêm 430 mẫu, 1 epoch | 8/30 strict; 25/30 semantic | Loại |

## Điều đã học được

- DPO 1.7B học phân biệt cặp validation nhưng không chuyển thành hành vi tốt trên bộ frozen eval.
- Qwen3-4B QLoRA chạy vừa RTX 4060 Laptop 8 GB ở context 512, khoảng 6,7 GB VRAM.
- Hardening 4B làm các câu safety an toàn hơn về nghĩa, nhưng còn 5 lỗi robotics thật hoặc thiếu thông tin an toàn.
- Gate từ khóa tạo 17 false negative trong vòng 4B hardened; vì vậy báo cáo giữ cả điểm strict và semantic, nhưng không dùng semantic review để bỏ qua lỗi thật.

## Lỗi chặn triển khai còn lại

1. Giới hạn khi thử robot thật lần đầu chưa đầy đủ.
2. Phân quyền PLC/AI chưa tách rõ safety controller khỏi AI.
3. Watchdog chưa đưa hệ thống về trạng thái an toàn khi controller treo.
4. Tự nêu giới hạn lực gần người mà không có tiêu chuẩn và đánh giá rủi ro.
5. Quy trình bảo trì thiếu lockout/tagout và xác minh zero-energy.

## Hướng v6

Vòng tiếp theo không tiếp tục train lặp lại cùng dữ liệu. Cần tạo tập teacher-distillation mới từ `qwen3:8b`, tách family khỏi eval, bổ sung curriculum robotics theo safety invariant, và chấm hai lớp: rule-based plus semantic review. Chỉ candidate không còn critical failure mới được chạy đủ 100 câu và cân nhắc tích hợp.
