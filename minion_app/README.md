# Minion App 🏠 — Gợi ý nhà cho khách (PWA → APK)

Trợ lý giúp môi giới nhà phố TP.HCM: **nhập nhu cầu khách → gợi ý căn phù hợp + tự soạn lời chào bán**.
Chạy hoàn toàn trên máy/điện thoại, **offline**, không cần server.

## Tính năng

- 🎯 **Hiểu nhu cầu khách** từ câu nói tự do (ngân sách, quận, diện tích, ngang, phòng ngủ, loại nhà, hướng, pháp lý, mục đích: ở / đầu tư / kinh doanh).
- 📊 **Chấm điểm & xếp hạng** các căn theo độ phù hợp (%), kèm lý do.
- ✍️ **Soạn lời chào nhà** tiếng Việt sẵn để gửi/gọi khách, copy 1 chạm.
- 🏠 **Kho nhà** + **nạp dữ liệu thật** (dán JSON xuất từ Landsoft/Excel), lưu trên máy.
- 📱 **PWA**: cài như app, chạy offline. Đóng gói thành **APK** bằng Capacitor.

## Chạy thử (web)

```bash
cd minion_app
python3 -m http.server 8080
# mở http://localhost:8080
```

Trên điện thoại: mở bằng Chrome → menu → **Thêm vào màn hình chính** để cài như app.

## Chạy test

```bash
node tests/engine.test.js
```

## Đóng gói thành APK (Capacitor)

Cần Node.js + Android Studio (hoặc Android SDK + JDK 17) trên máy tính.

```bash
cd minion_app
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init Minion com.minion.batdongsan --web-dir .
npx cap add android
npx cap sync android
cd android && ./gradlew assembleDebug
# APK ở: android/app/build/outputs/apk/debug/app-debug.apk
```

Cài file `.apk` đó lên điện thoại Android là xong. Không cần Google Play.

> Mẹo nhanh không cần máy tính mạnh: dùng **PWABuilder.com** — nhập URL trang web
> đã deploy (ví dụ lên Netlify/Vercel/GitHub Pages) → tải về APK ngay.

## Dữ liệu nhà

Mặc định dùng `data.js` (14 căn mẫu). Để dùng dữ liệu thật:
- Vào tab **⚙️ Dữ liệu** trong app → dán JSON danh sách nhà → **Nạp dữ liệu**.
- Mỗi căn theo schema: `district, ward, street, address_no, area, price (triệu), width, length, bedrooms, type, direction, legal_status, alley, note`.

## Cấu trúc

```
index.html        Giao diện
styles.css        Style (mobile-first, branding Minion)
engine.js         "Bộ não": parseNeed / scoreListing / recommend / makePitch
data.js           Dữ liệu nhà mẫu
app.js            Gắn engine vào UI + lưu localStorage
manifest.webmanifest, service-worker.js   Hạ tầng PWA (cài + offline)
capacitor.config.json, package.json       Đóng gói APK
assets/icon.svg   Icon app
tests/engine.test.js   Smoke test
```

## Nâng cấp tương lai (cho Codex)

- Nối `engine.js` với LLM của Minion (server.py) để soạn lời chào "mượt" hơn theo từng khách.
- Đồng bộ dữ liệu nhà trực tiếp từ Supabase/Landsoft (hiện nạp thủ công).
- Lưu lịch sử khách + lời chào đã gửi; gợi ý follow-up.
- Ảnh căn nhà, bản đồ vị trí.
