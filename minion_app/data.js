/*
 * Dữ liệu nhà mẫu — nhà phố TP.HCM.
 * Khớp schema Minion đang dùng (Landsoft/Supabase):
 *   district, ward, street, address_no, area, price (đơn vị: triệu),
 *   width (ngang, m), length (dài, m), bedrooms, type, direction,
 *   legal_status, alley (mô tả hẻm/mặt tiền), note
 *
 * Thay mảng này bằng dữ liệu thật của bạn (xuất từ Landsoft sang JSON)
 * hoặc nạp qua nút "Nạp dữ liệu" trong app.
 */
window.MINION_LISTINGS = [
  { id: "NP001", district: "Quận 7", ward: "Tân Phong", street: "Bùi Bằng Đoàn", address_no: "12",
    area: 90, width: 5, length: 18, bedrooms: 3, price: 7200, type: "nhà phố", direction: "Đông Nam",
    legal_status: "Sổ hồng riêng", alley: "Mặt tiền đường 12m", note: "Khu Phú Mỹ Hưng, gần trường quốc tế, nhà mới hoàn thiện." },
  { id: "NP002", district: "Gò Vấp", ward: "Phường 5", street: "Nguyễn Văn Nghi", address_no: "45/3",
    area: 56, width: 4, length: 14, bedrooms: 3, price: 4900, type: "nhà phố", direction: "Nam",
    legal_status: "Sổ hồng riêng", alley: "Hẻm xe hơi 6m", note: "Gần chợ Gò Vấp, khu dân cư yên tĩnh, thuận tiện đi lại." },
  { id: "NP003", district: "Bình Thạnh", ward: "Phường 13", street: "Nơ Trang Long", address_no: "88",
    area: 48, width: 4, length: 12, bedrooms: 2, price: 3950, type: "nhà phố", direction: "Tây",
    legal_status: "Sổ hồng riêng", alley: "Hẻm 4m xe máy", note: "Giá tốt cho người mua ở lần đầu, gần Emart, Vinhomes." },
  { id: "NP004", district: "Tân Bình", ward: "Phường 11", street: "Âu Cơ", address_no: "210",
    area: 64, width: 4.2, length: 15, bedrooms: 4, price: 8500, type: "nhà phố", direction: "Đông",
    legal_status: "Sổ hồng riêng", alley: "Mặt tiền đường Âu Cơ", note: "Tiện kinh doanh, mặt tiền sầm uất, đang cho thuê 25 triệu/tháng." },
  { id: "NP005", district: "Quận 12", ward: "Hiệp Thành", street: "Tô Ký", address_no: "30/12",
    area: 72, width: 5, length: 14.4, bedrooms: 3, price: 3600, type: "nhà phố", direction: "Bắc",
    legal_status: "Sổ hồng riêng", alley: "Hẻm xe hơi 5m", note: "Khu mới, không gian thoáng, giá mềm phù hợp gia đình trẻ." },
  { id: "NP006", district: "Quận 7", ward: "Phú Thuận", street: "Huỳnh Tấn Phát", address_no: "1005",
    area: 42, width: 4, length: 10.5, bedrooms: 2, price: 3200, type: "nhà phố", direction: "Đông Bắc",
    legal_status: "Sổ hồng riêng", alley: "Hẻm 3m xe máy", note: "Nhỏ gọn, giá tốt, hợp người độc thân hoặc đầu tư cho thuê." },
  { id: "NP007", district: "Thủ Đức", ward: "Linh Đông", street: "Phạm Văn Đồng", address_no: "159",
    area: 100, width: 5, length: 20, bedrooms: 4, price: 9800, type: "nhà phố", direction: "Nam",
    legal_status: "Sổ hồng riêng", alley: "Mặt tiền Phạm Văn Đồng", note: "Vị trí đẹp, tiện kinh doanh và ở, gần Gigamall." },
  { id: "NP008", district: "Bình Tân", ward: "Bình Trị Đông", street: "Số 7", address_no: "27",
    area: 60, width: 4, length: 15, bedrooms: 3, price: 4200, type: "nhà phố", direction: "Tây Nam",
    legal_status: "Sổ hồng riêng", alley: "Hẻm xe hơi 6m", note: "Khu Tên Lửa, dân trí cao, gần AEON Mall Bình Tân." },
  { id: "NP009", district: "Gò Vấp", ward: "Phường 14", street: "Quang Trung", address_no: "612",
    area: 80, width: 4.5, length: 17.8, bedrooms: 4, price: 6700, type: "nhà phố", direction: "Đông Nam",
    legal_status: "Sổ hồng riêng", alley: "Hẻm xe hơi tránh nhau", note: "Nhà 3 tầng đẹp, nội thất cơ bản, vào ở ngay." },
  { id: "NP010", district: "Quận 8", ward: "Phường 6", street: "Phạm Thế Hiển", address_no: "340",
    area: 50, width: 4, length: 12.5, bedrooms: 3, price: 3500, type: "nhà phố", direction: "Nam",
    legal_status: "Sổ hồng riêng", alley: "Hẻm 4m", note: "Giá mềm gần trung tâm, di chuyển qua Quận 5 nhanh." },
  { id: "NP011", district: "Tân Phú", ward: "Tân Quý", street: "Gò Dầu", address_no: "15",
    area: 68, width: 4, length: 17, bedrooms: 3, price: 5300, type: "nhà phố", direction: "Bắc",
    legal_status: "Sổ hồng riêng", alley: "Hẻm xe hơi 5m", note: "Khu an ninh, gần Celadon City, phù hợp gia đình có con nhỏ." },
  { id: "NP012", district: "Quận 7", ward: "Tân Quy", street: "Nguyễn Thị Thập", address_no: "404",
    area: 75, width: 5, length: 15, bedrooms: 3, price: 8800, type: "nhà phố", direction: "Tây Nam",
    legal_status: "Sổ hồng riêng", alley: "Mặt tiền kinh doanh", note: "Mặt tiền Nguyễn Thị Thập, đang kinh doanh cafe, dòng tiền tốt." },
  { id: "NP013", district: "Bình Thạnh", ward: "Phường 25", street: "Điện Biên Phủ", address_no: "21",
    area: 110, width: 6, length: 18.3, bedrooms: 5, price: 15500, type: "biệt thự", direction: "Đông",
    legal_status: "Sổ hồng riêng", alley: "Mặt tiền Điện Biên Phủ", note: "Biệt thự cao cấp, gần Landmark 81, đầu tư cho thuê hoặc ở sang." },
  { id: "NP014", district: "Quận 12", ward: "Thạnh Lộc", street: "Hà Huy Giáp", address_no: "78/5",
    area: 45, width: 4, length: 11, bedrooms: 2, price: 2350, type: "nhà cấp 4", direction: "Nam",
    legal_status: "Sổ hồng riêng", alley: "Hẻm 3m xe máy", note: "Tầm tiền thấp nhất khu, hợp vợ chồng trẻ mua ở hoặc đầu tư." }
];
