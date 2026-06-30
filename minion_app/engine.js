/*
 * engine.js — "bộ não" của Minion cho nghiệp vụ bán nhà phố.
 *
 *   parseNeed(text)            : câu mô tả nhu cầu khách (tiếng Việt) -> tiêu chí có cấu trúc
 *   scoreListing(need, house)  : chấm điểm 1 căn so với nhu cầu -> {score, reasons, fatal}
 *   recommend(need, listings)  : xếp hạng, trả về top phù hợp
 *   makePitch(need, house)     : soạn "lời chào nhà" tiếng Việt để gửi/gọi khách
 *
 * Thuần JavaScript, chạy offline trong app — không cần server, không cần mạng.
 */
(function (global) {
  "use strict";

  // ── Bỏ dấu tiếng Việt để so khớp linh hoạt ──────────────────────────────────
  function noAccent(s) {
    return (s || "")
      .toLowerCase()
      .replace(/đ/g, "d")
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "");
  }

  const DISTRICTS = [
    "Quận 1","Quận 2","Quận 3","Quận 4","Quận 5","Quận 6","Quận 7","Quận 8",
    "Quận 9","Quận 10","Quận 11","Quận 12","Bình Thạnh","Gò Vấp","Tân Bình",
    "Tân Phú","Phú Nhuận","Bình Tân","Thủ Đức","Nhà Bè","Hóc Môn","Bình Chánh"
  ];

  // ── Phân tích nhu cầu khách từ câu nói tự do ─────────────────────────────────
  function parseNeed(text) {
    const t = noAccent(text);
    const need = {
      raw: text, budgetMax: null, budgetMin: null, districts: [],
      areaMin: null, widthMin: null, bedroomsMin: null,
      type: null, direction: null, wantLegal: false, purpose: null
    };

    // Ngân sách (giá): "2 ty", "duoi 5 ty", "tam 3-4 ty", "khoang 2ty5", "800 trieu"
    const billions = [...t.matchAll(/(\d+(?:[.,]\d+)?)\s*(?:ty|tỷ|t\b)/g)].map(m => parseFloat(m[1].replace(",", ".")));
    // "2ty5" = 2.5 tỷ
    const billionsHalf = [...t.matchAll(/(\d+)\s*ty\s*(\d)/g)].map(m => parseFloat(m[1]) + parseFloat(m[2]) / 10);
    const allB = billionsHalf.length ? billionsHalf : billions;
    const millions = [...t.matchAll(/(\d{3,4})\s*(?:trieu|tr\b)/g)].map(m => parseFloat(m[1]));
    let prices = allB.map(b => b * 1000).concat(millions); // quy về triệu
    if (prices.length) {
      if (/duoi|toi da|khong qua|max|tam|khoang|tren duoi|<=|</.test(t) && prices.length === 1) {
        need.budgetMax = prices[0] * 1.05; // nới 5% cho "tầm/khoảng"
      } else if (prices.length >= 2) {
        need.budgetMin = Math.min(...prices);
        need.budgetMax = Math.max(...prices);
      } else {
        need.budgetMax = prices[0] * 1.08; // 1 con số -> coi là trần mềm
      }
    }

    // Quận / khu vực
    for (const d of DISTRICTS) {
      if (t.includes(noAccent(d))) need.districts.push(d);
    }
    // "quan 7", "q7", "q.7"
    const qm = [...t.matchAll(/\bq\.?\s*(\d{1,2})\b/g)].map(m => "Quận " + m[1]);
    for (const d of qm) if (!need.districts.includes(d)) need.districts.push(d);

    // Diện tích tối thiểu: "tren 50m2", "50m", "dien tich 60"
    const area = t.match(/(?:tren|>=|tu)?\s*(\d{2,3})\s*m(?:2|²|\b)/);
    if (area) need.areaMin = parseFloat(area[1]);
    // Kích thước ngang x dài: "4x15", "ngang 4"
    const wl = t.match(/(\d(?:[.,]\d)?)\s*[x\*]\s*(\d{1,2}(?:[.,]\d)?)/);
    if (wl) { need.widthMin = parseFloat(wl[1].replace(",", ".")); if (!need.areaMin) need.areaMin = Math.round(parseFloat(wl[1].replace(",",".")) * parseFloat(wl[2].replace(",","."))); }
    const ng = t.match(/ngang\s*(\d(?:[.,]\d)?)/);
    if (ng) need.widthMin = parseFloat(ng[1].replace(",", "."));

    // Số phòng ngủ: "3 phong ngu", "3pn"
    const pn = t.match(/(\d)\s*(?:phong ngu|pn|phong)/);
    if (pn) need.bedroomsMin = parseInt(pn[1]);

    // Loại nhà
    if (/biet thu/.test(t)) need.type = "biệt thự";
    else if (/cap 4|cap bon/.test(t)) need.type = "nhà cấp 4";
    else if (/can ho|chung cu/.test(t)) need.type = "căn hộ";
    else if (/nha pho|nha mat tien|nha hem/.test(t)) need.type = "nhà phố";

    // Hướng
    const dirs = [["dong nam","Đông Nam"],["dong bac","Đông Bắc"],["tay nam","Tây Nam"],
                  ["tay bac","Tây Bắc"],["dong","Đông"],["tay","Tây"],["nam","Nam"],["bac","Bắc"]];
    for (const [k, v] of dirs) { if (new RegExp("huong\\s*" + k).test(t)) { need.direction = v; break; } }

    // Pháp lý
    if (/so hong|so do|phap ly|chinh chu/.test(t)) need.wantLegal = true;

    // Mục đích
    if (/dau tu|cho thue|dong tien|sinh loi/.test(t)) need.purpose = "đầu tư";
    else if (/kinh doanh|buon ban|mat tien|mo shop|mo cua hang/.test(t)) need.purpose = "kinh doanh";
    else if (/de o|gia dinh|sinh song|an cu|o lau dai/.test(t)) need.purpose = "để ở";

    return need;
  }

  // ── Tiện ích ────────────────────────────────────────────────────────────────
  function priceText(triệu) {
    if (triệu == null) return "chưa có";
    return triệu >= 1000 ? (triệu / 1000).toFixed(triệu % 1000 ? 1 : 0).replace(/\.0$/, "") + " tỷ"
                         : Math.round(triệu) + " triệu";
  }
  function isFrontage(h) { return /mat tien/.test(noAccent(h.alley || "")); }
  function alleyCar(h) { return /xe hoi|xe hơi|o to|ô tô/.test(noAccent(h.alley || "")); }

  // ── Chấm điểm 1 căn so với nhu cầu ───────────────────────────────────────────
  function scoreListing(need, h) {
    let score = 0, max = 0; const reasons = []; let fatal = false;

    // Ngân sách (trọng số cao nhất)
    max += 35;
    if (need.budgetMax != null) {
      if (h.price <= need.budgetMax) {
        score += 35; reasons.push("Trong tầm ngân sách (" + priceText(h.price) + ")");
        if (need.budgetMin != null && h.price < need.budgetMin * 0.85) reasons.push("Giá thấp hơn khoảng mong muốn — thương lượng tốt");
      } else if (h.price <= need.budgetMax * 1.12) {
        score += 18; reasons.push("Nhỉnh hơn ngân sách một chút (" + priceText(h.price) + ") — có thể thương lượng");
      } else {
        fatal = true; reasons.push("Vượt ngân sách nhiều (" + priceText(h.price) + ")");
      }
    } else { score += 18; }

    // Khu vực
    max += 20;
    if (need.districts.length) {
      if (need.districts.includes(h.district)) { score += 20; reasons.push("Đúng khu vực mong muốn (" + h.district + ")"); }
      else { score += 0; reasons.push("Khác khu vực (" + h.district + ")"); }
    } else { score += 12; }

    // Diện tích
    max += 12;
    if (need.areaMin != null) {
      if (h.area >= need.areaMin) { score += 12; reasons.push("Diện tích đạt yêu cầu (" + h.area + "m²)"); }
      else if (h.area >= need.areaMin * 0.9) { score += 7; }
      else { reasons.push("Diện tích nhỏ hơn mong muốn (" + h.area + "m²)"); }
    } else { score += 7; }

    // Ngang
    max += 8;
    if (need.widthMin != null) {
      if ((h.width || 0) >= need.widthMin) { score += 8; reasons.push("Mặt tiền ngang " + h.width + "m đạt yêu cầu"); }
      else { score += 2; }
    } else { score += 5; }

    // Phòng ngủ
    max += 8;
    if (need.bedroomsMin != null) {
      if ((h.bedrooms || 0) >= need.bedroomsMin) { score += 8; reasons.push((h.bedrooms) + " phòng ngủ — đủ cho nhu cầu"); }
      else { reasons.push("Ít phòng ngủ hơn mong muốn"); }
    } else { score += 5; }

    // Loại nhà
    max += 7;
    if (need.type) {
      if (h.type === need.type) { score += 7; reasons.push("Đúng loại " + h.type); }
      else { score += 1; }
    } else { score += 4; }

    // Hướng
    max += 4;
    if (need.direction) {
      if (h.direction === need.direction) { score += 4; reasons.push("Hướng " + h.direction + " như mong muốn"); }
    } else { score += 2; }

    // Pháp lý
    max += 6;
    if (/so hong|so do/.test(noAccent(h.legal_status || ""))) {
      score += 6; if (need.wantLegal) reasons.push("Pháp lý rõ ràng: " + h.legal_status);
    }

    // Bonus theo mục đích
    if (need.purpose === "kinh doanh") {
      if (isFrontage(h)) { score += 8; max += 8; reasons.push("Mặt tiền — rất hợp kinh doanh/cho thuê"); }
      else { max += 8; }
    } else if (need.purpose === "đầu tư") {
      if (/cho thue|kinh doanh|dong tien|cafe/.test(noAccent(h.note || ""))) { score += 6; max += 8; reasons.push("Có dòng tiền/khai thác cho thuê được"); }
      else { max += 8; }
    } else if (need.purpose === "để ở") {
      if (alleyCar(h) && !isFrontage(h)) { score += 6; max += 8; reasons.push("Hẻm xe hơi yên tĩnh — hợp để ở"); }
      else { max += 8; }
    }

    const pct = Math.round((score / max) * 100);
    return { score: fatal ? 0 : pct, raw: score, reasons, fatal };
  }

  // ── Xếp hạng gợi ý ──────────────────────────────────────────────────────────
  function recommend(need, listings, k) {
    k = k || 3;
    const scored = listings.map(h => ({ house: h, ...scoreListing(need, h) }))
      .filter(x => !x.fatal)
      .sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  // ── Soạn "lời chào nhà" để gửi/gọi khách ─────────────────────────────────────
  function makePitch(need, h) {
    const addr = [h.address_no, h.street].filter(Boolean).join(" ");
    const size = (h.width && h.length) ? `${h.width} x ${h.length}m (${h.area}m²)` : `${h.area}m²`;
    const purpose = need.purpose || "để ở";

    const opens = {
      "để ở": "Em có căn này rất hợp để gia đình mình an cư,",
      "đầu tư": "Em có căn này khai thác dòng tiền tốt, hợp đầu tư,",
      "kinh doanh": "Em có căn mặt tiền tiện kinh doanh đúng nhu cầu mình,"
    };
    const typeLabel = h.type ? h.type.charAt(0).toUpperCase() + h.type.slice(1) : "Nhà";
    const lines = [];
    lines.push(`Dạ ${opens[purpose] || opens["để ở"]} anh/chị tham khảo nhé:`);
    lines.push("");
    lines.push(`🏠 ${typeLabel} ${addr}, ${h.district}`);
    lines.push(`• Diện tích: ${size}${h.bedrooms ? ` • ${h.bedrooms} phòng ngủ` : ""}`);
    lines.push(`• Giá: ${priceText(h.price)}${h.direction ? ` • Hướng ${h.direction}` : ""}`);
    if (h.legal_status) lines.push(`• Pháp lý: ${h.legal_status}`);
    if (h.alley) lines.push(`• ${h.alley}`);
    if (h.note) lines.push(`• ${h.note}`);
    lines.push("");

    // Câu nhấn theo nhu cầu
    const hooks = [];
    if (need.budgetMax != null && h.price <= need.budgetMax) hooks.push("đúng tầm tài chính mình đưa ra");
    if (need.districts.includes(h.district)) hooks.push(`ngay ${h.district} như mình muốn`);
    if (need.bedroomsMin && h.bedrooms >= need.bedroomsMin) hooks.push(`đủ ${h.bedrooms} phòng ngủ`);
    if (purpose === "kinh doanh" && isFrontage(h)) hooks.push("vị trí mặt tiền buôn bán thuận lợi");
    if (hooks.length) lines.push(`Căn này ${hooks.join(", ")}. `);
    lines.push("Anh/chị sắp xếp thời gian em dẫn đi xem thực tế nha. Em giữ chỗ ưu tiên cho mình ạ. 🙏");

    return lines.join("\n");
  }

  global.MinionEngine = { parseNeed, scoreListing, recommend, makePitch, priceText, noAccent };
})(typeof window !== "undefined" ? window : globalThis);

// Hỗ trợ chạy test bằng Node
if (typeof module !== "undefined" && module.exports) {
  module.exports = (typeof window !== "undefined" ? window : globalThis).MinionEngine;
}
