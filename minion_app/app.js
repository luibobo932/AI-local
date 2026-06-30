/* app.js — gắn engine vào giao diện, lưu dữ liệu trên máy (localStorage). */
(function () {
  "use strict";
  const E = window.MinionEngine;
  const $ = (s) => document.querySelector(s);

  const EXAMPLES = [
    "Khách cần nhà Quận 7 tầm 5 tỷ, 3 phòng ngủ để ở",
    "Nhà mặt tiền Gò Vấp hoặc Tân Bình để kinh doanh, dưới 9 tỷ",
    "Vợ chồng trẻ mua nhà nhỏ dưới 2.5 tỷ để ở",
    "Đầu tư căn cho thuê dòng tiền tốt khoảng 8 tỷ"
  ];

  let listings = loadListings();

  // ── Lưu / nạp dữ liệu ──
  function loadListings() {
    try {
      const saved = localStorage.getItem("minion_listings");
      if (saved) return JSON.parse(saved);
    } catch (e) {}
    return (window.MINION_LISTINGS || []).slice();
  }
  function saveListings(arr) {
    listings = arr;
    try { localStorage.setItem("minion_listings", JSON.stringify(arr)); } catch (e) {}
    updateCount();
  }
  function updateCount() { $("#count").textContent = listings.length + " căn"; }

  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 1800);
  }

  // ── Render gợi ý ──
  function runMatch() {
    const text = $("#need").value.trim();
    if (!text) { toast("Hãy nhập nhu cầu của khách"); return; }
    const need = E.parseNeed(text);
    renderParsed(need);
    const recs = E.recommend(need, listings, 4);
    const box = $("#results");
    if (!recs.length) {
      box.innerHTML = '<div class="empty"><div class="big">🤔</div><p>Chưa có căn nào khớp. Thử nới ngân sách hoặc đổi khu vực.</p></div>';
      return;
    }
    box.innerHTML = recs.map((r, i) => houseCard(r, need, i)).join("");
    bindCardActions(need, recs);
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderParsed(need) {
    const tags = [];
    if (need.budgetMax) tags.push("Ngân sách ≤ " + E.priceText(need.budgetMax));
    if (need.districts.length) tags.push("Khu vực: " + need.districts.join(", "));
    if (need.bedroomsMin) tags.push(need.bedroomsMin + "+ phòng ngủ");
    if (need.areaMin) tags.push("≥ " + need.areaMin + "m²");
    if (need.type) tags.push("Loại: " + need.type);
    if (need.purpose) tags.push("Mục đích: " + need.purpose);
    const p = $("#parsed");
    if (!tags.length) { p.classList.add("hidden"); return; }
    p.classList.remove("hidden");
    p.innerHTML = "<b>Minion hiểu:</b> " + tags.map(t => '<span class="tag">' + t + "</span>").join("");
  }

  function houseCard(r, need, idx) {
    const h = r.house;
    const size = (h.width && h.length) ? `${h.width}×${h.length}m (${h.area}m²)` : `${h.area}m²`;
    const reasons = r.reasons.filter(x => x.indexOf("Khác") !== 0 && x.indexOf("Ít ") !== 0 && x.indexOf("Diện tích nhỏ") !== 0).slice(0, 4);
    return `
    <div class="house" data-idx="${idx}">
      <div class="house-top">
        <div class="match" style="--p:${r.score * 3.6}deg"><span>${r.score}%</span></div>
        <div class="house-head">
          <h3>${cap(h.type)} ${h.street}</h3>
          <div class="addr">${[h.address_no, h.street].filter(Boolean).join(" ")}, ${h.district}</div>
        </div>
      </div>
      <div class="house-meta">
        <span><b>${E.priceText(h.price)}</b></span>
        <span>${size}</span>
        ${h.bedrooms ? `<span>${h.bedrooms} PN</span>` : ""}
        ${h.direction ? `<span>Hướng ${h.direction}</span>` : ""}
      </div>
      <div class="reasons">${reasons.map(x => `<div>${x}</div>`).join("")}</div>
      <div class="pitch-box hidden" id="pitch-${idx}"></div>
      <div class="house-actions">
        <button class="btn-pitch" data-idx="${idx}">✍️ Soạn lời chào</button>
        <button class="btn-copy" data-idx="${idx}">📋 Copy</button>
      </div>
    </div>`;
  }

  function bindCardActions(need, recs) {
    document.querySelectorAll(".btn-pitch").forEach(btn => {
      btn.onclick = () => {
        const i = +btn.dataset.idx;
        const box = $("#pitch-" + i);
        if (!box.dataset.text) box.dataset.text = E.makePitch(need, recs[i].house);
        box.textContent = box.dataset.text;
        box.classList.toggle("hidden");
      };
    });
    document.querySelectorAll(".btn-copy").forEach(btn => {
      btn.onclick = async () => {
        const i = +btn.dataset.idx;
        const text = E.makePitch(need, recs[i].house);
        try { await navigator.clipboard.writeText(text); toast("Đã copy lời chào ✓"); }
        catch (e) {
          const box = $("#pitch-" + i); box.dataset.text = text; box.textContent = text; box.classList.remove("hidden");
          toast("Đã hiện lời chào — bạn copy thủ công nhé");
        }
      };
    });
  }

  // ── Kho nhà ──
  function renderList() {
    const v = $("#view-list");
    v.innerHTML = "<h2 style='margin-bottom:10px'>Kho nhà (" + listings.length + ")</h2>" +
      listings.map(h => `
        <div class="mini-house">
          <h4>${cap(h.type)} ${h.street} — ${E.priceText(h.price)}</h4>
          <p>${[h.address_no, h.street].filter(Boolean).join(" ")}, ${h.district} • ${h.area}m²${h.bedrooms ? " • " + h.bedrooms + " PN" : ""}</p>
          <p>${h.note || ""}</p>
        </div>`).join("");
  }

  // ── Điều hướng tab ──
  function showView(view) {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.view === view));
    $("main").classList.toggle("hidden", view !== "match");
    $("#view-list").classList.toggle("hidden", view !== "list");
    $("#view-data").classList.toggle("hidden", view !== "data");
    if (view === "list") renderList();
  }

  // ── Khởi tạo ──
  function init() {
    updateCount();
    $("#examples").innerHTML = EXAMPLES.map(e => `<button>${e}</button>`).join("");
    document.querySelectorAll("#examples button").forEach(b => {
      b.onclick = () => { $("#need").value = b.textContent; runMatch(); };
    });
    $("#go").onclick = runMatch;
    document.querySelectorAll(".tab").forEach(t => t.onclick = () => showView(t.dataset.view));

    $("#loadData").onclick = () => {
      try {
        const arr = JSON.parse($("#dataInput").value);
        if (!Array.isArray(arr) || !arr.length) throw new Error("rỗng");
        saveListings(arr);
        $("#dataStatus").textContent = "✓ Đã nạp " + arr.length + " căn.";
        toast("Đã nạp " + arr.length + " căn");
      } catch (e) { $("#dataStatus").textContent = "✗ JSON không hợp lệ: " + e.message; }
    };
    $("#resetData").onclick = () => {
      saveListings((window.MINION_LISTINGS || []).slice());
      $("#dataStatus").textContent = "✓ Đã khôi phục dữ liệu mẫu.";
      toast("Đã khôi phục dữ liệu mẫu");
    };
  }

  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ""; }

  document.addEventListener("DOMContentLoaded", init);
})();
