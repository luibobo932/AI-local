/* Smoke test cho engine — chạy: node tests/engine.test.js */
const E = require("../engine.js");
global.window = global;
require("../data.js");
const L = global.MINION_LISTINGS;

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.error("✗ " + msg); } }

// parseNeed
let n = E.parseNeed("nhà quận 7 tầm 5 tỷ 3 phòng ngủ để ở");
ok(n.budgetMax >= 5000 && n.budgetMax <= 5600, "budget ~5 tỷ");
ok(n.districts.includes("Quận 7"), "nhận diện Quận 7");
ok(n.bedroomsMin === 3, "3 phòng ngủ");
ok(n.purpose === "để ở", "mục đích để ở");

n = E.parseNeed("mặt tiền kinh doanh dưới 9 tỷ Tân Bình");
ok(n.purpose === "kinh doanh", "mục đích kinh doanh");
ok(n.districts.includes("Tân Bình"), "nhận diện Tân Bình");

// recommend: không trả căn vượt ngân sách nhiều
n = E.parseNeed("dưới 2.5 tỷ để ở");
let recs = E.recommend(n, L, 5);
ok(recs.length >= 1, "có gợi ý dưới 2.5 tỷ");
ok(recs.every(r => r.house.price <= 2500 * 1.12), "không gợi ý căn vượt ngân sách nhiều");

// score giảm dần
n = E.parseNeed("nhà Gò Vấp kinh doanh dưới 9 tỷ");
recs = E.recommend(n, L, 4);
ok(recs[0].score >= recs[recs.length - 1].score, "điểm xếp giảm dần");

// pitch chứa thông tin căn
let pitch = E.makePitch(n, recs[0].house);
ok(pitch.includes("Giá") && pitch.includes(recs[0].house.district), "lời chào có giá + khu vực");
ok(!pitch.includes("Nhà nhà"), "không lặp 'Nhà nhà'");

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
