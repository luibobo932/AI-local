"""
Chuẩn bị dữ liệu TIẾNG VIỆT cho pre-training (char-level).

Khác với prepare.py (Shakespeare tiếng Anh), script này dùng một corpus
tiếng Việt nhúng sẵn → tokenizer char-level tự động bao gồm các ký tự có dấu
(à, ạ, ê, ộ, ữ...). Nhờ vậy model HỌC ĐƯỢC tiếng Việt thay vì coi dấu là ký tự lạ.

Usage:
    python data/prepare_vi.py                          # dùng corpus có sẵn
    python data/prepare_vi.py --input my_corpus.txt    # corpus tiếng Việt của bạn

Sau đó train:
    python train.py --dataset shakespeare --n_layer 4 --n_head 4 --n_embd 192 \
        --block_size 128 --batch_size 16 --max_iters 3000
"""

import argparse
import os
import pickle
import numpy as np


# ─── Corpus tiếng Việt nhúng sẵn ───────────────────────────────────────────────
# Đa dạng chủ đề: chào hỏi, tục ngữ, mô tả, hội thoại, kể chuyện.
# Càng nhiều và đa dạng, model học càng tốt. Bạn có thể thay bằng --input.

VIETNAMESE_CORPUS = """\
Xin chào, rất vui được gặp bạn. Hôm nay bạn khỏe không?
Tôi khỏe, cảm ơn bạn đã hỏi thăm. Còn bạn thì sao?
Hôm nay trời thật đẹp, nắng vàng trải khắp con đường làng.
Buổi sáng mùa thu, gió heo may nhè nhẹ thổi qua hàng cây.
Mẹ tôi nấu một nồi canh chua cá lóc thơm phức cả gian bếp.
Mùi cơm mới chín quyện với khói bếp khiến lòng người ấm áp.
Con mèo nhỏ nằm cuộn tròn bên cửa sổ, lim dim ngủ dưới nắng.
Đàn chim sẻ ríu rít chuyền cành trên cây bàng trước sân.
Dòng sông quê hương êm đềm chảy qua những cánh đồng lúa xanh.
Chiều về, khói lam chiều bay lên từ những mái nhà tranh.
Trẻ con nô đùa trên đê, thả diều giấy bay cao giữa trời.
Bà tôi kể chuyện cổ tích về cô Tấm hiền lành và quả thị thơm.
Ngày xưa có một chàng tiều phu nghèo sống bên bìa rừng.
Anh ta chăm chỉ làm lụng mỗi ngày để nuôi mẹ già yếu.
Một hôm, anh gặp một bà tiên hiện ra giữa rừng sâu.
Bà tiên ban cho anh một điều ước vì tấm lòng hiếu thảo.
Người Việt Nam coi trọng gia đình, tình làng nghĩa xóm.
Đi đâu xa cũng nhớ về quê hương, nhớ bữa cơm gia đình.
Học tập chăm chỉ thì sẽ thành công, lười biếng thì thất bại.
Có công mài sắt có ngày nên kim, kiên trì là chìa khóa.
Đói cho sạch, rách cho thơm, giữ phẩm giá dù nghèo khó.
Ăn quả nhớ kẻ trồng cây, uống nước nhớ nguồn cội.
Một con ngựa đau cả tàu bỏ cỏ, biết yêu thương đồng loại.
Lá lành đùm lá rách, người với người sống để yêu nhau.
Bạn thích ăn món gì nhất? Tôi thích phở bò và bún chả.
Phở là món ăn nổi tiếng của Hà Nội, nước dùng ngọt thanh.
Bánh mì Việt Nam giòn rụm, kẹp thịt và rau thơm tươi mát.
Cà phê sữa đá là thức uống quen thuộc của người Sài Gòn.
Mùa hè, một cốc chè đậu xanh mát lạnh thật là tuyệt vời.
Tết đến, nhà nhà gói bánh chưng xanh, treo câu đối đỏ.
Trẻ em được mừng tuổi, mặc áo mới, vui chơi khắp xóm làng.
Hoa đào nở thắm miền Bắc, hoa mai vàng rực rỡ miền Nam.
Tôi yêu tiếng nước tôi từ khi mới ra đời, người ơi.
Tiếng Việt giàu và đẹp, mỗi câu ca dao là một bài học quý.
Sách là người bạn tốt, mở ra chân trời kiến thức bao la.
Đọc một cuốn sách hay như trò chuyện cùng người thông thái.
Hãy sống tử tế, biết ơn và luôn cố gắng mỗi ngày bạn nhé.
Thất bại là mẹ thành công, đừng nản lòng khi gặp khó khăn.
Cảm ơn bạn rất nhiều. Chúc bạn một ngày tốt lành và bình an.
Hẹn gặp lại bạn lần sau. Tạm biệt và giữ gìn sức khỏe nhé.
Buổi tối, cả nhà quây quần bên mâm cơm, kể nhau nghe chuyện ngày.
Ánh trăng rằm sáng vằng vặc soi rõ con đường nhỏ ven sông.
Người nông dân thức dậy từ sớm, ra đồng cày cấy chăm lo mùa màng.
Giọt mồ hôi rơi trên luống cày đổi lấy bát cơm thơm dẻo.
Thầy cô là người lái đò thầm lặng đưa học trò qua sông tri thức.
Tôn sư trọng đạo là truyền thống tốt đẹp của dân tộc ta.
Mỗi sáng tôi tập thể dục, ăn sáng rồi đi làm đúng giờ.
Cuối tuần tôi thường về thăm ông bà và giúp mẹ dọn nhà.
Hà Nội mùa thu có hương hoa sữa nồng nàn trên từng góc phố.
Biển xanh cát trắng nắng vàng, sóng vỗ rì rào bên bờ cát mịn.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="File .txt corpus tiếng Việt của bạn")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Lặp corpus N lần để có đủ dữ liệu train (corpus nhỏ)")
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Đã đọc corpus từ {args.input}: {len(text):,} ký tự")
    else:
        # Lặp lại corpus để có đủ token train (corpus viết tay khá nhỏ)
        text = (VIETNAMESE_CORPUS + "\n") * args.repeat
        print(f"Dùng corpus tiếng Việt có sẵn × {args.repeat} = {len(text):,} ký tự")

    # ─── Char-level tokenizer ───
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary: {vocab_size} ký tự (gồm cả ký tự tiếng Việt có dấu)")
    print(f"  Mẫu: {''.join(chars[:80])}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # ─── Encode + chia train/val ───
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    os.makedirs(args.data_dir, exist_ok=True)
    train_data.tofile(os.path.join(args.data_dir, "train.bin"))
    val_data.tofile(os.path.join(args.data_dir, "val.bin"))

    with open(os.path.join(args.data_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"\n✓ Đã lưu:")
    print(f"  train.bin: {len(train_data):,} tokens")
    print(f"  val.bin:   {len(val_data):,} tokens")
    print(f"  meta.pkl:  vocab {vocab_size}")
    print(f"\nTrain ngay:")
    print(f"  python train.py --n_layer 4 --n_head 4 --n_embd 192 \\")
    print(f"      --block_size 128 --batch_size 16 --max_iters 3000 \\")
    print(f"      --eval_interval 300 --eval_iters 50 --log_interval 50 \\")
    print(f"      --warmup_iters 100 --lr_decay_iters 3000 --learning_rate 3e-3")


if __name__ == "__main__":
    main()
