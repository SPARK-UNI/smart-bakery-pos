import os
import shutil

ROOT_DIR = "data"  # đổi nếu khác
SPLITS = ["train", "valid", "test"]  # chỉnh tùy bạn
SUFFIXES = ["_multiple", "_mutiple"]  # hỗ trợ cả lỗi chính tả
DRY_RUN = False  # True để xem trước thay đổi, False để thực sự di chuyển

def strip_suffix(name: str) -> str:
    """Bỏ hậu tố multiple/mutiple (không phân biệt hoa thường) và khoảng trắng dư."""
    low = name.lower()
    for sfx in SUFFIXES:
        if low.endswith(sfx):
            return name[: -len(sfx)].rstrip(" _")
    return name

def ensure_dir(path: str):
    if not os.path.exists(path):
        if not DRY_RUN:
            os.makedirs(path, exist_ok=True)

def move_file(src: str, dst: str):
    """Di chuyển file, nếu trùng tên thì tự động thêm _dup-1, _dup-2..."""
    base, ext = os.path.splitext(dst)
    candidate = dst
    i = 1
    while os.path.exists(candidate):
        candidate = f"{base}_dup-{i}{ext}"
        i += 1
    if DRY_RUN:
        print(f"[DRY] move: {src} -> {candidate}")
    else:
        ensure_dir(os.path.dirname(candidate))
        shutil.move(src, candidate)

def merge_dir(src_dir: str, dst_dir: str) -> int:
    """Di chuyển toàn bộ nội dung src_dir vào dst_dir, trả về số file đã di chuyển."""
    moved = 0
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        rel = "" if rel == "." else rel
        target_root = os.path.join(dst_dir, rel)
        for f in files:
            src_path = os.path.join(root, f)
            dst_path = os.path.join(target_root, f)
            move_file(src_path, dst_path)
            moved += 1
    # xóa thư mục rỗng sau khi di chuyển
    if not DRY_RUN:
        # walk từ sâu lên để xóa rỗng
        for root, dirs, files in os.walk(src_dir, topdown=False):
            if not os.listdir(root):
                os.rmdir(root)
    return moved

def process_split(split_path: str):
    if not os.path.isdir(split_path):
        return
    # chỉ xét các thư mục con trực tiếp
    entries = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    # map: base_name -> [list các thư mục (gồm cả bản gốc và bản *_multiple)]
    groups = {}
    for d in entries:
        base = strip_suffix(d)
        groups.setdefault(base, []).append(d)

    total_moved = 0
    for base, variants in groups.items():
        # các thư mục có hậu tố cần gộp
        multiples = [v for v in variants if v.lower().endswith(tuple(SUFFIXES))]
        if not multiples:
            continue  # không có gì cần gộp
        dst_dir = os.path.join(split_path, base)
        ensure_dir(dst_dir)
        for m in multiples:
            src_dir = os.path.join(split_path, m)
            print(f"--> Merging '{m}'  ==>  '{base}'  in  {os.path.basename(split_path)}")
            moved = merge_dir(src_dir, dst_dir)
            total_moved += moved
            if DRY_RUN:
                print(f"[DRY] would remove empty dir: {src_dir}")
        # nếu tồn tại bản gốc bị trống sau gộp thì vẫn giữ (đã là đích)
    print(f"[{os.path.basename(split_path)}] moved files: {total_moved}")

def main():
    root_abs = os.path.abspath(ROOT_DIR)
    print(f"Root: {root_abs} | DRY_RUN={DRY_RUN}")
    for s in SPLITS:
        process_split(os.path.join(ROOT_DIR, s))

if __name__ == "__main__":
    main()
