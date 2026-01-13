import os
import re
import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def imread_unicode(path: str):
    # 解决 Windows 下中文路径 cv2.imread 读不到的问题
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ====== 配置 ======
EMP_DIR = os.path.join("employees")
OUT_DIR = os.path.join("gallery")
OUT_FILE = os.path.join(OUT_DIR, "gallery.npz")

# 文件名解析：10001_张三.jpg 或 10001-张三.png
NAME_RE = re.compile(r"^(?P<id>\d+)[_\-](?P<name>.+)$")

def parse_employee(filepath: str):
    base = os.path.splitext(os.path.basename(filepath))[0]
    m = NAME_RE.match(base)
    if not m:
        # 不符合格式就兜底：id=name=base
        return base, base
    return m.group("id"), m.group("name")

def largest_face(faces):
    # 取最大脸，更稳
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )[0]

def main():
    if not os.path.isdir(EMP_DIR):
        raise RuntimeError(f"找不到员工目录：{EMP_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    img_files = []
    for fn in os.listdir(EMP_DIR):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            img_files.append(os.path.join(EMP_DIR, fn))

    if not img_files:
        raise RuntimeError(f"{EMP_DIR} 里没有图片。请放入员工注册照（每人1张）。")

    # InsightFace 初始化（CPU）
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    ids, names, embs = [], [], []
    skipped = []

    for path in tqdm(img_files, desc="Building gallery"):
        img = imread_unicode(path)
        if img is None:
            skipped.append((path, "read_failed"))
            continue

        faces = app.get(img)
        if len(faces) == 0:
            skipped.append((path, "no_face"))
            continue

        face = largest_face(faces)

        emb = face.normed_embedding.astype(np.float32)  # 512维，已归一化
        emp_id, emp_name = parse_employee(path)

        ids.append(emp_id)
        names.append(emp_name)
        embs.append(emb)

    if not embs:
        raise RuntimeError("建库失败：没有任何图片检测到人脸。请换更清晰的正脸照片。")

    ids = np.array(ids, dtype=object)
    names = np.array(names, dtype=object)
    embs = np.stack(embs, axis=0)  # [N,512]

    np.savez_compressed(OUT_FILE, ids=ids, names=names, embs=embs)

    print("\n✅ 建库完成")
    print(f"  输出文件：{OUT_FILE}")
    print(f"  成功人数：{len(ids)}")
    if skipped:
        print(f"⚠️ 跳过图片：{len(skipped)}")
        for p, reason in skipped[:10]:
            print(f"  - {reason}: {p}")
        if len(skipped) > 10:
            print("  ...（仅显示前10条）")

if __name__ == "__main__":
    main()
