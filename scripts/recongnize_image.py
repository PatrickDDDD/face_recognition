import os
import sys
import numpy as np
import cv2
from insightface.app import FaceAnalysis

GALLERY_FILE = os.path.join("gallery", "gallery.npz")
THRESH = 0.45

def main():
    if len(sys.argv) < 2:
        print("用法：python scripts/recognize_image.py tests/test1.jpg")
        return

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        raise RuntimeError(f"找不到图片：{img_path}")

    if not os.path.isfile(GALLERY_FILE):
        raise RuntimeError(f"找不到向量库：{GALLERY_FILE}。请先运行：python scripts/build_gallery.py")

    data = np.load(GALLERY_FILE, allow_pickle=True)
    ids = data["ids"]
    names = data["names"]
    gallery = data["embs"].astype(np.float32)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) == 0:
        print("NO_FACE")
        return

    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        sims = gallery @ emb
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= THRESH:
            print(f"MATCH: {ids[best_idx]} {names[best_idx]}  score={best_score:.3f}")
        else:
            print(f"UNKNOWN: score={best_score:.3f}")

if __name__ == "__main__":
    main()
