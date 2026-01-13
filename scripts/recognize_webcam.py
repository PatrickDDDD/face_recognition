import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

# ====== 配置 ======
GALLERY_FILE = os.path.join("gallery", "gallery.npz")

# 经验阈值：越高越安全（更不容易认错人），但可能更容易 UNKNOWN
THRESH = 0.45

# 摄像头索引：None = 自动扫描；手动指定例如 2（常见 Iriun 在 1/2/3）
FORCE_CAM_INDEX = None

# 显示设置
WINDOW_NAME = "Face Recognition Demo (Green=Match, Red=Unknown)"


def try_open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows 更稳
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def find_working_cameras(max_index=12):
    working = []
    for i in range(max_index):
        cap = try_open_camera(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            working.append(i)
        cap.release()
    return working


def load_gallery():
    if not os.path.isfile(GALLERY_FILE):
        raise RuntimeError(f"找不到向量库：{GALLERY_FILE}。请先运行：python scripts/build_gallery.py")

    data = np.load(GALLERY_FILE, allow_pickle=True)
    ids = data["ids"]
    names = data["names"]
    embs = data["embs"].astype(np.float32)  # [N,512]
    return ids, names, embs


def get_windows_chinese_font():
    """尽量选择可用的中文字体（Windows）。"""
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    ]
    for p in font_candidates:
        if os.path.exists(p):
            return p
    return None


def put_text_pil(frame_bgr, text, x, y, color_bgr=(255, 255, 255), font_size=24):
    """
    在 OpenCV BGR 图像上绘制文本（支持中文）：
    - 用 Pillow 绘制，再转回 OpenCV。
    - color_bgr: OpenCV 的 BGR 颜色
    """
    font_path = get_windows_chinese_font()
    if font_path is None:
        # 没有中文字体则退化为 cv2.putText（中文可能显示为???）
        cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        return frame_bgr

    # Pillow 使用 RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Pillow 颜色是 RGB
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    font = ImageFont.truetype(font_path, font_size)

    draw.text((int(x), int(y)), str(text), font=font, fill=color_rgb)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def main():
    ids, names, gallery = load_gallery()

    # 初始化 InsightFace（CPU）
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # 选择摄像头
    if FORCE_CAM_INDEX is not None:
        cam_index = int(FORCE_CAM_INDEX)
        cap = try_open_camera(cam_index)
        if cap is None:
            raise RuntimeError(f"无法打开摄像头 index={cam_index}。请改 FORCE_CAM_INDEX 或设为 None 自动扫描。")
        print(f"✅ 使用手动指定摄像头 index={cam_index}")
    else:
        cams = find_working_cameras()
        if not cams:
            raise RuntimeError("未找到可用摄像头。请确认 Iriun Webcam 已启动且未被其他软件占用。")
        print("可用摄像头索引：", cams)
        # 默认策略：优先选非0（很多机器0是内置摄像头，Iriun常是1/2/3）
        cam_index = cams[0] if len(cams) == 1 else cams[1]
        cap = try_open_camera(cam_index)
        if cap is None:
            raise RuntimeError(f"无法打开摄像头 index={cam_index}")
        print(f"✅ 默认使用摄像头 index={cam_index}（如不是Iriun，把 FORCE_CAM_INDEX 改成正确索引）")

    print("按 Q 退出。")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        faces = app.get(frame)

        for f in faces:
            emb = f.normed_embedding.astype(np.float32)  # [512]
            sims = gallery @ emb  # [N] 余弦相似度（向量已归一化）
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            x1, y1, x2, y2 = f.bbox.astype(int)

            if best_score >= THRESH:
                # 识别成功：绿色框
                color = (0, 255, 0)
                # 显示：编号 + 中文名 + 分数
                label = f"{ids[best_idx]} {names[best_idx]}  {best_score:.2f}"
            else:
                # 未知：红色框
                color = (0, 0, 255)
                label = f"UNKNOWN  {best_score:.2f}"

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 画字（支持中文）
            # 字的位置：框上方；避免出界
            tx = int(x1)
            ty = int(max(0, y1 - 28))
            frame = put_text_pil(frame, label, tx, ty, color_bgr=color, font_size=24)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
