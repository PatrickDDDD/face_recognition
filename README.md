

# Face Recognition Demo (Local, InsightFace)

一个 **本地运行的人脸识别 Demo**，基于 **InsightFace（ArcFace）** 实现：

* 员工 **单张注册照片建库**
* 摄像头实时识别
* 已注册人员：显示 **编号 + 中文姓名（绿色框）**
* 未注册人员：显示 **UNKNOWN（红色框）**
* **完全本地运行**，不依赖云端服务

> 当前版本为 Demo / 原型阶段，适用于闸机、人脸核验、工业/办公场景 PoC。

---

## ✨ 功能特性

* ✅ 单人单照片注册（无需多张训练）
* ✅ 实时摄像头识别（支持 Iriun Webcam）
* ✅ 中文姓名显示（Pillow 渲染，解决 OpenCV 中文乱码）
* ✅ 未知人员识别（阈值控制）
* ✅ 员工照片 & 人脸向量 **不上传 GitHub**
* ✅ Windows + Python 3.11 友好

---

## 📁 项目结构

```text
face_recognition/
├─ employees/            # 员工注册照片（不进 Git）
│  └─ .gitkeep
├─ gallery/              # 人脸向量库（不进 Git）
│  └─ .gitkeep
├─ scripts/
│  ├─ build_gallery.py   # 构建人脸向量库
│  └─ webcam_recognize.py# 摄像头实时识别
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## 🧑‍💼 员工注册照片规范

* 每人 **一张正脸照片**
* 命名规则：

```text
员工编号_姓名.jpg
```

示例：

```text
001_张三.jpg
002_李四.jpg
003_王嘉尔.jpg
```

📌 建议：

* 正脸、无遮挡
* 清晰、光线均匀
* 分辨率 ≥ 256×256

---

## 🧠 技术方案说明

### 模型与框架

* **InsightFace**

  * 人脸检测：SCRFD
  * 人脸识别：ArcFace（512 维 embedding）
* 模型包：`buffalo_l`
* 推理方式：CPU（ONNX Runtime）

### 识别流程

1. 员工照片 → 提取 embedding → 保存为 `gallery.npz`
2. 摄像头帧 → 检测人脸 → 提取 embedding
3. 与 gallery 向量做 **余弦相似度**
4. 高于阈值 → 识别成功；否则 UNKNOWN

---

## 🚀 快速开始（Windows）

### 1️⃣ 创建虚拟环境（Python 3.11）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

---

### 2️⃣ 安装依赖

```powershell
pip install -r requirements.txt
```

> 如果是首次安装 InsightFace，系统可能需要
> **Visual Studio Build Tools（C++ x64）**

---

### 3️⃣ 放入员工照片

```text
employees/
├─ 001_张三.jpg
├─ 002_李四.jpg
└─ 003_王嘉尔.jpg
```

---

### 4️⃣ 构建人脸库

```powershell
python scripts/build_gallery.py
```

成功后输出：

```text
gallery/gallery.npz
```

---

### 5️⃣ 摄像头实时识别

```powershell
python scripts/webcam_recognize.py
```

* 绿色框：识别成功（编号 + 姓名）
* 红色框：UNKNOWN
* 按 `Q` 退出

📱 **Iriun Webcam**：

* 启动 Iriun App
* 若未识别到，可在脚本中指定摄像头索引

---

## 🔒 隐私与数据安全

* 员工照片（`employees/`）**不会提交到 GitHub**
* 人脸向量库（`gallery/*.npz`）**不会提交到 GitHub**
* `.gitignore` 已默认忽略以上内容
* Demo 仅用于本地测试 / PoC

---

## 📌 当前限制

* 单人单照片（未做多样本融合）
* 仅支持 CPU 推理
* 未接入闸机 / 门禁系统
* 未做活体检测（Liveness）

---

## 🛣️ 后续可扩展方向

* 🔐 闸机 / 门禁 / PLC 联动
* 🎥 活体检测（防照片攻击）
* 🧑‍🤝‍🧑 多人同时识别优化
* 📊 识别日志 & 出入记录
* 🖥️ Web / API 服务化
* 🧢 PPE 检测（安全帽、口罩）

---

## 🧑‍💻 作者

Patrick Ding
AI / Computer Vision / Industrial AI

---

## 📄 License

This project is provided for **demo and research purposes only**.

