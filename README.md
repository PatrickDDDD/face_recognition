太好了 🎉
下面是**更新后的 README.md（完整可直接替换版）**，我已经把你这一路踩过的关键点都整理进去了：**Windows / Python 3.11 / 中文名 / Iriun 摄像头 / InsightFace / VS Code**，非常适合后续给同事或领导演示。

---

# Face Recognition Demo（本地人脸识别 Demo）

一个 **基于 InsightFace 的本地人脸识别示例系统**，支持：

* 👤 **员工注册（每人 1 张照片）**
* 📷 **摄像头实时识别（支持 Iriun 手机摄像头）**
* 🟢 **已注册用户：绿色框 + 编号 + 中文姓名**
* 🔴 **未注册用户：红色框 + UNKNOWN**
* 🪟 **Windows 本地运行（CPU 即可）**

---

## 一、项目结构

```text
face_recognition/
├── employees/              # 员工注册照片（每人 1 张）
│   ├── 001_张三.jpg
│   ├── 002_李四.jpg
│   └── ...
├── gallery/
│   └── gallery.npz         # 人脸向量库（自动生成）
├── scripts/
│   ├── build_gallery.py    # 从 employees 建立人脸向量库
│   └── recognize_webcam.py # 摄像头实时识别
├── tests/                  # 预留
├── requirements.txt
├── README.md
└── .venv/                  # Python 虚拟环境（Python 3.11）
```

---

## 二、环境要求

* Windows 10 / 11
* Python **3.11.x**（强烈推荐）
* 摄像头（支持：

  * 内置摄像头
  * **Iriun Webcam（手机当摄像头）**）
* CPU 即可（无需 GPU）

---

## 三、环境准备（Windows）

### 1️⃣ 创建并激活虚拟环境（Python 3.11）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version   # 确认是 Python 3.11
```

---

### 2️⃣ 安装依赖

`requirements.txt` 内容如下：

```txt
insightface==0.7.3
onnxruntime==1.17.3
opencv-python==4.10.0.84
numpy==1.26.4
tqdm==4.66.4
pillow
```

安装：

```powershell
pip install -r requirements.txt
```

> ⚠️ **Windows 下首次安装 insightface 需要 C++ 编译环境**
> 请提前安装 **Build Tools for Visual Studio**，并勾选：
>
> * ✅ Desktop development with C++
> * ✅ MSVC v143
> * ✅ Windows 10/11 SDK
>
> 安装后，建议在 **x64 Native Tools Command Prompt for VS** 中执行 `pip install`.

---

## 四、员工注册（建库）

### 1️⃣ 准备员工照片

将员工照片放入 `employees/` 目录：

```text
employees/
├── 001_张三.jpg
├── 002_李四.jpg
├── 003_王五.jpg
```

要求：

* 每人 **1 张正脸照**
* 文件名格式：

  ```text
  员工编号_中文姓名.jpg
  ```
* 支持 **中文文件名**

---

### 2️⃣ 构建人脸向量库

```powershell
python scripts/build_gallery.py
```

首次运行会自动下载 InsightFace 官方模型（buffalo_l）。

成功示例输出：

```text
✅ 建库完成
输出文件：gallery\gallery.npz
成功人数：7
```

---

## 五、摄像头实时识别

### 1️⃣ 运行识别程序

```powershell
python scripts/recognize_webcam.py
```

程序行为：

* 🟢 已注册人员：**绿色框 + 编号 + 中文姓名 + 相似度**
* 🔴 未注册人员：**红色框 + UNKNOWN**
* 按 **Q** 键退出

---

### 2️⃣ 使用 Iriun Webcam（可选）

1. 手机安装 **Iriun Webcam**
2. 手机与电脑在同一网络
3. 打开 Iriun（手机端 + 电脑端）
4. 启动 `recognize_webcam.py`

如识别到的不是手机摄像头，可在 `recognize_webcam.py` 中指定：

```python
FORCE_CAM_INDEX = 1   # 或 2 / 3
```

---

## 六、关键实现说明

### ✅ 人脸算法

* InsightFace `buffalo_l`
* ArcFace（512 维 embedding）
* 余弦相似度匹配

### ✅ 中文姓名显示

* **OpenCV 原生 `cv2.putText` 不支持中文**
* 本项目使用 **Pillow + Windows 中文字体** 绘制文字
* 解决 `?????` / 方块问题

### ✅ 中文路径支持

* 使用 `np.fromfile + cv2.imdecode`
* 避免 `cv2.imread` 在 Windows 下读取中文路径失败

---

## 七、常见问题（FAQ）

### Q1：为什么只能识别编号，名字是 `?????`

A：OpenCV 不支持中文渲染，已通过 Pillow 解决（当前版本已修复）。

---

### Q2：建库时 `read_failed`

A：通常是：

* 图片路径包含中文
* 使用了 `cv2.imread`

本项目已修复。

---

### Q3：会影响其他 Python 项目吗？

A：不会。
本项目使用 **独立 `.venv` 虚拟环境**，与系统 Python 和其他项目完全隔离。

---

## 八、后续可扩展方向

* 🚪 接入闸机 / 门禁 IO
* ⛑️ 安全帽检测（YOLO + 人脸框联动）
* 📊 人员通行日志
* 🧠 多照片注册 / 多特征融合
* 🚀 GPU / 边缘设备部署

---


