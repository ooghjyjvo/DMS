import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite
import os
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import subprocess
import signal

# 参数配置
MODEL_PATH = './yolov4-tiny2.rknn'
INPUT_SIZE = 416
CLASSES = ['mouth_opened', 'mouth_closed', 'eyes_open', 'eyes_closed', 'face']

# 分层锚框定义
ANCHORS_PER_LAYER = [
    [[81, 82], [135, 169], [344, 319]],  # 13x13 层 (对应输出0)
    [[23, 27], [37, 58], [81, 82]]  # 26x26 层 (对应输出1)
]
STRIDES = [INPUT_SIZE // 13, INPUT_SIZE // 26]  # 32和16

# 疲劳检测参数
EYE_CLOSED_THRESHOLD = 1.5  # 眼睛闭合超过这个秒数视为疲劳
MOUTH_OPEN_THRESHOLD = 2.0  # 嘴巴张开超过这个秒数视为疲劳
BLINK_THRESHOLD = 0.3  # 眨眼时长阈值（秒）
BLINK_RATE_THRESHOLD = 15  # 每分钟眨眼次数低于此值视为疲劳（正常眨眼率约为15-20次/分钟）

# 尝试多个可能的摄像头设备路径
CAMERA_DEVICES = [
    '/dev/video11'  # 您原始尝试的路径
]

# 警报音频文件路径
ALARM_FILE = '/home/elf/bj.mp3'


# 警报管理器
class AlarmManager:
    def __init__(self):
        self.alarm_process = None
        self.is_playing = False
        self.last_play_time = 0
        self.min_interval = 3  # 两次警报之间的最小间隔（秒）

    def play_alarm(self):
        """播放警报音频"""
        current_time = time.time()
        # 检查是否已经在播放或者距离上次播放时间太近
        if self.is_playing or (current_time - self.last_play_time) < self.min_interval:
            return

        # 启动新线程播放音频
        threading.Thread(target=self._play_alarm_thread, daemon=True).start()
        self.last_play_time = current_time

    def _play_alarm_thread(self):
        """在后台线程中播放警报"""
        self.is_playing = True
        try:
            # 使用gst-launch命令播放音频
            command = [
                'gst-launch-1.0',
                'filesrc', f'location={ALARM_FILE}',
                '!', 'id3demux',
                '!', 'mpegaudioparse',
                '!', 'mpg123audiodec',
                '!', 'alsasink', 'device=plughw:1,0'
            ]
            self.alarm_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 等待音频播放完成
            self.alarm_process.wait()
        except Exception as e:
            print(f"播放警报时出错: {str(e)}")
        finally:
            self.is_playing = False

    def stop_alarm(self):
        """停止正在播放的警报"""
        if self.is_playing and self.alarm_process:
            try:
                # 发送SIGTERM信号终止进程
                self.alarm_process.terminate()
                self.alarm_process.wait(timeout=2)
            except Exception as e:
                print(f"停止警报时出错: {str(e)}")
            self.is_playing = False


# 状态跟踪器
class FatigueDetector:
    def __init__(self, alarm_manager):
        # 眼睛状态
        self.eye_state = "open"  # "open" or "closed"
        self.eye_state_start = time.time()

        # 嘴巴状态
        self.mouth_state = "closed"  # "opened" or "closed"
        self.mouth_state_start = time.time()

        # 眨眼计数器
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0

        # 疲劳状态
        self.fatigue_warning = False
        self.fatigue_start_time = 0
        self.fatigue_duration = 0
        self.current_eye_closed_duration = 0.0
        self.current_mouth_open_duration = 0.0

        # 日志文件
        self.log_file = f"fatigue_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_message("Fatigue detection started")

        # 警报管理器
        self.alarm_manager = alarm_manager

    def log_message(self, message):
        """记录日志信息"""
        with open(self.log_file, "a") as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
        print(message)

    def update_eye_state(self, detected_state, timestamp):
        """更新眼睛状态"""
        if detected_state != self.eye_state:
            # 状态变化时记录眨眼
            if self.eye_state == "open" and detected_state == "closed":
                # 开始眨眼
                self.blink_count += 1
                blink_duration = timestamp - self.eye_state_start
                if blink_duration < BLINK_THRESHOLD:
                    # 计算眨眼频率 (每分钟)
                    time_since_last_blink = timestamp - self.last_blink_time
                    self.last_blink_time = timestamp
                    if time_since_last_blink > 0:
                        self.blink_rate = 60 / time_since_last_blink

            # 更新状态
            self.eye_state = detected_state
            self.eye_state_start = timestamp

    def update_mouth_state(self, detected_state, timestamp):
        """更新嘴巴状态"""
        if detected_state != self.mouth_state:
            # 更新状态
            self.mouth_state = detected_state
            self.mouth_state_start = timestamp

    def check_fatigue(self, timestamp):
        """检查疲劳状态"""
        # 计算当前状态的持续时间
        if self.eye_state == "closed":
            self.current_eye_closed_duration = timestamp - self.eye_state_start
        else:
            self.current_eye_closed_duration = 0.0

        if self.mouth_state == "opened":
            self.current_mouth_open_duration = timestamp - self.mouth_state_start
        else:
            self.current_mouth_open_duration = 0.0

        # 检查眼睛疲劳
        eye_fatigue = self.current_eye_closed_duration > EYE_CLOSED_THRESHOLD

        # 检查嘴巴疲劳（打哈欠）
        mouth_fatigue = self.current_mouth_open_duration > MOUTH_OPEN_THRESHOLD

        # 检查低眨眼率
        blink_fatigue = self.blink_rate > 0 and self.blink_rate < BLINK_RATE_THRESHOLD

        # 确定新的疲劳状态
        new_fatigue_state = eye_fatigue or mouth_fatigue or blink_fatigue

        # 更新疲劳持续时间
        if new_fatigue_state:
            if not self.fatigue_warning:
                self.fatigue_warning = True
                self.fatigue_start_time = timestamp
                self.log_message("Fatigue detected! Take a break.")
                # 检测到新的疲劳状态时播放警报
                self.alarm_manager.play_alarm()
            self.fatigue_duration = timestamp - self.fatigue_start_time
        else:
            if self.fatigue_warning:
                self.log_message(f"Fatigue ended after {self.fatigue_duration:.1f} seconds")
                self.fatigue_warning = False
                self.fatigue_duration = 0

        return self.fatigue_warning, eye_fatigue, mouth_fatigue, blink_fatigue


def sigmoid_np(x):
    """高效实现sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def decode_yolo_output(outputs, confidence_thresh=0.5, iou_thresh=0.4):
    """解码YOLO输出"""
    boxes, confidences, class_ids = [], [], []

    for i, output in enumerate(outputs):
        # 获取形状信息
        if output.ndim == 4:
            # 格式为 (batch, channels, height, width)
            batch, channels, height, width = output.shape
        else:
            # 格式为 (height, width, channels) 或类似
            continue

        # 检查通道数是否符合YOLOv4-tiny要求
        if channels != 30:
            continue

        grid_size = height  # 网格尺寸是 height 或 width (应该相同)

        anchors = ANCHORS_PER_LAYER[i]
        stride = STRIDES[i]

        # 重塑输出为 (batch, anchors, features, grid, grid)
        # 特征 = 4坐标 + 1对象置信度 + 类别数(5)
        output = output.reshape(batch, 3, 10, grid_size, grid_size)

        # 遍历每个网格单元
        for y in range(grid_size):
            for x in range(grid_size):
                for a in range(3):
                    # 获取预测值
                    tx = output[0, a, 0, y, x]
                    ty = output[0, a, 1, y, x]
                    tw = output[0, a, 2, y, x]
                    th = output[0, a, 3, y, x]
                    obj = output[0, a, 4, y, x]
                    class_probs = output[0, a, 5:, y, x]

                    # 应用sigmoid激活函数
                    tx_sig = sigmoid_np(tx)
                    ty_sig = sigmoid_np(ty)
                    obj_sig = sigmoid_np(obj)
                    class_probs_sig = sigmoid_np(class_probs)

                    # 计算边界框坐标
                    bx = (tx_sig + x) * stride
                    by = (ty_sig + y) * stride
                    bw = anchors[a][0] * np.exp(tw)
                    bh = anchors[a][1] * np.exp(th)

                    # 转换为边界框坐标
                    x1 = bx - bw / 2
                    y1 = by - bh / 2
                    x2 = bx + bw / 2
                    y2 = by + bh / 2

                    # 获取类别
                    class_id = np.argmax(class_probs_sig)
                    confidence = obj_sig * class_probs_sig[class_id]

                    # 过滤低置信度
                    if confidence > confidence_thresh:
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    # 应用NMS（非极大值抑制）
    if boxes:
        # 确保boxes是浮点数类型
        boxes = np.array(boxes, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)

        # 如果所有框都是0，返回空列表
        if np.all(boxes == 0):
            return []

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                   confidence_thresh, iou_thresh)

        # 处理NMS结果
        if len(indices) > 0:
            results = []
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            for idx in indices:
                # 确保索引在范围内
                if idx < len(boxes):
                    results.append({
                        'box': boxes[idx],
                        'confidence': confidences[idx],
                        'class_id': class_ids[idx],
                        'class_name': CLASSES[class_ids[idx]]
                    })
            return results

    return []


def process_frame(frame, rknn, detector, current_time):
    """处理单帧图像并更新疲劳状态"""
    orig_h, orig_w = frame.shape[:2]

    # 图像预处理
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_input = np.expand_dims(img_resized, 0).astype(np.uint8)

    # 执行推理
    outputs = rknn.inference(inputs=[img_input])

    # 解码输出结果
    detections = decode_yolo_output(outputs, confidence_thresh=0.3)  # 降低阈值

    # 状态变量
    eyes_state = None
    mouth_state = None

    # 处理检测结果
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det['box']

            # 缩放回原始图像尺寸
            x1 = max(0, min(int(x1 * orig_w / INPUT_SIZE), orig_w - 1))
            y1 = max(0, min(int(y1 * orig_h / INPUT_SIZE), orig_h - 1))
            x2 = max(0, min(int(x2 * orig_w / INPUT_SIZE), orig_w - 1))
            y2 = max(0, min(int(y2 * orig_h / INPUT_SIZE), orig_h - 1))

            # 确保边界框有效
            if x1 < x2 and y1 < y2:
                # 绘制边界框和标签
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 检测眼睛和嘴巴状态
        eye_detections = [d for d in detections if d['class_name'] in ['eyes_open', 'eyes_closed']]
        mouth_detections = [d for d in detections if d['class_name'] in ['mouth_opened', 'mouth_closed']]

        # 取置信度最高的眼睛状态
        if eye_detections:
            eye_detection = max(eye_detections, key=lambda x: x['confidence'])
            eyes_state = "open" if eye_detection['class_name'] == 'eyes_open' else "closed"

        # 取置信度最高的嘴巴状态
        if mouth_detections:
            mouth_detection = max(mouth_detections, key=lambda x: x['confidence'])
            mouth_state = "opened" if mouth_detection['class_name'] == 'mouth_opened' else "closed"

    # 更新状态检测器
    if eyes_state is not None:
        detector.update_eye_state(eyes_state, current_time)

    if mouth_state is not None:
        detector.update_mouth_state(mouth_state, current_time)

    # 检查疲劳状态
    fatigue, eye_fatigue, mouth_fatigue, blink_fatigue = detector.check_fatigue(current_time)

    return frame, fatigue, eye_fatigue, mouth_fatigue, blink_fatigue


def create_camera_capture():
    """尝试创建摄像头捕获对象"""
    # 首先尝试使用V4L2直接捕获
    for device in CAMERA_DEVICES:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Successfully opened camera using V4L2: {device}")
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap

    # 如果V4L2失败，尝试GStreamer
    for device in CAMERA_DEVICES:
        gst_pipeline = (
            f'v4l2src device={device} ! '
            'video/x-raw, format=NV12, width=640, height=480, framerate=30/1 ! '
            'videoconvert ! '
            'video/x-raw, format=BGR ! '
            'appsink drop=true sync=false'
        )
        print(f"Trying GStreamer pipeline: {gst_pipeline}")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"Successfully opened camera using GStreamer: {device}")
            return cap

    return None


class CameraApp:
    def __init__(self, root, rknn, detector):
        self.root = root
        self.root.title("智能疲劳检测系统")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # 设置主题颜色
        self.bg_color = "#f5f7fa"
        self.panel_color = "#ffffff"
        self.header_color = "#3498db"
        self.button_color = "#3498db"
        self.warning_color = "#e74c3c"
        self.text_color = "#2c3e50"
        self.button_text_color = "#e74c3c"  # 按钮文本颜色改为红色

        # 设置中文字体
        self.title_font = ('SimHei', 16, 'bold')
        self.header_font = ('SimHei', 12, 'bold')
        self.normal_font = ('SimHei', 10)

        # 界面状态变量
        self.is_camera_running = False
        self.warning_active = False

        # RKNN和疲劳检测器
        self.rknn = rknn
        self.detector = detector

        # 摄像头相关变量
        self.cap = None
        self.frame = None

        # 创建界面
        self.create_style()
        self.create_widgets()

        # 信息栏初始消息
        self.update_info("系统就绪，请点击'开启摄像头'按钮")

    def create_style(self):
        """创建自定义样式"""
        style = ttk.Style()

        # 配置整体主题
        self.root.tk_setPalette(background=self.bg_color, foreground=self.text_color)

        # 配置标题栏
        style.configure('Header.TFrame', background=self.header_color)
        style.configure('Header.TLabel', background=self.header_color,
                        foreground='white', font=self.title_font)

        # 配置面板 - 使用TFrame和TLabel代替自定义样式
        style.configure('TFrame', background=self.panel_color)
        style.configure('TLabel', background=self.panel_color,
                        foreground=self.text_color)

        # 配置标签框架
        style.configure('TLabelframe', background=self.bg_color,
                        foreground=self.text_color)
        style.configure('TLabelframe.Label', font=self.header_font)

        # 配置按钮 - 将字体颜色改为红色，并修复font参数重复问题
        style.configure('Primary.TButton',
                        background=self.button_color,
                        foreground=self.button_text_color,
                        font=('SimHei', 12, 'bold'),  # 合并font参数
                        padding=8,
                        width=15)

        style.configure('Danger.TButton',
                        background=self.warning_color,
                        foreground=self.button_text_color,
                        font=('SimHei', 12, 'bold'),  # 合并font参数
                        padding=8,
                        width=15)

        # 配置滚动条
        style.configure('Vertical.TScrollbar', gripcount=0,
                        background=self.panel_color, darkcolor=self.header_color,
                        lightcolor=self.header_color, troughcolor=self.bg_color,
                        bordercolor=self.panel_color, arrowcolor='black')

    def create_widgets(self):
        """创建美化后的GUI界面"""
        # 顶部标题栏
        header_frame = ttk.Frame(self.root, style='Header.TFrame')
        header_frame.pack(fill=tk.X, ipady=10)

        title_label = ttk.Label(header_frame, text="智能疲劳检测系统", style='Header.TLabel')
        title_label.pack(padx=20)

        # 主内容区域
        content_frame = ttk.Frame(self.root, padding="15")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：摄像头显示区域
        left_frame = ttk.LabelFrame(content_frame, text="实时监控")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # 创建摄像头显示标签
        self.camera_frame = ttk.Frame(left_frame)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # 右侧：控制面板
        right_frame = ttk.Frame(content_frame, padding="0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 按钮区域
        button_frame = ttk.LabelFrame(right_frame, text="系统控制")
        button_frame.pack(fill=tk.X, pady=(0, 15), ipady=5)

        button_container = ttk.Frame(button_frame)
        button_container.pack(fill=tk.X, padx=10, pady=5)

        self.start_button = ttk.Button(button_container, text="开启摄像头",
                                       command=self.start_camera, style='Primary.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(button_container, text="关闭摄像头",
                                      command=self.stop_camera, style='Danger.TButton',
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        # 疲劳警告区域
        self.warning_frame = ttk.LabelFrame(right_frame, text="疲劳状态")
        self.warning_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15), ipady=5)

        self.warning_container = ttk.Frame(self.warning_frame)
        self.warning_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.warning_label = ttk.Label(self.warning_container, text="",
                                       font=('SimHei', 36, 'bold'), foreground=self.warning_color)
        self.warning_label.pack(expand=True)

        # 信息栏
        self.info_frame = ttk.LabelFrame(right_frame, text="系统信息")
        self.info_frame.pack(fill=tk.BOTH, expand=True, ipady=5)

        self.info_container = ttk.Frame(self.info_frame)
        self.info_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.info_text = tk.Text(self.info_container, wrap=tk.WORD, height=10,
                                 state=tk.DISABLED, font=self.normal_font,
                                 bg=self.panel_color, fg=self.text_color,
                                 bd=0, highlightthickness=0)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.info_container, command=self.info_text.yview,
                                  style='Vertical.TScrollbar')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

        # 底部状态栏
        status_frame = ttk.Frame(self.root, padding="10", style='Header.TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        status_label = ttk.Label(status_frame, text="系统状态: 就绪",
                                 style='Header.TLabel', font=self.normal_font)
        status_label.pack(anchor=tk.W)

    def start_camera(self):
        """开启摄像头"""
        if not self.is_camera_running:
            try:
                # 打开摄像头
                self.cap = create_camera_capture()
                if self.cap is None:
                    self.update_info("无法打开摄像头，请检查设备连接")
                    return

                # 更新状态和按钮
                self.is_camera_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                # 修改关闭摄像头按钮的文字颜色为黑色
                style = ttk.Style()
                style.configure('Danger.TButton', foreground='black')

                # 开始更新摄像头画面
                self.update_frame()
                self.update_info("摄像头已开启")
            except Exception as e:
                self.update_info(f"开启摄像头时出错: {str(e)}")

    def stop_camera(self):
        """关闭摄像头"""
        if self.is_camera_running:
            # 更新状态和按钮
            self.is_camera_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            # 恢复关闭摄像头按钮的文字颜色为红色
            style = ttk.Style()
            style.configure('Danger.TButton', foreground=self.button_text_color)

            # 释放摄像头资源
            if self.cap:
                self.cap.release()

            # 清空摄像头显示区域
            self.camera_label.config(image="")

            # 清空警告区域
            self.warning_label.config(text="")

            self.update_info("摄像头已关闭")

    def update_frame(self):
        """更新摄像头画面"""
        if self.is_camera_running:
            # 读取一帧
            ret, frame = self.cap.read()
            if ret:
                current_time = time.time()
                # 处理帧
                processed_frame, fatigue, eye_fatigue, mouth_fatigue, blink_fatigue = process_frame(
                    frame, self.rknn, self.detector, current_time)

                # 转换颜色空间（BGR到RGB）
                self.frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # 调整图像大小以适应窗口
                '''max_width = self.camera_label.winfo_width()
                max_height = self.camera_label.winfo_height()
                if max_width > 0 and max_height > 0:
                    height, width = self.frame.shape[:2]
                    ratio = min(max_width / width, max_height / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    # 调整图像大小
                    resized_frame = cv2.resize(self.frame, (new_width, new_height))

                    # 转换为Tkinter可用的图像格式
                    image = Image.fromarray(resized_frame)
                    photo = ImageTk.PhotoImage(image=image)'''
                 # 不调整图像大小，保持初始尺寸
                image = Image.fromarray(self.frame)
                photo = ImageTk.PhotoImage(image=image)

                # 更新标签显示
                self.camera_label.config(image=photo)
                self.camera_label.image = photo

                # 更新警告区域
                if fatigue:
                    self.warning_label.config(text="WARNING", foreground="red")
                else:
                    self.warning_label.config(text="")

                # 更新信息栏
                info_message = f"Eye state: {self.detector.eye_state}, " \
                               f"Eye closed: {self.detector.current_eye_closed_duration:.1f}s, " \
                               f"Mouth state: {self.detector.mouth_state}, " \
                               f"Mouth open: {self.detector.current_mouth_open_duration:.1f}s, " \
                               f"Blink rate: {self.detector.blink_rate:.1f} blinks/min"
                if fatigue:
                    info_message += f", FATIGUE WARNING: {self.detector.fatigue_duration:.1f}s"
                self.update_info(info_message)

            # 继续更新
            self.root.after(30, self.update_frame)

    def update_info(self, message):
        """更新信息栏内容"""
        # 获取当前时间
        current_time = time.strftime("%H:%M:%S")

        # 在信息文本中添加新消息
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"[{current_time}] {message}\n")
        self.info_text.see(tk.END)  # 滚动到底部
        self.info_text.config(state=tk.DISABLED)


def main():
    print("RK3588 Fatigue Detection System with RKNN Lite")

    # 初始化警报管理器
    alarm_manager = AlarmManager()

    # 初始化疲劳检测器
    detector = FatigueDetector(alarm_manager)

    # 初始化RKNN Lite对象
    rknn = RKNNLite()

    # 加载RKNN模型
    print('--> Loading RKNN model')
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f'Load RKNN model failed! Error: {ret}')
        return

    print('Model loaded successfully')

    # 初始化运行时环境
    print('--> Initializing runtime')
    ret = rknn.init_runtime()
    if ret != 0:
        print(f'Init runtime failed! Error: {ret}')
        rknn.release()
        return

    print('Runtime initialized (RKNN Lite)')

    # 创建Tkinter窗口
    root = tk.Tk()
    app = CameraApp(root, rknn, detector)

    # 设置窗口关闭时的处理
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, rknn, detector, alarm_manager))

    root.mainloop()

    # 最终日志信息
    detector.log_message(f"Total run time: {time.time() - detector.eye_state_start:.1f} seconds")
    detector.log_message(f"Average FPS: {0:.1f}")

    print("Resources released")


def on_close(root, rknn, detector, alarm_manager):
    """处理窗口关闭事件"""
    # 停止所有警报
    alarm_manager.stop_alarm()

    # 释放RKNN资源
    rknn.release()

    # 关闭窗口
    root.destroy()


if __name__ == '__main__':
    main()