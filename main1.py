import sys
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSlider, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


class YOLOv5Detector(QThread):
    update_result = pyqtSignal(int, str, QImage)  # 检测数量, 结果文本, 检测图像

    def __init__(self, weights, source, conf_thres, iou_thres, imgsz=(640, 640), device=''):
        super().__init__()
        self.weights = weights
        self.source = source
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device
        self.running = True

        # 初始化模型
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def run(self):
        """执行实际检测"""
        try:
            # 处理摄像头/视频/图像
            if self.source.isnumeric() or self.source.endswith('.streams'):
                cap = cv2.VideoCapture(int(self.source) if self.source.isnumeric() else self.source)
            else:
                cap = cv2.VideoCapture(self.source)

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行检测
                detections, result_text = self.detect_objects(frame)

                # 绘制检测结果
                annotated_frame = self.draw_detections(frame, detections)

                # 转换为QImage
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # 发送信号更新UI
                self.update_result.emit(len(detections), result_text, qt_image)

                # 控制帧率
                QThread.msleep(30)

            cap.release()
        except Exception as e:
            self.update_result.emit(0, f"检测出错: {str(e)}", QImage())

    def detect_objects(self, img):
        """执行YOLOv5检测"""
        # 预处理
        img0 = img.copy()
        img = cv2.resize(img0, self.imgsz)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(img.copy()).to(self.device)
        img = img.float() / 255.0  # 归一化
        if len(img.shape) == 3:
            img = img[None]  # 扩展为批处理

        # 推理
        pred = self.model(img, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        # 处理检测结果
        detections = []
        result_text = "检测结果:\n"

        for i, det in enumerate(pred):
            if len(det):
                # 调整框大小到原始图像尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    class_name = self.names[int(cls)]
                    confidence = float(conf)
                    detections.append((xyxy, class_name, confidence))
                    result_text += f"- {class_name} (置信度: {confidence:.2f})\n"

        return detections, result_text

    def draw_detections(self, img, detections):
        """在图像上绘制检测框"""
        for xyxy, class_name, confidence in detections:
            # 绘制矩形框
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        return img

    def stop(self):
        self.running = False
        self.wait()


class YOLOv5GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 目标检测系统")
        self.setGeometry(100, 100, 1200, 900)

        # 默认值
        self.weights_path = "yolov5s.pt"
        self.source_path = ""
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.detector = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()  # 改为水平布局

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        # 模型权重选择
        weights_group = QGroupBox("模型配置")
        weights_layout = QHBoxLayout()

        weights_label = QLabel("模型权重:")
        weights_label.setStyleSheet("font-size: 16px;")
        self.weights_edit = QLineEdit(self.weights_path)
        self.weights_edit.setStyleSheet("font-size: 16px; padding: 5px;")
        browse_weights_btn = QPushButton("浏览...")
        browse_weights_btn.setStyleSheet("font-size: 16px; padding: 5px 10px;")
        browse_weights_btn.clicked.connect(self.browse_weights)

        weights_layout.addWidget(weights_label)
        weights_layout.addWidget(self.weights_edit)
        weights_layout.addWidget(browse_weights_btn)
        weights_group.setLayout(weights_layout)
        weights_group.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        # 输入源选择
        source_group = QGroupBox("输入源")
        source_layout = QHBoxLayout()

        source_label = QLabel("输入源:")
        source_label.setStyleSheet("font-size: 16px;")
        self.source_edit = QLineEdit()
        self.source_edit.setStyleSheet("font-size: 16px; padding: 5px;")
        browse_source_btn = QPushButton("浏览...")
        browse_source_btn.setStyleSheet("font-size: 16px; padding: 5px 10px;")
        browse_source_btn.clicked.connect(self.browse_source)

        self.source_type = QComboBox()
        self.source_type.setStyleSheet("font-size: 16px; padding: 5px;")
        self.source_type.addItems(["图片", "视频", "摄像头"])
        self.source_type.currentIndexChanged.connect(self.update_source_type)

        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(browse_source_btn)
        source_layout.addWidget(self.source_type)
        source_group.setLayout(source_layout)
        source_group.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        # 阈值控制
        threshold_group = QGroupBox("检测参数")
        threshold_layout = QVBoxLayout()

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        conf_label.setStyleSheet("font-size: 16px;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.conf_thres * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_spinbox)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(self.conf_thres)
        self.conf_spinbox.setStyleSheet("font-size: 16px; padding: 5px;")
        self.conf_spinbox.valueChanged.connect(self.update_conf_slider)

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spinbox)

        # IoU阈值
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IoU阈值:")
        iou_label.setStyleSheet("font-size: 16px;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(int(self.iou_thres * 100))
        self.iou_slider.valueChanged.connect(self.update_iou_spinbox)
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setValue(self.iou_thres)
        self.iou_spinbox.setStyleSheet("font-size: 16px; padding: 5px;")
        self.iou_spinbox.valueChanged.connect(self.update_iou_slider)

        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_spinbox)

        threshold_layout.addLayout(conf_layout)
        threshold_layout.addLayout(iou_layout)
        threshold_group.setLayout(threshold_layout)
        threshold_group.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        # 操作按钮
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("开始检测")
        self.run_btn.setStyleSheet("font-size: 16px; background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_detection)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setStyleSheet("font-size: 16px; background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.stop_btn)

        # 添加控制组件到左侧面板
        control_layout.addWidget(weights_group)
        control_layout.addWidget(source_group)
        control_layout.addWidget(threshold_group)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)

        # 右侧结果显示面板
        result_panel = QWidget()
        result_layout = QVBoxLayout()

        # 检测结果显示
        self.detection_count_label = QLabel("检测数量: 0")
        self.detection_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; padding: 10px;")

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; padding: 10px;")

        # 结果文本区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: Consolas; font-size: 16px; padding: 10px;")

        result_layout.addWidget(self.detection_count_label)
        result_layout.addWidget(self.image_label, 1)
        result_layout.addWidget(self.result_text)
        result_panel.setLayout(result_layout)

        # 添加左右面板到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(result_panel, 2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def browse_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "PyTorch 文件 (*.pt);;所有文件 (*)"
        )
        if file_path:
            self.weights_edit.setText(file_path)

    def browse_source(self):
        if self.source_type.currentText() == "图片":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*)"
            )
        else:  # 视频
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)"
            )

        if file_path:
            self.source_edit.setText(file_path)
            self.display_original_image(file_path)  # 显示原始图像

    def update_source_type(self):
        if self.source_type.currentText() == "摄像头":
            self.source_edit.setText("0")
            self.source_edit.setEnabled(False)
        else:
            self.source_edit.setEnabled(True)
            self.source_edit.clear()

    def update_conf_spinbox(self, value):
        self.conf_spinbox.setValue(value / 100)

    def update_conf_slider(self, value):
        self.conf_slider.setValue(int(value * 100))

    def update_iou_spinbox(self, value):
        self.iou_spinbox.setValue(value / 100)

    def update_iou_slider(self, value):
        self.iou_slider.setValue(int(value * 100))

    def run_detection(self):
        # 获取所有参数
        weights = self.weights_edit.text()
        source = self.source_edit.text()
        conf_thres = self.conf_spinbox.value()
        iou_thres = self.iou_spinbox.value()

        if not weights or not source:
            self.result_text.setPlainText("错误: 请先选择模型权重和输入源！")
            return

        # 清空之前的结果
        self.detection_count_label.setText("检测数量: 检测中...")
        self.result_text.setPlainText("正在初始化检测器...")

        # 禁用开始按钮，启用停止按钮
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 停止之前的检测线程
        if self.detector and self.detector.isRunning():
            self.detector.stop()

        # 创建并启动检测线程
        self.detector = YOLOv5Detector(weights, source, conf_thres, iou_thres)
        self.detector.update_result.connect(self.update_results)
        self.detector.start()

    def stop_detection(self):
        if self.detector and self.detector.isRunning():
            self.detector.stop()
            self.result_text.append("\n检测已停止")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def update_results(self, count, result_text, qt_image):
        # 更新检测数量显示
        self.detection_count_label.setText(f"检测数量: {count}")

        # 更新结果文本
        self.result_text.setPlainText(result_text)

        # 更新图像显示
        if not qt_image.isNull():
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 重新启用开始按钮
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def display_original_image(self, file_path):
        """显示原始图像"""
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # 窗口大小改变时调整图像大小
        if hasattr(self, 'image_label') and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

    def closeEvent(self, event):
        # 窗口关闭时停止检测线程
        if self.detector and self.detector.isRunning():
            self.detector.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv5GUI()
    window.show()
    sys.exit(app.exec_())