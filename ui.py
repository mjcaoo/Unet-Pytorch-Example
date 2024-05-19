import os
import shutil

import torch
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QTabWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QTextBrowser,
)


def cv2_to_pixmap(cv2_img):
    if cv2_img.ndim == 2:
        height, width = cv2_img.shape
        bytesPerLine = width
        qImg = QImage(
            cv2_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8
        )
        return QPixmap.fromImage(qImg)
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


class UnetDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()

        return model

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)

        return self.model(img_tensor)

    def postprocess(self, pred):
        res = np.array(pred.data.cpu()[0])[0]
        res[res >= 0.98] = 255
        res[res < 0.98] = 0

        return res

    def __call__(self, image):
        pred = self.predict(image)
        res = self.postprocess(pred)

        return res


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.Unet = UnetDetector("runs/train/exp2/checkpoints/best.pt")
        self.cap = None
        self.camera_timer = QTimer()
        self.initMainWindow()
        self.tabNames = ["项目介绍", "数据集", "图像检测", "实时监测", "作者信息"]
        self.train_images = [
            os.path.join("data/ISIC2018-demo/images", i)
            for i in os.listdir("data/ISIC2018-demo/images")
        ]
        self.train_masks = [
            i.replace("images", "masks").replace(".jpg", "_segmentation.png")
            for i in self.train_images
        ]
        self.images_count = len(self.train_images)
        self.images_index = 0

        self.tabs = [QWidget() for i in range(len(self.tabNames))]
        self.initTabs()

        self.tab1UI()  # tab1：项目介绍
        self.tab2UI()  # tab2：数据集
        self.tab3UI()  # tab3：图像检测
        self.tab4UI()  # tab4：实时监测
        self.tab5UI()  # tab5：作者信息

    def initMainWindow(self):
        self.setWindowTitle("《机器学习》课程设计")
        self.setWindowIcon(QIcon("data/ui/favicon.png"))
        self.setStyleSheet('font: 14pt "微软雅黑";')
        self.setGeometry(400, 300, 1200, 800)
        self.setFixedSize(1200, 800)

    def initTabs(self):
        icons = [
            "data/ui/home.png",
            "data/ui/data.png",
            "data/ui/detect-c.png",
            "data/ui/camera-icon.png",
            "data/ui/profile-fill.png",
        ]
        for i in range(len(self.tabs)):
            self.addTab(self.tabs[i], self.tabNames[i])
            self.setTabIcon(i, QIcon(icons[i]))
            self.setTabToolTip(i, self.tabNames[i])

    def tab1UI(self):
        # tab1：项目介绍
        layout = QVBoxLayout()
        project_desc = QTextBrowser()
        html_content = """
        <h1>皮肤病变区域分割任务</h1>
        <p>ISIC 2018 数据集在 UNet 模型上的应用</p>
        <h2>摘要</h2>
        <p>皮肤癌是全球最常见的癌症类型之一，其早期检测对于治疗成功至关重要。随着深度学习技术的发展，计算机辅助诊断（CAD）系统在皮肤病变分割和识别方面展现出巨大潜力。本项目综述旨在介绍如何利用 ISIC 2018 数据集训练 UNet 模型，以实现对皮肤病变区域的高精度分割。</p>
        <h2>ISIC2018数据集</h2>
        <p>ISIC 2018 数据集提供了丰富的、经过专家标注的图像，适用于皮肤病变分割任务。数据集的多样性和质量使其成为训练和测试深度学习模型的理想选择。</p>
        <ul>
            <li>大量图像：超过 3000 张高分辨率皮肤病变图像。</li>
            <li>详细的标注：每张图像都有精确的像素级病变区域标注。</li>
            <li>多样性：包含多种类型的皮肤病变，包括良性和恶性病变。</li>
        </ul>
        <h2>UNet模型</h2>
        <p>UNet 是一种用于图像分割的卷积神经网络结构，由编码器和解码器组成。编码器用于提取图像特征，解码器用于生成分割结果。UNet 模型在医学图像分割任务中表现优异，已被广泛应用于皮肤病变分割、肺部结节检测等任务。</p>
        <h2>实验结果</h2>
        <p>待补充...</p>
        """
        project_desc.append(html_content)
        layout.addWidget(project_desc)
        self.tabs[0].setLayout(layout)

    def tab2UI(self):
        # tab2：数据集
        layout = QVBoxLayout()
        images = QHBoxLayout()
        imageLeft = QLabel()
        imageLeft.resize(480, 480)
        imageLeft.setPixmap(QPixmap(self.train_images[self.images_index]))
        imageLeft.setAlignment(Qt.AlignCenter)  # type: ignore
        imageLeft.setScaledContents(True)
        self.datasetImageLeft = imageLeft
        imageRight = QLabel()
        imageRight.resize(480, 480)
        imageRight.setPixmap(QPixmap(self.train_masks[self.images_index]))
        imageRight.setAlignment(Qt.AlignCenter)  # type: ignore
        imageRight.setScaledContents(True)
        self.datasetImageRight = imageRight
        images.addWidget(imageLeft)
        images.addWidget(imageRight)

        buttons = QHBoxLayout()
        button1 = QPushButton("上一张")
        button2 = QPushButton("下一张")
        button1.clicked.connect(self.toggle_previous_image)
        button2.clicked.connect(self.toggle_next_image)
        buttons.addSpacing(400)
        buttons.addWidget(button1)
        buttons.addSpacing(100)
        buttons.addWidget(button2)
        buttons.addSpacing(400)

        html_content = """
        <h2>ISIC2018数据集</h2>
        <p>ISIC 2018 数据集提供了丰富的、经过专家标注的图像，适用于皮肤病变分割任务。数据集的多样性和质量使其成为训练和测试深度学习模型的理想选择。</p>
        <ul>
            <li>大量图像：超过 3000 张高分辨率皮肤病变图像。</li>
            <li>详细的标注：每张图像都有精确的像素级病变区域标注。</li>
            <li>多样性：包含多种类型的皮肤病变，包括良性和恶性病变。</li>
        </ul>
        """
        datasetInfo = QLabel(html_content)

        layout.addWidget(datasetInfo)
        layout.addSpacing(20)
        layout.addLayout(images)
        layout.addLayout(buttons)
        layout.setAlignment(Qt.AlignCenter)  # type: ignore
        self.tabs[1].setLayout(layout)

    def tab3UI(self):
        # tab3：图像检测
        layout = QVBoxLayout()
        images = QHBoxLayout()
        imageLeft = QLabel()
        imageLeft.resize(480, 480)
        imageLeft.setPixmap(QPixmap("data/ui/upload.png").scaled(480, 480))
        imageLeft.setAlignment(Qt.AlignCenter)  # type: ignore
        imageLeft.setScaledContents(True)
        self.imageUpload = imageLeft
        imageRight = QLabel()
        imageRight.resize(480, 480)
        imageRight.setPixmap(QPixmap("data/ui/result.png").scaled(480, 480))
        imageRight.setAlignment(Qt.AlignCenter)  # type: ignore
        imageRight.setScaledContents(True)
        self.imagePred = imageRight
        images.addWidget(imageLeft)
        images.addSpacing(10)
        images.addWidget(imageRight)

        buttons = QHBoxLayout()
        button1 = QPushButton("上传图像")
        button2 = QPushButton("开始检测")
        button1.clicked.connect(self.upload_image)
        button2.clicked.connect(self.check_image_from_file)
        buttons.addSpacing(80)
        buttons.addWidget(button1)
        buttons.addSpacing(170)
        buttons.addWidget(button2)
        buttons.addSpacing(80)

        layout.addLayout(images)
        layout.addSpacing(40)
        layout.addLayout(buttons)
        layout.setAlignment(Qt.AlignCenter)  # type: ignore
        self.tabs[2].setLayout(layout)

    def tab4UI(self):
        # tab4：实时监测
        layout = QVBoxLayout()
        images = QHBoxLayout()
        imageLeft = QLabel()
        imageLeft.resize(480, 480)
        imageLeft.setPixmap(QPixmap("data/ui/camera.png").scaled(480, 480))
        imageLeft.setAlignment(Qt.AlignCenter)  # type: ignore
        imageLeft.setScaledContents(True)
        self.cameraInput = imageLeft
        imageRight = QLabel()
        imageRight.resize(480, 480)
        imageRight.setPixmap(QPixmap("data/ui/result.png").scaled(480, 480))
        imageRight.setAlignment(Qt.AlignCenter)  # type: ignore
        imageRight.setScaledContents(True)
        self.cameraOut = imageRight
        images.addWidget(imageLeft)
        images.addSpacing(10)
        images.addWidget(imageRight)

        buttons = QHBoxLayout()
        button1 = QPushButton("开启摄像头")
        button2 = QPushButton("关闭摄像头")
        button1.clicked.connect(self.open_camera)
        button2.clicked.connect(self.close_camera)
        buttons.addSpacing(80)
        buttons.addWidget(button1)
        buttons.addSpacing(170)
        buttons.addWidget(button2)
        buttons.addSpacing(80)

        layout.addLayout(images)
        layout.addSpacing(40)
        layout.addLayout(buttons)
        layout.setAlignment(Qt.AlignCenter)  # type: ignore
        self.tabs[3].setLayout(layout)

    def tab5UI(self):
        # tab5：作者信息
        layout = QVBoxLayout()
        project_desc = QTextBrowser()
        html_content = """
        <br/><br/><br/>
        <div style='margin-left: 450;'>
            <img src='data/ui/author.jpg' width='200'/>
            <p>学生：曹明杰</p>
            <p>指导老师：王坤</p>
            <p>班级：计算机（拔尖）221</p>
            <p>学号：3191001335</p>
        </div>
        """
        project_desc.append(html_content)
        layout.addWidget(project_desc)
        self.tabs[4].setLayout(layout)

    def toggle_previous_image(self):
        self.images_index = (
            self.images_index - 1 if self.images_index > 0 else self.images_count - 1
        )
        self.datasetImageLeft.setPixmap(QPixmap(self.train_images[self.images_index]))
        self.datasetImageRight.setPixmap(QPixmap(self.train_masks[self.images_index]))

    def toggle_next_image(self):
        self.images_index = (
            self.images_index + 1 if self.images_index < self.images_count - 1 else 0
        )
        self.datasetImageLeft.setPixmap(QPixmap(self.train_images[self.images_index]))
        self.datasetImageRight.setPixmap(QPixmap(self.train_masks[self.images_index]))

    def upload_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open File", "data/ui", "Image Files (*.jpg *.png)"
        )
        if filename:
            suffix = filename.split(".")[-1]
            temp_path = os.path.join("data/ui", f"temp_upload.{suffix}")
            try:
                shutil.copy(filename, temp_path)
            except shutil.SameFileError:
                pass
            self.imageUpload.setPixmap(QPixmap(temp_path).scaled(480, 480))

    def check_image_from_file(self):
        filename = "data/ui/temp_upload.jpg"
        assert os.path.exists(filename), "Please upload an image first."
        image = cv2.imread(filename)
        res = self.Unet(image)
        cv2.imwrite("data/ui/temp_pred.jpg", res)
        self.imagePred.setPixmap(QPixmap("data/ui/temp_pred.jpg").scaled(480, 480))

    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.camera_timer.start(1000 // 24)
            self.camera_timer.timeout.connect(self.check_image_from_camera)

    def check_image_from_camera(self):
        ret, frame = self.cap.read()  # type: ignore
        if not ret:
            assert False, "Failed to open camera."
        res = self.Unet(frame)
        cv2.imwrite("data/ui/temp_camera.jpg", res)
        self.cameraInput.setPixmap(cv2_to_pixmap(frame).scaled(480, 480))
        self.cameraOut.setPixmap(cv2_to_pixmap(res).scaled(480, 480))

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.camera_timer.stop()
        self.cap = None
        self.cameraInput.setPixmap(QPixmap("data/ui/camera.png").scaled(480, 480))
        self.cameraOut.setPixmap(QPixmap("data/ui/result.png").scaled(480, 480))


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
