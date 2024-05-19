import glob
import numpy as np
import torch
import os
import cv2
import time
import matplotlib.pyplot as plt

from utils.utils import Saver, attempt_mkdir, dice_coefficient, intersection_over_union


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    # 保存器
    saver = Saver("runs/predict")
    # 加载网络
    net = torch.load("runs/train/exp2/checkpoints/best.pt")
    net.to(device=device)
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob("../datasets/ISIC2018/train/images/*.jpg")
    # 遍历素有图片
    for test_path in tests_path[:10]:
        # 读取图片
        t0 = time.time()
        origin_img = cv2.imread(test_path)
        label = cv2.imread(
            test_path.replace("images", "masks").replace(".jpg", "_segmentation.png")
        )
        # 转为灰度图
        img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        t1 = time.time()
        pred = net(img_tensor)
        t2 = time.time()
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.98] = 255
        pred[pred < 0.98] = 0

        # 计算dice和iou
        label_ = cv2.resize(label, (512, 512))
        label_ = cv2.cvtColor(label_, cv2.COLOR_BGR2GRAY)
        label_ = label_.reshape(1, label_.shape[0], label_.shape[1])
        if label_.max() > 1:
            label_ = label_ / 255 # type: ignore
        pred_ = pred.copy() / 255
        pred_ = pred_.reshape(1, pred_.shape[0], pred_.shape[1])
        dice_score = dice_coefficient(torch.from_numpy(pred_), torch.from_numpy(label_))
        iou_score = intersection_over_union(torch.from_numpy(pred_), torch.from_numpy(label_))
        # print(f"Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")
        
        # 保存结果地址
        save_name = os.path.basename(os.path.splitext(test_path)[0] + "_pred.png")
        save_path = os.path.join("images", save_name)
        pred = cv2.resize(pred, (origin_img.shape[1], origin_img.shape[0]))
        # saver.save_image(save_name, pred)
        t3 = time.time()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(origin_img)
        ax[0].set_title("Origin Image")
        ax[1].imshow(label)
        ax[1].set_title("Label")
        ax[2].imshow(pred)
        ax[2].set_title("Predict")
        fig.text(0.5, 0.1, f"Dice: {dice_score:.4f}, IoU: {iou_score:.4f}", fontsize=12, ha='center')
        os.makedirs(os.path.join(saver.saver_path, "images"), exist_ok=True)
        plt.savefig(os.path.join(saver.saver_path, "images", save_name))
        print(
            f"Saving to:{save_path} preprocess time:{(t1-t0)*1000}ms predict time:{(t2-t1)*1000}ms postprocess time:{(t3-t2)*1000}ms"
        )

    plt.close()
