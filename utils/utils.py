import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def clamp(value, min_value, max_value):
    """限制值在[min_value, max_value]的范围内"""
    return max(min_value, min(value, max_value))


def dice_coefficient(predicted, target, smooth=1e-8):
    intersection = torch.sum(predicted * target, dim=(1, 2))
    union = torch.sum(predicted, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # 使用clamp函数将Dice系数的值限制在0到1之间
    # dice = clamp(dice.mean().item(), 0, 1)
    dice = dice.mean().item()
    return dice


def intersection_over_union(predicted, target, smooth=1e-8):
    intersection = torch.sum(predicted * target, dim=(1, 2))
    union = (
        torch.sum(predicted, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) - intersection
    )
    iou = (intersection + smooth) / (union + smooth)

    # 使用clamp函数将IoU的值限制在0到1之间
    # iou = clamp(iou.mean().item(), 0, 1)
    iou = iou.mean().item()
    return iou


def attempt_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        idx = 1
        basename = os.path.basename(dir_path)
        basedir = os.path.dirname(dir_path)
        while os.path.exists(os.path.join(basedir, f"{basename}{idx}")):
            idx += 1
        dir_path = os.path.join(basedir, f"{basename}{idx}")
        os.makedirs(dir_path, exist_ok=True)

    return dir_path


class Saver:
    def __init__(self, dir="runs"):
        self.saver_path = self.get_saver_path(dir)

    def get_saver_path(self, dir, name="exp"):
        save_dir = attempt_mkdir(os.path.join(dir, name))

        return save_dir

    def save_model(self, model, name="model.pt"):
        os.makedirs(os.path.join(self.saver_path, "checkpoints"), exist_ok=True)
        torch.save(model, os.path.join(self.saver_path, "checkpoints", name))

    def save_output(self, output_datas):
        # 保存训练过程中的数据
        output_df = pd.DataFrame(output_datas)
        output_df.to_csv(os.path.join(self.saver_path, "output.csv"), index=False)

        # 绘制训练数据图表
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0, 0].plot(output_df["epoch"], output_df["train_loss"], label="train_loss")
        ax[0, 0].set_title("Train Loss")
        ax[0, 0].legend()
        ax[0, 1].plot(output_df["epoch"], output_df["train_dice"], label="train_dice")
        ax[0, 1].set_title("Train Dice")
        ax[0, 1].legend()
        ax[0, 2].plot(output_df["epoch"], output_df["train_iou"], label="train_iou")
        ax[0, 2].set_title("Train IoU")
        ax[0, 2].legend()
        ax[1, 0].plot(output_df["epoch"], output_df["val_loss"], label="val_loss")
        ax[1, 0].set_title("Val Loss")
        ax[1, 0].legend()
        ax[1, 1].plot(output_df["epoch"], output_df["val_dice"], label="val_dice")
        ax[1, 1].set_title("Val Dice")
        ax[1, 1].legend()
        ax[1, 2].plot(output_df["epoch"], output_df["val_iou"], label="val_iou")
        ax[1, 2].set_title("Val IoU")
        ax[1, 2].legend()
        plt.savefig(os.path.join(self.saver_path, "output.png"))

    def save_hyps(self, hyps: dict):
        import yaml

        with open(os.path.join(self.saver_path, "hyps.yaml"), "w") as f:
            yaml.dump(hyps, f)
    
    def save_image(self, save_path, image):
        save_path = os.path.join(self.saver_path, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    # Example usage
    # Assuming predicted and target are torch tensors with batch dimension
    predicted = torch.tensor(
        [[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[1, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
        dtype=torch.float32,
    )

    # Compute Dice coefficient
    dice = dice_coefficient(predicted, target)
    print("Dice Coefficient:", dice)

    # Compute IoU
    iou = intersection_over_union(predicted, target)
    print("IoU:", iou)

    saver = Saver()
    print(saver.saver_path)
