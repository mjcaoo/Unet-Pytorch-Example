import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model.unet_model import UNet
from utils.dataset import ISIC_Loader
from utils.utils import dice_coefficient, intersection_over_union, Saver

from tqdm import tqdm
import yaml


def train_net(
    net,
    device,
    train_data_path,
    val_data_path,
    epochs=1,
    batch_size=1,
    lr=0.00001,
    resize_shape=(512, 512),
    num_workers=4,
    pin_memory=True,
    seed=42,
):
    saver = Saver("runs/train")
    saver.save_hyps(
        {
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "resize_shape": resize_shape,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "seed": seed,
        }
    )
    torch.manual_seed(seed)
    # 加载训练集
    train_dataset = ISIC_Loader(train_data_path, resize_shape)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # 加载验证集
    val_dataset = ISIC_Loader(val_data_path, resize_shape)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = 0

    # 保存输出信息
    output_datas = list()

    # 训练
    for epoch in range(1, epochs + 1):
        epoch_output_data = dict()
        epoch_output_data["epoch"] = epoch

        # 训练模式
        net.train()
        # 初始化train_loss、dice和iou
        train_loss = 0
        train_dice_score = 0
        train_iou_score = 0
        # 按照batch_size开始训练
        train_bar = tqdm(train_loader)
        for image, label in train_bar:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)  # x
            label = label.to(device=device, dtype=torch.float32)  # y
            # 使用网络参数，输出预测结果
            pred = net(image)  # y_pred

            # 计算loss
            loss = criterion(pred, label)  # loss是y与y_pred之间的差距
            train_loss += loss.item()
            # 计算dice和iou
            pred_ = pred.clone().detach()
            pred_[pred_ >= 0.5] = 1
            pred_[pred_ < 0.5] = 0
            train_dice = dice_coefficient(pred_, label)
            train_iou = intersection_over_union(pred_, label)
            train_dice_score += train_dice
            train_iou_score += train_iou

            # 更新参数
            loss.backward()  # 将loss反向传播
            optimizer.step()
            scheduler.step()

            train_bar.set_description(f"Epoch {epoch}: train loss: {loss.item():.6f}")

        train_loss /= len(train_loader)
        train_dice_score /= len(train_loader)
        train_iou_score /= len(train_loader)

        print(
            f"Epoch {epoch}: train loss: {train_loss:.6f}, dice: {train_dice_score:.4f}, iou: {train_iou_score:.4f}"
        )

        epoch_output_data["train_loss"] = train_loss
        epoch_output_data["train_dice"] = train_dice_score
        epoch_output_data["train_iou"] = train_iou_score

        # 每隔1个epoch验证一次
        if epoch % 1 == 0:
            # 验证模式
            net.eval()

            val_loss = 0
            val_dice_score = 0
            val_iou_score = 0
            # 不更新参数
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                for image, label in val_bar:
                    # 将数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    # 使用网络参数，输出预测结果
                    pred = net(image)

                    # 计算loss
                    loss = criterion(pred, label)
                    val_loss += loss.item()
                    # 计算dice和iou
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    val_dice = dice_coefficient(pred, label)
                    val_iou = intersection_over_union(pred, label)
                    val_dice_score += val_dice
                    val_iou_score += val_iou

                    val_bar.set_description(
                        f"Epoch {epoch}: val loss: {loss.item():.6f}"
                    )

            val_loss /= len(val_loader)
            val_dice_score /= len(val_loader)
            val_iou_score /= len(val_loader)

            print(
                f"Epoch {epoch}: val loss: {val_loss:.6f}, dice: {val_dice_score:.4f}, iou: {val_iou_score:.4f}"
            )

            epoch_output_data["val_loss"] = val_loss
            epoch_output_data["val_dice"] = val_dice_score
            epoch_output_data["val_iou"] = val_iou_score

            if val_dice_score > best_val_dice:
                best_val_dice = val_dice_score
                # 保存最好的模型
                saver.save_model(net, "best.pt")
                print(
                    f"Best model saved at epoch {epoch}, val_dice: {val_dice_score:.4f}"
                )

        if epoch % 5 == 0:
            # 保存完整模型
            saver.save_model(net, f"epoch{epoch}.pt")

        output_datas.append(epoch_output_data)

    # 保存训练过程中的数据
    saver.save_output(output_datas)


if __name__ == "__main__":
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载网络，图片单通道，分类为1
    net = UNet(n_channels=1, n_classes=1)
    # 移动到deivce中
    net.to(device=device)
    # 指定训练集地址及训练超参数
    with open("data/hyps.yaml", "r") as file:
        args = yaml.safe_load(file)
    # 开始训练
    train_net(net, device, **args)
