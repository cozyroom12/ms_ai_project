import argparse
import os
import pandas as pd

import numpy as np
import torch
from torch import nn
from torchvision.models.efficientnet import efficientnet_b0
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from customdataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.utils import class_weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "./diabetic-retinopathy-detection/trainLabels.csv/"
train_df = pd.read_csv(f"{path}trainLabels.csv")
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['level'].values)
class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
class CheckPoint:
    def __init__(self, model, optimizer, criterion):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.start_epochs = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def train(self, train_loader, val_loader, args):
        best_val_acc = 0.0

        for epoch in range(self.start_epochs, args.epochs):

            train_correct = 0.0
            train_loss = 0.0
            val_correct = 0.0
            val_loss = 0.0
            self.model.train()

            train_loader_iter = tqdm(train_loader,
                                     desc=(f"Epoch: {epoch + 1} / {args.epochs}"),
                                     leave=False)
            for index, (data, target) in enumerate(train_loader_iter):

                data, target = data.float().to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                train_correct += (pred == target).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)
            train_loader_iter.set_postfix({"Loss": loss.item()})

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)

            print(f"train accuracy : {train_acc}")

            if epoch % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.float().to(self.device)
                        target = target.to(self.device)

                        output = self.model(data)

                        pred = output.argmax(dim=1, keepdim=True)

                        val_correct += pred.eq(target.view_as(pred)).sum().item()
                        val_loss += self.criterion(output, target).item()

                val_loss /= len(val_loader)
                val_acc = val_correct / len(val_loader.dataset)

                self.val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)


                if val_acc > best_val_acc:
                    with open("metrics.txt", "a") as f:
                        f.write(f"Epoch [{epoch + 1} / {args.epochs}] , Train loss [{train_loss:.4f}],"
                                f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
                                f"Val ACC [{val_acc:.4f}]\n")
                    print(f"Epoch {epoch + 1} / {args.epochs} : Train loss : {train_loss:.3f}, ",
                          f"Val loss : {val_loss:.3f}, Train Acc : {train_acc:.3f}, Val Acc : {val_acc:.3f}")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_losses": self.train_loss_list,
                        "train_accs": self.train_acc_list,
                        "val_losses": self.val_loss_list,
                        "val_accs": self.val_acc_list
                    }, args.checkpoint_path.replace(".pt",
                                                    "_best.pt"))
                    best_val_acc = val_acc

            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_loss_list,
                "train_accs": self.train_acc_list,
                "val_losses": self.val_loss_list,
                "val_accs": self.val_acc_list
            }, args.checkpoint_path.replace(".pt", f"_{epoch}.pt"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_loss_list,
            "train_accs": self.train_acc_list,
            "val_losses": self.val_loss_list,
            "val_accs": self.val_acc_list
        }, args.checkpoint_path.replace(".pt", "_last.pt"))

    def load_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epochs = ckpt["epoch"]
        self.train_loss_list = ckpt["train_losses"]
        self.train_acc_list = ckpt["train_accs"]
        self.val_loss_list = ckpt["val_losses"]
        self.val_acc_list = ckpt["val_accs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--checkpoint_path", type=str,
                        default="./weight/duabetic_retinopathy_efiicient_b0_checkpoint.pt")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight/")

    parser.add_argument("--resume_training", action="store_true")

    parser.add_argument("--learning_rate", type=float,
                        default=0.001)
    parser.add_argument("--weight_decay", type=float,
                        default=1e-2)
    parser.add_argument("--batch_size", type=int,
                        default=24)
    parser.add_argument("--num_workers", type=int,
                        default=4)

    parser.add_argument("--resume_epoch", type=int,
                        default=0)

    parser.add_argument("--train_dir", type=str,
                        default="./final/train")

    parser.add_argument("--val_dir", type=str,
                        default="./final/val")

    args = parser.parse_args()

    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)


    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 5)


    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss(weight=class_weights)

    classifier = CheckPoint(model, optimizer, criterion)

    if args.resume_training:
        classifier.load_ckpt(
            args.checkpoint_path.replace(".pt",
                                         f"{args.resume_epoch}.pt"))

    train_transform = A.Compose([
            # A.Blur(),
            # A.Flip(),
            # A.RandomBrightnessContrast(),
            # A.ShiftScaleRotate(),
            # A.ElasticTransform(),
            # A.Transpose(),
            # A.GridDistortion(),
            # A.HueSaturationValue(),
            # A.CLAHE(),
            # A.CoarseDropout(),
            A.Resize(512, 512),
            ToTensorV2(),
        ])
    val_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

    train_dataset = CustomDataset(args.train_dir,
                                  train_transform)
    val_dataset = CustomDataset(args.val_dir,
                                val_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    classifier.train(train_dataloader, val_dataloader, args)