import argparse
import os 
import torch 
import matplotlib.pyplot as plt 
import pandas as pd 
from torchvision.models.efficientnet import EfficientNet_B3_Weights, efficientnet_b3
from torch.optim import AdamW 
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, Linear, Dropout
from torch.utils.data import DataLoader 
from customdataset import RetDataset 
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm  
from loss_weight import class_weights

from torch.optim.lr_scheduler import StepLR

class RetClassifier:
    def __init__(self, model, optimizer, criterion):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        
        for epoch in range(self.start_epochs, self.start_epochs+args.epochs+1):
            train_correct = 0.0
            train_loss = 0.0
            val_correct = 0.0
            val_loss = 0.0
            self.model.train()

            train_loader_iter = tqdm(train_loader,
                                     desc=(f"Epoch: {epoch + 1} / {self.start_epochs+args.epochs}"),
                                     leave=False)

            for index, (data, label) in enumerate(train_loader_iter):
                data = data.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(data)
                #print(outputs.shape) # torch.Size([16, 1000])

                loss = self.criterion(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                train_correct += (pred == label).sum().item()

                scheduler.step()

            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)

            print(f'train accuracy: {train_acc}')

            if epoch % 1  == 0:
                self.model.eval()
                with torch.no_grad():
                    for data, label in val_loader:
                        data = data.float().to(self.device)
                        label = label.to(self.device)

                        output = self.model(data)
                        
                        pred = output.argmax(dim=1, keepdim=True)

                        val_correct += pred.eq(label.view_as(pred)).sum().item()
                        val_loss += self.criterion(output, label).item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / len(val_loader.dataset)

                self.val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)

                print(f"Epoch {epoch + 1} / {self.start_epochs+args.epochs} : Train loss : {train_loss:.3f}, ",
                      f"Val loss : {val_loss:.3f}, Train Acc : {train_acc:.3f}, Val Acc : {val_acc:.3f}")
                
                if val_acc > best_val_acc:
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
        
        self.save_results_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_results_to_csv(self):
        df = pd.DataFrame({
            'Train Loss' : self.train_loss_list,
            'Train Accuracy': self.train_acc_list,
            'Validation Loss' : self.val_loss_list,
            'Validation Accuracy' : self.val_acc_list
        })
        df.to_csv('train_val_results_sports.csv', index=False)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_loss_list, label='Train loss')
        plt.plot(self.val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('sports_loss_plot.png')

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_acc_list, label='Train Accuracy')
        plt.plot(self.val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('sports_acc_plot.png')

    def load_ckpt(self, ckpt_file):
        '''
        model.state_dict()
        optimizer.state_dict()
        '''
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epochs = ckpt["epoch"]

        self.train_loss_list = ckpt["train_losses"]
        self.train_acc_list = ckpt["train_accs"]
        self.val_loss_list = ckpt["val_losses"]
        self.val_acc_list = ckpt["val_accs"]

# EfficientNet B3 with drop_out
class RegularizedEfficientNetB3(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.5):
        super(RegularizedEfficientNetB3, self).__init__()
        self.efficientnet = efficientnet_b3(EfficientNet_B3_Weights)
        self.efficientnet.classifier = nn.Sequential(
            self.efficientnet.classifier,
            nn.Dropout(dropout_prob),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        )
    def forward(self, x):
        return self.efficientnet(x)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--checkpoint_path", type=str,
                        default="./weight/ret_classifier_effb3.pt")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight/")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4)
    parser.add_argument("--weight_decay", type=float,
                        default=5e-4)
    parser.add_argument("--batch_size", type=int,
                        default=16)
    parser.add_argument("--num_workers", type=int,
                        default=4)
    parser.add_argument("--resume_epoch", type=int,
                        default=0)
    parser.add_argument("--train_dir", type=str,
                        default="./dataset_resized_cropped")
    parser.add_argument("--val_dir", type=str,
                        default="./dataset_resized_cropped")

    
    args = parser.parse_args()
    
    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)

    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    # model.classifier[1].out_features = 5

    num_classes = 5
    in_features = model.classifier[1].in_features
    model.classifier = Linear(in_features, num_classes)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss(weight=class_weights)

    classifier = RetClassifier(model, optimizer, criterion)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    if args.resume_training:
        resume_epoch = args.resume_epoch  # Assign the value to a variable
        classifier.load_ckpt(
            args.checkpoint_path.replace(".pt", f"_{resume_epoch}.pt"))
        classifier.start_epochs = resume_epoch
    
    print(f"args.resume_epoch: {args.resume_epoch}")


    train_transform = A.Compose([
        A.Resize(width=512, height=512),
        #A.RandomCrop(height=728, width=728),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        #A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    train_dataset = RetDataset(args.train_dir, mode='train',
                                  transform=train_transform)
    val_dataset = RetDataset(args.val_dir, mode='val',
                                transform=val_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    
    print('training is about to start')
    try:
        classifier.train(train_dataloader, val_dataloader, args)
    except Exception as e:
        print(f"An error occurred: {e}")