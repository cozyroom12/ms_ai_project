
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_b0
from customdataset import CustomDataset
import albumentations as A

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 5)




    model.load_state_dict(torch.load(f="./efficientnet_b0_best.pt"))

    val_transforms = A.Compose([A.Resize(512, 512), ToTensorV2()])

    test_dataset = CustomDataset("./remove_black/val/", transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    from tqdm import tqdm
    correct = 0
    with torch.no_grad() :
        for data, target in tqdm(test_loader) :
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()


    print("test set : Acc {}/{} [{:.0f}]%\n".format(
        correct, len(test_loader.dataset),
        100*correct / len(test_loader.dataset)
    ))


#

if __name__ == "__main__" :
    main()