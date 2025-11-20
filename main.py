import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 * 2, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 100),
        )

    def forward(self, inp):
        inp = self.conv(inp)
        inp = inp.view(-1, 512 * 2 * 2)
        inp = self.fc(inp)
        return inp


def train(model, criterion, optimizer, dataloader, device):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, device, dataloader):
    model.eval()
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pre = output.argmax(axis=1, keepdims=True)
        correct += pre.eq(target.view_as(pre)).sum().item()
    print(f"testing acc {correct / 100}%")
    return correct / 100


if __name__ == "__main__":
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(0.25),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_dataset = datasets.CIFAR100(
        root="./DATA", train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.CIFAR100(
        root="./DATA", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=12,
    )
    model = CNN()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = torch.nn.DataParallel(model)

    best_acc = 0
    for x in range(100):
        train(model, criterion, optimizer, train_loader, device)
        acc = test(model, device, test_loader)
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./best_model.pth")
            print("保存最佳模型")
        print(f"epoch {x} finished")
    print(f"best acc is {best_acc}%")
