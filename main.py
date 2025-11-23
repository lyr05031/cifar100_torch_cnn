import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import analyze

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = CNN.conv_factory(3, 64, DROP)
        self.conv2 = CNN.conv_factory(64, 128, DROP)
        self.conv3 = CNN.conv_factory(128, 256, DROP)
        self.conv4 = CNN.conv_factory(256, 512, DROP)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 * 2, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(DROP),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(DROP),
            torch.nn.Linear(256, 100),
        )

    @staticmethod
    def conv_factory(in_channels, out_channels, rate):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(rate),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inp):
        inp = self.conv1(inp)
        inp = self.conv2(inp)
        inp = self.conv3(inp)
        inp = self.conv4(inp)

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
    print(f"testing acc {correct * 100 / train_dataset.__len__()}%")
    return correct / train_dataset.__len__()


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
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=8,
    )

    device = "mps"
    best_acc = 0
    for y in range(9):
        DROP = y/10
        acc_list = []
        model = CNN()
        model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        criterion = torch.nn.CrossEntropyLoss()
        for x in range(1000):
            train(model, criterion, optimizer, train_loader, device)
            acc = test(model, device, test_loader)
            scheduler.step(acc)
            acc_list.append(acc*100)
            if acc > best_acc:
                best_acc = acc
            print(f"drop = {DROP} || epoch {x} finished")
        analyze.save_floats_to_excel(
            float_list=acc_list,
            excel_filename=f"drop={DROP}.xlsx",
            sheet_name="DATA",
        )
        print(f"✅ drop = {DROP} || best acc is {best_acc*100}%")

