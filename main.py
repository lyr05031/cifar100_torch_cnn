import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import analyze
import random
import numpy as np


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
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=0.9),
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
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        pre = output.argmax(axis=1, keepdims=True)
        correct += pre.eq(target.view_as(pre)).sum().item()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"training acc {correct * 100 / train_dataset.__len__()}%")
    return correct / train_dataset.__len__()


@torch.no_grad()
def test(model, device, dataloader):
    model.eval()
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pre = output.argmax(axis=1, keepdims=True)
        correct += pre.eq(target.view_as(pre)).sum().item()
    print(f"testing acc {correct * 100 / test_dataset.__len__()}%")
    return correct / test_dataset.__len__()


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
    )

    device = "mps"

    torch.manual_seed(42)
    torch.mps.manual_seed(42)

    for y in range(0, 10):
        best_acc = 0
        DROP = y / 10
        test_acc_list = []
        train_acc_list = []
        model = CNN()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        criterion = torch.nn.CrossEntropyLoss()
        x = 0
        cnt = 0
        while 1:
            train_acc = train(model, criterion, optimizer, train_loader, device)
            test_acc = test(model, device, test_loader)
            print(f"drop = {DROP} || epoch {x} finished")
            scheduler.step(test_acc)
            test_acc_list.append(test_acc * 100)
            train_acc_list.append(train_acc * 100)
            x += 1
            cnt += 1
            if test_acc > best_acc:
                best_acc = test_acc
                cnt = 0
            if cnt >= 20:
                print(
                    f"drop = {y / 10} complete\n total epoch {x} best acc {best_acc * 100}%"
                )
                del model, optimizer, scheduler, criterion
                torch.mps.empty_cache()
                break
        analyze.save_floats_to_excel(
            float_list=test_acc_list,
            excel_filename=f"drop={DROP}_test.xlsx",
            sheet_name="DATA",
        )
        analyze.save_floats_to_excel(
            float_list=train_acc_list,
            excel_filename=f"drop={DROP}_train.xlsx",
            sheet_name="DATA",
        )
        print(f"drop = {y / 10} saved")
