import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import convert, prepare_qat

from args import args
from distill import train_knowledge_distillation
from MLP import MLP, get_default_qat_qconfig_per_tensor
import prune
from resnet import ResNet18


MAX_TRAIN_NOISE = 0.05
IMAGE_SIZE = 14
LAYER_NUM = 3
TRAIN_EPOCHS = 20
HIDDEN_LAYER_SIZE = 12
DISTILL_EPOCHS = 5
PRUNE_AMOUNT = 0.3
BATCH_SIZE = 64
VAL_SIZE = 10000
SEED = 42
TEACHER_CHECKPOINT = "teacher_resnet18.pth"
STUDENT_QAT_CHECKPOINT = "student_qat.pth"
STUDENT_QUANTIZED_CHECKPOINT = "student_quantized.pth"
USE_QUANTIZED_EVAL = True
CALIBRATION_BATCHES = 10


def set_args_defaults():
    args.image_size = IMAGE_SIZE
    args.layer_num = LAYER_NUM
    args.train_epoch = TRAIN_EPOCHS
    args.hidden_layer_size = HIDDEN_LAYER_SIZE
    args.distill_epoch = DISTILL_EPOCHS
    args.prune_amount = PRUNE_AMOUNT


def make_noisy_collate(max_noise):
    def collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        if max_noise > 0:
            sigma = random.uniform(0.0, max_noise)
            noise = torch.randn_like(images) * sigma
            images = torch.clamp(images + noise, 0.0, 1.0)
        return images, labels

    return collate


def accuracy_from_logits(logits, labels):
    _, predicted = torch.max(logits.data, 1)
    return (predicted == labels).sum().item(), labels.size(0)


def train_teacher_epoch(model, loader, optimizer, device):
    model.train()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct, total = accuracy_from_logits(outputs, labels)
        total_correct += correct
        total_samples += total
    return running_loss, total_correct / total_samples


def evaluate_teacher(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct, total = accuracy_from_logits(outputs, labels)
            total_correct += correct
            total_samples += total
    return total_correct / total_samples


def train_student_epoch(model, loader, optimizer, device):
    model.train()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct, total = accuracy_from_logits(outputs, labels)
        total_correct += correct
        total_samples += total
    return running_loss, total_correct / total_samples


def evaluate_student(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = torch.flatten(inputs, start_dim=1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            correct, total = accuracy_from_logits(outputs, labels)
            total_correct += correct
            total_samples += total
    return total_correct / total_samples


def evaluate_student_with_noise(model, dataset, device, max_noise, runs=10):
    accuracies = []
    for _ in range(runs):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=make_noisy_collate(max_noise),
        )
        accuracies.append(evaluate_student(model, loader, device))
    return sum(accuracies) / len(accuracies)


def build_quantized_model(model_qat, loader, device):
    model_qat.eval()
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            if batch_index >= CALIBRATION_BATCHES:
                break
            inputs = torch.flatten(inputs, start_dim=1).to(device)
            model_qat(inputs)
    model_quantized = convert(model_qat, inplace=False)
    model_quantized.eval()
    with torch.no_grad():
        inputs, labels = next(iter(loader))
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        model_quantized(inputs)
    return model_quantized


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_args_defaults()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(dataset) - VAL_SIZE
    trainset, valset = torch.utils.data.random_split(
        dataset,
        [train_size, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_noisy_collate(MAX_TRAIN_NOISE),
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    teacher_model = ResNet18().to(device)
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = "cpu"
    try:
        teacher_state = torch.load(TEACHER_CHECKPOINT, map_location=map_location)
        teacher_model.load_state_dict(teacher_state)
        print(f"Loaded teacher checkpoint: {TEACHER_CHECKPOINT}")
    except FileNotFoundError:
        teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(TRAIN_EPOCHS):
            _, train_acc = train_teacher_epoch(teacher_model, trainloader, teacher_optimizer, device)
            val_acc = evaluate_teacher(teacher_model, valloader, device)
            print(
                f"Teacher Epoch {epoch + 1}/{TRAIN_EPOCHS} - "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )
        torch.save(teacher_model.state_dict(), TEACHER_CHECKPOINT)
        print(f"Saved teacher checkpoint: {TEACHER_CHECKPOINT}")

    num_i = IMAGE_SIZE * IMAGE_SIZE
    student_model = MLP(num_i, HIDDEN_LAYER_SIZE, 10).to(device)
    student_model.qconfig = get_default_qat_qconfig_per_tensor()
    model_qat = prepare_qat(student_model)

    loaded_student = False
    if os.path.exists(STUDENT_QAT_CHECKPOINT):
        try:
            model_qat.load_state_dict(
                torch.load(STUDENT_QAT_CHECKPOINT, map_location=map_location)
            )
            model_qat.eval()
            if USE_QUANTIZED_EVAL:
                model_quantized = build_quantized_model(model_qat, trainloader, device)
                eval_model = model_quantized
            else:
                eval_model = model_qat
            loaded_student = True
            print(f"Loaded student QAT checkpoint: {STUDENT_QAT_CHECKPOINT}")
        except RuntimeError as error:
            print(
                "Failed to load QAT checkpoint, will retrain and overwrite: "
                f"{error}"
            )

    if not loaded_student:
        student_optimizer = optim.Adam(model_qat.parameters())
        for epoch in range(TRAIN_EPOCHS):
            _, train_acc = train_student_epoch(model_qat, trainloader, student_optimizer, device)
            val_acc = evaluate_student(model_qat, valloader, device)
            if epoch > 5:
                model_qat.apply(torch.ao.quantization.disable_observer)
            if epoch > 3:
                model_qat.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print(
                f"Student Epoch {epoch + 1}/{TRAIN_EPOCHS} - "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

        train_knowledge_distillation(
            teacher=teacher_model,
            student=model_qat,
            train_loader=trainloader,
            epochs=DISTILL_EPOCHS,
            learning_rate=0.005,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            device=device,
        )

        model_qat = prune.prune_model(model_qat, PRUNE_AMOUNT)
        model_qat.eval()

        torch.save(model_qat.state_dict(), STUDENT_QAT_CHECKPOINT)
        print(f"Saved student QAT checkpoint: {STUDENT_QAT_CHECKPOINT}")

        if USE_QUANTIZED_EVAL:
            try:
                model_quantized = build_quantized_model(model_qat, trainloader, device)
                eval_model = model_quantized
                torch.save(model_quantized.state_dict(), STUDENT_QUANTIZED_CHECKPOINT)
                print(
                    f"Saved student quantized checkpoint: {STUDENT_QUANTIZED_CHECKPOINT}"
                )
            except RuntimeError as error:
                print(
                    "Quantized conversion failed after retraining. "
                    f"Falling back to QAT eval: {error}"
                )
                eval_model = model_qat
        else:
            eval_model = model_qat

    noise_levels = [i / 10 for i in range(0, 11)]
    noise_accs = []
    for max_noise in noise_levels:
        acc = evaluate_student_with_noise(eval_model, testset, device, max_noise, runs=10)
        noise_accs.append(acc)
        print(f"Test Noise Max {max_noise:.1f} - Acc: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, noise_accs, marker="o")
    plt.xlabel("Max Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Noise Robustness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_robustness.png")


if __name__ == "__main__":
    main()
