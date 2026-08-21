"""Reproducible experiments for Enhanced Hybrid Redundancy-Aware Pruning.

The script evaluates structured filter pruning on five referenced image
classification datasets and three architectures: SimpleCNN, VGG-11-BN, and a
true CIFAR ResNet-20.  ResNet pruning is applied to each residual block's
internal convolution so that channel removal is physical while residual tensor
shapes remain valid.

Full experimental grid:
    python s9.py --full-grid

Quick pipeline check:
    python s9.py --quick

Useful focused run:
    python s9.py --datasets CIFAR10 CIFAR100 --architectures ResNet20 VGG11BN
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_DATASETS = ("CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "MNIST")
SUPPORTED_ARCHITECTURES = ("SimpleCNN", "VGG11BN", "ResNet20")
SUPPORTED_METHODS = (
    "Uniform-Magnitude",
    "Adaptive-Magnitude",
    "Adaptive-Correlation",
    "E-HRAP",
)

DATASET_REFERENCES = {
    "CIFAR10": "A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.",
    "CIFAR100": "A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.",
    "SVHN": "Y. Netzer et al., Reading Digits in Natural Images with Unsupervised Feature Learning, 2011.",
    "FashionMNIST": "H. Xiao, K. Rasul, and R. Vollgraf, Fashion-MNIST, arXiv:1708.07747, 2017.",
    "MNIST": "Y. LeCun et al., Gradient-Based Learning Applied to Document Recognition, Proc. IEEE, 1998.",
}

ARCHITECTURE_REFERENCES = {
    "SimpleCNN": "Custom three-layer CNN; fully specified in the experiment manifest.",
    "VGG11BN": "K. Simonyan and A. Zisserman, Very Deep Convolutional Networks, ICLR, 2015.",
    "ResNet20": "K. He et al., Deep Residual Learning for Image Recognition, CVPR, 2016 (CIFAR 6n+2 variant, n=3).",
}


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DatasetManager:
    @staticmethod
    def get_dataset(name="CIFAR10", data_root="./data", download=True):
        if name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset {name!r}; choose from {SUPPORTED_DATASETS}")

        if name == "CIFAR10":
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR10(data_root, train=True, download=download, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(data_root, train=False, download=download, transform=test_transform)
            return trainset, testset, 10, 3

        if name == "CIFAR100":
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            trainset = torchvision.datasets.CIFAR100(data_root, train=True, download=download, transform=train_transform)
            testset = torchvision.datasets.CIFAR100(data_root, train=False, download=download, transform=test_transform)
            return trainset, testset, 100, 3

        if name == "SVHN":
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4524, 0.4525, 0.4690), (0.2194, 0.2266, 0.2285)),
            ])
            trainset = torchvision.datasets.SVHN(data_root, split="train", download=download, transform=train_transform)
            testset = torchvision.datasets.SVHN(data_root, split="test", download=download, transform=test_transform)
            return trainset, testset, 10, 3

        grayscale_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset_class = (
            torchvision.datasets.FashionMNIST if name == "FashionMNIST" else torchvision.datasets.MNIST
        )
        trainset = dataset_class(data_root, train=True, download=download, transform=grayscale_transform)
        testset = dataset_class(data_root, train=False, download=download, transform=grayscale_transform)
        return trainset, testset, 10, 1


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, channels=None, input_channels=3):
        super().__init__()
        channels = channels or [64, 128, 256]
        self.channels = list(channels)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        feature_size = 32 // (2 ** len(channels))
        self.fc1 = nn.Linear(channels[-1] * feature_size * feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.pool(self.relu(bn(conv(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

    def get_prunable_layers(self):
        return [
            {"name": f"conv{i + 1}", "module": conv, "bn_module": bn}
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers))
        ]


class ConfigurableVGG11BN(nn.Module):
    CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, num_classes=10, channels=None, input_channels=3):
        super().__init__()
        channels = list(channels or [v for v in self.CFG if isinstance(v, int)])
        self.channels = channels
        layers = []
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = input_channels
        channel_index = 0
        for token in self.CFG:
            if token == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            out_channels = channels[channel_index]
            conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            layers.extend([conv, bn, nn.ReLU(inplace=True)])
            in_channels = out_channels
            channel_index += 1
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))

    def get_prunable_layers(self):
        return [
            {"name": f"features.conv{i + 1}", "module": conv, "bn_module": bn}
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers))
        ]


class CifarBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, hidden_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, hidden_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ConfigurableResNet20(nn.Module):
    """True CIFAR ResNet-20 (6n+2, n=3) with prunable block interiors.

    Each block's first convolution is physically pruned.  The second
    convolution retains the stage output width, which preserves the residual
    addition while still reducing parameters and MACs.
    """

    def __init__(self, num_classes=10, hidden_channels=None, input_channels=3):
        super().__init__()
        stage_widths = [16] * 3 + [32] * 3 + [64] * 3
        hidden_channels = list(hidden_channels or stage_widths)
        if len(hidden_channels) != 9:
            raise ValueError("ResNet20 requires nine hidden-channel values")
        self.hidden_channels = hidden_channels
        self.stem_conv = nn.Conv2d(input_channels, 16, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList()
        in_planes = 16
        for i, (planes, hidden) in enumerate(zip(stage_widths, hidden_channels)):
            stride = 2 if i in (3, 6) else 1
            self.blocks.append(CifarBasicBlock(in_planes, planes, hidden, stride))
            in_planes = planes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))

    def get_prunable_layers(self):
        return [
            {"name": f"block{i + 1}.conv1", "module": block.conv1, "bn_module": block.bn1}
            for i, block in enumerate(self.blocks)
        ]


def build_model(name, num_classes, input_channels):
    if name == "SimpleCNN":
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    if name == "VGG11BN":
        return ConfigurableVGG11BN(num_classes=num_classes, input_channels=input_channels)
    if name == "ResNet20":
        return ConfigurableResNet20(num_classes=num_classes, input_channels=input_channels)
    raise ValueError(f"Unsupported architecture {name!r}")


def copy_bn(old_bn, new_bn, indices=None):
    if indices is None:
        indices = torch.arange(old_bn.num_features, device=old_bn.weight.device)
    new_bn.weight.data.copy_(old_bn.weight.data[indices])
    new_bn.bias.data.copy_(old_bn.bias.data[indices])
    new_bn.running_mean.data.copy_(old_bn.running_mean.data[indices])
    new_bn.running_var.data.copy_(old_bn.running_var.data[indices])
    new_bn.num_batches_tracked.data.copy_(old_bn.num_batches_tracked.data)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_macs(model, input_channels, image_size=32):
    macs = []
    hooks = []

    def conv_hook(module, _inputs, output):
        output_elements = output.numel() // output.shape[0]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups
        macs.append(output_elements * kernel_ops)

    def linear_hook(module, _inputs, _output):
        macs.append(module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, input_channels, image_size, image_size, device=next(model.parameters()).device))
    for hook in hooks:
        hook.remove()
    model.train(was_training)
    return int(sum(macs))


def benchmark_latency(model, input_channels, warmup=20, repeats=100):
    model.eval()
    sample = torch.randn(1, input_channels, 32, 32, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


class ComprehensivePruner:
    def __init__(self, model, train_loader, test_loader, architecture, num_classes, input_channels):
        self.original_model = model
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels

    def evaluate_model(self, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                predictions = model(inputs).argmax(dim=1)
                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()
        return 100.0 * correct / max(total, 1)

    def calculate_layer_sensitivity(self, layer, sample_size=300):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        batch_scores = []
        captured_gradients = []

        def forward_hook(_module, _inputs, output):
            output.register_hook(lambda gradient: captured_gradients.append(gradient.detach()))

        handle = layer.register_forward_hook(forward_hook)
        samples_seen = 0
        try:
            for inputs, labels in self.train_loader:
                if samples_seen >= sample_size:
                    break
                remaining = sample_size - samples_seen
                inputs = inputs[:remaining].to(DEVICE, non_blocking=True)
                labels = labels[:remaining].to(DEVICE, non_blocking=True)
                self.model.zero_grad(set_to_none=True)
                loss = criterion(self.model(inputs), labels)
                loss.backward()
                if captured_gradients:
                    gradient = captured_gradients[-1]
                    if torch.isfinite(gradient).all():
                        batch_scores.append(torch.linalg.vector_norm(gradient).item())
                captured_gradients.clear()
                samples_seen += inputs.size(0)
        finally:
            handle.remove()
        return float(np.mean(batch_scores)) if batch_scores else 1.0

    @staticmethod
    def calculate_filter_importance(bn_layer):
        return bn_layer.weight.detach().abs().cpu()

    def calculate_channel_correlation(self, layer, sample_size=200):
        activations = []

        def hook(_module, _inputs, output):
            activations.append(output.detach().cpu())

        handle = layer.register_forward_hook(hook)
        self.model.eval()
        samples_seen = 0
        try:
            with torch.no_grad():
                for inputs, _labels in self.train_loader:
                    if samples_seen >= sample_size:
                        break
                    remaining = sample_size - samples_seen
                    inputs = inputs[:remaining].to(DEVICE, non_blocking=True)
                    self.model(inputs)
                    samples_seen += inputs.size(0)
        finally:
            handle.remove()
        if not activations:
            return torch.zeros(layer.out_channels)
        tensor = torch.cat(activations, dim=0)[:sample_size].float()
        if tensor.ndim != 4:
            return torch.zeros(layer.out_channels)
        channels = tensor.shape[1]
        if channels == 1:
            return torch.zeros(1)
        flattened = tensor.permute(0, 2, 3, 1).reshape(-1, channels)
        flattened -= flattened.mean(dim=0, keepdim=True)
        denominator = max(flattened.shape[0] - 1, 1)
        covariance = flattened.T.mm(flattened) / denominator
        std = covariance.diag().clamp_min(0).sqrt()
        correlation = covariance / (std[:, None] * std[None, :] + 1e-8)
        absolute = correlation.abs().clamp(0, 1)
        redundancy = (absolute.sum(dim=1) - absolute.diag()) / (channels - 1)
        return redundancy.cpu()

    @staticmethod
    def normalize_scores(scores):
        minimum = scores.min()
        maximum = scores.max()
        if (maximum - minimum).item() < 1e-8:
            return torch.full_like(scores, 0.5)
        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def adaptive_rates(sensitivities, layer_names, base_rate, alpha=0.7):
        sensitivity = torch.tensor(sensitivities, dtype=torch.float64).clamp_min(0)
        normalized = sensitivity / sensitivity.sum().clamp_min(1e-12)
        rates = base_rate * (1.0 - alpha * normalized)
        return {
            name: float(rate.clamp(0.0, 0.95).item())
            for name, rate in zip(layer_names, rates)
        }

    def prune_model(self, method="E-HRAP", base_prune_rate=0.3, alpha=0.7, beta=0.3,
                    sensitivity_samples=300, correlation_samples=200):
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {method!r}; choose from {SUPPORTED_METHODS}")
        started = time.time()
        layer_info = self.model.get_prunable_layers()
        layer_names = [item["name"] for item in layer_info]

        adaptive = method != "Uniform-Magnitude"
        if adaptive:
            sensitivities = [
                self.calculate_layer_sensitivity(item["module"], sensitivity_samples)
                for item in layer_info
            ]
            pruning_rates = self.adaptive_rates(sensitivities, layer_names, base_prune_rate, alpha)
        else:
            sensitivities = [math.nan] * len(layer_info)
            pruning_rates = {name: base_prune_rate for name in layer_names}

        pruned_indices = {}
        retained_channels = []
        for item in layer_info:
            name = item["name"]
            magnitude = self.calculate_filter_importance(item["bn_module"])
            if method in ("Adaptive-Correlation", "E-HRAP"):
                redundancy = self.calculate_channel_correlation(item["module"], correlation_samples)
                uniqueness = 1.0 - redundancy
            if method in ("Uniform-Magnitude", "Adaptive-Magnitude"):
                score = magnitude
            elif method == "Adaptive-Correlation":
                score = uniqueness
            else:
                score = (
                    (1.0 - beta) * self.normalize_scores(magnitude)
                    + beta * self.normalize_scores(uniqueness)
                )
            filter_count = item["module"].out_channels
            prune_count = int(math.floor(pruning_rates[name] * filter_count))
            keep_count = max(1, filter_count - prune_count)
            indices = torch.topk(score, keep_count, largest=True).indices.sort().values
            pruned_indices[name] = indices.to(item["module"].weight.device)
            retained_channels.append(keep_count)

        base_accuracy = self.evaluate_model(self.model)
        params_before = count_parameters(self.model)
        macs_before = estimate_macs(self.model, self.input_channels)
        new_model = self.create_pruned_model(retained_channels).to(DEVICE)
        self.copy_pruned_weights(new_model, pruned_indices)
        pruned_accuracy = self.evaluate_model(new_model)
        params_after = count_parameters(new_model)
        macs_after = estimate_macs(new_model, self.input_channels)
        self.model = new_model
        return {
            "method": method,
            "target_prune_rate": base_prune_rate,
            "base_accuracy": base_accuracy,
            "pruned_accuracy": pruned_accuracy,
            "parameter_reduction_pct": 100.0 * (params_before - params_after) / params_before,
            "mac_reduction_pct": 100.0 * (macs_before - macs_after) / macs_before,
            "params_before": params_before,
            "params_after": params_after,
            "macs_before": macs_before,
            "macs_after": macs_after,
            "pruning_time_sec": time.time() - started,
            "layer_names": json.dumps(layer_names),
            "layer_sensitivities": json.dumps(sensitivities),
            "layer_pruning_rates": json.dumps(pruning_rates),
            "retained_channels": json.dumps(retained_channels),
        }

    def create_pruned_model(self, retained_channels):
        if self.architecture == "SimpleCNN":
            return SimpleCNN(self.num_classes, retained_channels, self.input_channels)
        if self.architecture == "VGG11BN":
            return ConfigurableVGG11BN(self.num_classes, retained_channels, self.input_channels)
        if self.architecture == "ResNet20":
            return ConfigurableResNet20(self.num_classes, retained_channels, self.input_channels)
        raise ValueError(f"Unsupported architecture {self.architecture}")

    def copy_pruned_weights(self, new_model, indices_by_name):
        if self.architecture in ("SimpleCNN", "VGG11BN"):
            self.copy_sequential_weights(new_model, indices_by_name)
        elif self.architecture == "ResNet20":
            self.copy_resnet_weights(new_model, indices_by_name)

    def copy_sequential_weights(self, new_model, indices_by_name):
        old_info = self.original_model.get_prunable_layers()
        new_info = new_model.get_prunable_layers()
        previous_indices = None
        for old_item, new_item in zip(old_info, new_info):
            indices = indices_by_name[old_item["name"]]
            weight = old_item["module"].weight.data[indices]
            if previous_indices is not None:
                weight = weight[:, previous_indices]
            new_item["module"].weight.data.copy_(weight)
            copy_bn(old_item["bn_module"], new_item["bn_module"], indices)
            previous_indices = indices

        if self.architecture == "SimpleCNN":
            old = self.original_model
            new = new_model
            spatial = (32 // (2 ** len(old.conv_layers))) ** 2
            expanded = torch.cat([
                torch.arange(int(index) * spatial, (int(index) + 1) * spatial, device=previous_indices.device)
                for index in previous_indices
            ])
            new.fc1.weight.data.copy_(old.fc1.weight.data[:, expanded])
            new.fc1.bias.data.copy_(old.fc1.bias.data)
            new.fc2.load_state_dict(old.fc2.state_dict())
        else:
            new_model.classifier.weight.data.copy_(self.original_model.classifier.weight.data[:, previous_indices])
            new_model.classifier.bias.data.copy_(self.original_model.classifier.bias.data)

    def copy_resnet_weights(self, new_model, indices_by_name):
        old = self.original_model
        new_model.stem_conv.load_state_dict(old.stem_conv.state_dict())
        new_model.stem_bn.load_state_dict(old.stem_bn.state_dict())
        new_model.fc.load_state_dict(old.fc.state_dict())
        for i, (old_block, new_block) in enumerate(zip(old.blocks, new_model.blocks)):
            indices = indices_by_name[f"block{i + 1}.conv1"]
            new_block.conv1.weight.data.copy_(old_block.conv1.weight.data[indices])
            copy_bn(old_block.bn1, new_block.bn1, indices)
            new_block.conv2.weight.data.copy_(old_block.conv2.weight.data[:, indices])
            new_block.bn2.load_state_dict(old_block.bn2.state_dict())
            if isinstance(old_block.shortcut, nn.Sequential):
                new_block.shortcut.load_state_dict(old_block.shortcut.state_dict())

    def fine_tune(self, epochs=10, lr=0.0005):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        best_accuracy = -1.0
        best_state = None
        for epoch in range(epochs):
            self.model.train()
            progress = tqdm(self.train_loader, desc=f"Fine-tune {epoch + 1}/{epochs}", leave=False)
            for inputs, labels in progress:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(inputs), labels)
                loss.backward()
                optimizer.step()
            accuracy = self.evaluate_model(self.model)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_accuracy


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.results_path = self.output_dir / "full_grid_results.csv"

    def make_loaders(self, dataset_name, seed):
        trainset, testset, num_classes, input_channels = DatasetManager.get_dataset(
            dataset_name, self.args.data_root, not self.args.no_download
        )
        if self.args.quick:
            trainset = Subset(trainset, range(min(len(trainset), 1024)))
            testset = Subset(testset, range(min(len(testset), 512)))
        generator = torch.Generator().manual_seed(seed)
        common = {
            "batch_size": self.args.batch_size,
            "num_workers": self.args.workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
            "generator": generator,
            "persistent_workers": self.args.workers > 0,
        }
        train_loader = DataLoader(trainset, shuffle=True, **common)
        test_loader = DataLoader(testset, shuffle=False, **common)
        return train_loader, test_loader, num_classes, input_channels

    @staticmethod
    def evaluate(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                predictions = model(inputs).argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / max(total, 1)

    def train_baseline(self, model, train_loader, test_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.baseline_lr, weight_decay=1e-4)
        best_accuracy = -1.0
        best_state = None
        for epoch in range(self.args.baseline_epochs):
            model.train()
            progress = tqdm(train_loader, desc=f"Baseline {epoch + 1}/{self.args.baseline_epochs}", leave=False)
            for inputs, labels in progress:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            accuracy = self.evaluate(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return best_accuracy, best_state

    def save_results(self):
        pd.DataFrame(self.results).to_csv(self.results_path, index=False)

    def write_manifest(self):
        manifest = {
            "created": datetime.now().isoformat(),
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "configuration": vars(self.args),
            "dataset_references": {name: DATASET_REFERENCES[name] for name in self.args.datasets},
            "architecture_references": {name: ARCHITECTURE_REFERENCES[name] for name in self.args.architectures},
            "architecture_definitions": {
                "SimpleCNN": "Conv64-BN-ReLU-Pool, Conv128-BN-ReLU-Pool, Conv256-BN-ReLU-Pool, FC512, FC(classes).",
                "VGG11BN": "VGG-11 convolution layout with BatchNorm and adaptive global average pooling for 32x32 inputs.",
                "ResNet20": "CIFAR 6n+2 ResNet with n=3 blocks per stage and widths 16/32/64; only block-internal conv1 filters are structurally removed.",
            },
        }
        (self.output_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self):
        self.write_manifest()
        for seed in self.args.seeds:
            set_seed(seed)
            for dataset_name in self.args.datasets:
                train_loader, test_loader, num_classes, input_channels = self.make_loaders(dataset_name, seed)
                for architecture in self.args.architectures:
                    print(f"\n=== Seed {seed} | {dataset_name} | {architecture} ===")
                    baseline_model = build_model(architecture, num_classes, input_channels).to(DEVICE)
                    baseline_accuracy, baseline_state = self.train_baseline(
                        baseline_model, train_loader, test_loader
                    )
                    baseline_latency = benchmark_latency(
                        baseline_model, input_channels, self.args.latency_warmup, self.args.latency_repeats
                    )
                    for method in self.args.methods:
                        for rate in self.args.rates:
                            print(f"-- {method} | target={rate:.0%}")
                            model = build_model(architecture, num_classes, input_channels).to(DEVICE)
                            model.load_state_dict(baseline_state)
                            pruner = ComprehensivePruner(
                                model, train_loader, test_loader, architecture, num_classes, input_channels
                            )
                            try:
                                result = pruner.prune_model(
                                    method=method,
                                    base_prune_rate=rate,
                                    alpha=self.args.alpha,
                                    beta=self.args.beta,
                                    sensitivity_samples=self.args.sensitivity_samples,
                                    correlation_samples=self.args.correlation_samples,
                                )
                                final_accuracy = pruner.fine_tune(
                                    self.args.fine_tune_epochs, self.args.fine_tune_lr
                                )
                                pruned_latency = benchmark_latency(
                                    pruner.model,
                                    input_channels,
                                    self.args.latency_warmup,
                                    self.args.latency_repeats,
                                )
                                result.update({
                                    "status": "ok",
                                    "seed": seed,
                                    "dataset": dataset_name,
                                    "architecture": architecture,
                                    "baseline_training_accuracy": baseline_accuracy,
                                    "final_accuracy": final_accuracy,
                                    "accuracy_change": final_accuracy - baseline_accuracy,
                                    "recovery_efficiency_eta": (
                                        (final_accuracy - result["pruned_accuracy"])
                                        / max(result["parameter_reduction_pct"], 1e-8)
                                        * 100.0
                                    ),
                                    "baseline_latency_ms": baseline_latency,
                                    "pruned_latency_ms": pruned_latency,
                                    "speedup": baseline_latency / max(pruned_latency, 1e-12),
                                    "error": "",
                                })
                            except Exception as exc:
                                result = {
                                    "status": "error",
                                    "seed": seed,
                                    "dataset": dataset_name,
                                    "architecture": architecture,
                                    "method": method,
                                    "target_prune_rate": rate,
                                    "error": repr(exc),
                                }
                                print(f"ERROR: {exc!r}")
                            self.results.append(result)
                            self.save_results()
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    del baseline_model
        self.print_summary()

    def print_summary(self):
        frame = pd.DataFrame(self.results)
        successful = frame[frame["status"] == "ok"] if not frame.empty else frame
        if successful.empty:
            print("No successful experiment was recorded.")
            return
        columns = [
            "final_accuracy",
            "accuracy_change",
            "parameter_reduction_pct",
            "mac_reduction_pct",
            "speedup",
        ]
        summary = successful.groupby(
            ["dataset", "architecture", "method", "target_prune_rate"]
        )[columns].agg(["mean", "std"]).round(3)
        print("\n=== Aggregate summary ===")
        print(summary.to_string())
        summary.to_csv(self.output_dir / "full_grid_summary.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=["CIFAR10", "CIFAR100"])
    parser.add_argument("--architectures", nargs="+", choices=SUPPORTED_ARCHITECTURES, default=["ResNet20", "VGG11BN", "SimpleCNN"])
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./pruning_results/full_grid")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--baseline-lr", type=float, default=0.001)
    parser.add_argument("--fine-tune-lr", type=float, default=0.0005)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--sensitivity-samples", type=int, default=300)
    parser.add_argument("--correlation-samples", type=int, default=200)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use all five datasets and pruning rates 10%% through 50%% (very long run).",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small end-to-end pipeline check.")
    args = parser.parse_args()
    if args.full_grid:
        args.datasets = list(SUPPORTED_DATASETS)
        args.rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    if any(rate <= 0 or rate >= 1 for rate in args.rates):
        parser.error("all pruning rates must be strictly between 0 and 1")
    if args.quick:
        args.datasets = [args.datasets[0]]
        args.architectures = [args.architectures[0]]
        args.methods = ["E-HRAP"]
        args.rates = [0.3]
        args.seeds = [args.seeds[0]]
        args.baseline_epochs = 1
        args.fine_tune_epochs = 1
        args.sensitivity_samples = min(args.sensitivity_samples, 64)
        args.correlation_samples = min(args.correlation_samples, 64)
        args.latency_warmup = 2
        args.latency_repeats = 10
    return args


def main():
    args = parse_args()
    Path("./logs").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TeeLogger(f"./logs/execution_log_{timestamp}.txt")
    sys.stdout = logger
    try:
        print("Enhanced Hybrid Redundancy-Aware Pruning (E-HRAP)")
        print(f"Device: {DEVICE}")
        print(json.dumps(vars(args), indent=2))
        ExperimentRunner(args).run()
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
