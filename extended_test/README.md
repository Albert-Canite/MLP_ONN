# 扩展测试

这个目录把 CIFAR-10 的扩展实验拆成了几份独立脚本，分别对应你要的 3 类工作：

1. `train_hardware_models.py`
   使用与现有主逻辑一致的硬件因素思路训练 CIFAR-10：
   输入加噪声，卷积/全连接层改为量化版本，反向梯度使用低比特并注入噪声。
   支持 `ResNet-18` 和 `VGG-11`。

2. `train_software_models.py`
   在纯软件环境下训练 CIFAR-10，不加入任何硬件因素。
   同样支持 `ResNet-18` 和 `VGG-11`。

3. `evaluate_inference_modes.py`
   将训练得到的 4 个模型分别放到两种环境下推理：
   纯理想软件环境；
   硬件环境（前向量化 + 输入噪声）。

## 文件说明

- `common.py`: 公共配置、路径、checkpoint 读写
- `data_utils.py`: CIFAR-10 数据集与噪声变换
- `models.py`: ResNet-18 / VGG-11 构建，以及硬件层替换
- `train_hardware_models.py`: 训练硬件模型
- `train_software_models.py`: 训练软件模型
- `evaluate_inference_modes.py`: 统一评估 4 个模型在 2 种推理环境下的精度差异

## 使用方式

先训练纯软件模型：

```bash
python 扩展测试/train_software_models.py --arch all --epochs 30
```

再训练硬件模型：

```bash
python 扩展测试/train_hardware_models.py --arch all --epochs 30
```

最后统一做推理评估：

```bash
python 扩展测试/evaluate_inference_modes.py
```

## 输出位置

训练好的模型和汇总结果会放在：

`扩展测试/checkpoints/`

默认会生成：

- `software_resnet18_cifar10.pth`
- `software_vgg11_cifar10.pth`
- `hardware_resnet18_cifar10.pth`
- `hardware_vgg11_cifar10.pth`
- `inference_comparison.json`

## 当前实现假设

- `ResNet-18` 和 `VGG-11` 使用 `torchvision` 标准结构，最后分类头改成 10 类。
- 硬件推理环境重点模拟前向链路，因此推理时不再施加反向梯度噪声。
- 纯软件训练得到的模型在硬件环境推理前，会先做一次量化统计校准。
