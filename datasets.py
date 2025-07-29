import pickle
import torch
from torch.utils.data import Dataset
from Arguments import Arguments

Dataloaders = lambda args: MNISTDataLoaders(args, 'MNIST')


from torchquantum.dataset import MNIST
def MNISTDataLoaders(args, task):
    if task in ('MNIST', 'MNIST-10'):
        FAHION = False
    else:
        FAHION = True
    dataset = MNIST(
        root='data',
        train_valid_split_ratio=args.train_valid_split_ratio,
        center_crop=args.center_crop,
        resize=args.resize,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=args.digits_of_interest,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=FAHION,
        n_train_samples=None
        )
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            batch_size = args.batch_size
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = len(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True)

    return dataflow['train'], dataflow['valid'], dataflow['test']
class MyDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        img = self.data[idx]
        digit=self.labels[idx]
        return {"image": img, "digit": digit}


def reshape_to_target(tensor):
    import math
    import torch.nn as nn
    """
    将 (m, 1, n) 的 Tensor 转换为 (m, 1, 16)
    处理逻辑：
    1. 如果 n < 16：用0填充到16
    2. 如果 n > 16 且是完全平方数：转为2D后池化到4x4=16
    3. 其他情况：用1D池化降到16
    """
    m, _, n = tensor.shape

    if n == 16:
        return tensor

    # 情况1：n < 16，填充0
    if n < 16:
        pad_size = 16 - n
        return torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)

    # 情况2：n > 16 且是完全平方数
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        # 转为2D (假设可以合理reshape)
        try:
            # 先转为 (m, 1, sqrt_n, sqrt_n)
            tensor_2d = tensor.view(m, 1, sqrt_n, sqrt_n)
            # 自适应池化到 (4,4)
            pool = nn.AdaptiveAvgPool2d((4, 4))
            pooled = pool(tensor_2d)
            return pooled.view(m, 1, 16)
        except:
            # 如果reshape失败，降级到1D池化
            pass

    # 情况3：其他情况使用1D池化
    pool = nn.AdaptiveAvgPool1d(16)
    return pool(tensor)
def create_dataloader(args,train,test):
    from torch.utils.data.sampler import RandomSampler
    import numpy as np
    train_data = reshape_to_target(torch.from_numpy(train.iloc[:, :-1].values.astype(np.float32)).unsqueeze(1))
    train_labels = torch.from_numpy(train.iloc[:, -1].values.astype(np.int64))
    test_data = reshape_to_target(torch.from_numpy(test.iloc[:, :-1].values.astype(np.float32)).unsqueeze(1))
    test_labels = torch.from_numpy(test.iloc[:, -1].values.astype(np.int64))
    train_labels =torch.where(train_labels == -1, torch.tensor(0), train_labels)
    test_labels = torch.where(test_labels == -1, torch.tensor(0),test_labels)
    train_dateset = MyDataset(train_data, train_labels)
    test_dateset = MyDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dateset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dateset)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dateset,
        batch_size=args.batch_size,
        sampler=RandomSampler(test_dateset)
    )
    return train_loader, test_loader, test_loader
def mylinearly_separable(args,n_features):
    import pandas as pd
    train=pd.read_csv(f'data/linear/linearly_separable_{n_features}d_train.csv',header=None)
    test=pd.read_csv(f'data/linear/linearly_separable_{n_features}d_test.csv',header=None)
    return create_dataloader(args, train, test)