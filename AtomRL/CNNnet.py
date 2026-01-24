import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class AtomRearrangementNet(nn.Module):
    def __init__(self, M):
        super(AtomRearrangementNet, self).__init__()
        
        # 卷积层部分
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入通道1，输出通道32，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入通道32，输出通道64，3x3卷积核
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输入通道64，输出通道128，3x3卷积核
        
        # 计算CNN输出的flatten后的维度
        self.flatten_size = 128 * M * M  # 最后一个卷积层的输出通道数 * 输入矩阵的大小
        
        # 全连接层部分
        # 预测方向d（2个类别）
        self.fc_d = nn.Linear(self.flatten_size, 2)
        
        # 预测n_x（选择的行/列个数）和P_x（选择的行/列的概率分布）
        self.fc_nx = nn.Linear(self.flatten_size, M)  # n_x 是一个长度为M的one-hot编码
        self.fc_Px = nn.Linear(self.flatten_size, M)  # P_x 是一个长度为M的one-hot编码
        
        # 预测n_y（移动维度选择的个数）和P_y1/P_y2（移动维度的起始点和终点概率）
        self.fc_ny = nn.Linear(self.flatten_size, M)  # n_y 是一个长度为M的one-hot编码
        self.fc_Py1 = nn.Linear(self.flatten_size, M)  # P_y1 是一个长度为M的one-hot编码
        self.fc_Py2 = nn.Linear(self.flatten_size, M)  # P_y2 是一个长度为M的one-hot编码

    def forward(self, x):
        # 输入x的维度是 (batch_size, 1, M, M)，即每个样本是一个M x M的矩阵
        x = F.relu(self.conv1(x))  # 卷积层1，ReLU激活
        x = F.relu(self.conv2(x))  # 卷积层2，ReLU激活
        x = F.relu(self.conv3(x))  # 卷积层3，ReLU激活
        
        # 将特征图展平（flatten）
        x = x.view(x.size(0), -1)  # (batch_size, 128 * M * M)
        
        # 预测方向d
        d = self.fc_d(x)  # 输出方向d的one-hot编码概率
        d = F.softmax(d, dim=1)  # 使用Softmax获得概率分布
        
        # 预测n_x（选择的行/列个数）和P_x（选择的行/列的概率分布）
        n_x = self.fc_nx(x)  # 输出选择的行/列个数（one-hot）
        P_x = self.fc_Px(x)  # 输出选择的行/列概率
        
        # 预测n_y（移动维度的选择个数）和P_y1/P_y2（移动维度的起始点和终点概率）
        n_y = self.fc_ny(x)  # 输出选择的移动维度个数（one-hot）
        P_y1 = self.fc_Py1(x)  # 输出移动维度起始点概率
        P_y2 = self.fc_Py2(x)  # 输出移动维度终点点概率
        
        # 使用Softmax将所有概率转化为分布
        n_x = F.softmax(n_x, dim=1)
        P_x = F.softmax(P_x, dim=1)
        n_y = F.softmax(n_y, dim=1)
        P_y1 = F.softmax(P_y1, dim=1)
        P_y2 = F.softmax(P_y2, dim=1)

        return d, n_x, P_x, n_y, P_y1, P_y2

class AtomRearrangementDataset(Dataset):
    def __init__(self, input_data, targets):
        self.input_data = input_data  # 输入数据 (batch_size, 1, M, M)
        self.targets = targets        # 目标标签，包含d, n_x, P_x, n_y, P_y1, P_y2

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.targets[idx]

# 训练过程
def train(model, train_loader, optimizer, epochs=10):
    model.train()  # 设置为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            # 清零梯度
            optimizer.zero_grad()

            # 将输入和标签移动到GPU（如果有的话）
            inputs, targets = inputs.cuda(), targets.cuda()

            # 前向传播
            d, n_x, P_x, n_y, P_y1, P_y2 = model(inputs)

            # 计算损失
            loss_d = criterion_d(d, targets['d'])
            loss_nx = criterion_nx(n_x, targets['n_x'])
            loss_ny = criterion_ny(n_y, targets['n_y'])
            loss_Px = criterion_Px(P_x, targets['P_x'])
            loss_Py1 = criterion_Py1(P_y1, targets['P_y1'])
            loss_Py2 = criterion_Py2(P_y2, targets['P_y2'])

            # 总损失
            total_loss = loss_d + loss_nx + loss_Px + loss_ny + loss_Py1 + loss_Py2

            # 反向传播
            total_loss.backward()

            # 优化
            optimizer.step()

            # 统计损失
            running_loss += total_loss.item()
            if i % 100 == 99:  # 每100个batch输出一次
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

def validate(model, val_loader):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存
        total_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            d, n_x, P_x, n_y, P_y1, P_y2 = model(inputs)
            loss_d = criterion_d(d, targets['d'])
            loss_nx = criterion_nx(n_x, targets['n_x'])
            loss_ny = criterion_ny(n_y, targets['n_y'])
            loss_Px = criterion_Px(P_x, targets['P_x'])
            loss_Py1 = criterion_Py1(P_y1, targets['P_y1'])
            loss_Py2 = criterion_Py2(P_y2, targets['P_y2'])
            total_loss += (loss_d + loss_nx + loss_Px + loss_ny + loss_Py1 + loss_Py2).item()
        print(f"Validation Loss: {total_loss / len(val_loader):.4f}")

if __name__ == '__main__':
    M = 16  # 假设阵列的大小是 M x M
    model = AtomRearrangementNet(M)

    input_data = torch.randn(1, 1, M, M)

    d, n_x, P_x, n_y, P_y1, P_y2 = model(input_data)

    print(f"方向预测 d: {d}")  # (batch_size, 2)
    print(f"行/列个数预测 n_x: {n_x}")  # (batch_size, M)
    print(f"行/列选择概率 P_x: {P_x}")  # (batch_size, M)
    print(f"移动维度个数预测 n_y: {n_y}")  # (batch_size, M)
    print(f"移动维度起始点概率 P_y1: {P_y1}")  # (batch_size, M)
    print(f"移动维度终点点概率 P_y2: {P_y2}")  # (batch_size, M)
    
    input_data = ...
    targets = ...
    
    train_dataset = AtomRearrangementDataset(input_data, targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion_d = nn.CrossEntropyLoss()  # 方向预测，二分类
    criterion_nx = nn.CrossEntropyLoss()  # 选择的行/列个数n_x，one-hot编码
    criterion_ny = nn.CrossEntropyLoss()  # 选择的移动维度个数n_y，one-hot编码
    criterion_Px = nn.CrossEntropyLoss()  # 选择的行/列概率P_x，softmax后的概率
    criterion_Py1 = nn.CrossEntropyLoss()  # 移动维度起始点的概率P_y1，softmax后的概率
    criterion_Py2 = nn.CrossEntropyLoss()  # 移动维度终点点的概率P_y2，softmax后的概率
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, optimizer, epochs=10)