import torch
import torch.nn as nn
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader

# 1.预处理
# 读取文本
# 按照utf-8读出，且文本中的换行符更改为逗号
with open("new_year_web/new_year_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().replace("\n", "，")
chars = list(set(text))  # 提取所有唯一字符
char2idx = {c: i for i, c in enumerate(chars)}  # 字符→索引
idx2char = {i: c for i, c in enumerate(chars)}  # 索引→字符
vocab_size = len(chars)  # 字符表大小
seq_len = 10  # 序列长度：用前10个字预测第11个，可调（如8/12）
step = 1  # 步长
inputs, targets = [], []

# 构建训练样本
for i in range(0, len(text) - seq_len, step):
    inputs.append([char2idx[c] for c in text[i:i + seq_len]])
    targets.append(char2idx[text[i + seq_len]])
inputs = torch.LongTensor(inputs)
targets = torch.LongTensor(targets)

# 划分训练集（小数据量无需测试集）
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 批次大小，可调

import json

with open("new_year_web/backend/char2idx_train.json", "w", encoding="utf-8") as f:
    json.dump(char2idx, f, ensure_ascii=False, indent=2)

with open("new_year_web/backend/idx2char_train.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx2char.items()}, f, ensure_ascii=False, indent=2)

print("训练时的字符映射表已保存：char2idx_train.json、idx2char_train.json")


# 2.搭建模型
class NewYearModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super(NewYearModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 字符嵌入
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # 输出层→预测下一个字符

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, hidden


# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动用GPU/CPU
model = NewYearModel(vocab_size).to(device)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率
# 损失函数
criterion = nn.CrossEntropyLoss()

# 3.训练模型
epochs = 200
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    if (epoch + 1) % 20 == 0:  # 每20轮打印一次损失
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
# 保存模型（可选，下次直接加载不用再训练）
torch.save(model.state_dict(), "new_year_web/backend/new_year_model.pth")

# 4.生成祝福语
def generate_greeting(start_char, max_len=20):
    """
    start_char: 开头字符（如"新""春"）
    max_len: 祝福语最大长度
    """
    model.eval()
    if start_char not in char2idx:
        start_char = "新"  # 默认开头
    # 初始化输入
    input_seq = [char2idx[start_char]]
    hidden = None
    greeting = start_char  # 最终生成的祝福语
    with torch.no_grad():
        for _ in range(max_len - 1):

            x = torch.LongTensor(input_seq).reshape(1, -1).to(device)  # 移除最后一个1
            output, hidden = model(x, hidden)
            # 随机选择（避免每次生成完全一样，temperature调随机性）
            output = output / 0.8
            pred_idx = torch.multinomial(torch.softmax(output, dim=1), num_samples=1).item()
            pred_char = idx2char[pred_idx]
            greeting += pred_char
            # 更新输入序列（保留最后seq_len个字符）
            input_seq.append(pred_idx)
            if len(input_seq) > seq_len:
                input_seq = input_seq[1:]
            # 遇到句号/逗号停止（可选）
            if pred_char in ["。", "，"] and len(greeting) > 5:
                break
    return greeting
# 测试生成：输入不同开头字
print("生成结果：")
for start in ["新", "春", "阖", "龙"]:
    print(generate_greeting(start))