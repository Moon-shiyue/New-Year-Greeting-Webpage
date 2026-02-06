import torch
import json
import os

# 注意：这里的NewYearModel要和你训练时的模型类完全一致！
class NewYearModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# 【修改1】跳过重新处理语料，直接加载训练时保存的字符映射表
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
char2idx_path = os.path.join(BASE_DIR, "char2idx_train.json")
idx2char_train_path = os.path.join(BASE_DIR, "idx2char_train.json")

# 检查训练时的映射表是否存在
if not os.path.exists(char2idx_path) or not os.path.exists(idx2char_train_path):
    raise FileNotFoundError("请先把训练时保存的char2idx_train.json、idx2char_train.json放到backend文件夹下！")

# 加载训练时的字符映射表
with open(char2idx_path, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
with open(idx2char_train_path, "r", encoding="utf-8") as f:
    idx2char_json = json.load(f)
    idx2char = {int(k): v for k, v in idx2char_json.items()}

# 【关键】vocab_size和训练模型完全一致（237）
vocab_size = len(char2idx)
seq_len = 10  # 和训练时的序列长度完全一致，不能改！

# 2. 加载训练好的pth模型，转CPU（后端推理用CPU足够）
device = torch.device("cpu")
model = NewYearModel(vocab_size).to(device)
# 加载pth模型（请确保new_year_model.pth在backend文件夹下）
model.load_state_dict(torch.load("new_year_model.pth", map_location=device))
model.eval()  # 评估模式，关闭dropout

# 3. 转换为ONNX格式（适配低版本PyTorch，删除不支持的参数和动态维度）
dummy_input = torch.LongTensor([[0]*seq_len]).to(device)  # 示例输入，形状[1, seq_len]（固定为[1,10]）
onnx_path = "new_year_model.onnx"
# 导出ONNX模型（删除use_legacy_export和dynamic_axes，适配低版本PyTorch）
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],  # 输入名，后端推理时要对应
    output_names=["output", "hidden"],  # 输出名
    opset_version=12,  # 适配低版本onnxruntime，不要改太高
    export_params=True,  # 必须为True，导出训练好的权重（低版本PyTorch可显式指定）
    do_constant_folding=True  # 优化常量折叠，提升后端推理速度
)


# 4. 保存（复用）字符映射表（供后端main.py使用，和训练时一致）
with open("char2idx.json", "w", encoding="utf-8") as f:
    json.dump(char2idx, f, ensure_ascii=False, indent=2)
with open("idx2char.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx2char.items()}, f, ensure_ascii=False, indent=2)

print("模型转换完成！后端文件已生成：")
print(f"1. ONNX模型：{onnx_path}")
print("2. 字符映射表：char2idx.json、idx2char.json")
print(f"✅ 字符表大小：{vocab_size}（和训练模型完全一致，无形状不匹配）")
# 验证文件是否生成
for file in [onnx_path, "char2idx.json", "idx2char.json"]:
    if os.path.exists(file):
        print(f"✅ {file} 生成成功")
    else:
        print(f"❌ {file} 生成失败")