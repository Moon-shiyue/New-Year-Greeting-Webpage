import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import json
import os
import random

# 初始化FastAPI应用
app = FastAPI(title="新年祝福生成API", version="1.0")

# 解决跨域问题（前端和后端端口不同，必须加，否则前端调用接口会报错）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，包括 origin: null（file:// 协议）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法（GET/POST/OPTIONS 等）
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["*"],  # 暴露所有响应头，让浏览器能读取到跨域头
)
# 加载模型和字符映射表（启动时只加载一次，提升推理速度）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 加载字符映射表
with open(os.path.join(BASE_DIR, "char2idx.json"), "r", encoding="utf-8") as f:
    char2idx = json.load(f)
with open(os.path.join(BASE_DIR, "idx2char.json"), "r", encoding="utf-8") as f:
    idx2char_json = json.load(f)
    idx2char = {int(k): v for k, v in idx2char_json.items()}  # 键转int

# 模型配置（和训练时一致，不能改）
SEQ_LEN = 10
MAX_LEN = 25
# 可选开头字（和之前一致，随机选择）
START_CHARS = ["新", "春", "阖", "龙", "福", "财", "万", "吉"]

# 加载ONNX模型
ort_sess = ort.InferenceSession(
    os.path.join(BASE_DIR, "new_year_model.onnx"),
    providers=["CPUExecutionProvider"]  # 纯CPU推理，无需GPU，适配所有设备
)

def generate_greeting(start_char: str = None) -> str:
    START_CHARS = ["新", "春", "阖", "福", "财", "吉", "龙", "年"]
    if start_char is None or start_char not in char2idx:
        start_char = random.choice(START_CHARS)
    start_idx = char2idx[start_char]
    input_seq = [start_idx]
    greeting = start_char

    for _ in range(18):  # 控制长度
        while len(input_seq) < SEQ_LEN:
            input_seq.insert(0, start_idx)
        input_arr = [input_seq[-SEQ_LEN:]]
        outputs = ort_sess.run(None, {"input": input_arr})
        logits = outputs[0][0]

        # 过滤逗号
        comma_idx = char2idx.get("，", -1)
        if comma_idx != -1:
            logits[comma_idx] = -1e9

        # 温度采样（核心：让句子更自然，不僵硬）
        temperature = 0.7
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_idx = np.random.choice(len(probs), p=probs)

        pred_char = idx2char.get(pred_idx, "")
        if not pred_char or not '\u4e00' <= pred_char <= '\u9fff':
            continue

        # 防连续重复
        if len(greeting) > 0 and pred_char == greeting[-1]:
            continue

        greeting += pred_char
        input_seq.append(pred_idx)

        if len(greeting) >= 16:
            break

    # 兜底保证完整
    if len(greeting) < 8:
        greeting += "万事顺遂"
    if greeting[-1] not in "。！":
        greeting += "。"
    return greeting

# 提供GET接口：/generate  调用即可生成祝福，支持指定开头字（可选参数）
@app.get("/generate", summary="生成新年祝福语")
async def generate(start_char: str = None):
    try:
        greeting = generate_greeting(start_char)
        return JSONResponse(content={"code": 200, "msg": "生成成功", "greeting": greeting})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")

# 根路径接口：测试后端是否运行成功
@app.get("/", summary="测试接口")
async def root():
    return {"code": 200, "msg": "新年祝福生成后端服务运行正常！", "tips": "访问/generate生成祝福"}

# 运行后端（直接执行该文件即可，无需额外命令）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",  # 允许本机/局域网/公网访问
        port=8000,       # 后端端口，可修改（如8888）
        reload=False      # 关闭热重载，修改代码后自动重启，开发时用
    )