# New-Year-Greeting-Webpage
通过AI写的一个小的新年祝福网页

# 使用流程
1.生成模型
先通过new_year_generator.py文件生成训练模型（当然，也可以直接用文件夹里训练好的）

2.安装 cloudflared
Windows：下载 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
重命名为 cloudflared.exe

3.将下载的cloudflared通过终端打开
终端进入 cloudflared.exe 所在目录，执行
cloudflared.exe tunnel --url http://localhost:8000

4.获取公网地址
终端输出类似 https://xxxx.trycloudflare.com
复制该地址

5.更新前端 API_URL
修改 frontend 文件夹index.html 中：
const API_URL = "https://xxxx.trycloudflare.com/generate";

6.运行
确保上述操作正常运行后，点开 new_year_web 文件夹中的 backend 文件夹，运行 main.py

7.之后就可以通过网页进行访问啦~

# 注意
该模型很小，仅是笨蛋大学生练手之作，所以生成的祝福词是不连贯且问题很多（发给亲友可能被打）
