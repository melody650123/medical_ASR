# Step 1: 使用官方 Python 輕量版作為基底
FROM python:3.10-slim

# Step 2: 安裝系統層級的工具 (librosa 讀取 mp3 必須用到 ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 3: 設定容器內的工作目錄
WORKDIR /app

# Step 4: 複製環境清單並安裝套件
# 先複製 requirements 可以利用 Docker 快取機制，加速之後的編譯
COPY requirements.txt .

# 在這行之前加入升級 pip 的指令
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Step 5: 複製專案所有程式碼到容器中
COPY . .

# Step 6: 宣告容器要對外開放的埠號 (與 uvicorn 一致)
EXPOSE 8000

# Step 7: 啟動指令 (使用 uvicorn 執行你的 FastAPI)
# --host 0.0.0.0 是關鍵，這樣外部才能連進容器
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]