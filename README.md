# 醫療 ASR 台語翻譯 API

本專案已封裝為 Docker 鏡像，請按照以下步驟啟動：

### 1. 準備環境
請在根目錄建立 `.env` 檔案，並填入你的 API Key：
`GOOGLE_API_KEY=你的金鑰`

### 2. 建立 Docker 鏡像 (第一次執行較久)
docker build -t medical-asr-app .

### 3. 啟動服務
docker run -p 8000:8000 --env-file .env medical-asr-app

### 4. 測試 API
開啟瀏覽器前往：http://localhost:8000/docs
