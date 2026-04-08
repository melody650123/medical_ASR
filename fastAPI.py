import os
import io
import time
import torch
import librosa
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, AutoProcessor
from google import genai
from dotenv import load_dotenv

# --- 1. 環境配置與安全檢查 ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise ValueError("❌ 錯誤：找不到 GEMINI_API_KEY，請確認 .env 檔案是否存在")

app = FastAPI(title="醫療台語 ASR 翻譯系統 (MMS + Gemini)")

# --- 2. 初始化模型 (全域載入) ---
MODEL_ID = "facebook/mms-1b-all"
TARGET_LANG = "nan"  # 台語閩南語代碼
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

print(f"📦 正在載入 MMS 模型至 {DEVICE}...")

# 修正：確保載入時處理詞表大小不一的問題，防止 Index out of bounds
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    target_lang=TARGET_LANG,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# 載入台語專用的 Adapter
model.load_adapter(TARGET_LANG)
processor.tokenizer.set_target_lang(TARGET_LANG)
model.eval()

# 初始化 Gemini Client
client = genai.Client(api_key=GEMINI_KEY)
print("✅ 模型與 API 預熱完成，伺服器準備就緒。")

# --- 3. 核心 API 路由 ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"📥 接收到檔案: {file.filename}")
        start_overall = time.time()
        
        # 1. 讀取與預處理音訊
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="檔案為空")

        try:
            # 強制轉換為單聲道 (mono) 與 16000Hz 採樣率
            audio, _ = librosa.load(io.BytesIO(content), sr=16000, mono=True)
        except Exception as e:
            print(f"❌ Librosa 報錯: {e}")
            return {"status": "error", "message": "音訊解碼失敗，請確認環境已安裝 ffmpeg"}

        # 2. MMS ASR 辨識
        print("🎙️ ASR 推論中...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        
        # 如果是 GPU 則使用半精度優化效能
        if DEVICE == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
        
        # 關鍵修正：使用 clamp 防止 CUDA Index 越界崩潰
        vocab_size = logits.shape[-1]
        ids = torch.argmax(logits, dim=-1)[0]
        ids = torch.clamp(ids, min=0, max=vocab_size - 1)
        
        transcription = processor.decode(ids)
        asr_time = time.time() - start_overall
        print(f"📝 辨識結果: {transcription}")

        # 3. Gemini 醫療翻譯
        print("🤖 呼叫 Gemini 進行校正翻譯...")
        prompt = (
            "你是一位精通台語與醫學術語的專業翻譯官。背景為「醫院診間問診」。"
            "輸入是由 ASR 產生的台羅拼音 (Tai-lo)，可能包含錯誤發音。"
            "任務：將醫病對話的「台羅拼音 (Tai-lo)」翻譯成精確的「台灣繁體中文」，格式僅需提供中文翻譯，不要有其他多餘。"
            f"\n待翻譯內容：{transcription}"
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview', 
                contents=prompt
            )
            translation = response.text.strip()
        except Exception as e:
            print(f"❌ Gemini API 報錯: {e}")
            translation = "翻譯服務暫時無法使用"

        total_latency = time.time() - start_overall
        
        # 4. 回傳 JSON 結果
        print(f"✅ 處理完成，總耗時: {total_latency:.2f}s")
        return {
            "status": "success",
            "data": {
                "transcription_tailo": transcription,
                "translation_zh": translation,
            },
            "performance": {
                "asr_duration": f"{asr_time:.2f}s",
                "total_duration": f"{total_latency:.2f}s"
            }
        }

    except Exception as e:
        # 在終端機印出完整錯誤路徑，方便 Debug
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")

# --- 4. 啟動進入點 ---
if __name__ == "__main__":
    # 使用 0.0.0.0 讓同區域網路內的組員也能連線
    uvicorn.run(app, host="0.0.0.0", port=8000)