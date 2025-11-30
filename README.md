# AI Health Dashboard

Motive AI
一個使用 Streamlit 構建的個人健康儀表板應用程序，提供姿勢檢測、身體成分分析和健康建議功能。
組員：邱奕翔、呂韋勳、林柏宇

## 功能特色

- 🏃 **姿勢檢測**: 使用 AI 分析運動姿勢並提供改進建議
- 📊 **健康儀表板**: 睡眠、心率、活動數據的可視化
- 🤖 **AI 健康建議**: 基於個人數據的智能健康建議
- 📄 **InBody 分析**: 上傳身體成分報告進行 AI 分析
- 📈 **數據追蹤**: 記錄和分析健康數據趨勢

## 環境設置

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 環境變數配置

1. 複製環境變數範本：
   ```bash
   cp .env.example .env
   ```

2. 編輯 `.env` 文件，填入您的 API 金鑰：
   ```
   GEMINI_API_KEY_2=your_actual_gemini_api_key_here
   ```

3. 獲取 Gemini API 金鑰：
   - 前往 [Google AI Studio](https://makersuite.google.com/app/apikey)
   - 創建新的 API 金鑰
   - 將金鑰複製到 `.env` 文件中

### 3. 運行應用程序

```bash
streamlit run app.py
```

應用程序將在 http://localhost:8501 上運行。

## 項目結構

```
├── app.py                 # 主應用程序入口
├── modules/
│   ├── data_dashboard.py  # 健康儀表板模組
│   ├── pose_detection.py  # 姿勢檢測模組
│   └── ...
├── mock_data.json         # 模擬數據文件
├── requirements.txt       # Python 依賴
├── .env.example          # 環境變數範本
└── .gitignore            # Git 忽略文件
```

## 安全注意事項

- 🔒 **API 金鑰安全**: API 金鑰已移至環境變數，不會提交到版本控制
- 🚫 **不要分享 .env 文件**: .env 文件已被 .gitignore 排除
- 🔑 **API 金鑰管理**: 請妥善保管您的 Gemini API 金鑰

## 技術棧

- **前端**: Streamlit
- **AI**: Google Gemini API
- **視覺處理**: OpenCV, MediaPipe
- **數據處理**: Pandas, NumPy
- **可視化**: Plotly, Matplotlib
- **PDF 生成**: ReportLab

## 授權

此項目僅供學習和個人使用。