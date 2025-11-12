import streamlit as st
import sys
import os

# 獲取項目根目錄
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 將 modules 目錄加入 sys.path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "modules"))

# 導入各個頁面模組
from modules import pose_detection, home, data_dashboard, ai_recommendations, system_features, target_audience

# 設置頁面配置
st.set_page_config(
    page_title="Motiv A.I. - AI運動科學的未來",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 導航側邊欄
st.sidebar.title("🏋️ Motiv A.I.")
page = st.sidebar.radio(
    "選擇功能",
    ["首頁", "🎥 動作偵測", "數據整合", "AI 建議", "系統特色", "適用對象"]
)

# 根據選擇的頁面顯示內容
if page == "🎥 動作偵測":
    pose_detection.show()
elif page == "首頁":
    home.show()
elif page == "數據整合":
    data_dashboard.show(PROJECT_ROOT)
elif page == "AI 建議":
    ai_recommendations.show()
elif page == "系統特色":
    system_features.show()
elif page == "適用對象":
    target_audience.show()

# ============================================
# 頁腳
# ============================================
st.divider()
st.write("© 2025 Motiv A.I. 版權所有。")
