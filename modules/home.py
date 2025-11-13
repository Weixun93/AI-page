import streamlit as st


def show():
    """首頁內容"""
    st.title("您的隨身 AI 運動科學教練")
    st.subheader("Motiv A.I. - AI運動科學的未來")
    
    st.write("""
    運用 AI 與電腦視覺，將專業教練的即時監控帶到每個人身邊，取代昂貴的設備與課程，推動全民健康、永續運動的未來。
    """)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🚀 開始您的科學化訓練", key="cta_hero", width='stretch'):
            st.info("前往『數據整合』頁面開始吧！")
    
    st.divider()
    
    # 核心功能區塊
    st.header("讓專業指導，無所不在")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.write("### ✓ AI 姿勢偵測與即時回饋")
            st.write("""
偵測跑步、深蹲等運動姿勢，當姿勢正確時標示綠色，錯誤則標示紅色。
並提供動作計次數、對稱性分析與穩定性指標。
            """)
    
    with col2:
        with st.container(border=True):
            st.write("### ✓ 多元健康數據整合")
            st.write("""
整合 InBody (體脂、骨骼肌量)、健檢 (心率、血壓、血糖) 
及穿戴式設備資料。自動生成個人化運動健康報告。
            """)
    
    with col3:
        with st.container(border=True):
            st.write("### ✓ 生成式 AI 個人化建議")
            st.write("""
將複雜數據轉化為清晰且可執行的建議。
根據姿勢分析生成動態建議，提供飲食與生活習慣建議。
            """)
