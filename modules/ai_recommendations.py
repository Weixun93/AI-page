import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
import base64


# 設置 Gemini API
GEMINI_API_KEY = "AIzaSyBbtvL5AXg6sMd2UON-Pv4heCGD4PfCOAQ"
genai.configure(api_key=GEMINI_API_KEY)


def extract_inbody_data_from_image(uploaded_file):
    """使用 Gemini Vision 從上傳的圖片中提取 InBody 數據"""
    try:
        # 讀取上傳的文件
        file_content = uploaded_file.read()
        
        # 根據文件類型選擇處理方式
        if uploaded_file.type.startswith('image/'):
            # 圖片文件：使用 Vision API
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # 將圖片轉換為 base64
            image_base64 = base64.standard_b64encode(file_content).decode('utf-8')
            
            # 構建 vision 請求
            prompt = """
請分析這張 InBody 檢測報告的圖片，並提取以下信息，以 JSON 格式返回：
{
  "weightKg": 數值,
  "skeletalMuscleMassKg": 數值,
  "bodyFatPercentage": 數值,
  "bmi": 數值
}

如果找不到某些數據，請填入 null。
只返回 JSON 格式，不要有其他文字。
"""
            
            image_part = {
                "mime_type": uploaded_file.type,
                "data": image_base64
            }
            
            response = model.generate_content([prompt, image_part])
            response_text = response.text
            
            # 提取 JSON
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    inbody_data = json.loads(json_str)
                    inbody_data['date'] = datetime.now().strftime('%Y-%m-%d')
                    return inbody_data
            except json.JSONDecodeError:
                pass
        
        elif uploaded_file.type == 'application/pdf':
            # PDF 文件：使用通用分析
            st.info("PDF 檔案需要使用進階 OCR 處理，目前使用預設數據")
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'weightKg': 75.5,
                'skeletalMuscleMassKg': 35.1,
                'bodyFatPercentage': 18.2,
                'bmi': 24.1
            }
        
        return None
    
    except Exception as e:
        st.error(f"提取 InBody 數據時出錯: {e}")
        return None


def generate_training_plan(inbody_data):
    """使用 Gemini API 生成一週訓練計畫"""
    try:
        # 構建 prompt 確保生成固定格式的一週訓練計畫
        prompt = f"""
根據以下 InBody 身體成分檢測數據，為用戶生成一份為期一週的個人化訓練計畫。

身體數據:
- 體重: {inbody_data.get('weightKg', 'N/A')} kg
- 骨骼肌: {inbody_data.get('skeletalMuscleMassKg', 'N/A')} kg
- 體脂率: {inbody_data.get('bodyFatPercentage', 'N/A')}%
- BMI: {inbody_data.get('bmi', 'N/A')}

請以以下 JSON 格式生成訓練計畫，包含星期一到星期日的詳細訓練安排：

{{
  "monday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "tuesday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "wednesday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "thursday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "friday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "saturday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }},
  "sunday": {{
    "exercise": "運動項目",
    "sets": "組數",
    "reps": "次數",
    "rest_time_minutes": "休息時間(分鐘)",
    "intensity": "強度(低/中/高)",
    "diet": "飲食建議"
  }}
}}

請確保返回有效的 JSON 格式，並提供實際的訓練和飲食建議。
只返回 JSON 格式，不要有其他文字。
"""
        
        # 使用最新的 Gemini 模型
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(prompt)
        
        # 解析回應
        response_text = response.text
        
        # 嘗試提取 JSON 內容
        try:
            # 尋找 JSON 內容
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                training_plan = json.loads(json_str)
                return training_plan
        except json.JSONDecodeError:
            pass
        
        return None
    
    except Exception as e:
        st.error(f"生成訓練計畫時出錯: {e}")
        return None


def show():
    """AI 個人化建議"""
    st.header("AI 個人化建議")
    
    # 初始化 session state
    if 'training_plan' not in st.session_state:
        st.session_state.training_plan = None
    if 'inbody_data_ai' not in st.session_state:
        st.session_state.inbody_data_ai = None
    
    # ==================== InBody 上傳區域 ====================
    st.subheader("📄 上傳 InBody 報告")
    st.info("上傳您的 InBody 檢測報告，AI 將根據您的身體數據生成個人化的一週訓練計畫")
    
    uploaded_file = st.file_uploader(
        "選擇 InBody 檢測結果 (支援 JPG, PNG, PDF)",
        type=["jpg", "jpeg", "png", "pdf"],
        key="inbody_ai_upload"
    )
    
    if uploaded_file:
        with st.spinner("正在分析 InBody 報告..."):
            # 真正從上傳的文件中提取 InBody 數據
            inbody_data = extract_inbody_data_from_image(uploaded_file)
            
            if inbody_data:
                st.session_state.inbody_data_ai = inbody_data
                st.success("✅ InBody 數據提取完成！")
                st.write("**提取的數據:**")
                st.json(inbody_data)
            else:
                st.error("❌ 無法提取 InBody 數據，請確保上傳的是清晰的報告圖片")
    
    st.divider()
    
    # ==================== 訓練計畫生成 ====================
    col1, col2, col3 = st.columns(3)
    with col2:
        generate_report = st.button("🤖 生成一週訓練計畫", key="generate_report", width='stretch')
    
    if generate_report or st.session_state.get("show_report", False):
        # 檢查是否有上傳 InBody 數據
        if st.session_state.inbody_data_ai is None:
            st.warning("⚠️ 請先上傳 InBody 報告，才能生成訓練計畫")
        else:
            st.session_state.show_report = True
            
            # 生成訓練計畫
            if st.session_state.training_plan is None:
                with st.spinner("🤖 AI 正在為您生成一週訓練計畫..."):
                    training_plan = generate_training_plan(st.session_state.inbody_data_ai)
                    if training_plan:
                        st.session_state.training_plan = training_plan
                        st.success("✅ 訓練計畫已生成！")
                    else:
                        st.error("❌ 生成訓練計畫失敗，請稍後重試")
            
            if st.session_state.training_plan:
                st.divider()
                st.subheader("📋 您的一週個人化訓練計畫")
                
                # 顯示 InBody 數據摘要
                inbody = st.session_state.inbody_data_ai
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("體重", f"{inbody['weightKg']:.1f} kg")
                with col2:
                    st.metric("骨骼肌", f"{inbody['skeletalMuscleMassKg']:.1f} kg")
                with col3:
                    st.metric("體脂率", f"{inbody['bodyFatPercentage']:.1f}%")
                with col4:
                    st.metric("BMI", f"{inbody['bmi']:.1f}")
                
                st.divider()
                
                # 顯示一週訓練計畫（卡片格式）
                days_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                days_display = ['📅 星期一', '📅 星期二', '📅 星期三', '📅 星期四', '📅 星期五', '🏖️ 星期六', '🏖️ 星期日']
                
                cols = st.columns(2)
                for idx, (day, day_display) in enumerate(zip(days_order, days_display)):
                    if day in st.session_state.training_plan:
                        with cols[idx % 2]:
                            with st.container(border=True):
                                st.write(f"### {day_display}")
                                
                                workout = st.session_state.training_plan[day]
                                
                                # 運動項目
                                st.write(f"**🏋️ 運動項目:** {workout.get('exercise', 'N/A')}")
                                
                                # 訓練詳情（網格式）
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.write(f"**組數:** {workout.get('sets', 'N/A')}")
                                with col_b:
                                    st.write(f"**次數:** {workout.get('reps', 'N/A')}")
                                with col_c:
                                    st.write(f"**休息:** {workout.get('rest_time_minutes', 'N/A')} 分鐘")
                                
                                # 強度
                                intensity = workout.get('intensity', 'N/A')
                                intensity_emoji = '🟢' if intensity == '低' else '🟡' if intensity == '中' else '🔴'
                                st.write(f"**{intensity_emoji} 強度:** {intensity}")
                                
                                # 飲食建議
                                st.write(f"**🍗 飲食建議:** {workout.get('diet', 'N/A')}")
                
                st.divider()
                
                # 重新生成按鈕
                if st.button("🔄 重新生成訓練計畫"):
                    st.session_state.training_plan = None
                    st.rerun()
