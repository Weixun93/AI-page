import streamlit as st
import json
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 從環境變數獲取 Gemini API Key
GEMINI_API_KEY_2 = os.getenv('GEMINI_API_KEY_2')
if not GEMINI_API_KEY_2:
    st.error("❌ 找不到 GEMINI_API_KEY_2 環境變數。請檢查 .env 文件是否存在且包含正確的 API 金鑰。")
    st.stop()

# 配置Gemini API
genai.configure(api_key=GEMINI_API_KEY_2)

def analyze_inbody_file(file_bytes, file_type):
    """使用Gemini API分析InBody文件並提取關鍵數值"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """
請仔細分析這份InBody身體成分分析報告，提取以下關鍵數值：

- 身高 (height)：單位為cm
- 體重 (weight)：單位為kg
- 體脂肪率 (body_fat_percentage)：單位為%
- 骨骼肌重量 (skeletal_muscle_mass)：單位為kg
- BMI：身體質量指數

請以JSON格式返回結果，格式如下：
{
"height": 數值或null,
"weight": 數值或null,
"body_fat_percentage": 數值或null,
"skeletal_muscle_mass": 數值或null,
"bmi": 數值或null
}

如果找不到某個數值，請設為null。
只返回JSON，不要其他文字。
"""
        
        # 設置mime type
        if file_type in ['jpg', 'jpeg']:
            mime_type = "image/jpeg"
        elif file_type == 'png':
            mime_type = "image/png"
        elif file_type == 'pdf':
            mime_type = "application/pdf"
        else:
            return None
        
        # 創建文件part
        file_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(file_bytes).decode()
        }
        
        response = model.generate_content([prompt, file_part])
        
        # 清理響應文本
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        # 解析JSON
        data = json.loads(text)
        return data
        
    except Exception as e:
        st.error(f"分析InBody數據時出錯：{e}")
        return None


def load_mock_data(project_root):
    """從 JSON 檔案讀取模擬資料"""
    json_path = os.path.join(project_root, "mock_data.json")
    
    if not os.path.exists(json_path):
        st.error(f"找不到 mock_data.json 檔案，路徑：{json_path}")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"讀取 JSON 文件時出錯：{e}")
        return None


def save_data_to_json(project_root, mock_data):
    """將更新的數據保存到 JSON 檔案"""
    json_path = os.path.join(project_root, "mock_data.json")
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存 JSON 文件時出錯：{e}")
        return False


def generate_ai_health_recommendations(mock_data):
    """
    使用 Gemini API 根據所有健康數據生成個人化建議
    """
    try:
        # 準備數據摘要
        sleep_data = st.session_state.sleep_data
        vitals_data = st.session_state.vitals_data
        activity_data = st.session_state.activity_data
        inbody_data = st.session_state.inbody_data

        # 構建分析提示
        prompt = f"""
請根據以下健康數據，為用戶提供全面的個人化健康建議：

**睡眠數據：**
- 總睡眠時數：{sleep_data['totalHours']} 小時
- 睡眠分數：{sleep_data['sleepScore']}/100
- 深度睡眠：{sleep_data['deepHours']} 小時
- REM 睡眠：{sleep_data['remHours']} 小時
- 淺度睡眠：{sleep_data['lightHours']} 小時

**心率與恢復數據：**
- 靜息心率：{vitals_data['restingHeartRateBpm']} BPM
- 心率變異性：{vitals_data['heartRateVariabilityMs']} ms

**活動數據：**
- 今日消耗卡路里：{activity_data['todayCaloriesBurnt']} kcal
- 每日目標卡路里：{activity_data['todayCaloriesGoal']} kcal
- 每週總距離：{activity_data['weeklyTotalDistanceKm']} km

**身體成分數據：**
{f"- 身高：{inbody_data['heightCm']} cm" if inbody_data and inbody_data.get('heightCm') else "- 身高：未檢測"}
{f"- 體重：{inbody_data['weightKg']} kg" if inbody_data and inbody_data.get('weightKg') else "- 體重：未檢測"}
{f"- 體脂肪率：{inbody_data['bodyFatPercentage']}%" if inbody_data and inbody_data.get('bodyFatPercentage') else "- 體脂肪率：未檢測"}
{f"- 骨骼肌重量：{inbody_data['skeletalMuscleMassKg']} kg" if inbody_data and inbody_data.get('skeletalMuscleMassKg') else "- 骨骼肌重量：未檢測"}
{f"- BMI：{inbody_data['bmi']}" if inbody_data and inbody_data.get('bmi') else "- BMI：未檢測"}

請提供以下內容的建議：
1. 整體健康狀態評估
2. 睡眠改善建議
3. 運動與活動建議
4. 營養與飲食建議
5. 具體可行的行動計劃

請用繁體中文回答，保持專業且鼓勵性的語氣，提供具體的建議和目標。
"""

        # 使用 Gemini API 生成建議
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"AI 分析時發生錯誤: {str(e)}"


def generate_health_report_pdf(mock_data):
    """生成健康報告PDF"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # 自定義樣式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1  # 居中
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=20
    )
    
    normal_style = styles['Normal']
    
    story = []
    
    # 標題
    story.append(Paragraph("Health Dashboard Report", title_style))
    story.append(Paragraph(f"Generated for: {mock_data['userName']}", subtitle_style))
    story.append(Paragraph(f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # InBody 數據表格
    if st.session_state.inbody_data:
        current_inbody = st.session_state.inbody_data
        
        # 計算差異 (與歷史數據比較)
        if len(mock_data['inbody']['history']) > 0:
            last_inbody = mock_data['inbody']['history'][-1]
            inbody_diff = {
                'weightKg': current_inbody['weightKg'] - last_inbody['weightKg'],
                'skeletalMuscleMassKg': current_inbody['skeletalMuscleMassKg'] - last_inbody['skeletalMuscleMassKg'],
                'bodyFatPercentage': current_inbody['bodyFatPercentage'] - last_inbody['bodyFatPercentage']
            }
        else:
            inbody_diff = {'weightKg': 0, 'skeletalMuscleMassKg': 0, 'bodyFatPercentage': 0}
        
        story.append(Paragraph("InBody Body Composition", subtitle_style))
        
        inbody_data = [
            ['Metric', 'Current Value', 'Change'],
            ['Weight (kg)', f"{current_inbody['weightKg']:.1f}", f"{inbody_diff['weightKg']:.1f}"],
            ['Skeletal Muscle (kg)', f"{current_inbody['skeletalMuscleMassKg']:.1f}", f"{inbody_diff['skeletalMuscleMassKg']:.1f}"],
            ['Body Fat (%)', f"{current_inbody['bodyFatPercentage']:.1f}", f"{inbody_diff['bodyFatPercentage']:.1f}"],
            ['BMI', f"{current_inbody['bmi']:.1f}", '-']
        ]
        
        inbody_table = Table(inbody_data)
        inbody_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(inbody_table)
        story.append(Spacer(1, 20))
    
    # 睡眠數據
    current_sleep = st.session_state.sleep_data
    story.append(Paragraph("Sleep & Recovery Metrics", subtitle_style))
    
    sleep_data = [
        ['Metric', 'Value'],
        ['Total Sleep Hours', f"{current_sleep['totalHours']} hours"],
        ['Sleep Score', f"{current_sleep['sleepScore']}/100"],
        ['Deep Sleep', f"{current_sleep['deepHours']} hours"],
        ['REM Sleep', f"{current_sleep['remHours']} hours"],
        ['Light Sleep', f"{current_sleep['lightHours']} hours"]
    ]
    
    sleep_table = Table(sleep_data)
    sleep_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(sleep_table)
    story.append(Spacer(1, 20))
    
    # 心率數據
    current_vitals = st.session_state.vitals_data
    story.append(Paragraph("Heart Rate & Recovery", subtitle_style))
    
    vitals_data = [
        ['Metric', 'Value'],
        ['Resting Heart Rate', f"{current_vitals['restingHeartRateBpm']} BPM"],
        ['Heart Rate Variability', f"{current_vitals['heartRateVariabilityMs']} ms"]
    ]
    
    vitals_table = Table(vitals_data)
    vitals_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(vitals_table)
    story.append(Spacer(1, 20))
    
    # 圓餅圖 - 睡眠結構
    story.append(Paragraph("Sleep Structure Analysis", subtitle_style))
    
    # 創建圓餅圖
    fig, ax = plt.subplots(figsize=(6, 4))
    sleep_labels = ['Deep Sleep', 'REM Sleep', 'Light Sleep']
    sleep_values = [current_sleep['deepHours'], current_sleep['remHours'], current_sleep['lightHours']]
    colors_pie = ['#FF9999', '#66B2FF', '#99FF99']
    
    ax.pie(sleep_values, labels=sleep_labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax.axis('equal')
    ax.set_title('Last Night Sleep Structure')
    
    # 保存圖表到buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # 添加圖片到PDF
    img = Image(buf)
    img.drawHeight = 3*inch
    img.drawWidth = 4*inch
    story.append(img)
    
    # 活動數據
    story.append(Spacer(1, 20))
    story.append(Paragraph("Activity Summary", subtitle_style))
    
    activity_data = st.session_state.activity_data
    
    activity_table_data = [
        ['Metric', 'Value'],
        ['Today Calories Burned', f"{activity_data['todayCaloriesBurnt']} kcal"],
        ['Daily Goal', f"{activity_data['todayCaloriesGoal']} kcal"],
        ['Weekly Distance', f"{activity_data['weeklyTotalDistanceKm']} km"]
    ]
    
    activity_table = Table(activity_table_data)
    activity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(activity_table)
    
    # 生成PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def show(project_root):
    """顯示數據儀表板的主要函數"""
    mock_data = load_mock_data(project_root)
    if mock_data is None:
        st.stop()

    # 初始化 session state 用於存儲用戶輸入的數據
    if 'inbody_data' not in st.session_state:
        st.session_state.inbody_data = None
    if 'sleep_data' not in st.session_state:
        st.session_state.sleep_data = mock_data['sleep']['lastNight'].copy()
    if 'vitals_data' not in st.session_state:
        st.session_state.vitals_data = mock_data['vitals']['weeklyHistory'][-1].copy()
    if 'activity_data' not in st.session_state:
        st.session_state.activity_data = mock_data['activity'].copy()

    st.header(f"👤 {mock_data['userName']} 的健康儀表板")

    # ==================== 1. 數據輸入控制面板 ====================
    st.subheader("📝 數據管理")
    
    tab_sync, tab_manual = st.tabs(["🔄 自動同步", "✏️ 手動輸入"])
    
    with tab_sync:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.session_state.inbody_data:
                st.caption(f"✅ InBody: {st.session_state.inbody_data['date']}")
            else:
                st.caption("❌ InBody: 未上傳數據")
        with col2:
            st.caption(f"✅ 睡眠數據: {st.session_state.sleep_data['date']}")
        with col3:
            if st.button("🔄 同步所有數據"):
                with st.spinner("正在從您的裝置同步最新資料..."):
                    # 重新載入 JSON 文件數據
                    updated_data = load_mock_data(project_root)
                    if updated_data:
                        # 更新 session state 為最新的文件數據
                        st.session_state.sleep_data = updated_data['sleep']['lastNight'].copy()
                        st.session_state.vitals_data = updated_data['vitals']['weeklyHistory'][-1].copy()
                        st.session_state.activity_data = updated_data['activity'].copy()
                        
                    time.sleep(1.5) # 模擬載入時間
                st.toast("✅ 資料同步完成！已載入最新數據")
                st.rerun()  # 重新載入頁面以顯示最新數據
    
    with tab_manual:
        st.write("手動輸入或更新您的健康數據：")
        
        # 睡眠數據手動輸入
        with st.expander("😴 睡眠數據", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_total_sleep = st.number_input("總睡眠時數 (小時)", 
                                                min_value=0.0, max_value=24.0, 
                                                value=float(st.session_state.sleep_data['totalHours']),
                                                step=0.5)
                new_deep_sleep = st.number_input("深度睡眠 (小時)", 
                                               min_value=0.0, max_value=24.0, 
                                               value=float(st.session_state.sleep_data['deepHours']),
                                               step=0.1)
                new_rem_sleep = st.number_input("REM 睡眠 (小時)", 
                                              min_value=0.0, max_value=24.0, 
                                              value=float(st.session_state.sleep_data['remHours']),
                                              step=0.1)
            with col2:
                new_light_sleep = st.number_input("淺度睡眠 (小時)", 
                                                min_value=0.0, max_value=24.0, 
                                                value=float(st.session_state.sleep_data['lightHours']),
                                                step=0.1)
                new_sleep_score = st.slider("睡眠分數", 0, 100, 
                                          int(st.session_state.sleep_data['sleepScore']))
            
            if st.button("💾 保存睡眠數據"):
                # 更新 session state
                st.session_state.sleep_data.update({
                    'totalHours': new_total_sleep,
                    'deepHours': new_deep_sleep,
                    'remHours': new_rem_sleep,
                    'lightHours': new_light_sleep,
                    'sleepScore': new_sleep_score,
                    'date': time.strftime('%Y-%m-%d')
                })
                
                # 同步到 JSON 文件
                mock_data['sleep']['lastNight'] = st.session_state.sleep_data.copy()
                if save_data_to_json(project_root, mock_data):
                    st.success("✅ 睡眠數據已更新並同步到文件！")
                    st.rerun()  # 重新載入頁面以顯示更新
                else:
                    st.error("❌ 數據保存失敗")
        
        # 心率數據手動輸入
        with st.expander("❤️ 心率數據", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_resting_hr = st.number_input("靜息心率 (BPM)", 
                                               min_value=40, max_value=120, 
                                               value=int(st.session_state.vitals_data['restingHeartRateBpm']))
            with col2:
                new_hrv = st.number_input("心率變異性 (ms)", 
                                        min_value=10, max_value=200, 
                                        value=int(st.session_state.vitals_data['heartRateVariabilityMs']))
            
            if st.button("💾 保存心率數據"):
                # 更新 session state
                st.session_state.vitals_data.update({
                    'restingHeartRateBpm': new_resting_hr,
                    'heartRateVariabilityMs': new_hrv,
                    'date': time.strftime('%Y-%m-%d')
                })
                
                # 同步到 JSON 文件 - 更新最新記錄並添加到歷史記錄
                mock_data['vitals']['weeklyHistory'][-1] = st.session_state.vitals_data.copy()
                if save_data_to_json(project_root, mock_data):
                    st.success("✅ 心率數據已更新並同步到文件！")
                    st.rerun()  # 重新載入頁面以顯示更新
                else:
                    st.error("❌ 數據保存失敗")
        
        # 活動數據手動輸入
        with st.expander("🏃 活動數據", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_calories_burnt = st.number_input("今日消耗卡路里", 
                                                   min_value=0, max_value=5000, 
                                                   value=int(st.session_state.activity_data['todayCaloriesBurnt']))
                new_calories_goal = st.number_input("每日目標卡路里", 
                                                  min_value=500, max_value=5000, 
                                                  value=int(st.session_state.activity_data['todayCaloriesGoal']))
            with col2:
                new_weekly_distance = st.number_input("每週總距離 (km)", 
                                                    min_value=0.0, max_value=200.0, 
                                                    value=float(st.session_state.activity_data['weeklyTotalDistanceKm']),
                                                    step=0.1)
            
            if st.button("💾 保存活動數據"):
                # 更新 session state
                st.session_state.activity_data.update({
                    'todayCaloriesBurnt': new_calories_burnt,
                    'todayCaloriesGoal': new_calories_goal,
                    'weeklyTotalDistanceKm': new_weekly_distance
                })
                
                # 同步到 JSON 文件
                mock_data['activity'] = st.session_state.activity_data.copy()
                if save_data_to_json(project_root, mock_data):
                    st.success("✅ 活動數據已更新並同步到文件！")
                    st.rerun()  # 重新載入頁面以顯示更新
                else:
                    st.error("❌ 數據保存失敗")

        # 新增活動記錄功能
        with st.expander("📝 新增活動記錄", expanded=True):
            st.write("記錄您的新活動：")
            
            col1, col2 = st.columns(2)
            with col1:
                activity_type = st.selectbox("活動類型", 
                                           ["跑步", "騎自行車", "游泳", "瑜伽", "重量訓練", "其他"],
                                           key="activity_type")
                activity_duration = st.number_input("持續時間 (分鐘)", 
                                                  min_value=1, max_value=300, 
                                                  value=30, step=5)
                activity_calories = st.number_input("消耗卡路里", 
                                                  min_value=0, max_value=1000, 
                                                  value=200, step=10)
            with col2:
                activity_distance = st.number_input("距離 (公里)", 
                                                  min_value=0.0, max_value=50.0, 
                                                  value=5.0 if activity_type == "跑步" else 0.0, 
                                                  step=0.1)
                activity_date = st.date_input("活動日期", value=pd.to_datetime('today'))
                activity_notes = st.text_input("備註 (選填)", placeholder="例如：晨跑、公園跑步等")
            
            if st.button("➕ 添加活動記錄", key="add_activity"):
                # 創建新活動記錄
                new_activity = {
                    'type': activity_type,
                    'date': activity_date.strftime('%Y-%m-%d'),
                    'durationMinutes': activity_duration,
                    'caloriesBurnt': activity_calories,
                    'distanceKm': activity_distance if activity_distance > 0 else None,
                    'notes': activity_notes if activity_notes else None
                }
                
                # 添加到活動數據的 recentActivities 列表
                if 'recentActivities' not in st.session_state.activity_data:
                    st.session_state.activity_data['recentActivities'] = []
                
                st.session_state.activity_data['recentActivities'].insert(0, new_activity)  # 插入到最前面
                
                # 只在活動日期是今天時才更新今日消耗卡路里
                today_date = pd.to_datetime('today').strftime('%Y-%m-%d')
                if activity_date.strftime('%Y-%m-%d') == today_date:
                    st.session_state.activity_data['todayCaloriesBurnt'] += activity_calories
                
                # 更新總距離
                st.session_state.activity_data['weeklyTotalDistanceKm'] += activity_distance
                
                # 同步到 JSON 文件
                mock_data['activity'] = st.session_state.activity_data.copy()
                if save_data_to_json(project_root, mock_data):
                    st.success(f"✅ 活動記錄已添加！{activity_type} {activity_duration}分鐘，消耗{activity_calories}卡路里")
                    st.rerun()  # 重新載入頁面以顯示新記錄
                else:
                    st.error("❌ 活動記錄保存失敗")

    st.divider()

    # ==================== 2. InBody 指標 (需要上傳紙本資料) ====================
    st.subheader("📊 InBody 身體成分")
    
    if st.session_state.inbody_data is None:
        # 顯示上傳區域
        st.info("📄 請上傳您的 InBody 檢測紙本資料以查看身體成分分析")
        
        uploaded_file = st.file_uploader(
            "上傳 InBody 檢測結果 (支援 JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            key="inbody_upload"
        )
        
        if uploaded_file:
            with st.spinner("正在使用AI分析 InBody 數據..."):
                # 讀取文件
                file_bytes = uploaded_file.read()
                file_type = uploaded_file.type.split('/')[-1].lower()
                
                # 使用Gemini分析文件
                inbody_data = analyze_inbody_file(file_bytes, file_type)
                
                if inbody_data:
                    # 將數據存儲到session state
                    extracted_data = {
                        'date': time.strftime('%Y-%m-%d'),
                        'heightCm': inbody_data.get('height'),
                        'weightKg': inbody_data.get('weight'),
                        'bodyFatPercentage': inbody_data.get('body_fat_percentage'),
                        'skeletalMuscleMassKg': inbody_data.get('skeletal_muscle_mass'),
                        'bmi': inbody_data.get('bmi')
                    }
                    
                    st.session_state.inbody_data = extracted_data
                    
                    # 同步到 JSON 文件 - 添加到歷史記錄
                    if 'inbody' not in mock_data:
                        mock_data['inbody'] = {'history': []}
                    
                    # 添加新記錄到歷史
                    mock_data['inbody']['history'].append(extracted_data)
                    
                    if save_data_to_json(project_root, mock_data):
                        st.success("✅ InBody 數據分析完成並已同步到文件！")
                    else:
                        st.success("✅ InBody 數據分析完成！(文件同步失敗)")
                    
                    st.rerun()  # 重新載入頁面以顯示數據
                else:
                    st.error("❌ 無法分析InBody數據，請檢查文件是否清晰可讀。")
    else:
        # 顯示 InBody 數據
        current_inbody = st.session_state.inbody_data
        
        # 顯示提取的數值
        st.subheader("📊 提取的身體成分數據")
        
        # 創建列來顯示數值
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if current_inbody.get('heightCm'):
                st.metric(label="身高 (cm)", value=f"{current_inbody['heightCm']:.1f}")
            else:
                st.metric(label="身高 (cm)", value="未檢測")
        
        with col2:
            if current_inbody.get('weightKg'):
                st.metric(label="體重 (kg)", value=f"{current_inbody['weightKg']:.1f}")
            else:
                st.metric(label="體重 (kg)", value="未檢測")
        
        with col3:
            if current_inbody.get('bodyFatPercentage'):
                st.metric(label="體脂肪率 (%)", value=f"{current_inbody['bodyFatPercentage']:.1f}")
            else:
                st.metric(label="體脂肪率 (%)", value="未檢測")
        
        with col4:
            if current_inbody.get('skeletalMuscleMassKg'):
                st.metric(label="骨骼肌重量 (kg)", value=f"{current_inbody['skeletalMuscleMassKg']:.1f}")
            else:
                st.metric(label="骨骼肌重量 (kg)", value="未檢測")
        
        with col5:
            if current_inbody.get('bmi'):
                st.metric(label="BMI", value=f"{current_inbody['bmi']:.1f}")
            else:
                st.metric(label="BMI", value="未檢測")
        
        st.divider()
        
        # 舊的顯示邏輯（如果需要比較）
        # 計算差異 (與歷史數據比較)
        if len(mock_data['inbody']['history']) > 0 and current_inbody.get('weightKg') and current_inbody.get('skeletalMuscleMassKg') and current_inbody.get('bodyFatPercentage'):
            last_inbody = mock_data['inbody']['history'][-1]
            inbody_diff = {
                'weightKg': current_inbody['weightKg'] - last_inbody['weightKg'],
                'skeletalMuscleMassKg': current_inbody['skeletalMuscleMassKg'] - last_inbody['skeletalMuscleMassKg'],
                'bodyFatPercentage': current_inbody['bodyFatPercentage'] - last_inbody['bodyFatPercentage']
            }
        else:
            inbody_diff = {'weightKg': 0, 'skeletalMuscleMassKg': 0, 'bodyFatPercentage': 0}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if current_inbody.get('weightKg'):
                st.metric(
                    label="體重 (kg)",
                    value=f"{current_inbody['weightKg']:.1f}",
                    delta=f"{inbody_diff['weightKg']:.1f}"
                )
            else:
                st.metric(label="體重 (kg)", value="未檢測")
        
        with col2:
            if current_inbody.get('skeletalMuscleMassKg'):
                st.metric(
                    label="骨骼肌 (kg)",
                    value=f"{current_inbody['skeletalMuscleMassKg']:.1f}",
                    delta=f"{inbody_diff['skeletalMuscleMassKg']:.1f}"
                )
            else:
                st.metric(label="骨骼肌 (kg)", value="未檢測")
        
        with col3:
            if current_inbody.get('bodyFatPercentage'):
                st.metric(
                    label="體脂率 (%)",
                    value=f"{current_inbody['bodyFatPercentage']:.1f}",
                    delta=f"{inbody_diff['bodyFatPercentage']:.1f}"
                )
            else:
                st.metric(label="體脂率 (%)", value="未檢測")
        
        with col4:
            if current_inbody.get('bmi'):
                st.metric(
                    label="BMI",
                    value=f"{current_inbody['bmi']:.1f}"
                )
            else:
                st.metric(label="BMI", value="未檢測")
        
        # 重新上傳按鈕和清除數據按鈕
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 重新上傳 InBody 數據"):
                st.session_state.inbody_data = None
                st.rerun()
        with col_btn2:
            if st.button("🗑️ 清除當前 InBody 數據"):
                st.session_state.inbody_data = None
                st.rerun()
        
        # InBody 趨勢圖 (如果有歷史數據)
        if len(mock_data['inbody']['history']) > 0:
            # 添加當前數據到歷史數據中進行顯示
            trend_data = mock_data['inbody']['history'] + [current_inbody]
            inbody_df = pd.DataFrame(trend_data)
            inbody_df['date'] = pd.to_datetime(inbody_df['date'])
            
            fig_inbody = px.line(
                inbody_df.melt(id_vars='date', value_vars=['weightKg', 'skeletalMuscleMassKg', 'bodyFatPercentage']),
                x="date",
                y="value",
                color="variable",
                title="身體組成趨勢圖",
                markers=True,
                labels={"date": "日期", "value": "數值", "variable": "指標"}
            )
            st.plotly_chart(fig_inbody, width='stretch')

    st.divider()
    
    # ==================== 3. 睡眠與核心指標 ====================
    st.subheader("😴 睡眠與心率")
    col1, col2 = st.columns(2)

    with col1:
        # 睡眠指標
        st.write("#### 睡眠品質")
        current_sleep = st.session_state.sleep_data
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(
                label="昨晚睡眠",
                value=f"{current_sleep['totalHours']} 小時"
            )
        with col_s2:
            st.metric(
                label="睡眠分數",
                value=f"{current_sleep['sleepScore']}"
            )
        
        # 睡眠圓餅圖
        sleep_labels = ['深度睡眠', 'REM 睡眠', '淺度睡眠']
        sleep_values = [current_sleep['deepHours'], current_sleep['remHours'], current_sleep['lightHours']]
        fig_sleep_pie = go.Figure(data=[go.Pie(
            labels=sleep_labels, 
            values=sleep_values, 
            hole=.4,
            pull=[0.05, 0.05, 0.05]
        )])
        fig_sleep_pie.update_layout(title_text="昨晚睡眠結構", height=300, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_sleep_pie, width='stretch')

    with col2:
        # 心率指標
        st.write("#### 恢復指標")
        current_vitals = st.session_state.vitals_data
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric(
                label="靜止心率 (BPM)",
                value=f"{current_vitals['restingHeartRateBpm']}"
            )
            st.caption("代表基礎心肺健康")
        
        with col_v2:
            hrv_value = current_vitals['heartRateVariabilityMs']
            hrv_color = "normal" if hrv_value > 60 else "inverse"
            hrv_delta = "恢復良好" if hrv_value > 60 else "注意疲勞"
            
            st.metric(
                label="心率變異 (HRV)",
                value=f"{hrv_value} ms",
                delta=hrv_delta,
                delta_color=hrv_color
            )
            st.caption("越高代表恢復越好")

        # 睡眠與心率趨勢圖
        vitals_df = pd.DataFrame(mock_data['vitals']['weeklyHistory'])
        vitals_df['date'] = pd.to_datetime(vitals_df['date'])
        
        fig_vitals = px.line(
            vitals_df.melt(id_vars='date', value_vars=['restingHeartRateBpm', 'heartRateVariabilityMs']),
            x="date",
            y="value",
            color="variable",
            title="每週恢復趨勢",
            markers=True,
            labels={"date": "日期", "value": "數值", "variable": "指標"}
        )
        st.plotly_chart(fig_vitals, width='stretch')

    
    st.divider()
    
    # ==================== 4. 今日卡路里進度 ====================
    st.subheader("🔥 今日卡路里消耗")
    activity_data = st.session_state.activity_data
    cal_progress = activity_data['todayCaloriesBurnt'] / activity_data['todayCaloriesGoal']
    
    # 確保進度條不超過 1.0
    cal_progress = min(cal_progress, 1.0) 
    
    st.progress(cal_progress, text=f"{activity_data['todayCaloriesBurnt']} / {activity_data['todayCaloriesGoal']} 大卡")
    
    st.divider()
    
    # ==================== 5. 近期活動 ====================
    st.subheader("🏃 近期活動")
    activity_data = st.session_state.activity_data
    for activity in activity_data['recentActivities']:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{activity['type']}**")
            
            with col2:
                st.write(f"📅 {activity['date']}")
            
            with col3:
                st.write(f"⏱️ {activity['durationMinutes']} 分鐘")
            
            with col4:
                st.write(f"🔥 {activity['caloriesBurnt']} 大卡")
            
            # 顯示距離，如果距離大於0
            if activity.get('distanceKm') and activity['distanceKm'] > 0:
                st.write(f"📍 距離: {activity['distanceKm']} 公里")
            
            # 顯示備註，如果有的話
            if activity.get('notes'):
                st.write(f"📝 {activity['notes']}")

    # ==================== PDF 導出和 AI 建議按鈕 ====================
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🤖 生成 AI 健康建議", key="ai_recommendations", width='stretch'):
            with st.spinner("正在分析您的健康數據並生成個人化建議..."):
                ai_recommendations = generate_ai_health_recommendations(mock_data)
                
                if ai_recommendations and not ai_recommendations.startswith("AI 分析時發生錯誤"):
                    st.success("✅ AI 健康建議生成完成！")
                    
                    # 顯示 AI 建議 - 滿版顯示
                    st.subheader("🧠 AI 個人化健康建議")
                    st.markdown(ai_recommendations)
                    
                    # 提供下載建議的選項
                    st.download_button(
                        label="📥 下載 AI 建議",
                        data=ai_recommendations,
                        file_name=f"ai_health_recommendations_{mock_data['userName']}_{time.strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        key="download_ai_recommendations"
                    )
                else:
                    st.error(f"❌ {ai_recommendations}")
    
    with col2:
        if st.button("📄 Export PDF Report", key="export_pdf", width='stretch'):
            with st.spinner("Generating PDF report..."):
                pdf_data = generate_health_report_pdf(mock_data)
                
                # 創建下載鏈接
                b64_pdf = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="health_report_{mock_data["userName"]}_{time.strftime("%Y%m%d_%H%M%S")}.pdf" target="_blank">📥 Download PDF Report</a>'
                
                st.success("✅ PDF report generated successfully!")
                st.markdown(href, unsafe_allow_html=True)