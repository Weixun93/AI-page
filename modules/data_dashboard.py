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


def show(project_root):
    """數據整合 - 健康地圖儀表板"""
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
                    time.sleep(1.5) # 模擬載入時間
                st.toast("✅ 資料同步完成！")
    
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
                st.session_state.sleep_data.update({
                    'totalHours': new_total_sleep,
                    'deepHours': new_deep_sleep,
                    'remHours': new_rem_sleep,
                    'lightHours': new_light_sleep,
                    'sleepScore': new_sleep_score,
                    'date': time.strftime('%Y-%m-%d')
                })
                st.success("✅ 睡眠數據已更新！")
        
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
                st.session_state.vitals_data.update({
                    'restingHeartRateBpm': new_resting_hr,
                    'heartRateVariabilityMs': new_hrv,
                    'date': time.strftime('%Y-%m-%d')
                })
                st.success("✅ 心率數據已更新！")
        
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
                st.session_state.activity_data.update({
                    'todayCaloriesBurnt': new_calories_burnt,
                    'todayCaloriesGoal': new_calories_goal,
                    'weeklyTotalDistanceKm': new_weekly_distance
                })
                st.success("✅ 活動數據已更新！")

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
            with st.spinner("正在分析 InBody 數據..."):
                time.sleep(2)  # 模擬分析時間
                
                # 模擬從圖片中提取數據 (實際應用中需要 OCR)
                # 這裡使用隨機數據來模擬
                mock_inbody_data = {
                    'date': time.strftime('%Y-%m-%d'),
                    'weightKg': np.random.uniform(65, 85),
                    'skeletalMuscleMassKg': np.random.uniform(30, 40),
                    'bodyFatPercentage': np.random.uniform(15, 25),
                    'bmi': np.random.uniform(20, 28)
                }
                
                st.session_state.inbody_data = mock_inbody_data
                st.success("✅ InBody 數據分析完成！")
                st.rerun()  # 重新載入頁面以顯示數據
    else:
        # 顯示 InBody 數據
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
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="體重 (kg)",
                value=f"{current_inbody['weightKg']:.1f}",
                delta=f"{inbody_diff['weightKg']:.1f}"
            )
        
        with col2:
            st.metric(
                label="骨骼肌 (kg)",
                value=f"{current_inbody['skeletalMuscleMassKg']:.1f}",
                delta=f"{inbody_diff['skeletalMuscleMassKg']:.1f}"
            )
        
        with col3:
            st.metric(
                label="體脂率 (%)",
                value=f"{current_inbody['bodyFatPercentage']:.1f}",
                delta=f"{inbody_diff['bodyFatPercentage']:.1f}"
            )
        
        with col4:
            st.metric(
                label="BMI",
                value=f"{current_inbody['bmi']:.1f}"
            )
        
        # 重新上傳按鈕
        if st.button("🔄 重新上傳 InBody 數據"):
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
            
            if activity['type'] == "跑步":
                st.write(f"📍 距離: {activity['distanceKm']} 公里")

    # ==================== PDF 導出按鈕 ====================
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Export PDF Report", key="export_pdf", width='stretch'):
            with st.spinner("Generating PDF report..."):
                pdf_data = generate_health_report_pdf(mock_data)
                
                # 創建下載鏈接
                b64_pdf = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="health_report_{mock_data["userName"]}_{time.strftime("%Y%m%d_%H%M%S")}.pdf" target="_blank">📥 Download PDF Report</a>'
                
                st.success("✅ PDF report generated successfully!")
                st.markdown(href, unsafe_allow_html=True)


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