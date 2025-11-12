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

    st.header(f"👤 {mock_data['userName']} 的健康儀表板")

    # ==================== 1. 模擬同步按鈕 (快速加分) ====================
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.caption(f"上次同步：InBody ({mock_data['inbody']['lastUpdated'].split('T')[0]})")
    with col2:
        st.caption(f"上次同步：Apple Watch ({mock_data['sleep']['lastNight']['date']})")
    with col3:
        if st.button("🔄 立即同步"):
            with st.spinner("正在從您的裝置同步最新資料..."):
                time.sleep(1.5) # 模擬載入時間
            st.toast("✅ 資料同步完成！")
    
    st.divider()

    # 從 history 讀取最新資料
    current_inbody = mock_data['inbody']['history'][-1]
    inbody_diff = mock_data['inbody']['diff']
    current_sleep = mock_data['sleep']['lastNight']
    current_vitals = mock_data['vitals']['weeklyHistory'][-1]

    # ==================== 2. InBody 指標 (含情境說明) ====================
    st.subheader("📊 InBody 身體成分")
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
    
    # InBody 趨勢圖 (立即執行)
    inbody_df = pd.DataFrame(mock_data['inbody']['history'])
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
    st.plotly_chart(fig_inbody, use_container_width=True)

    st.divider()
    
    # ==================== 3. 睡眠與核心指標 (含情境說明) ====================
    st.subheader("😴 睡眠與心率")
    col1, col2 = st.columns(2)

    with col1:
        # 睡眠指標
        st.write("#### 睡眠品質")
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
        
        # 睡眠圓餅圖 (立即執行)
        sleep_labels = ['深度睡眠', 'REM 睡眠', '淺度睡眠']
        sleep_values = [current_sleep['deepHours'], current_sleep['remHours'], current_sleep['lightHours']]
        fig_sleep_pie = go.Figure(data=[go.Pie(
            labels=sleep_labels, 
            values=sleep_values, 
            hole=.4,
            pull=[0.05, 0.05, 0.05]
        )])
        fig_sleep_pie.update_layout(title_text="昨晚睡眠結構", height=300, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_sleep_pie, use_container_width=True)

    with col2:
        # 心率指標
        st.write("#### 恢復指標")
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

        # 睡眠與心率趨勢圖 (立即執行)
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
        st.plotly_chart(fig_vitals, use_container_width=True)

    
    st.divider()
    
    # ==================== 4. 今日卡路里進度 (無變動) ====================
    st.subheader("🔥 今日卡路里消耗")
    cal_progress = mock_data['activity']['todayCaloriesBurnt'] / mock_data['activity']['todayCaloriesGoal']
    
    # 確保進度條不超過 1.0
    cal_progress = min(cal_progress, 1.0) 
    
    st.progress(cal_progress, text=f"{mock_data['activity']['todayCaloriesBurnt']} / {mock_data['activity']['todayCaloriesGoal']} 大卡")
    
    st.divider()
    
    # ==================== 5. 近期活動 (無變動) ====================
    st.subheader("🏃 近期活動")
    for activity in mock_data['activity']['recentActivities']:
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
        if st.button("📄 Export PDF Report", key="export_pdf", use_container_width=True):
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
    current_inbody = mock_data['inbody']['history'][-1]
    inbody_diff = mock_data['inbody']['diff']
    
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
    current_sleep = mock_data['sleep']['lastNight']
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
    current_vitals = mock_data['vitals']['weeklyHistory'][-1]
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
    
    activity_data = [
        ['Metric', 'Value'],
        ['Today Calories Burned', f"{mock_data['activity']['todayCaloriesBurnt']} kcal"],
        ['Daily Goal', f"{mock_data['activity']['todayCaloriesGoal']} kcal"],
        ['Weekly Distance', f"{mock_data['activity']['weeklyTotalDistanceKm']} km"]
    ]
    
    activity_table = Table(activity_data)
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