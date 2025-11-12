import streamlit as st
import plotly.graph_objects as go


def show():
    """AI 個人化建議"""
    st.header("AI 個人化建議")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        generate_report = st.button("🤖 產出建議報告", key="generate_report", use_container_width=True)
    
    if generate_report or st.session_state.get("show_report", False):
        st.session_state.show_report = True
        
        st.subheader("🎯 您的 AI 建議報告")
        
        # 姿勢分析
        with st.expander("🏃 姿勢分析", expanded=True):
            st.write("""
您的深蹲有輕微的膝蓋內夾 (Knee Valgus) 狀況。建議您在課表中加入「臀中肌」與「核心穩定」訓練，例如：
- 彈力帶側走
- 鳥狗式
- 單腳臀橋
            """)
        
        # 體態建議
        with st.expander("💪 體態建議 (InBody)", expanded=True):
            st.write("""
您的骨骼肌重 (SMM) 35.1kg 表現良好，但體脂率 (PBF) 18.2% 略高於標準。

**建議:**
- 在飲食中適度提高蛋白質攝取 (每日 1.6-2.0g/kg)
- 在訓練後加入 20 分鐘的有氧運動
- 增加肌力訓練頻率至每週 4 次
            """)
        
        # 恢復與生活
        with st.expander("😴 恢復與生活", expanded=True):
            st.write("""
您昨晚的睡眠 7.5 小時品質不錯，但靜止心率 58bpm 相比上週平均 (55bpm) 略高，可能處於輕微疲勞。

**今日訓練建議:**
- 降低訓練強度 10%
- 注重動態伸展與放鬆
- 多攝取電解質與水分
            """)
        
        # 可視化圖表
        st.divider()
        st.subheader("📈 健康指標評分")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 雷達圖
            categories = ["肌肉量", "體脂率", "睡眠", "訓練頻率", "恢復度"]
            values = [8, 6, 8, 7, 6]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='您的狀態',
                fillcolor='rgba(0, 82, 204, 0.3)',
                line=dict(color='#0052cc')
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=400,
                title_text="整體健康狀態"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**各項指標評分:**")
            st.progress(0.8, text="肌肉量: 8/10")
            st.progress(0.6, text="體脂率: 6/10")
            st.progress(0.8, text="睡眠: 8/10")
            st.progress(0.7, text="訓練頻率: 7/10")
            st.progress(0.6, text="恢復度: 6/10")
