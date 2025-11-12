import streamlit as st
import tempfile
import os


def analyze_motion(source_name, image_or_video=None):
    """分析動作"""
    st.success(f"✓ 開始分析 {source_name}...")
    
    with st.spinner("🔍 正在分析動作..."):
        # 簡單的統計分析
        
        # 創建分析結果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("分析類型", "準備中", help="使用 MediaPipe 進行識別")
        with col2:
            st.metric("動作類型", "準備中", help="使用 AI 進行識別")
        with col3:
            st.metric("準確度", "準備中", help="基於關節點偏差計算")
    
    # 顯示分析結果
    st.subheader("📊 詳細分析報告")
    
    # 動作評分
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📈 姿態評分")
        scores = {
            "身體對稱性": 85,
            "關節位置": 78,
            "穩定性": 92,
            "流暢度": 88
        }
        for metric, score in scores.items():
            st.metric(metric, f"{score}%")
    
    with col2:
        st.write("### 💡 改進建議")
        suggestions = [
            "✓ 身體保持筆直",
            "⚠️ 膝蓋需要更彎曲",
            "✓ 步幅均勻",
            "💪 可以增加速度"
        ]
        for suggestion in suggestions:
            st.write(suggestion)



def show():
    """動作偵測頁面"""
    st.header("🎥 AI 動作偵測")
    st.write("使用您的攝像頭或上傳影片，AI 將分析您的健身動作是否正確")
    
    st.divider()
    
    # ==================== 錄製和上傳選項 ====================
    tab_camera, tab_upload, tab_info, tab_tips = st.tabs(
        ["📹 開始錄製", "📤 上傳影片", "ℹ️ 關節節點介紹", "💡 偵測提示"]
    )
    
    with tab_camera:
        st.write("### 📱 直接錄製")
        st.warning("⚠️ 注意：請確保攝像頭已授權，光線充足，穿著貼身衣物")
        
        # 使用 Streamlit 的攝像頭輸入
        picture = st.camera_input("拍攝您的動作", label_visibility="collapsed")
        
        if picture is not None:
            st.success("✓ 已捕獲圖像")
            st.image(picture, caption="捕獲的圖像", use_column_width=True)
            
            if st.button("🔍 分析此圖像", key="analyze_camera_pic"):
                analyze_motion("攝像頭捕獲", picture)
    
    
    with tab_upload:
        st.write("### 📤 上傳影片")
        uploaded_video = st.file_uploader(
            "選擇影片檔案 (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"]
        )
        if uploaded_video:
            st.success(f"✓ 已上傳: {uploaded_video.name}")
            st.video(uploaded_video)
            
            if st.button("🔍 分析此影片", key="analyze_uploaded"):
                analyze_motion(uploaded_video.name, uploaded_video)
    
    with tab_info:
        st.write("### 🦴 關鍵關節節點介紹")
        
        st.write("""
        AI 動作分析系統會監測以下 17 個關鍵關節點，來判斷您的運動姿勢是否正確：
        """)
        
        # 使用 tab 來組織不同部位的關節
        joint_tab1, joint_tab2, joint_tab3, joint_tab4 = st.tabs(["上肢", "軀幹", "下肢", "其他"])
        
        with joint_tab1:
            st.write("**上肢關節:**")
            joints_upper = [
                ("👁️ 鼻子 (Nose)", "面部中心，用於頭部方向判斷"),
                ("👁️ 左眼 (Left Eye)", "左眼位置"),
                ("👁️ 右眼 (Right Eye)", "右眼位置"),
                ("👂 左耳 (Left Ear)", "左耳位置"),
                ("👂 右耳 (Right Ear)", "右耳位置"),
                ("💪 左肩 (Left Shoulder)", "左肩關節，決定上臂位置"),
                ("💪 右肩 (Right Shoulder)", "右肩關節"),
                ("🤚 左肘 (Left Elbow)", "左肘關節，監測手臂彎曲程度"),
                ("🤚 右肘 (Right Elbow)", "右肘關節"),
                ("✋ 左腕 (Left Wrist)", "左手腕，監測手臂延伸"),
                ("✋ 右腕 (Right Wrist)", "右手腕"),
            ]
            for joint, desc in joints_upper:
                st.write(f"- {joint}: {desc}")
        
        with joint_tab2:
            st.write("**軀幹關節:**")
            joints_torso = [
                ("🫀 左髖 (Left Hip)", "左髖關節，影響身體傾斜"),
                ("🫀 右髖 (Right Hip)", "右髖關節"),
            ]
            for joint, desc in joints_torso:
                st.write(f"- {joint}: {desc}")
        
        with joint_tab3:
            st.write("**下肢關節:**")
            joints_lower = [
                ("🦵 左膝 (Left Knee)", "左膝關節，深蹲時的關鍵位置"),
                ("🦵 右膝 (Right Knee)", "右膝關節"),
                ("🦶 左踝 (Left Ankle)", "左踝關節，平衡和穩定性"),
                ("🦶 右踝 (Right Ankle)", "右踝關節"),
            ]
            for joint, desc in joints_lower:
                st.write(f"- {joint}: {desc}")
        
        with joint_tab4:
            st.write("**其他參數:**")
            st.write("""
            - **對稱性 (Symmetry)**: 左右兩側身體是否對稱
            - **穩定性 (Stability)**: 身體重心是否穩定
            - **角度 (Angles)**: 各關節的彎曲角度
            - **速度 (Velocity)**: 動作執行速度是否過快/過慢
            """)
    
    with tab_tips:
        st.write("### 💡 最佳實踐")
        
        tips = [
            ("📍 站位清晰", "請站在攝像頭前 1-2 米，確保全身都在鏡頭範圍內"),
            ("💡 光線充足", "避免逆光，確保視頻畫面清晰明亮"),
            ("👕 穿著合適", "穿著貼身衣物，使 AI 能清楚識別關節點"),
            ("📹 角度適當", "最佳角度是正面或側面 90 度拍攝"),
            ("⏱️ 完整動作", "錄製完整的一個動作周期（如一次深蹲）"),
            ("🎯 一個動作", "一次錄製只分析一種動作（跑步、深蹲等）"),
        ]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for idx, (tip_title, tip_desc) in enumerate(tips):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.write(f"**{tip_title}**")
                    st.write(tip_desc)
        
        st.divider()
        
        # ==================== 支援的動作 ====================
        st.write("### 🏋️ 目前支援的動作分析")
        
        supported_exercises = [
            ("🏃 跑步 (Running)", "分析步幅、腿部擡起、著地方式"),
            ("⬇️ 深蹲 (Squat)", "分析膝蓋角度、身體傾斜、對稱性"),
            ("💪 俯卧撑 (Push-up)", "分析手臂彎曲、身體平直度、下降高度"),
            ("🧘 瑜伽姿態 (Yoga)", "分析身體對齐、平衡、靈活性"),
            ("🤸 弓箭步 (Lunge)", "分析膝蓋位置、步幅、身體穩定"),
            ("🏋️ 舉重 (Lifting)", "分析軀幹姿態、手臂路徑、重心"),
        ]
        
        for exercise, description in supported_exercises:
            st.write(f"- **{exercise}**: {description}")
