import streamlit as st
import tempfile
import os
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time
import subprocess
import platform
import logging

# 抑制 TensorFlow 和 MediaPipe 的日誌
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)


def calculate_angle(a, b, c):
    """
    計算三點所形成的夾角，b 為中心點
    a, b, c: (x, y)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    if denom == 0:
        return np.nan
    cosine = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def calculate_trunk_lean(shoulder, hip):
    """
    軀幹前傾角：Hip -> Shoulder 與「垂直向上」的夾角
    影像座標 y 軸向下為正，所以垂直向上是 (0, -1)
    """
    shoulder = np.array(shoulder)
    hip = np.array(hip)
    v = shoulder - hip           # hip -> shoulder
    vertical = np.array([0, -1.0])
    denom = (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-8)
    if denom == 0:
        return np.nan
    cosine = np.dot(v, vertical) / denom
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def record_from_webcam(output_video_path):
    """
    從本地攝像頭錄制視頻，即時顯示節點
    返回: (錄制成功與否, 錄制時長秒數)
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    connections = mp_pose.POSE_CONNECTIONS

    # 打開本地攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ 無法開啟攝像頭，請檢查攝像頭是否已連接")
        return False, 0

    # 設置攝像頭分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 設定影片輸出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, input_fps, (width, height))

    st.info("📹 攝像頭已啟動！")
    frame_placeholder = st.empty()
    timer_placeholder = st.empty()
    frame_count = 0
    start_time = time.time()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 鏡像翻轉便於自拍
                frame = cv2.flip(frame, 1)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    # 繪製骨架和節點
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, connections,
                        landmark_drawing_spec=pose_style
                    )

                # 添加狀態文字
                cv2.putText(
                    frame, f"Recording... Frame: {frame_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                # 寫入輸出影片
                out.write(frame)

                # 轉換為 RGB 以在 Streamlit 中顯示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, width='stretch')

                # 更新計時器
                elapsed_time = time.time() - start_time
                timer_placeholder.metric("⏱️ 錄製時長", f"{elapsed_time:.1f} 秒")

                # 檢查停止標誌
                if st.session_state.get('stop_recording', False):
                    break

                frame_count += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time
    return True, elapsed_time


def analyze_video_pose(video_path):
    """
    分析已錄制的視頻中的姿勢，返回分析數據
    """
    mp_pose = mp.solutions.pose
    smooth_buffer_size = 5
    elbow_R_buffer = deque(maxlen=smooth_buffer_size)
    wrist_R_buffer = deque(maxlen=smooth_buffer_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("無法開啟影片")

    # 取得影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    data_rows = []
    frame_idx = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    h, w, _ = frame.shape
                    lm = results.pose_landmarks.landmark

                    def get_xy(name):
                        p = lm[mp_pose.PoseLandmark[name].value]
                        return np.array([p.x * w, p.y * h])

                    # 右側關節座標
                    shoulder_R = get_xy("RIGHT_SHOULDER")
                    elbow_R = get_xy("RIGHT_ELBOW")
                    wrist_R = get_xy("RIGHT_WRIST")
                    hip_R = get_xy("RIGHT_HIP")
                    knee_R = get_xy("RIGHT_KNEE")
                    ankle_R = get_xy("RIGHT_ANKLE")

                    # 左側關節座標
                    shoulder_L = get_xy("LEFT_SHOULDER")
                    hip_L = get_xy("LEFT_HIP")
                    knee_L = get_xy("LEFT_KNEE")
                    ankle_L = get_xy("LEFT_ANKLE")

                    # 平滑化右手節點
                    elbow_R_buffer.append(elbow_R)
                    wrist_R_buffer.append(wrist_R)

                    if len(elbow_R_buffer) == smooth_buffer_size:
                        elbow_R = np.mean(elbow_R_buffer, axis=0)
                        wrist_R = np.mean(wrist_R_buffer, axis=0)

                    # 角度計算
                    right_elbow_angle = calculate_angle(shoulder_R, elbow_R, wrist_R)
                    right_shoulder_angle = calculate_angle(elbow_R, shoulder_R, hip_R)
                    right_hip_angle = calculate_angle(shoulder_R, hip_R, knee_R)
                    right_knee_angle = calculate_angle(hip_R, knee_R, ankle_R)

                    left_hip_angle = calculate_angle(shoulder_L, hip_L, knee_L)
                    left_knee_angle = calculate_angle(hip_L, knee_L, ankle_L)

                    # 軀幹前傾角
                    mid_shoulder = (
                        (shoulder_R[0] + shoulder_L[0]) / 2,
                        (shoulder_R[1] + shoulder_L[1]) / 2
                    )
                    mid_hip = (
                        (hip_R[0] + hip_L[0]) / 2,
                        (hip_R[1] + hip_L[1]) / 2
                    )
                    trunk_lean_deg = calculate_trunk_lean(mid_shoulder, mid_hip)

                    # 記錄數據
                    time_sec = frame_idx / input_fps
                    data_rows.append({
                        "frame": frame_idx,
                        "time": time_sec,
                        "right_elbow_angle": right_elbow_angle,
                        "right_shoulder_angle": right_shoulder_angle,
                        "right_hip_angle": right_hip_angle,
                        "right_knee_angle": right_knee_angle,
                        "left_hip_angle": left_hip_angle,
                        "left_knee_angle": left_knee_angle,
                        "trunk_lean_deg": trunk_lean_deg,
                        "right_ankle_x": float(ankle_R[0]),
                        "right_ankle_y": float(ankle_R[1]),
                        "left_ankle_x": float(ankle_L[0]),
                        "left_ankle_y": float(ankle_L[1]),
                    })

                # 更新進度條
                if total_frames > 0:
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"分析中... {frame_idx}/{total_frames} 幀")

                frame_idx += 1

    finally:
        cap.release()
        progress_bar.empty()
        status_text.empty()

    return data_rows, input_fps, width, height


def generate_pose_analysis_plots(data_rows):
    """
    生成姿勢分析圖表
    """
    if len(data_rows) == 0:
        return None, None, None, None, None

    times = [row["time"] for row in data_rows]

    # 1. 右膝角度隨時間變化
    right_knee_angles = [row["right_knee_angle"] for row in data_rows]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times, right_knee_angles, linewidth=2, color='#1f77b4')
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Right Knee Angle (deg)", fontsize=12)
    ax1.set_title("Right Knee Angle over Time", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2. 左右膝角度比較
    left_knee_angles = [row["left_knee_angle"] for row in data_rows]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(times, right_knee_angles, label="Right Knee", linewidth=2, color='#ff7f0e')
    ax2.plot(times, left_knee_angles, label="Left Knee", linewidth=2, color='#2ca02c')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Knee Angle (deg)", fontsize=12)
    ax2.set_title("Left vs Right Knee Angle over Time", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # 3. 軀幹前傾角度
    trunk_lean = [row["trunk_lean_deg"] for row in data_rows]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(times, trunk_lean, linewidth=2, color='#d62728')
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Trunk Lean (deg)", fontsize=12)
    ax3.set_title("Trunk Lean Angle over Time", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    # 4. 右腳踝垂直位移
    right_ankle_y = [row["right_ankle_y"] for row in data_rows]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(times, right_ankle_y, linewidth=2, color='#9467bd')
    ax4.set_xlabel("Time (s)", fontsize=12)
    ax4.set_ylabel("Right Ankle Y (pixel)", fontsize=12)
    ax4.set_title("Right Ankle Vertical Trajectory", fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    # 5. 右腳踝在畫面中的 2D 路徑
    right_ankle_x = [row["right_ankle_x"] for row in data_rows]
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.plot(right_ankle_x, right_ankle_y, linewidth=2, color='#8c564b')
    ax5.set_xlabel("Right Ankle X (pixel)", fontsize=12)
    ax5.set_ylabel("Right Ankle Y (pixel)", fontsize=12)
    ax5.set_title("Right Ankle 2D Trajectory", fontsize=14, fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig1, fig2, fig3, fig4, fig5


def display_analysis_results(data_rows):
    """
    顯示姿勢分析結果，包括圖表、統計和數據下載
    """
    if len(data_rows) == 0:
        st.warning("沒有分析數據可顯示")
        return

    # 顯示基本統計
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("總幀數", f"{len(data_rows)}", help="已分析的影片幀數")
    with col2:
        duration = data_rows[-1]["time"] if data_rows else 0
        st.metric("分析時間", f"{duration:.1f}秒", help="分析的持續時間")
    with col3:
        fps = len(data_rows) / duration if duration > 0 else 0
        st.metric("平均 FPS", f"{fps:.1f}", help="每秒處理幀數")

    # 生成圖表
    st.subheader("📊 姿勢分析圖表")
    fig1, fig2, fig3, fig4, fig5 = generate_pose_analysis_plots(data_rows)

    # 顯示圖表
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "右膝角度", "膝蓋比較", "軀幹傾斜", "腳踝垂直軌跡", "腳踝2D軌跡"
    ])

    with tab1:
        st.pyplot(fig1, width='stretch')
    with tab2:
        st.pyplot(fig2, width='stretch')
    with tab3:
        st.pyplot(fig3, width='stretch')
    with tab4:
        st.pyplot(fig4, width='stretch')
    with tab5:
        st.pyplot(fig5, width='stretch')

    # 計算平均值和統計
    st.subheader("📈 姿勢統計摘要")

    # 提取數據
    right_knee_angles = [row["right_knee_angle"] for row in data_rows if not np.isnan(row["right_knee_angle"])]
    left_knee_angles = [row["left_knee_angle"] for row in data_rows if not np.isnan(row["left_knee_angle"])]
    trunk_lean_angles = [row["trunk_lean_deg"] for row in data_rows if not np.isnan(row["trunk_lean_deg"])]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### 🦵 膝蓋角度")
        if right_knee_angles:
            st.metric("右膝平均角度", f"{np.mean(right_knee_angles):.1f}°")
            st.metric("右膝最大角度", f"{np.max(right_knee_angles):.1f}°")
            st.metric("右膝最小角度", f"{np.min(right_knee_angles):.1f}°")
        if left_knee_angles:
            st.metric("左膝平均角度", f"{np.mean(left_knee_angles):.1f}°")

    with col2:
        st.write("### 🫀 軀幹傾斜")
        if trunk_lean_angles:
            st.metric("平均傾斜角度", f"{np.mean(trunk_lean_angles):.1f}°")
            st.metric("最大傾斜角度", f"{np.max(trunk_lean_angles):.1f}°")

    with col3:
        st.write("### 📊 整體評分")
        # 簡單的評分邏輯
        symmetry_score = 100 - abs(np.mean(right_knee_angles) - np.mean(left_knee_angles)) if right_knee_angles and left_knee_angles else 0
        stability_score = 100 - np.std(trunk_lean_angles) if trunk_lean_angles else 0

        st.metric("左右對稱性", f"{max(0, min(100, symmetry_score)):.1f}%")
        st.metric("姿勢穩定性", f"{max(0, min(100, stability_score)):.1f}%")

    # 提供數據下載
    st.subheader("💾 下載分析數據")

    # 創建CSV數據
    csv_data = []
    for row in data_rows:
        csv_data.append({
            "幀數": row["frame"],
            "時間(秒)": row["time"],
            "右肘角度": row["right_elbow_angle"],
            "右肩角度": row["right_shoulder_angle"],
            "右臀角度": row["right_hip_angle"],
            "右膝角度": row["right_knee_angle"],
            "左臀角度": row["left_hip_angle"],
            "左膝角度": row["left_knee_angle"],
            "軀幹傾斜": row["trunk_lean_deg"],
            "右踝X": row["right_ankle_x"],
            "右踝Y": row["right_ankle_y"],
            "左踝X": row["left_ankle_x"],
            "左踝Y": row["left_ankle_y"],
        })

    df = pd.DataFrame(csv_data)

    # CSV下載
    csv_buffer = df.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 下載CSV數據",
        data=csv_buffer,
        file_name=f"pose_analysis_{int(time.time())}.csv",
        mime="text/csv",
        key="download_csv"
    )

    # 顯示數據預覽
    st.subheader("📋 數據預覽")
    st.dataframe(df.head(20), width='stretch')


def analyze_uploaded_video(video_file):
    """分析上傳的影片"""
    st.success(f"✓ 開始分析 {video_file.name}...")

    try:
        # 保存上傳的影片到臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_video_path = temp_file.name

        # 分析影片姿勢
        st.subheader("🔍 正在分析...")
        data_rows, fps, width, height = analyze_video_pose(temp_video_path)

        # 清理臨時文件
        os.unlink(temp_video_path)

        if len(data_rows) == 0:
            st.error("❌ 未檢測到任何姿勢數據，請檢查影片是否清晰且包含人體動作")
            return

        st.success("✅ 分析完成！")

        # 顯示分析結果
        display_analysis_results(data_rows)

    except Exception as e:
        st.error(f"❌ 分析過程中發生錯誤: {str(e)}")


def show():
    """動作偵測頁面"""
    st.header("🎥 AI 動作偵測")
    st.write("使用您的攝像頭實時錄製並分析，或上傳影片進行分析")
    
    st.divider()
    
    # 初始化 session state
    if 'stop_recording' not in st.session_state:
        st.session_state.stop_recording = False
    
    # ==================== 錄製和上傳選項 ====================
    tab_camera, tab_upload, tab_info, tab_tips = st.tabs(
        ["📹 攝像頭錄製", "📤 上傳影片", "ℹ️ 關節節點介紹", "💡 偵測提示"]
    )
    
    with tab_camera:
        st.write("### 📱 即時攝像頭錄製和分析")
        st.warning("⚠️ 注意：請確保攝像頭已授權，光線充足，穿著貼身衣物")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ 開始錄製", key="start_recording", type="primary", width='stretch'):
                st.session_state.stop_recording = False
                
                # 定義輸出路徑
                output_video = "webcam_recording.mp4"
                
                # 開始錄製
                success, duration = record_from_webcam(output_video)
                
                if success and os.path.exists(output_video):
                    st.success(f"✅ 錄製完成！時長：{duration:.1f} 秒")
                    
                    # 顯示分析選項
                    st.info("現在開始分析錄制的影片...")
                    data_rows, fps, width, height = analyze_video_pose(output_video)
                    
                    if len(data_rows) > 0:
                        st.success("✅ 分析完成！")
                        st.divider()
                        display_analysis_results(data_rows)
                    else:
                        st.error("❌ 未偵測到任何姿勢數據")
                else:
                    st.error("❌ 錄製失敗")

        with col2:
            if st.button("⏹️ 停止錄製", key="stop_recording_btn", width='stretch'):
                st.session_state.stop_recording = True
                st.info("⏸️ 正在停止錄製...")

    with tab_upload:
        st.write("### 📤 上傳影片進行分析")
        uploaded_video = st.file_uploader(
            "選擇影片檔案 (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"]
        )
        
        if uploaded_video:
            st.success(f"✓ 已上傳: {uploaded_video.name}")
            st.video(uploaded_video)

            # 顯示影片資訊
            col1, col2 = st.columns(2)
            with col1:
                st.metric("檔案大小", f"{uploaded_video.size / (1024*1024):.1f} MB")
            with col2:
                st.metric("檔案類型", uploaded_video.type)

            if st.button("🔍 分析此影片", key="analyze_uploaded", type="primary", width='stretch'):
                analyze_uploaded_video(uploaded_video)
    
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
