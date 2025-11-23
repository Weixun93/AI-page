import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv
import matplotlib.pyplot as plt


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


def main():
    # 初始化 Mediapipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    connections = mp_pose.POSE_CONNECTIONS

    # 移動平均緩衝區（右手）
    smooth_buffer_size = 5
    elbow_R_buffer = deque(maxlen=smooth_buffer_size)
    wrist_R_buffer = deque(maxlen=smooth_buffer_size)

    # 打開本地攝像頭（0 表示默認攝像頭）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝像頭，請檢查攝像頭是否已連接")
        return

    # 設置攝像頭分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 取得攝像頭資訊
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 30.0  # 默認 30 FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 記錄時長（秒），若不想限制可設為 None
    DURATION = 30  # 秒
    max_frames = int(DURATION * input_fps) if DURATION is not None else None

    print(f"📹 攝像頭已開啟: 分辨率 {width}x{height}, FPS: {input_fps:.2f}, "
          f"設定錄製時長: {DURATION if DURATION is not None else '無限制'} 秒")
    print("💡 按 'Q' 或 'ESC' 鍵停止錄製")

    # 設定影片輸出（保存錄制的視頻）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("webcam_output.mp4", fourcc, input_fps, (width, height))

    # 用來存每一幀跑步數據
    data_rows = []
    frame_idx = 0

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
                # 若有設定 DURATION，超過幀數就停止
                if max_frames is not None and frame_idx >= max_frames:
                    break

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

                    # --- 平滑化右手節點 ---
                    elbow_R_buffer.append(elbow_R)
                    wrist_R_buffer.append(wrist_R)

                    if len(elbow_R_buffer) == smooth_buffer_size:
                        elbow_R = np.mean(elbow_R_buffer, axis=0)
                        wrist_R = np.mean(wrist_R_buffer, axis=0)

                    # ====== 專業相關角度計算 ======
                    right_elbow_angle = calculate_angle(shoulder_R, elbow_R, wrist_R)
                    right_shoulder_angle = calculate_angle(elbow_R, shoulder_R, hip_R)
                    right_hip_angle = calculate_angle(shoulder_R, hip_R, knee_R)
                    right_knee_angle = calculate_angle(hip_R, knee_R, ankle_R)

                    left_hip_angle = calculate_angle(shoulder_L, hip_L, knee_L)
                    left_knee_angle = calculate_angle(hip_L, knee_L, ankle_L)

                    # 軀幹前傾角（左右平均）
                    mid_shoulder = (
                        (shoulder_R[0] + shoulder_L[0]) / 2,
                        (shoulder_R[1] + shoulder_L[1]) / 2
                    )
                    mid_hip = (
                        (hip_R[0] + hip_L[0]) / 2,
                        (hip_R[1] + hip_L[1]) / 2
                    )
                    trunk_lean_deg = calculate_trunk_lean(mid_shoulder, mid_hip)

                    # ✅ 每一幀紀錄數據
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

                    # ✅ 常亮 Running Form OK
                    cv2.putText(
                        frame, "Running Form OK", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
                    )

                    # 繪製骨架
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, connections,
                        landmark_drawing_spec=pose_style
                    )

                # 寫入輸出影片
                out.write(frame)

                # 實時顯示畫面
                cv2.imshow("Running Posture Detection (Live)", frame)
                
                # 按 'Q' 或 'ESC' 鍵停止
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # ESC
                    print("\n⏹️ 錄製已停止")
                    break

                frame_idx += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print("✅ 偵測完成，開始輸出 CSV 與圖表...")

    # ====== 將數據輸出成 CSV 檔 ======
    if len(data_rows) > 0:
        csv_file = "webcam_metrics.csv"
        fieldnames = list(data_rows[0].keys())

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)

        print(f"✅ 跑步數據已輸出成 CSV：{csv_file}")

        # ====== 視覺化：從 data_rows 直接畫圖 ======
        times = [row["time"] for row in data_rows]

        # 1. 右膝角度隨時間變化
        right_knee_angles = [row["right_knee_angle"] for row in data_rows]
        plt.figure()
        plt.plot(times, right_knee_angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Right Knee Angle (deg)")
        plt.title("Right Knee Angle over Time")
        plt.tight_layout()
        plt.savefig("plot_right_knee_angle_over_time.png", dpi=200)
        plt.close()

        # 2. 左右膝角度比較（對稱性檢查）
        left_knee_angles = [row["left_knee_angle"] for row in data_rows]
        plt.figure()
        plt.plot(times, right_knee_angles, label="Right Knee")
        plt.plot(times, left_knee_angles, label="Left Knee")
        plt.xlabel("Time (s)")
        plt.ylabel("Knee Angle (deg)")
        plt.title("Left vs Right Knee Angle over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_knee_angle_left_vs_right.png", dpi=200)
        plt.close()

        # 3. 軀幹前傾角度
        trunk_lean = [row["trunk_lean_deg"] for row in data_rows]
        plt.figure()
        plt.plot(times, trunk_lean)
        plt.xlabel("Time (s)")
        plt.ylabel("Trunk Lean (deg)")
        plt.title("Trunk Lean Angle over Time")
        plt.tight_layout()
        plt.savefig("plot_trunk_lean_over_time.png", dpi=200)
        plt.close()

        # 4. 右腳踝垂直位移（可以看步頻 / 垂直震盪）
        right_ankle_y = [row["right_ankle_y"] for row in data_rows]
        plt.figure()
        plt.plot(times, right_ankle_y)
        plt.xlabel("Time (s)")
        plt.ylabel("Right Ankle Y (pixel)")
        plt.title("Right Ankle Vertical Trajectory")
        plt.gca().invert_yaxis()  # 影像座標 y 向下，反轉較符合直覺
        plt.tight_layout()
        plt.savefig("plot_right_ankle_vertical_trajectory.png", dpi=200)
        plt.close()

        # 5. 右腳踝在畫面中的 2D 路徑（x-y）
        right_ankle_x = [row["right_ankle_x"] for row in data_rows]
        plt.figure()
        plt.plot(right_ankle_x, right_ankle_y)
        plt.xlabel("Right Ankle X (pixel)")
        plt.ylabel("Right Ankle Y (pixel)")
        plt.title("Right Ankle 2D Trajectory")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("plot_right_ankle_2d_trajectory.png", dpi=200)
        plt.close()

        print("✅ 圖表已輸出：")
        print("   - plot_right_knee_angle_over_time.png")
        print("   - plot_knee_angle_left_vs_right.png")
        print("   - plot_trunk_lean_over_time.png")
        print("   - plot_right_ankle_vertical_trajectory.png")
        print("   - plot_right_ankle_2d_trajectory.png")
    else:
        print("⚠️ 沒有偵測到任何姿勢數據，未產生 CSV。")

    print("✅ 全部完成，錄制視頻：webcam_output.mp4, CSV：webcam_metrics.csv")


if __name__ == "__main__":
    main()
