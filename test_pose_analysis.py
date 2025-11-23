#!/usr/bin/env python3
"""
測試姿勢分析功能的腳本
"""

import sys
import os
sys.path.insert(0, 'modules')

from pose_detection import analyze_video_pose

def test_pose_analysis():
    """測試姿勢分析功能"""
    video_path = "run.mp4"

    if not os.path.exists(video_path):
        print(f"❌ 找不到測試影片: {video_path}")
        return

    print(f"📹 開始分析影片: {video_path}")

    try:
        # 分析影片
        data_rows, fps, width, height = analyze_video_pose(video_path)

        print("✅ 分析完成！")
        print(f"   總幀數: {len(data_rows)}")
        print(f"   FPS: {fps:.1f}")
        print(f"   解析度: {width}x{height}")

        if len(data_rows) > 0:
            print("   數據範例:")
            sample = data_rows[0]
            for key, value in sample.items():
                if isinstance(value, float) and not str(value).lower() == 'nan':
                    print(f"      {key}: {value:.2f}")
                elif not str(value).lower() == 'nan':
                    print(f"      {key}: {value}")
        else:
            print("   ⚠️ 未檢測到任何姿勢數據")

    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {e}")

if __name__ == "__main__":
    test_pose_analysis()