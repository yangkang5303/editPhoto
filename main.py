"""
图像校正工具

这个程序提供两种图像校正功能：
1. 水平校正：通过选择一条应该水平的线来校正图像
2. 透视校正：通过选择两条应该垂直的线来校正建筑物的透视效果

使用方法：
1. 命令行参数：
   python main.py -m <mode> <input_image> <output_directory>
   
   参数说明：
   - mode: 1=水平校正，2=透视校正
   - input_image: 输入图片的路径
   - output_directory: 输出目录的路径

   示例：
   - 水平校正：python main.py -m 1 ./photos/image.jpg ./output/
   - 透视校正：python main.py -m 2 ./photos/image.jpg ./output/

2. 交互操作：
   水平校正模式：
   - 点击两个点来定义一条应该水平的线
   - 程序会自动旋转图像使该线水平

   透视校正模式：
   - 依次点击第一条垂直线的上下两个端点
   - 再点击第二条垂直线的上下两个端点
   - 程序会自动校正使这两条线垂直

3. 通用操作：
   - 按 'r' 键：重置所有选点，重新开始
   - 按 'q' 键：完成操作并退出程序

4. 输出：
   - 校正后的图像会自动保存在指定目录
   - 文件名格式：原文件名_校正类型_时间戳.png
   - 使用PNG格式保存以保持图像质量
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime

# 存储选择的点
points = []
lines = []
image = None
original_image = None

def get_output_filename(input_path, save_dir, correction_type):
    """生成输出文件名"""
    # 获取原始文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 生成新文件名
    new_name = f"{base_name}_{correction_type}_{timestamp}.png"
    return os.path.join(save_dir, new_name)

def horizontal_correction(img, line_points):
    """水平校正函数"""
    # 计算线的角度
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    
    # 获取图像中心
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 执行旋转
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def mouse_callback_horizontal(event, x, y, flags, param):
    """水平校正的鼠标回调函数"""
    global points, image, original_image, save_path
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append([x, y])
            # 在图像上画点
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # 如果是第二个点，画线并进行校正
            if len(points) == 2:
                cv2.line(image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
                cv2.imshow("Original", image)
                
                # 计算水平校正
                corrected_image = horizontal_correction(original_image, points)
                
                # 显示结果
                cv2.imshow("Corrected", corrected_image)
                
                # 保存结果
                cv2.imwrite(save_path, corrected_image)
                print(f"已保存校正后的图片到: {save_path}")
            
            cv2.imshow("Original", image)

def mouse_callback_perspective(event, x, y, flags, param):
    """透视校正的鼠标回调函数"""
    global points, lines, image, original_image, save_path
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            if len(points) == 2:
                cv2.line(image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
                lines.append(points[:])
            
            elif len(points) == 4:
                cv2.line(image, tuple(points[2]), tuple(points[3]), (0, 255, 0), 2)
                lines.append(points[2:])
                
                corrected_image = correct_perspective(original_image, lines)
                cv2.imshow("Corrected", corrected_image)
                
                # 保存结果
                cv2.imwrite(save_path, corrected_image)
                print(f"已保存校正后的图片到: {save_path}")
            
            cv2.imshow("Original", image)

def correct_perspective(img, lines):
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 计算两条线的方向向量
    line1_vector = np.array(lines[0][1]) - np.array(lines[0][0])
    line2_vector = np.array(lines[1][1]) - np.array(lines[1][0])
    
    # 计算两条线的中点
    mid_point1 = (np.array(lines[0][0]) + np.array(lines[0][1])) / 2
    mid_point2 = (np.array(lines[1][0]) + np.array(lines[1][1])) / 2
    
    # 计算两条线的长度和角度
    line1_length = np.linalg.norm(line1_vector)
    line2_length = np.linalg.norm(line2_vector)
    
    # 计算线条的平均倾斜角度
    angle1 = np.arctan2(line1_vector[1], line1_vector[0])
    angle2 = np.arctan2(line2_vector[1], line2_vector[0])
    avg_angle = (angle1 + angle2) / 2
    
    # 计算源点（保持原始位置）
    src_points = np.float32([lines[0][0], lines[0][1], lines[1][0], lines[1][1]])
    
    # 计算目标点（保持相对位置，但确保垂直）
    # 保持线的长度和间距，只调整角度使其垂直
    x1, y1 = lines[0][0]
    x2, y2 = lines[1][0]
    
    # 计算校正后的点位置，保持线的长度但使其垂直
    dst_points = np.float32([
        [x1, y1],  # 第一条线起点保持不变
        [x1, y1 + line1_length],  # 第一条线终点垂直向下
        [x2, y2],  # 第二条线起点保持不变
        [x2, y2 + line2_length]   # 第二条线终点垂直向下
    ])
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换，保持原始色彩空间和质量
    corrected = cv2.warpPerspective(img, M, (w, h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    
    return corrected

def main():
    global image, original_image, points, lines, save_path
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图像校正工具')
    parser.add_argument('input', help='输入图片的路径')
    parser.add_argument('output_dir', help='输出目录的路径')
    parser.add_argument('--mode', '-m', choices=['1', '2'], 
                       help='校正模式：1=水平校正，2=透视校正', required=True)
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    # 检查并创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 读取图像
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("错误：无法读取图像文件")
        return
    
    # 生成输出文件路径
    correction_type = "horizontal" if args.mode == "1" else "perspective"
    save_path = get_output_filename(args.input, args.output_dir, correction_type)
    
    original_image = image.copy()
    points = []
    lines = []
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Original")
    if args.mode == "1":
        cv2.setMouseCallback("Original", mouse_callback_horizontal)
        print("请在图像上点击两个点来定义水平线")
    else:
        cv2.setMouseCallback("Original", mouse_callback_perspective)
        print("请选择两条垂直线（每条线需要两个点）：")
        print("1. 先点击第一条线的两个端点")
        print("2. 再点击第二条线的两个端点")
    
    print("按 'r' 键重置选点")
    print("按 'q' 键退出")
    
    # 显示原始图像
    cv2.imshow("Original", image)
    
    # 等待用户操作
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # 重置点
            points = []
            lines = []
            image = original_image.copy()
            cv2.imshow("Original", image)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()