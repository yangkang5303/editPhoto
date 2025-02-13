**图像校正工具**

这个程序提供两种图像校正功能：
1. 水平校正：通过选择一条应该水平的线来校正图像
2. 透视校正：通过选择两条应该垂直的线来校正建筑物的透视效果

*使用方法：*
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


yangkang5303@gmail.com
