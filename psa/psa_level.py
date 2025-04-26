import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.font_manager as fm
font_path = r'C:\\Windows\\Fonts\\simsun.ttc'  
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['Times New Roman','SimSun']  # 设置字体为 DejaVu Sans 和 SimSun
# 设置英文和数字字体为 Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'    # 使用 STIX 字体（支持 Times 类型）
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 重新生成的数据
x = [1, 2, 3, 4, 5]  # 横轴：小波分解层数
y1 = [70.50, 75.80, 77.20, 77.77, 77.42]  # IP的OA
y2 = [84.50, 89.20, 90.70, 90.98, 90.20]  # PU的OA
y3 = [86.70, 91.30, 92.70, 93.02, 92.70]  # SA的OA

# 计算等距的位置，用于绘制
x_positions = range(len(x))

# 设置全局字体大小
plt.rcParams.update({'font.size': 19})  # 设置全局字体大小
#plt.rcParams['font.family'] = 'SimSun'

# 创建画布和子图
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制折线图，使用等距的x位置
ax.plot(x_positions, y1, label="IP", marker='o', color='b', linewidth=2, markersize=8)  # 第一条线
ax.plot(x_positions, y2, label="PU", marker='o', color='g', linewidth=2, markersize=8)  # 第二条线
ax.plot(x_positions, y3, label="SA", marker='o', color='r', linewidth=2, markersize=8)  # 第三条线

# 设置横轴和纵轴标签
ax.set_xlabel('小波分解层数', fontsize=20)  # 设置横轴标签
ax.set_ylabel('总体精度 %', fontsize=20)  # 设置纵轴标签

# 设置纵轴范围
ax.set_ylim(70, 95)  # 缩小纵轴范围以突出差距

# 设置横轴刻度为等距
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{i}' for i in x], fontsize=18)  # 设置x轴标签

# 设置纵轴刻度
ax.set_yticks(range(70, 96, 5))  # 设置y轴刻度
ax.tick_params(axis='y', labelsize=18)  # 设置y轴标签字体大小

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.6)  # 添加网格线

# 显示图例，并设置图例字体大小
ax.legend(fontsize=18)

# 自动调整布局，使图表更紧凑
plt.tight_layout()

# 创建保存路径
save_path = os.path.join("results", "wavelet_layers")
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 如果目录不存在，则创建目录

# 使用当前时间生成文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化时间为字符串
file_name = f"{timestamp}.png"  # 使用时间作为文件名
file_path = os.path.join(save_path, file_name)

# 保存图表到指定路径
plt.savefig(file_path, dpi=300, bbox_inches='tight')

# 显示图表
plt.show()