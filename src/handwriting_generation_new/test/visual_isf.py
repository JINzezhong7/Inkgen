import numpy as np
import matplotlib.pyplot as plt
from PyInk import ReadIsf


my_isf_path = "numbers/isfs/no_priming_0.isf"


strokes = ReadIsf(my_isf_path)
print("There are %d strokes in the sample." % (len(strokes)))

fig, ax = plt.subplots(1, 1, figsize=(15, 6))
fig.suptitle(f"Plotting strokes from {my_isf_path}")


for stroke in strokes:
    # 注意：如果你想让 Y 轴正方向朝下，可以在绘制时对 y 坐标取负值
    ax.scatter(stroke[0, :], -stroke[1, :], s=1, color = 'black')

# 添加网格并显示
ax.grid()
plt.show()
