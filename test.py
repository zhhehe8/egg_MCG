import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.arange(0, 1, 1/fs)  # 1秒时长

# 生成测试信号（含高频成分）
f_high = 600  # 高于Nyquist频率
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*f_high*t)

plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.title('时域波形检查（600Hz成分应混叠为400Hz）')
plt.xlabel('Time (s)')
plt.grid(True)
