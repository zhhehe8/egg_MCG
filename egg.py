import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt



# 从提供的原始数据中加载Bx和By列的磁场强度
data = np.loadtxt('d17_e17_t1.txt', skiprows=2, encoding="utf-8")  # 跳过头部描述行
Bx = data[:, 0]   # 通道1（38Bx）
By = data[:, 1]   # 通道2（38By）
fs = 1000         # 采样率1kHz
t = np.arange(len(Bx)) / fs

print(By)