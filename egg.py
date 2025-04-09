# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt



# # 从提供的原始数据中加载Bx和By列的磁场强度
# data = np.loadtxt('d17_e17_t1.txt', skiprows=2, encoding="utf-8")  # 跳过头部描述行
# Bx = data[:, 0]   # 通道1（38Bx）
# By = data[:, 1]   # 通道2（38By）
# fs = 1000         # 采样率1kHz
# t = np.arange(len(Bx)) / fs

# print(By)





import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from scipy.io import loadmat  # 如果数据是.mat格式
import matplotlib as mpl
import pywt                    # 小波变换库


# ===============================================
# 1. 设置全局绘图样式
# ===============================================

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
mpl.rcParams['axes.unicode_minus'] = False    # 负号显示


# 读取文本文件 
data = np.loadtxt('egg_d13_B24_t3.txt', skiprows=2, encoding="utf-8")  # 跳过头部描述行
fs = 1000  # 采样率

# ===============================================
# 2. 滤波处理
# ===============================================

# 带通滤波器设计 (0.5-30Hz)
nyq = 0.5 * fs
b, a = signal.butter(3, [0.5/nyq, 30/nyq], btype='bandpass')
signals = signal.filtfilt(b, a, data, axis=0)

# 设计带阻滤波器（滤除50Hz工频干扰）
b_stop, a_stop = signal.butter(3, [49/nyq, 51/nyq], btype='bandstop')
signals = signal.filtfilt(b_stop, a_stop, signals, axis=0)

# # ===============================================
# # 3. STFT 时频分析 (双通道)
# # ===============================================

# plt.figure('STFT Analysis', figsize=(10, 8), facecolor='w')

# # 公共参数设置
# win_length = 800       # 窗长
# overlap = 160          # 重叠长度
# nfft = 1024            # FFT点数
# freq_lim = 20          # 最大显示频率
# dB_range = [-40, 40]   # 动态范围
# colormap = plt.cm.coolwarm  # 颜色映射

# for ch in range(2):
#     plt.subplot(2, 1, ch+1)
    
#     # 计算STFT
#     f, t, Sxx = signal.spectrogram(
#         signals[:, ch],
#         fs=fs,
#         window=signal.windows.hamming(win_length),
#         nperseg=win_length,
#         noverlap=overlap,
#         nfft=nfft,
#         mode='magnitude'
#     )
    
#     # 截取0-20Hz频段
#     valid_freq = f <= freq_lim
#     Sxx_db = 10 * np.log10(Sxx[valid_freq, :] + 1e-10)  # 转换为dB并避免log(0)
    
#     # 小波变换



#     # 绘制时频图
#     plt.pcolormesh(t, f[valid_freq], Sxx_db, 
#                   vmin=dB_range[0], vmax=dB_range[1], 
#                   cmap=colormap, shading='auto')
    
#     # 标注谐波位置
#     for harmonic in range(1, 6):
#         plt.axhline(harmonic, color='white', ls='--', alpha=0.3)
    
#     plt.colorbar(label='Power (dB)')
#     plt.title(f'Signal {ch+1} STFT (0-20Hz)')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.ylim(0, freq_lim)
#     plt.grid(True, alpha=0.3)
    
# plt.tight_layout()

# ===============================================
# 3. 小波变换参数设置
# ===============================================

plt.figure('Wavelet Analysis', figsize=(10, 8), facecolor='w')

wavelet = 'morl'  # 选择Morlet小波（适合时频分析）
scales = np.arange(1, 128)  # 尺度范围
frequencies = pywt.scale2frequency(wavelet, scales) * fs  # 转换为实际频率
freq_lim = 20  # 最大显示频率
dB_range = [-40, 40]  # 动态范围

for ch in range(2):
    plt.subplot(2, 1, ch+1)
    
    # 执行连续小波变换(CWT)
    coeffs, freqs = pywt.cwt(signals[:, ch], scales, wavelet, sampling_period=1/fs)
    
    # 转换为分贝单位并限制频率范围
    valid_freq = freqs <= freq_lim
    cwt_power = np.abs(coeffs[valid_freq, :])
    cwt_db = 10 * np.log10(cwt_power + 1e-10)  # 避免log(0)
    
    # 生成时间轴
    t = np.arange(signals.shape[0]) / fs     # 时间轴计算
    
    # 绘制时频图
    plt.pcolormesh(t, freqs[valid_freq], cwt_db,
                  vmin=dB_range[0], vmax=dB_range[1],
                  cmap=plt.cm.coolwarm, shading='auto')
    
    # 标注特征频率
    for harmonic in range(1, 6):
        plt.axhline(harmonic, color='white', ls='--', alpha=0.3)
    
    plt.colorbar(label='Magnitude (dB)')
    plt.title(f'Channel {ch+1} - CWT ({wavelet} Wavelet)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, freq_lim)
    plt.grid(True, alpha=0.3)

plt.tight_layout()



# ===============================================
# 4. 时域信号展示
# ===============================================
plt.figure('Time Domain', figsize=(10, 4), facecolor='w')
t_axis = np.arange(signals.shape[0]) / fs  # 时间轴计算
for ch in range(2):
    plt.plot(t_axis, signals[:, ch], label=f'Signal {ch+1}')

plt.legend(loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Signals (0.5-20Hz)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


