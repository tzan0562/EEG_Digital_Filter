import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq, fftshift

# 1. 讀取與整理資料
# 確保這個 csv 檔與你的 python 程式碼放在同一個資料夾
df = pd.read_csv("EEG data - 工作表1.csv")

fs = 200.0  # 採樣頻率 200 Hz
nyq = 0.5 * fs 

# 建立時間軸 (Row Index / 200 Hz = Time in seconds)
df['Time'] = np.arange(len(df)) / fs

# 根據頻譜分析對應的亂碼欄位
bands = {
    'Delta': {'range': (0.5, 4.0), 'ref_col': 'delta'},
    'Theta': {'range': (4.0, 8.0), 'ref_col': 'theta'},
    'Alpha': {'range': (8.0, 13.0), 'ref_col': 'alpha'},
    'Beta':  {'range': (13.0, 30.0), 'ref_col': 'beta'}
}

# 2. 定義濾波與 FFT 函數
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_two_sided_fft(signal, fs):
    N = len(signal)
    signal_no_dc = signal - np.mean(signal) # 去除直流成分 (DC Offset)
    
    yf = fft(signal_no_dc.values if isinstance(signal_no_dc, pd.Series) else signal_no_dc)
    xf = fftfreq(N, 1/fs)
    
    # 助教要求：不截掉負頻率，將 0Hz 移至中央
    yf_shifted = fftshift(yf)
    xf_shifted = fftshift(xf)
    amplitude = np.abs(yf_shifted) / N
    return xf_shifted, amplitude

# 3. 根據時間切割資料
df_closed = df[(df['Time'] >= 0) & (df['Time'] < 20)]  # 閉眼 0~20 秒
df_open   = df[(df['Time'] >= 20) & (df['Time'] < 40)] # 睜眼 20~40 秒

raw_closed = df_closed['EEG(uV)']
raw_open   = df_open['EEG(uV)']

# 4. 繪圖設定 (4x4 圖表)
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
fig.suptitle('EEG Two-Sided FFT (Designed vs Reference, Eyes Closed vs Open)', fontsize=16, y=0.98)

for i, (band_name, info) in enumerate(bands.items()):
    low, high = info['range']
    ref_col = info['ref_col']
    
    # --- A. 自己設計的濾波器 (Designed) ---
    designed_closed = butter_bandpass_filter(raw_closed, low, high, fs)
    designed_open   = butter_bandpass_filter(raw_open, low, high, fs)
    
    # --- B. BIOPAC 對照組 (Reference) ---
    ref_closed = df_closed[ref_col]
    ref_open   = df_open[ref_col]
    
    # --- C. 計算雙邊 FFT ---
    x_dc, y_dc = compute_two_sided_fft(designed_closed, fs)
    x_rc, y_rc = compute_two_sided_fft(ref_closed, fs)
    x_do, y_do = compute_two_sided_fft(designed_open, fs)
    x_ro, y_ro = compute_two_sided_fft(ref_open, fs)
    
    # 找出該頻帶的最大振幅以統一這四張圖的 Y 軸尺度 (方便肉眼比較)
    max_y = max(np.max(y_dc), np.max(y_rc), np.max(y_do), np.max(y_ro)) * 1.1 
    if max_y == 0: max_y = 0.1
    
    # 繪圖迴圈分配
    plots_data = [
        (x_dc, y_dc, f'Designed, eyes closed - {band_name}', 'blue'),
        (x_rc, y_rc, f'Reference, eyes closed - {band_name}', 'red'),
        (x_do, y_do, f'Designed, eyes open - {band_name}', 'blue'),
        (x_ro, y_ro, f'Reference, eyes open - {band_name}', 'red')
    ]
    
    for j in range(4):
        ax = axes[i, j]
        x_data, y_data, title, color = plots_data[j]
        
        ax.plot(x_data, y_data, color=color, linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-100, 100) # 設定 X 軸 -100 到 100 Hz
        ax.set_ylim(0, max_y)
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)

# 放寬上下圖(hspace)與左右圖(wspace)的間距
plt.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.95, hspace=0.7, wspace=0.3)
plt.savefig('EEG_FFT_Analysis.png', dpi=300) # 儲存高解析度圖片
plt.show()