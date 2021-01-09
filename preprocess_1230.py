# %%
import math
import wave
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import numpy
import librosa
import logmmse
import soundfile as sf
# %%
pathhh = './dataset/'
src_dir = pathhh[:-1]
dst_dir = pathhh[:-1]+'_denoise'
os.mkdir(dst_dir)
dirs=os.listdir(src_dir)
print(dirs)
for d in dirs:
    filenames = os.listdir(src_dir+'/'+d)
    os.mkdir(dst_dir+'/'+d)
#     print(filenames)
    for file in filenames:
        try:
            #讀檔
            filename = src_dir+'/'+ d +'/'+file
            y, sr = librosa.load(filename)
            # print(y)

            #降噪
            yyy=logmmse.logmmse(y, sr ,initial_noise=50)
            #輸出wav檔
            # librosa.output.write_wav(dst_dir+'/'+ d +'/'+file, yyy, sr)
            sf.write(dst_dir+'/'+ d +'/'+file, yyy, sr) #, 'PCM_24'
            # break
        except Exception as e:
            print("error")
            sf.write(dst_dir+'/'+ d +'/'+file, y, sr)

        
# %%
pathhh = './dataset/'
src_dir = pathhh[:-1]+'_denoise/'   #"D:/dataset/test_data_autoclass_denoise/"
dst_dir = pathhh[:-1]+'_spectrogram/'   #"D:/dataset/test_data_spectrogram/"

os.mkdir(dst_dir)#建立新資料夾放產出的頻譜圖
dirs=os.listdir(src_dir)#得到分類資料夾名稱
print(dirs)

count=0

for d in dirs:
    filenames = os.listdir(src_dir+'/'+d)#取出每種分類
    os.mkdir(dst_dir+'/'+d)#建立各分類的資料夾
    for file in filenames:
        #讀檔
        filename = src_dir+'/'+ d +'/'+file
        sample_rate,signal = wav.read(filename)
        signal = numpy.append(signal[0],signal[1:] - 0.97 * signal[:-1])
    
        #設定參數
        time_window = 25 #每窗幾毫秒
        window_length = sample_rate // 1000 * time_window
        x=np.linspace(0, window_length - 1, window_length, dtype = np.int64)#返回區間內的均勻數字
        w = np.hamming(window_length)
    
    
        # 分幀進行加窗
        p_begin = 0
        p_end = p_begin + window_length
        while p_end < len(signal):
            frame = signal[p_begin:p_end]

            # 加窗
            frame = frame * w

            # 進行快速傅里葉變換
            frame_fft = np.abs(fft(frame))

            #將每幀合併
            if p_begin==0:
                fftsignal=frame_fft
                newsignal=frame
            else:
                fftsignal=np.hstack((newsignal,frame_fft))
                newsignal=np.hstack((newsignal,frame))
     
            p_begin = p_end
            p_end += window_length
        
        
        #畫頻譜圖
        plt.figure(figsize=(7, 7))#固定大小
        plt.specgram(newsignal, Fs = sample_rate, scale_by_freq=True,vmax=10,vmin=-60, sides='default', cmap='rainbow')

        plt.savefig(dst_dir+'/'+d+'/'+file+'.jpg')#存圖
        
# %%
