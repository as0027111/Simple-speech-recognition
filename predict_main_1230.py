

# coding: utf-8

# In[ ]:
from __future__ import print_function

import os
import shutil
import pyaudio 
import wave
import librosa
import logmmse
import soundfile as sf

import numpy
import math
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import glob
import cv2
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint ,EarlyStopping
from keras.optimizers import SGD, Adam
from pydub.silence import split_on_silence
from pydub import AudioSegment
from keras.models import load_model


import datetime
import pickle
import os.path

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from urllib.request import urlopen
from gtts import gTTS
from pygame import mixer
import requests
import sched
import json
import tempfile
import time
import random
import datetime
from function import stock_crawl, speak, weather, breakfast, Give_time, calendar

# In[ ]:

# https://blog.csdn.net/zzZ_CMing/article/details/81739193
#從麥克風收音5秒
def get_audio(filepath):
    aa = str(input("是否開始錄音？   （y/n）"))
    if aa == str("y") :
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
#         RATE = 11025                # 采样率
        RATE = 44100                # 采样率
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = filepath
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*"*10, "開始錄音，請在5秒內輸入語音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*"*10, "錄音結束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif aa == str("n"):
        exit()
    else:
        print("輸入無效，請重新輸入")
        get_audio(filepath)

# %%
def denoise(input_filepath):
    output_filepath = input_filepath+'denoise/'
    input_filename = "input.wav" 
    denoise_filename = "denoise.wav" 
    filename = input_filepath+input_filename
    os.mkdir(output_filepath)
    y, sr = librosa.load(filename)
    #降噪
    yyy=logmmse.logmmse(y, sr ,initial_noise=50)
    # librosa.output.write_wav(output_filepath+denoise_filename, yyy, sr)
    sf.write(output_filepath+denoise_filename, yyy, sr)

# %%
def cut_sentence(input_fp):
    input_filepath = input_fp + 'denoise/'
    output_filepath = input_fp + 'cutsentence/'
    denoise_filename = "denoise.wav" 
    cut1_filename = 'cutsentence'
    src_dir = input_filepath + denoise_filename
    dst_dir = output_filepath + cut1_filename
    os.mkdir(output_filepath)
    #讀檔
    filename = src_dir
    print(filename)
    sound = AudioSegment.from_file(filename)
    
    #該音檔的最大分貝
    loudness = sound.dBFS
    
    #切割
    audio_chunks=split_on_silence(sound,min_silence_len=1250,silence_thresh=(loudness-10))
    
    #每個切好的檔案分別輸出成wav檔
    count_syllable=0
    for i,chunk in enumerate(audio_chunks):
        chunk.export(dst_dir +'-' +str(i)+'.wav',format='wav')
        count_syllable+=1
    print(count_syllable)#切幾段


# %%
def wav2fig(input_fp):
    src_dir = input_fp + "denoise/"
    dst_dir = input_fp + "spectrogram/"
    os.mkdir(dst_dir)#建立資料夾放產出的頻譜圖
    filenames=os.listdir(src_dir)
    print(filenames)

    #每個經句子切割的檔案輪流再進行字詞切割
    for file in filenames:
        #讀檔
        filename = src_dir+'/'+file
        sample_rate,signal = wav.read(filename)
        signal = numpy.append(signal[0],signal[1:] - 0.97 * signal[:-1])
        
        #設定參數
        time_window = 25 #每窗幾毫秒
        window_length = sample_rate // 1000 * time_window
        x=np.linspace(0, window_length - 1, window_length, dtype = np.int64)#返回區間內的均勻數字
        w = np.hamming(window_length)#窗函式
        
        
        # 分幀進行加窗
        p_begin = 0
        p_end = p_begin + window_length
        while p_end < len(signal):
            frame = signal[p_begin:p_end]
    
            
            frame = frame * w # 加窗
            frame_fft = np.abs(fft(frame)) # 進行快速傅里葉變換
            frame_log = np.log(frame_fft) # 取對數，求db
            
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
        plt.figure(figsize=(7, 7))#固定圖的大小
        plt.specgram(fftsignal, Fs = sample_rate, scale_by_freq=True,vmax=10,vmin=-60, sides='default', cmap='rainbow')

        plt.savefig(dst_dir+'/'+file+'.jpg')#存圖
        plt.show()

# %%

def predict(path):
    # path = 'D:/dataset/microphone/spectrogram_cut/'
    # 讀取圖片並resize，轉為可丟入cnn的形式
    imgs = []
    fpath = []
    for im in glob.glob(path + '/*.jpg'):
        #print('reading the images:%s' % (im))
        img = cv2.imread(im)             #opencv讀圖片
        img = cv2.resize(img, (150, 150))  #resize
        imgs.append(img)                 #資料
    #     labels.append(idx)               #種類
        fpath.append(path+im)            #資料路徑
        
    fpaths = np.asarray(fpath, np.string_)
    data = np.asarray(imgs, np.float32)
    print(data.shape)  
    data = data.astype('float32') / 255.
    print('data',data.shape)

    # 模型載入
    model = load_model('./model/0106_17.h5')
    
    # 預測與比對
    y_pred = model.predict_classes(data)
    y_pred_probibility = model.predict(data)
    print(y_pred)
    print(y_pred_probibility)
    print("['breakfast','calender','stock','time', 'weather']")
    # ['breakfast','calender','stock','time', 'weather']
    if y_pred[0]==0:
        breakfast()
    elif y_pred[0]==1:  
        # print()
        calendar()  
    elif y_pred[0]==2:
        stock_list = ['0050', 't00']
        stock_crawl(stock_list)
    elif y_pred[0]==3:
        Give_time()
    elif y_pred[0]==4:
        now = weather()

# %%
def main():
    # delete_dir = 'D:/dataset/microphone/'
    input_filename = "input.wav" # 麦克风采集的语音输入 
    input_filepath = "./microphone_input/" # 输入文件的path 


    shutil.rmtree(input_filepath) #將所有資料刪除並重新建立一個資料夾放等下的檔案
    os.mkdir(input_filepath)
    in_path = input_filepath + input_filename
    get_audio(in_path) # 收音5秒
    denoise(input_filepath) # 去除噪音
    # cut_sentence(input_filepath) # 切掉前後空白的部分/若有多句則切開
    wav2fig(input_filepath) # 轉頻譜圖
    
    predictPath = input_filepath + "spectrogram/"
    predict(predictPath)
    
# %%
if __name__ == '__main__':
    main()
# %%
