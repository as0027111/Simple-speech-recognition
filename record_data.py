
# coding: utf-8

# In[ ]:


# import os
# import shutil

# delete_dir = 'D:/dataset/microphone/'

# #將所有資料刪除並重新建立一個資料夾放等下的檔案
# shutil.rmtree(delete_dir)
# os.mkdir(delete_dir)


# In[ ]:


# https://blog.csdn.net/zzZ_CMing/article/details/81739193
#從麥克風收音5秒
import pyaudio 
import wave 
import os
# input_filename = "input.wav" # 麦克风采集的语音输入 
# input_filepath = "D:/dataset/microphone/" # 输入文件的path 
# in_path = input_filepath + input_filename

# %%
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
        get_audio(in_path)
# %%
# input_filename = "input.wav" # 麦克风采集的语音输入
cat=['breakfast','calender','stock','time','weather'] 
k = 4 
input_filepath = "./new_dataset_2/"+ cat[k] + '/'  # 输入文件的path
os.mkdir(input_filepath)
# in_path = input_filepath + input_filename
for i in range(100):
    input_filename = cat[k] + '_' + str(i)+".wav"
    in_path = input_filepath + input_filename
    get_audio(in_path)



# %%