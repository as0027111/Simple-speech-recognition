from __future__ import print_function
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
import pandas as pd
import datetime

timer = sched.scheduler(time.time, time.sleep)
def stock_crawl(targets):
    stock_list = '|'.join('tse_{}.tw'.format(target) for target in targets)
    #print(stock_list)
    try:
        query_url = "http://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch="+ stock_list
        data = json.loads(urlopen(query_url).read())
        #print(data)
        column = ['c','n','z','o','h','l','y']
        df = pd.DataFrame(data['msgArray'], columns=column)
        df.columns = ['股票代號','公司簡稱','當盤成交價','開盤價','最高價','最低價','昨收價']
        df.insert(7, "漲跌百分比" , 0.0)
        #print(df)
        # 新增漲跌百分比
        for x in range(len(df.index)):
            if df['當盤成交價'].iloc[x] != '-':
                df.iloc[x, [2,3,4,5,6]] = df.iloc[x, [2,3,4,5,6]].astype(float) # 2-8 欄本來是string型態
                df['漲跌百分比'].iloc[x] = (df['當盤成交價'].iloc[x] - df['昨收價'].iloc[x])/df['昨收價'].iloc[x] * 100

        time = datetime.datetime.now() # 紀錄更新時間  
        print("更新時間:" + str(time.hour)+":"+str(time.minute))
        '''
        start_time = datetime.datetime.strptime(str(time.date())+'9:30', '%Y-%m-%d%H:%M')
        end_time =  datetime.datetime.strptime(str(time.date())+'13:30', '%Y-%m-%d%H:%M')
        
        # 判斷爬蟲終止條件
        if time >= start_time and time <= end_time:
            timer.enter(1, 0, stock_crawl, argument=(targets,))
        '''
        # show table
        #df = df.style.applymap(tableColor, subset=['漲跌百分比'])
        for x in range(len(df.index)):
            print(df.iloc[x])
    except:
        print('International Error, check you connection.')

    return
#### call of stock_crawl()
stock_list = ['0050', 't00']
# stock_crawl(stock_list)

def speak(sentence):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text = sentence,lang='zh-tw')
        tts.save('{}.mp3'.format(fp.name))
        mixer.init(frequency=22050)
        mixer.music.load('{}.mp3'.format(fp.name))
        mixer.music.play()

def weather():
    print('weather:\n')
    dataid = 'F-D0047-077' #鄉鎮天氣預報-臺南市未來2天天氣預報
    api_key = 
    load_format = 'JSON'
    try:
        query_url = 'https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/'+dataid+'?Authorization='+api_key+'&format='+load_format
        data = json.loads(urlopen(query_url).read())
        df = pd.DataFrame(data['cwbopendata']['dataset']['locations']['location'])
        east = df.iloc[31]
        east = east.loc['weatherElement']
        temperature = pd.DataFrame(east[10]['time']).loc[:8, 'startTime':'elementValue'] #8*3=24hr
        now_describe = temperature.elementValue[0]
        now_describe = now_describe['value'].split('。')[:4]
        print('台南市東區的天氣：', now_describe)
        speak('台南市東區的天氣：')
        time.sleep(3)
        for i in now_describe:
            speak(i)
            #print(len(i))
            time.sleep(len(i)*0.4)
    except:
        print('International Error, check you connection.')
        return None
    return now_describe
#### call of weather()
#now = weather()
#print('end')

def breakfast():
    breakfast_list = ['阿寶','提克','濰克','飯捲','拉雅漢堡','seven eleven','捷米','今天不吃']
    speak('今天吃')
    time.sleep(1)
    speak(breakfast_list[random.randrange(len(breakfast_list))])
    time.sleep(1.5)
#### call of breakfast()
# breakfast()

def Give_time():
    current_time = datetime.datetime.now() # 紀錄更新時間  
    if current_time.hour > 18:
        HR = current_time.hour-12
        now_time = "現在時間:"+"晚上"+str(HR)+"點"+str(current_time.minute)+"分"
    elif current_time.hour > 12:
        HR = current_time.hour-12
        now_time = "現在時間:"+"下午"+str(HR)+"點"+str(current_time.minute)+"分"
    else:
        now_time = "現在時間:"+"早上"+str(current_time.hour)+"點"+str(current_time.minute)+"分"
    speak(now_time)
    time.sleep(len(now_time)*0.5)
#### call of Give_time()
#Give_time()


'''
source: https://developers.google.com/calendar/quickstart/python
        Step 1: Turn on the Google Calendar API.
        Step 2: Install the Google Client library.
                -pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
        Step 3: Run the sample.
'''
# If modifying these scopes, delete the file token.pickle.
def calendar():
    SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_id.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)

    # Call the Calendar API
    now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
    print('Getting the upcoming 10 events') 
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                        maxResults=10, singleEvents=True, #最多回傳10項
                                        orderBy='startTime').execute() 
    events = events_result.get('items', [])

    if not events:
        print('No upcoming events found.')
        return
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        #print(start, event['summary']) 
        
    # 整理輸出格式 show(df[事件,開始日期,開始時間,結束日期,結束時間])
    columns = ['summary','start','end']
    df = pd.DataFrame(events, columns=columns)
    df.columns = ['事件','開始日期','結束日期']
    df.insert(2, "開始時間" , 0.0)
    df.insert(4, "結束時間" , 0.0)
    for x in range(len(df.index)):
        start = (df['開始日期'].iloc[x]['dateTime']).split('T')
        start_date, start_time = start[0], start[1].split('+')[0]
        df['開始日期'].iloc[x], df['開始時間']= start_date, start_time
        end = (df['結束日期'].iloc[x]['dateTime']).split('T')
        end_date, end_time = end[0], end[1].split('+')[0]
        df['結束日期'].iloc[x], df['結束時間']= end_date, end_time

    for x in range(len(df.index)):
        print(df.iloc[x])
#### call of calendar()
# calendar()