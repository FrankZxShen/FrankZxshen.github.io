---
title: （摸鱼）pcr角色语音收录
abbrlink: 9a7a7324
date: 2023-05-09 15:51:24
tags:
categories:
top_img:
---

使用jupyter lab

```
# 测试网址是否可用
import time
import requests
print("看响应参数是不是200判断url是否能访问")
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'Referer': 'http://www.kuwo.cn/search/list?key=%E5%91%A8%E6%9D%B0%E4%BC%A6',
    'csrf': 'RUJ53PGJ4ZD',
    'Cookie': 'Hm_lvt_cdb524f42f0ce19b169a8071123a4797=1577029678,1577034191,1577034210,1577076651; Hm_lpvt_cdb524f42f0ce19b169a8071123a4797=1577080777; kw_token=RUJ53PGJ4ZD'
}
response = requests.get(
    "https://patchwiki.biligame.com/images/pcr/7/70/2l86cmxdl1uc2pannakd6fj125h0f4h.mp3", headers=headers)
print(response)
```

```
# 测试音频是否存在
from bs4 import BeautifulSoup  
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'Referer': 'http://www.kuwo.cn/search/list?key=%E5%91%A8%E6%9D%B0%E4%BC%A6',
    'csrf': 'RUJ53PGJ4ZD',
    'Cookie': 'Hm_lvt_cdb524f42f0ce19b169a8071123a4797=1577029678,1577034191,1577034210,1577076651; Hm_lpvt_cdb524f42f0ce19b169a8071123a4797=1577080777; kw_token=RUJ53PGJ4ZD'
}
url = 'https://wiki.biligame.com/pcr/%E8%8E%89%E7%8E%9B' 
res = requests.get(url).text 
content = BeautifulSoup(res, "html.parser") 
data = content.find_all('div', attrs={'class': 'audio-wrapper'}) 
audio_list = [] 
for d in data: 
    plist = d.find('audio')['src'] 
    audio_list.append(plist) 
print(audio_list)
```

```
# 爬取一个角色音频
import os
def download_audio(audio_l,c_name): 
    if not os.path.exists(c_name): 
        os.mkdir(c_name) 
    for i in audio_l: 
        audio = requests.get(i) 
        audio_name = i.split('/')[7] 
        with open(c_name + '\\' + audio_name, 'wb') as f: 
            f.write(audio.content)

def get_mp3_names(res):
    content = BeautifulSoup(res, "html.parser") 
    data = content.find_all('div', attrs={'class': 'audio-wrapper'}) 
    audio_list = [] 
    for d in data: 
        plist = d.find('audio')['src'] 
        audio_list.append(plist) 
    print("audio_list:")
    print(audio_list)
    return audio_list

print("开始爬取...")
print("#####################################################")
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'Referer': 'http://www.kuwo.cn/search/list?key=%E5%91%A8%E6%9D%B0%E4%BC%A6',
    'csrf': 'RUJ53PGJ4ZD',
    'Cookie': 'Hm_lvt_cdb524f42f0ce19b169a8071123a4797=1577029678,1577034191,1577034210,1577076651; Hm_lpvt_cdb524f42f0ce19b169a8071123a4797=1577080777; kw_token=RUJ53PGJ4ZD'
}
url = 'https://wiki.biligame.com/pcr/%E8%8E%89%E7%8E%9B' 
c_name = '莉玛'
print("角色名：",c_name)
try:
    os.getcwd()
    res = requests.get(url).text 
    audio_list = get_mp3_names(res)
    os.chdir('./audio/')
    download_audio(audio_list, c_name)
except Exception as e:
    print("未知错误：",e)

time.sleep(0.5)
print("#####################################################")
print("爬取结束...")
```

```
# 音频编号
os.chdir('../')
os.getcwd()
path = './audio/莉玛'
i = 1
for filename in os.listdir(path):
    if filename.endswith('.mp3'):
        os.rename(os.path.join(path, filename), os.path.join(path, str(i) + '.wav'))
        i += 1
```

其他方面可能有点问题，得去手动填充数据