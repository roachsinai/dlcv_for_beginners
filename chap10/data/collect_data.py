#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: loveNight
# @Date:   2015-10-28 19:59:24
# @Last Modified by:   roachsinai
# @Last Modified time: 2018-03-29 16:44:38


import urllib
import requests
import os
import re
import time
import threading
from datetime import datetime as dt
from multiprocessing.dummy import Pool
from multiprocessing import Queue
from multiprocessing import Process
import multiprocessing


class BaiduImgDownloader(object):

    """百度图片下载工具，目前只支持单个关键词"""

    # 解码网址用的映射表
    str_table = {
        '_z2C$q': ':',
        '_z&e3B': '.',
        'AzdH3F': '/'
    }

    char_table = {
        'w': 'a',
        'k': 'b',
        'v': 'c',
        '1': 'd',
        'j': 'e',
        'u': 'f',
        '2': 'g',
        'i': 'h',
        't': 'i',
        '3': 'j',
        'h': 'k',
        's': 'l',
        '4': 'm',
        'g': 'n',
        '5': 'o',
        'r': 'p',
        'q': 'q',
        '6': 'r',
        'f': 's',
        'p': 't',
        '7': 'u',
        'e': 'v',
        'o': 'w',
        '8': '1',
        'd': '2',
        'n': '3',
        '9': '4',
        'c': '5',
        'm': '6',
        '0': '7',
        'b': '8',
        'l': '9',
        'a': '0'
    }

    re_objURL = re.compile(r'"objURL":"(.*?)".*?"type":"(.*?)"')
    re_downNum = re.compile(r"已下载\s(\d+)\s张图片")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.71 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, sdch",
    }

    SUPPORTED_FORMATS = ['jpg', 'png', 'jpeg']

    def __init__(self, class_id, word, pic_num_each_class, info_path='info', processNum=30):
        if " " in word:
            raise AttributeError("本脚本仅支持单个关键字")
        self.dirpath = '{:0>3d}'.format(class_id)
        self.word = word
        self.pic_num = pic_num_each_class
        self.jsonUrlFile = '{}{}{}_jsonUrl.txt'.format(info_path, os.sep, self.dirpath)
        self.logFile =  '{}{}{}_logInfo.txt'.format(info_path, os.sep, self.dirpath)
        self.errorFile =  '{}{}{}_errorUrl.txt'.format(info_path, os.sep, self.dirpath)
        if not os.path.exists(info_path):
            os.mkdir(info_path)
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        if os.path.exists(self.jsonUrlFile):
            os.remove(self.jsonUrlFile)
        if os.path.exists(self.logFile):
        	os.remove(self.logFile)
        if os.path.exists(self.errorFile):
        	os.remove(self.errorFile)

        self.char_table = {ord(key): ord(value)
                           for key, value in BaiduImgDownloader.char_table.items()}
        self.pool = Pool(processNum)
        self.session = requests.Session()
        self.session.headers = BaiduImgDownloader.headers
        self.queue = Queue()
        self.messageQueue = Queue()
        self.index = 0 # 图片起始编号，牵涉到计数，不要更改
        self.promptNum = 100 # 下载几张图片提示一次
        self.lock = threading.Lock()
        self.delay = 1.5  # 网络请求太频繁会被封
        self.QUIT = "QUIT"  # Queue中表示任务结束
        self.printPrefix = "**" # 用于指定在控制台输出

    def start(self):
        # 控制台输出线程
        t = threading.Thread(target=self.__log)
        t.setDaemon(True)
        t.start()
        self.messageQueue.put(self.printPrefix + "脚本开始执行")
        start_time = dt.now()
        urls = self.__buildUrls()
        self.messageQueue.put(self.printPrefix + "已获取 %s 个Json请求网址" % len(urls))
        # 解析出所有图片网址，该方法会阻塞直到任务完成
        self.pool.map(self.__resolveImgUrl, urls)
        while self.queue.qsize():
            imgs = self.queue.get()
            self.pool.map_async(self.__downImg, imgs)
        self.pool.close()
        self.pool.join()
        self.messageQueue.put(self.printPrefix + "下载完成！已下载 %s 张图片，总用时 %s" %
                              (self.index, dt.now() - start_time))
        self.messageQueue.put(self.printPrefix + "请到 %s 查看结果！" % self.dirpath)
        self.messageQueue.put(self.printPrefix + "错误信息保存在 %s" % self.errorFile)
        self.messageQueue.put(self.QUIT)


    def __log(self):
        """控制台输出，加锁以免被多线程打乱"""
        with open(self.logFile, "w", encoding = "utf-8") as f:
            while True:
                message = self.messageQueue.get()
                if message == self.QUIT:
                    break
                message = str(dt.now()) + " " + message
                if self.printPrefix  in message:
                    print(message)
                elif "已下载" in message:
                    # 下载N张图片提示一次
                    downNum = self.re_downNum.findall(message)
                    if downNum and int(downNum[0]) % self.promptNum == 0:
                        print(message)
                f.write(message + '\n')
                f.flush()

    def __getIndex(self):
        """获取文件编号"""
        self.lock.acquire()
        try:
            return self.index
        finally:
            self.index += 1
            self.lock.release()

    def decode(self, url):
        """解码图片URL
        解码前：
        ippr_z2C$qAzdH3FAzdH3Ffl_z&e3Bftgwt42_z&e3BvgAzdH3F4omlaAzdH3Faa8W3ZyEpymRmx3Y1p7bb&mla
        解码后：
        http://s9.sinaimg.cn/mw690/001WjZyEty6R6xjYdtu88&690
        """
        # 先替换字符串
        for key, value in self.str_table.items():
            url = url.replace(key, value)
        # 再替换剩下的字符
        return url.translate(self.char_table)

    def __buildUrls(self):
        """json请求网址生成器"""
        word = urllib.parse.quote(self.word)
        url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
        time.sleep(self.delay)
        urls = [url.format(word=word, pn=x)
                for x in range(0, self.pic_num + 1, 60)]
        with open(self.jsonUrlFile, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(url + "\n")
        return urls

    def __resolveImgUrl(self, url):
        """从指定网页中解析出图片URL"""
        time.sleep(self.delay)
        html = self.session.get(url, timeout = 15).content.decode('utf-8', 'ignore')
        datas = self.re_objURL.findall(html)
        imgs = [Image(self.decode(x[0]), x[1]) for x in datas]
        self.messageQueue.put(self.printPrefix + "已解析出 %s 个图片网址" % len(imgs))
        self.queue.put(imgs)

    def __downImg(self, img):
        """下载单张图片，传入的是Image对象"""
        """在爬取的URL中，下载制定格式的图像"""
        if img.type in self.SUPPORTED_FORMATS:
            imgUrl = img.url
                # self.messageQueue.put("线程 %s 正在下载 %s " %
            #          (threading.current_thread().name, imgUrl))
            try:
                time.sleep(self.delay)
                res = self.session.get(imgUrl, timeout = 15)
                message = None
                if str(res.status_code)[0] == "4":
                    message = "\n%s： %s" % (res.status_code, imgUrl)
                elif "text/html" in res.headers["Content-Type"]:
                    message = "\n无法打开图片： %s" % imgUrl
            except Exception as e:
                message = "\n抛出异常： %s\n%s" % (imgUrl, str(e))
            finally:
                if message:
                    self.messageQueue.put(message)
                    self.__saveError(message)
                    return
            index = self.__getIndex()
            # index从0开始
            self.messageQueue.put("已下载 %s 张图片：%s" % (index + 1, imgUrl))
            filename = os.path.join(self.dirpath, str(index) + "." + img.type)
            filename = "{}{}{:0>6d}.{}".format(self.dirpath, os.sep, index ,img.type)
            with open(filename, "wb") as f:
                f.write(res.content)

    def __saveError(self, message):
        self.lock.acquire()
        try:
            with open(self.errorFile, "a", encoding="utf-8") as f:
                f.write(message)
        finally:
            self.lock.release()


class Image(object):

    """图片类，保存图片信息"""

    def __init__(self, url, type):
        super(Image, self).__init__()
        self.url = url
        self.type = type

def down_image(foods, start_index, images_per_food):
    for id, food in enumerate(foods):
        down = BaiduImgDownloader(start_index + id, food, images_per_food)
        print("开始爬取××{}××图片，{}张...".format(food, images_per_food))
        down.start()


if __name__ == '__main__':
    num_cpus =multiprocessing.cpu_count()
    with open('keywords.txt', 'r') as f:
        foods = f.read().split()
    print("欢迎使用百度图片下载脚本！\n目前仅支持单个关键词。")
    print("=" * 50)

    processes = []

    food_per_proc = len(foods) // num_cpus
    residuals = len(foods) % num_cpus
    end_index = 0
    for i in range(num_cpus):
        start_index = end_index
        end_index = start_index + food_per_proc
        if residuals != 0:
            end_index += 1
            residuals -= 1
        processes.append(Process(target=down_image, args=(foods[start_index: end_index], start_index, 100)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
