# -*- coding: utf-8 -*-
import scrapy
import logging
import urllib
import os
import sys
import glob
import re
import json 
from scrapy.selector import Selector
import logging
import time
logfile_name = time.ctime(time.time()).replace(' ', '_')
if not os.path.exists('logs/'):
    os.mkdir('logs/')
logging.basicConfig(filename=f'logs/{logfile_name}.log', filemode='a+',
                    format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class BaikeSpider(scrapy.Spider):
    name = 'baike'
    allowed_domains = ['baike.baidu.com']
    items = json.load(open("../entity.json"))
    start_urls = ['https://baike.baidu.com/item/'+e["ent"] for e in items]
    olds = set()

    print("We will crawl %d documents" % len(start_urls))
    # record and settings 
    data = []
    tot = 0
    save_step = 1000

    def parse(self, response):
        item_name = re.sub('/', '', re.sub('https://baike.baidu.com/item/',
                                           '', urllib.parse.unquote(response.url)))
        # pass the item which is crawled
        if item_name in self.olds:
            return
        # stroe main-content in data
        try:
            self.data.append(
                {
                    'ent': item_name,
                    'text': ''.join(response.xpath('//div[@class="lemma-summary"]').xpath('//div[@class="para"]//text()').getall())
                })
        except:
            pass
        # update olds
        self.olds.add(item_name)

        self.tot += 1
        sys.stdout.write("Processed: %d documents\r" % self.tot)
        sys.stdout.flush()
        if self.tot % self.save_step == 0:
            print("")
            print("----------------------------%d documents saved.----------------------------" % self.tot)
            json.dump(self.data, open("data.json", 'w'))
