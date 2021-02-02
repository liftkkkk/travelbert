
## Spider for Baidu-Baike
### Introduction
This directory are modified from github repo [WEB_KG](https://github.com/lixiang0/WEB_KG).

### Requirements
- python 3
- scrapy

### Quick Start：
Firstly, you should place a json-fromatted file containing items you want to crawl  in `.` directory.  We have placed the file`entity.json` in `.`. 
And then, run the following commands in linux shell or windows powershell.

```bash
cd baike
scrapy crawl baike
```
Finally, you will get `data.json` in directory `baike/`. The data in `data.json` is a python list in which each item is a python dict. And the format is as follows：
```python
[
	{'_id': "北京故宫", "text": "北京故宫又名北京紫禁城,...."},
	... other items ...
]
```

### Processed Data
We have processed a piece of data. The data is placed on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/c22873c68e0f4b7399c3/).

