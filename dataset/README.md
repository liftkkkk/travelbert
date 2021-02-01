# dataset

## 预训练语料更新

### SCI-bert语料
[v2021014纯净无噪版400token长度消除english-全国景点-旅游攻略+百度正文+摘要-纯文本语料库（sci-bert训练）](https://cloud.tsinghua.edu.cn/f/38fe0a8ed7b645918ccf/?dl=1)

数据量：2.79亿token

数据格式：

每行是400左右token，除了每一部分剩下的不够400token。

[v20210112纯净无噪版400token长度-全国景点-旅游攻略+百度正文+摘要-纯文本语料库（sci-bert训练）](https://cloud.tsinghua.edu.cn/f/86b275355bad44e9a32d/?dl=1)

数据量：2.79亿token

数据格式：

每行是400左右token，除了每一部分剩下的不够400token。

[v20201229纯净无噪版全国景点-旅游攻略+百度正文+摘要-纯文本语料库（sci-bert训练）](https://cloud.tsinghua.edu.cn/f/0371b3ddc9a74afca926/?dl=1)

数据量：2.3亿token

数据格式：

每行是最多两句话。

[v20201224全国景点-旅游攻略+百度正文+摘要-纯文本语料库（sci-bert训练）](https://cloud.tsinghua.edu.cn/f/2afc5d341a1340f1a90b/?dl=1)

数据格式：

每行是一段文字，包含多句话。

[v20201224全国景点-旅游攻略-纯文本语料库](https://cloud.tsinghua.edu.cn/f/2e6382c3bc914b098866/?dl=1)

### KAST语料
[v20210112-纯净无噪音版400token长度-全国景点加入content-table（kast训练）](https://cloud.tsinghua.edu.cn/f/3d72f657cda345cab8ff/?dl=1)

数据格式
```
{"header":北京故宫\t\t...\t\t,"text":[{"title":摘要,"content":[400token,400token,...]},{"title":"历史沿革","content":[400token,400token,...], ...}}
```
title用的是离着文本最近的一级标题，没有找到title的记作"title":"正文"。header和原来文件内容一样，景点的名字。

[v20201230-12万纯净无噪音版全国景点加入content-table全都用上（kast训练）](https://cloud.tsinghua.edu.cn/f/db040d3d66754f65941b/?dl=1)

以前的百科文本包含太多噪音

数据格式
```
{"header":北京故宫\t\t...\t\t,"text":[{"title":摘要,"content":[第1段,第2段,...]},{"title":"历史沿革","content":[第1段,第2段,...], ...}}
```
没有找到title的记作"title":"正文"。header和原来文件内容一样，景点的名字。

[v20201222-12万全国景点加入content-table全都用上（kast训练）](https://cloud.tsinghua.edu.cn/f/df5bda40b2ef4850b71d/?dl=1)

数据格式
```
北京故宫\t\t...\t\t摘要\t\t正文片段1\t\t正文片段2\t\t...\t\t正文片段n
```
每一个摘要和正文片段，都可单独作为wklm的文本，每个片段最长长度600 characters。这里面插入了content-table目录结构，无需额外处理， [unused11] 标出了摘要，[unused12]标出了一级目录 [unused13] 标出了二级目录

### 知识图谱实体

[v20201209-12万全国景点原始数据](https://cloud.tsinghua.edu.cn/f/ad565e2bca3a42f49973/?dl=1) === [词表](utils/全国景点) === [处理之后的数据](https://cloud.tsinghua.edu.cn/f/9f42433b7bd54c789491/?dl=1)

原始数据格式 format_baidu.json
```
{"url":"北京故宫", "html":"<html>...</html>"}
```
处理之后的数据 [处理工具](https://github.com/iamlockelightning/BaiduProcess)
```
baidubd_infobox.txt.clean.txt infobox
baidubd_abstract.txt.clean.txt 摘要
baidubd_article.txt.clean.txt 全文
```

[v20201126-九千实体高质量](https://cloud.tsinghua.edu.cn/f/7166f3bc2e7043b69359/?dl=1) === [下载词汇表](utils/20201126人工精细版)

[v20201120-KG三元组](https://cloud.tsinghua.edu.cn/f/1792b4937ff74f45a79b/?dl=1) === [词汇表](utils/第二次扩展后图谱实体)

[v20201109-6类实体的属性过滤下载](https://cloud.tsinghua.edu.cn/f/05a07d9f074c4e2fae91/?dl=1) === [词汇表](utils/entities)

[v20201009-8类实体的属性过滤下载](https://cloud.tsinghua.edu.cn/f/6f25b788b3d34d2fb7cf/?dl=1)

[v20201006-旅游知识图谱8类实体的属性三元组下载](https://cloud.tsinghua.edu.cn/f/ebf73ffec08c4994bffc/?dl=1)

[Crawler and data](utils/baike-spider/)


## 评测任务数据集更新
[v20201227全国景点-旅游攻略-travel-qa（travel-qa task）](https://cloud.tsinghua.edu.cn/f/a719ef3b5ac94f8086af/?dl=1)

每个问题有30个左右的候选答案，只有一个是最佳答案，最佳答案标签为1，其他答案标签为0.问题\t\t答案\t\tlabel
```
我持有港澳通行证但没签注，请问可以过关时在海关直接签注吗？急在线等		如果是广东省户籍，可以在深圳的各个口岸大厅，自助机上办理，立等可取。其他省份，回户口所在地的出入境管理局办事大厅，自助机上办理。每次每地15元，立等可取。如有帮助，麻烦您点一下采纳。谢谢了🙏		1
我持有港澳通行证但没签注，请问可以过关时在海关直接签注吗？急在线等		过关的时候海关不可能给你直接签注的，如果你是广东的卡式通行证，可以找家关口最近的自助签证大厅自助签证，一分钟搞定签注		0
我持有港澳通行证但没签注，请问可以过关时在海关直接签注吗？急在线等		不需要，直接港澳通行证及香港签注就可以了，现在都可以自助过关，很方便，望采纳		0
```

[NER数据集](utils/tner)