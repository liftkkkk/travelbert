# dataset

## 预训练语料更新

### 纯文本语料
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

### HKLM语料
[v20210112-纯净无噪音版400token长度-全国景点加入content-table（hklm训练）](https://cloud.tsinghua.edu.cn/f/3d72f657cda345cab8ff/?dl=1)

数据格式
```
{"header":北京故宫\t\t...\t\t,"text":[{"title":摘要,"content":[400token,400token,...]},{"title":"历史沿革","content":[400token,400token,...], ...}}
```
title用的是离着文本最近的一级标题，没有找到title的记作"title":"正文"。header和原来文件内容一样，景点的名字。

[v20201230-12万纯净无噪音版全国景点加入content-table全都用上（hklm训练）](https://cloud.tsinghua.edu.cn/f/db040d3d66754f65941b/?dl=1)

以前的百科文本包含太多噪音

数据格式
```
{"header":北京故宫\t\t...\t\t,"text":[{"title":摘要,"content":[第1段,第2段,...]},{"title":"历史沿革","content":[第1段,第2段,...], ...}}
```
没有找到title的记作"title":"正文"。header和原来文件内容一样，景点的名字。

[v20201222-12万全国景点加入content-table全都用上（hklm训练）](https://cloud.tsinghua.edu.cn/f/df5bda40b2ef4850b71d/?dl=1)

数据格式
```
北京故宫\t\t...\t\t摘要\t\t正文片段1\t\t正文片段2\t\t...\t\t正文片段n
```
每一个摘要和正文片段，都可单独作为wklm的文本，每个片段最长长度600 characters。这里面插入了content-table目录结构，无需额外处理， [unused11] 标出了摘要，[unused12]标出了一级目录 [unused13] 标出了二级目录

### 知识图谱实体

[v20201209-12万全国景点原始数据](https://cloud.tsinghua.edu.cn/f/ad565e2bca3a42f49973/?dl=1) === [词表](../Tourism-KG/全国景点) === [处理之后的数据](https://cloud.tsinghua.edu.cn/f/9f42433b7bd54c789491/?dl=1)

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

[v20201126-九千实体高质量](https://cloud.tsinghua.edu.cn/f/7166f3bc2e7043b69359/?dl=1) === [下载词汇表](../Tourism-KG/20201126人工精细版)

[v20201120-KG三元组](https://cloud.tsinghua.edu.cn/f/1792b4937ff74f45a79b/?dl=1) === [词汇表](../Tourism-KG/第二次扩展后图谱实体)

[v20201109-6类实体的属性过滤下载](https://cloud.tsinghua.edu.cn/f/05a07d9f074c4e2fae91/?dl=1) === [词汇表](../Tourism-KG/entities)

[v20201009-8类实体的属性过滤下载](https://cloud.tsinghua.edu.cn/f/6f25b788b3d34d2fb7cf/?dl=1)

[v20201006-旅游知识图谱8类实体的属性三元组下载](https://cloud.tsinghua.edu.cn/f/ebf73ffec08c4994bffc/?dl=1)

[Crawler and data](../Tourism-KG/baike-spider/)

### 知识图谱三元组
[knowldge triples describing tourist attractions](https://cloud.tsinghua.edu.cn/f/4a831744700c4f6f9146/?dl=1)
