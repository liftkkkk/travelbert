import requests
import json
import sys


# url  = "http://166.111.68.66:9019/XLink/linkingSubmit.action"
# url="https://xlink.xlore.org/"
url="http://10.1.1.68:8081/EntityLinkingWeb/linkingSubmit.action"
text = "中华人民共和国成立于 1949.10.1"
lang = "zh"
data = {"text": text, "lang": lang}

request_result = requests.post(url, data)
print(request_result)
sys.exit()
link_result = json.loads(request_result.text)
print(link_result)
sys.exit()

url = "https://api.xlore.org/sigInfo?uri=http://xlore.org/instance/"
entity_id = "eni1"
entity_info_res = requests.get(url + entity_id)
entity_info = json.loads(entity_info_res.text)