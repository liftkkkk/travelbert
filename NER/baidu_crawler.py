from urllib import request
from urllib.parse import quote
import pandas, json

def get_html(name):
	if name.strip() == "":
		return None
	with request.urlopen(r"https://baike.baidu.com/item/{}".format(quote(name))) as f:
		s = f.read().decode("utf8")
		return s

if __name__ == "__main__":

	df = pandas.read_csv("union.csv",skip_blank_lines=True,header=None)

	with open("baike.json", "a+") as f:
		for index, row in df.iterrows():
			if index < 0:
				continue
			s = get_html(row[0])
			if s is not None:
				json.dump({"name":row[0], "data":s}, f, ensure_ascii=False)
				f.write("\n")

			print("{}:{} is finished".format(index, row[0]))





