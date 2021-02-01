import glob
import pandas as pd

files = glob.glob("/Users/zhuhongyin/Downloads/44" + '/**/*.csv', recursive=True)
files.sort()

# my_path/     the dir
# **/       every file and dir under my_path
# *.txt     every file that ends with '.txt'

types = ["菜品","组织机构","建筑","文物","人物","门店","品牌","景点"]

processed_types = [""]
data = []

for f in files:
	print(f)
	entity_type = None
	for t in types:
		if t in f:
			entity_type = t
	
	if processed_types[-1] != entity_type:
		# print(dft)
		if len(data)>0:
			print("write file {}".format(processed_types[-1]))
			dft = pd.DataFrame(data, columns=['hash','entity', 'attribute', 'value'])
			dft = dft.sort_values('hash')
			dft.pop('hash')
			dft.to_csv("{}_filter.csv".format(processed_types[-1]), index= False)
			data=[]

	df = pd.read_csv(f) 
	df.fillna("", inplace = True)

	for index, row in df.iterrows():
		if row[1] == "http://travel.org/AttractionUrl":
			continue
		if row[1] == "http://travel.org/MainAttractions":
			continue
		if row[1] == "http://www.w3.org/2000/01/rdf-schema#label":
			continue
		if row[1] == "http://travel.org/ImageLink":
			continue
		if row[1] == "http://travel.org/图片":
			continue
		if "http://" in row[2]:
			continue

		data.append([hash(row[0]), row[0], row[1], row[2]])
		# print(row)

	processed_types.append(entity_type)

if len(data)>0:
	print("write file {}".format(processed_types[-1]))
	dft = pd.DataFrame(data, columns=['hash','entity', 'attribute', 'value'])
	dft = dft.sort_values('hash')
	dft.pop('hash')
	dft.to_csv("{}_filter.csv".format(processed_types[-1]), index= False)

	


