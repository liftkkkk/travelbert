import json 
import os 


def convert():
    data = []
    files = os.listdir(".")
    for file in files:
        type = file.split(".")[0]
        f = open(file, encoding="UTF8")
        lines = f.readlines()
        for line in lines:
            data.append({"ent":line.strip(), 'type':type})
    print("The number of all entities is %d" % (len(data)))
    json.dump(data, open("entity.json", 'w'))



if __name__ == "__main__":
    convert()
