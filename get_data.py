import unicodedata
import json
import numpy as np
import io


def read_txt(filename):
	with io.open(filename, "r", encoding="utf-8") as f:
		text = f.read()
	return text

def write(filename, text):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def read_json(path):
    with open(path, encoding='utf-8-sig') as json_file:
        json_data = json.load(json_file)
    return json_data



# print(read_txt('./data/Domain/Phone/pos.txt'))
