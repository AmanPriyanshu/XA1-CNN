import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def reader(path='./data/train/'):
	texts, tags = [], []
	for tag, path_seg in enumerate([path+i for i in ['neg', 'pos']]):
		for file in tqdm([path_seg+'/'+i for i in os.listdir(path_seg)]):
			with open(file, 'r', errors='ignore') as f:
				texts.append(f.read())
				tags.append(tag)
	csv = pd.DataFrame({'texts': texts, 'tags': tags})
	csv.to_csv(path+"data.csv", index=False)

if __name__ == '__main__':
	reader('./data/train/')
	reader('./data/test/')