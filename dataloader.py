import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
import os

class MovieDataset(torch.utils.data.Dataset):
	def __init__(self, path, path_x='./data/train.pt', max_len=128, min_occurences=25, max_occurences=1000, shuffle=False, seed=0):
		self.min_occurences = min_occurences
		self.max_occurences = max_occurences
		self.max_len = max_len
		self.df = pd.read_csv(path)
		self.df = self.df.values
		np.random.seed(seed)
		np.random.shuffle(self.df)
		self.preprocess()
		self.vocab, self.vocab_count = self.make_vocab()
		self.sentences = [i for i in self.df.T[0]]
		if not os.path.exists(path_x):
			self.x = []
			for sentence in tqdm(self.sentences, desc="Converting to Torch"):
				self.x.append(self.convert_sentence(sentence))
			self.x = torch.stack(self.x)
			torch.save(self.x, path_x)
		else:
			self.x = torch.load(path_x)

	def preprocess(self):
		for idx, (text, _) in enumerate(tqdm(self.df, desc="Pre-Processing")):
			self.df[idx][0] = ' '.join([i for i in re.sub(r'[^a-zA-Z ]+', ' ', text).strip().split() if i != ''])

	def make_vocab(self):
		all_words = []
		for idx, (text, _ ) in enumerate(tqdm(self.df, desc="Generating Vocab")):
			all_words.extend(text.split())
		all_words = np.array(all_words)
		vocab, vocab_count = np.unique(all_words, return_counts=True)
		indexes = np.argsort(vocab_count)[::-1]
		vocab = vocab[indexes]
		vocab_count = vocab_count[indexes]
		valid_indexes = np.argwhere(np.all([self.min_occurences<=vocab_count, vocab_count<=self.max_occurences], axis=0)).flatten()
		vocab = vocab[valid_indexes]
		vocab_count = vocab_count[valid_indexes]
		return [i for i in vocab], vocab_count

	def get_vocab_length(self):
		return len(self.vocab)+1

	def convert_sentence(self, sentence):
		sentence = [self.vocab.index(word)+1 if word in self.vocab else 0 for word in sentence.split()]
		if len(sentence)<self.max_len:
			sentence += [0 for _ in range(self.max_len-len(sentence))]
		return torch.tensor(sentence[:self.max_len])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.x[idx], torch.tensor([float(self.df[idx][1])])

if __name__ == '__main__':
	md = MovieDataset('./data/train.csv')
	md_dataloader = torch.utils.data.DataLoader(md, batch_size=64, shuffle=True)
	for batch_x, batch_y in md_dataloader:
		print(batch_x.shape, batch_y.shape)