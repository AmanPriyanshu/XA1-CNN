import torch
from tqdm import tqdm

class XA1CNN(torch.nn.Module):
	def __init__(self, vocab_size, max_len, embedding_dim, n_gram=3, n_cnn=16, linear_layers=5, flatten=False):
		super(XA1CNN, self).__init__()
		self.vocab_size = vocab_size
		self.n_gram = n_gram
		self.embedding_dim = embedding_dim
		self.linear_layers = linear_layers
		self.n_cnn = n_cnn
		self.max_len = max_len
		self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
		self.cnn1d = torch.nn.Conv1d(self.max_len, self.n_cnn, self.n_gram)
		self.flatten = flatten
		self.flattener = torch.nn.Flatten()
		if flatten:
			self.neural_network = torch.nn.Sequential(*self.generate_network((self.n_cnn*(self.embedding_dim-2*(self.n_gram//2))), 1, 3))
		else:
			self.neural_network = torch.nn.Sequential(*self.generate_network(self.n_cnn, 1, 3))

	def generate_network(self, in_features, out_features, num_layers):
		layers = []
		layers_n = [in_features]+[out_features+i*(in_features - out_features)//num_layers for i in range(num_layers)][::-1]
		for i in range(1, num_layers):
			layers.append(torch.nn.Linear(layers_n[i-1], layers_n[i]))
			layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Linear(layers_n[i], out_features))
		return layers

	def forward(self, x, return_indices=False):
		x_embedded = self.embedding(x)
		x_cnn = self.cnn1d(x_embedded)
		if self.flatten:
			x_flattened = self.flattener(x_cnn)
			pred = self.neural_network(x_flattened)
			return torch.sigmoid(pred)
		else:
			x_cnn_masked, indexes = torch.max(x_cnn, dim=2)
			pred = self.neural_network(x_cnn_masked)
			if return_indices:
				return torch.sigmoid(pred), indexes
			else:
				return torch.sigmoid(pred)