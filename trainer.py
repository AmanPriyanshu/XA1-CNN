import torch
from dataloader import MovieDataset
from model import XA1CNN
from tqdm import tqdm
from matplotlib import pyplot as plt

def test(model, test_dataloader, criterion, epoch):
	model.eval()
	bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
	running_loss, running_acc = 0, 0
	for batch_idx, (batch_x, batch_y) in bar:
		preds = model(batch_x)
		loss = criterion(preds, batch_y)
		running_loss += loss.item()
		preds = preds.detach()
		y_pred = preds
		y_pred[y_pred<0.5] = 0
		y_pred[y_pred>=0.5] = 1
		acc = torch.mean((y_pred==batch_y).float())
		running_acc += acc.item()
		bar.set_description(str({"mode":"VALIDATION", "epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
	bar.close()

def train(train_dataloader, test_dataloader, epochs=100):
	model = XA1CNN(md.get_vocab_length(), md.max_len, 64)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	criterion = torch.nn.BCELoss()
	progress= {"epoch": [], "loss": [], "acc": []}
	for epoch in range(epochs):
		model.train()
		running_loss, running_acc = 0, 0
		bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
		for batch_idx, (batch_x, batch_y) in bar:
			optimizer.zero_grad()
			preds = model(batch_x)
			loss = criterion(preds, batch_y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			preds = preds.detach()
			y_pred = preds
			y_pred[y_pred<0.5] = 0
			y_pred[y_pred>=0.5] = 1
			acc = torch.mean((y_pred==batch_y).float())
			running_acc += acc.item()
			bar.set_description(str({"mode":"TRAINING", "epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
		progress["epoch"].append(epoch+1)
		progress["loss"].append(running_loss/(batch_idx+1))
		progress["acc"].append(running_acc/(batch_idx+1))
		bar.close()
		test(model, test_dataloader, criterion, epoch)
	torch.save(model.state_dict(), "./models/model.pt")
	progress = pd.DataFrame(progress)
	progress.to_csv("progress.csv", index=False)

if __name__ == '__main__':
	md = MovieDataset('./data/train.csv')
	md_dataloader = torch.utils.data.DataLoader(md, batch_size=32, shuffle=True)
	md_test = MovieDataset('./data/test.csv')
	md_dataloader_test = torch.utils.data.DataLoader(md_test, batch_size=32, shuffle=True)
	train(md_dataloader, md_dataloader_test, 150)