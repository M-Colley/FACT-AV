import sympy as sympy
import torch
from dataset import TrustDataset
from network import Model
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from matplotlib.pylab import plt

from pathlib import Path

# Get the parent directory and construct the path to the data folder
data_folder = Path(__file__).parent.parent / "data"

# Construct the full file path
file_path = data_folder / "all_combined_prepared_with_demographics.xlsx"

results_folder = Path(__file__).parent.parent / "results" / "MLP"

epochs = 2000
batch_size = 16
learning_rate = 1e-4

train_dataset = TrustDataset(file_path, split="train")
valid_dataset = TrustDataset(file_path, split="valid")
test_dataset = TrustDataset(file_path, split="test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = Model(input_size=34).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
f1score = MulticlassF1Score(num_classes=5).cuda()
accuracy = MulticlassAccuracy(num_classes=5).cuda()


print(f"N steps: {len(train_loader)*epochs}")

history = []
for epoch in tqdm(range(epochs)):
    model.train()
    train_measures = [[] for i in range(3)]
    for x,y in train_loader:
        x = x.cuda()
        y = y.squeeze().cuda()
        logits = model(x)
        pred = logits.softmax(dim=-1)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = accuracy(pred, y)
        f1 = f1score(pred, y)
        train_measures[0].append(loss.item())
        train_measures[1].append(acc.item())
        train_measures[2].append(f1.item())     
    
     
    model.eval()
    valid_measures = [[] for i in range(3)]
    for x, y in valid_dataset:
        x = x.cuda()
        y = y.cuda().view(1)
        logits = model(x).view(1, -1)
        pred = logits.softmax(dim=-1)
        loss = criterion(logits, y)
        acc = accuracy(pred, y)
        f1 = f1score(pred, y)
        valid_measures[0].append(loss.item())
        valid_measures[1].append(acc.item())
        valid_measures[2].append(f1.item())
    
    history.append({
        "Train_Loss": np.mean(train_measures[0]),
        "Train_Acc": np.mean(train_measures[1]),
        "Train_F1": np.mean(train_measures[2]),
        "Valid_Loss": np.mean(valid_measures[0]),
        "Valid_Acc": np.mean(valid_measures[1]),
        "Valid_F1": np.mean(valid_measures[2]),
    })
    
    

    if epoch % 10 == 0 and epoch > 0:
 
        # Generate a sequence of integers to represent the epoch numbers
        eps = range(1, epoch+2)
        
        # Plot and label the training and validation loss values
        #plt.plot(epochs, [h["Loss"] for h in history], label='Training Loss')
        plt.plot(eps, [h["Train_Acc"] for h in history], label='Train Accuracy')
        plt.plot(eps, [h["Valid_Acc"] for h in history], label='Valid Accuracy')
        
        # Add in a title and axes labels
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        # Set the tick locations
        plt.xticks(np.arange(0, epochs, 20))
        plt.yticks(np.arange(0, 1, 0.2))
        
        # Display the plot
        plt.legend(loc='best')
        plt.savefig(f"epoch{epoch}.jpg")
        #plt.show()
        plt.close()
        
        

### EVAL

model.eval()
test_measures = [[] for i in range(3)]
for x, y in test_dataset:
    x = x.cuda()
    y = y.cuda().view(1)
    logits = model(x).view(1, -1)
    pred = logits.softmax(dim=-1)
    loss = criterion(logits, y)
    acc = accuracy(pred, y)
    f1 = f1score(pred, y)
    test_measures[0].append(loss.item())
    test_measures[1].append(acc.item())
    test_measures[2].append(f1.item())
    
print(f"Test Loss: {np.mean(test_measures[0])}")
print(f"Test Acc: {np.mean(test_measures[1])}")
print(f"Test F1: {np.mean(test_measures[2])}")

### SAVE WEIGHTS
file_path = results_folder / f"test_f1_{np.mean(test_measures[2]):0.5f}.pt"
torch.save(model.state_dict(), file_path)