import sympy as sympy
import torch
from dataset import TrustDataset
from network import Model
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


test_dataset = TrustDataset("all_combined_prepared_with_demographics.xlsx", split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = Model(input_size=34).cuda()
model.load_state_dict(torch.load("test_f1_0.74157.pt", map_location="cuda:0"))
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
f1score = MulticlassF1Score(num_classes=5).cuda()
accuracy = MulticlassAccuracy(num_classes=5).cuda()

model.eval()
test_measures = [[] for i in range(3)]
y_true = []
y_pred = []
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
    y_true.append(y.item())
    y_pred.append(torch.argmax(pred, dim=-1).item())

print(f"Test Loss: {np.mean(test_measures[0])}")
print(f"Test Acc: {np.mean(test_measures[1])}")
print(f"Test F1: {np.mean(test_measures[2])}")



disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=[f"Trust Level {i+1}" for i in range(5)],
    cmap=plt.cm.Blues,
    normalize="pred",
)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("Trust Estimation")
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.pdf")
plt.savefig("confusion_matrix.jpg")