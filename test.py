import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score
)
import torch.nn.functional as F
import pandas as pd
import random
import os
import model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     

set_seed(42) 


USE_CUDA = True
batch_size = 64
n_epochs = 30
best_num_cap = 64 
res = 64
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)
final_model_path = os.path.join(save_dir, "final_model_lnc-test-num64.pth")
train_data = pd.read_csv("/home/scao/CGR/Datasets/feature/lncTrain_rev.txt")
test_data = pd.read_csv("/home/scao/CGR/Datasets/feature/lncTest_rev.txt")
 



X_train_forward, X_train_reverse = [], []

for i in train_data['sequence']:
    vals = np.array(i.split(" "), dtype=np.float32)
    half = vals.size // 2
    f = vals[:half].reshape(res, res)
    r = vals[half:].reshape(res, res)
    X_train_forward.append(f)
    X_train_reverse.append(r)

y_train = train_data['label'][:len(X_train_forward)]

X_train = np.stack([np.array(X_train_forward), np.array(X_train_reverse)], axis=1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_test_forward, X_test_reverse = [], []

for i in test_data['sequence']:
    vals = np.array(i.split(" "), dtype=np.float32)
    half = vals.size // 2
    f = vals[:half].reshape(res, res)
    r = vals[half:].reshape(res, res)
    X_test_forward.append(f)
    X_test_reverse.append(r)

y_test = test_data['label'][:len(X_test_forward)]


X_test = np.stack([np.array(X_test_forward), np.array(X_test_reverse)], axis=1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("开始训练......")

capsule_net = model.CapsNet(Primary_capsule_num=best_num_cap, in_channels=2)
if USE_CUDA:
    capsule_net = capsule_net.cuda()

optimizer = Adam(capsule_net.parameters(), lr=1e-3, betas=(0.9, 0.999))

for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    for batch_id, (data, target) in enumerate(train_loader):
        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
        data, target = Variable(data), Variable(target)
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked, _, _ = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(capsule_net.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.4f}")


torch.save(capsule_net.state_dict(), final_model_path)
print(f"\n✅ Final model saved at: {final_model_path}")



# best_num_cap = 32  
# capsule_net = model.CapsNet(Primary_capsule_num=best_num_cap, in_channels=2)
# capsule_net.load_state_dict(torch.load('/home/scao/CGR/saved_models/final_model_mi.pth'))
# capsule_net = capsule_net.to("cuda")


capsule_net.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.inference_mode():
    for data, target in test_loader:
        if USE_CUDA:
            data = data.cuda()

        output, reconstructions, masked, _, _ = capsule_net(data)
        

        capsule_norms = torch.norm(output, dim=2) 
        probs = F.softmax(capsule_norms, dim=1).data.cpu().numpy() 
        preds = torch.argmax(capsule_norms, dim=1).data.cpu().numpy()
        labels = target.numpy()  
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs[:, 1]) 


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


cm = confusion_matrix(all_labels, all_preds)
TN, FP, FN, TP = cm.ravel()

ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
MCC = matthews_corrcoef(all_labels, all_preds)
try:
    AUC = roc_auc_score(all_labels, all_probs)
except ValueError:
    AUC = np.nan


SN = TP / (TP + FN + 1e-8)  # Sensitivity (Recall)
#SP = TN / (TN + FP + 1e-8)  # Specificity


F1 = f1_score(all_labels, all_preds)
Precision = precision_score(all_labels, all_preds)
Recall = recall_score(all_labels, all_preds)


print("\n===== Test Results =====")
print(f"ACC: {ACC:.4f}")
print(f"F1: {F1:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")
print(f"MCC: {MCC:.4f}")
print(f"AUC: {AUC:.4f}")
print("=========================")



