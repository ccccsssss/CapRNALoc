import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, confusion_matrix, classification_report, matthews_corrcoef, f1_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import model
import pandas as pd
import copy
import os
import sys

USE_CUDA = True
torch.manual_seed(42)
np.random.seed(42)
if USE_CUDA:
    torch.cuda.manual_seed_all(42)


batch_size = 64
n_epochs = 70
res = 64
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "best_model_lnc_wochannel.pth")

fig = pd.read_csv("/home/scao/CGR/Datasets/feature/lncTrain_rev.txt")

X_forward, X_reverse = [], []

for i in fig['sequence']:
    vals = np.array(i.split(" "), dtype=np.float32)
    half = vals.size // 2
    f = vals[:half].reshape(res, res)
    r = vals[half:].reshape(res, res)
    X_forward.append(f)
    X_reverse.append(r)

y = fig['label'][:len(X_forward)]


X = np.stack([np.array(X_forward), np.array(X_reverse)], axis=1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


dataset = TensorDataset(X, y)
num_cap_list = [32]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


global_best_mcc = 0
global_best_info = {"num_cap": None, "fold": None, "epoch": None}


fold_metrics = {"ACC": [], "MCC": [], "F1": [], "Precision": [], "Recall": [],"AUC": []}

for num_cap in num_cap_list:
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        capsule_net = model.CapsNet(Primary_capsule_num=num_cap, in_channels=2)
        if USE_CUDA:
            capsule_net = capsule_net.cuda()
        optimizer = Adam(capsule_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        best_mcc = 0

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

            avg_train_loss = train_loss / len(train_loader)
            print(f"Train Loss: {avg_train_loss:.4f}")


            capsule_net.eval()
            all_preds, all_trues, all_probs = [], [], []
            with torch.inference_mode():
                correct_val, TP_val, FN_val, FP_val, TN_val = 0, 0, 0, 0, 0
                for data, target in val_loader:

                    if USE_CUDA:
                        data, target = data.cuda(), target.cuda()
                    output, reconstructions, masked, _, _ = capsule_net(data)


                    capsule_norms = torch.norm(output, dim=2)
                    probs = F.softmax(capsule_norms, dim=1).data.cpu().numpy()[:, 1]
                    pred_labels = np.argmax(masked.data.cpu().numpy(), 1)

                    true_labels = target.data.cpu().numpy()
                    all_preds.extend(pred_labels)
                    all_trues.extend(true_labels)
                    all_probs.extend(probs)
                    correct_val += np.sum(pred_labels == true_labels)
                    TP_val += np.sum((pred_labels == 1) & (true_labels == 1))
                    FN_val += np.sum((pred_labels == 0) & (true_labels == 1))
                    FP_val += np.sum((pred_labels == 1) & (true_labels == 0))
                    TN_val += np.sum((pred_labels == 0) & (true_labels == 0))

                avg_val_accuracy = correct_val / len(val_loader.dataset)
                recall_val = TP_val / (TP_val + FN_val + 1e-8)
                precision_val = TP_val / (TP_val + FP_val + 1e-8)
                mcc_val = (TP_val * TN_val - FP_val * FN_val) / np.sqrt((TP_val+FP_val)*(TP_val+FN_val)*(TN_val+FP_val)*(TN_val+FN_val) + 1e-8)

                print(f"num_cap={num_cap}, Fold {fold + 1}, Epoch {epoch + 1}/{n_epochs}")
                print(f"val Accuracy: {avg_val_accuracy:.4f}, val Recall: {recall_val:.4f}, val Precision: {precision_val:.4f},val MCC: {mcc_val:.4f}")


                if global_best_mcc < mcc_val:
                    global_best_mcc = mcc_val
                    global_best_info = {
                        "num_cap": num_cap,
                        "fold": fold + 1,
                        "epoch": epoch + 1
                    }
                    torch.save(capsule_net.state_dict(), best_model_path)
                    print(f"New best model saved (overwritten): {best_model_path}")


        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        all_probs = np.array(all_probs)

        ACC = np.mean(all_preds == all_trues)
        MCC = matthews_corrcoef(all_trues, all_preds)
        F1 = f1_score(all_trues, all_preds)
        Precision = precision_score(all_trues, all_preds)
        Recall = recall_score(all_trues, all_preds)


        try:
            AUC = roc_auc_score(all_trues, all_probs)
        except ValueError:
            AUC = np.nan

        fold_metrics["ACC"].append(ACC)
        fold_metrics["MCC"].append(MCC)
        fold_metrics["F1"].append(F1)
        fold_metrics["Precision"].append(Precision)
        fold_metrics["Recall"].append(Recall)
        fold_metrics["AUC"].append(AUC)

        print(f"Fold {fold + 1} Finished | ACC={ACC:.4f}, MCC={MCC:.4f}, "
              f"F1={F1:.4f}, Precision={Precision:.4f}, Recall={Recall:.4f}, AUC={AUC:.4f}\n")


    mean_ACC = np.nanmean(fold_metrics["ACC"])
    mean_MCC = np.nanmean(fold_metrics["MCC"])
    mean_F1 = np.nanmean(fold_metrics["F1"])
    mean_Precision = np.nanmean(fold_metrics["Precision"])
    mean_Recall = np.nanmean(fold_metrics["Recall"])
    mean_AUC = np.nanmean(fold_metrics["AUC"])

    print("\n==== 5-Fold Average Metrics ====")
    print(f"Mean ACC: {mean_ACC:.4f}")
    print(f"Mean F1 : {mean_F1:.4f}")
    print(f"Mean Precision: {mean_Precision:.4f}")
    print(f"Mean Recall: {mean_Recall:.4f}")
    print(f"Mean MCC: {mean_MCC:.4f}")
    print(f"Mean AUC: {mean_AUC:.4f}")


print("\n==== Training Finished ====")
print(f"Best Overall -> num_cap={global_best_info['num_cap']}, "
    f"Fold={global_best_info['fold']}, Epoch={global_best_info['epoch']}, "
    f"MCC={global_best_mcc:.4f}")
print(f"Best model saved at: {best_model_path}")
