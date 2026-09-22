import os
import copy
import random
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import cicrRNA_model as model





SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res = 64
n_splits = 5
max_epochs = 80
patience = 10
min_delta = 1e-4


num_cap_list = [4, 8, 16, 32]
learning_rate_list = 1e-3
batch_size_list =  16


train_file = "./Datasets"

save_dir = "./grid_search_results"
os.makedirs(save_dir, exist_ok=True)

result_csv = os.path.join(save_dir, "grid_search_results.csv")
best_params_txt = os.path.join(save_dir, "best_params.txt")
best_cv_model_path = os.path.join(save_dir, "best_cv_fold_model.pth")





print(f"Device: {DEVICE}")
print(f"Reading training data: {train_file}")

fig = pd.read_csv(train_file)

X_forward = []
X_reverse = []

for value in fig["sequence"]:
    vals = np.array(value.split(" "), dtype=np.float32)

    if vals.size != 2 * res * res:
        raise ValueError(
            f"Invalid CGR vector length: {vals.size}. "
            f"Expected {2 * res * res} for two {res}x{res} channels."
        )

    half = vals.size // 2
    forward = vals[:half].reshape(res, res)
    reverse = vals[half:].reshape(res, res)

    X_forward.append(forward)
    X_reverse.append(reverse)

X = np.stack(
    [np.array(X_forward), np.array(X_reverse)],
    axis=1
)
X = torch.tensor(X, dtype=torch.float32)

y_np = fig["label"].to_numpy(dtype=np.int64)
y = torch.tensor(y_np, dtype=torch.long)

dataset = TensorDataset(X, y)

print(f"Samples: {len(dataset)}")
print(f"Input shape: {tuple(X.shape)}")
print(f"Class 0: {(y_np == 0).sum()}, Class 1: {(y_np == 1).sum()}")





def evaluate(model_net, data_loader):
    model_net.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for data, target in data_loader:
            data = data.to(DEVICE, non_blocking=True)

            output, reconstructions, masked, _, _ = model_net(data)



            capsule_norms = torch.sqrt((output ** 2).sum(dim=2))
            if capsule_norms.dim() == 3:
                capsule_norms = capsule_norms.squeeze(-1)

            probs = F.softmax(capsule_norms, dim=1)
            preds = torch.argmax(capsule_norms, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(target.numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    acc = np.mean(all_preds == all_labels)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = np.nan

    return {
        "ACC": acc,
        "MCC": mcc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
    }





def train_one_fold(
    train_index,
    val_index,
    num_cap,
    learning_rate,
    batch_size,
    fold_id,
):
    set_seed(SEED + fold_id)

    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)

    generator = torch.Generator()
    generator.manual_seed(SEED + fold_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    capsule_net = model.CapsNet(
        Primary_capsule_num=num_cap,
        in_channels=2,
    ).to(DEVICE)

    optimizer = Adam(
        capsule_net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
    )

    best_mcc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        capsule_net.train()
        train_loss = 0.0

        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            target_onehot = F.one_hot(target, num_classes=2).float()

            optimizer.zero_grad()

            output, reconstructions, masked, _, _ = capsule_net(data)
            loss = capsule_net.loss(
                data,
                output,
                target_onehot,
                reconstructions,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(capsule_net.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_metrics = evaluate(capsule_net, val_loader)
        current_mcc = val_metrics["MCC"]

        print(
            f"Fold {fold_id} | Epoch {epoch:03d}/{max_epochs} | "
            f"Loss={avg_train_loss:.4f} | "
            f"ACC={val_metrics['ACC']:.4f} | "
            f"MCC={current_mcc:.4f} | "
            f"F1={val_metrics['F1']:.4f} | "
            f"AUC={val_metrics['AUC']:.4f}"
        )


        if current_mcc > best_mcc + min_delta:
            best_mcc = current_mcc
            best_epoch = epoch
            best_state = copy.deepcopy(capsule_net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, Best MCC={best_mcc:.4f}"
            )
            break


    capsule_net.load_state_dict(best_state)
    best_metrics = evaluate(capsule_net, val_loader)
    best_metrics["BestEpoch"] = best_epoch
    
    return best_metrics, copy.deepcopy(best_state)





kf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=SEED,
)

parameter_combinations = list(
    product(
        num_cap_list,
        learning_rate_list,
        batch_size_list,
    )
)

print("\n============================================================")
print("Grid Search Started")
print("============================================================")
print(f"Total parameter combinations: {len(parameter_combinations)}")
print(f"num_cap: {num_cap_list}")
print(f"learning_rate: {learning_rate_list}")
print(f"batch_size: {batch_size_list}")
print(f"Early stopping patience: {patience}")
print("Selection criterion: mean 5-fold validation MCC")
print("============================================================\n")


grid_results = []

global_best_mean_mcc = -np.inf
global_best_params = None
global_best_fold_state = None
global_best_fold_mcc = -np.inf


for config_id, (num_cap, learning_rate, batch_size) in enumerate(
    parameter_combinations,
    start=1,
):
    print("\n############################################################")
    print(
        f"Configuration {config_id}/{len(parameter_combinations)} | "
        f"num_cap={num_cap}, lr={learning_rate}, batch_size={batch_size}"
    )
    print("############################################################")

    fold_results = []
    config_best_fold_state = None
    config_best_fold_mcc = -np.inf

    for fold, (train_index, val_index) in enumerate(
        kf.split(np.zeros(len(y_np)), y_np),
        start=1,
    ):
        print(f"\n---------- Fold {fold}/{n_splits} ----------")

        metrics, fold_state = train_one_fold(
            train_index=train_index,
            val_index=val_index,
            num_cap=num_cap,
            learning_rate=learning_rate,
            batch_size=batch_size,
            fold_id=fold,
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold} best | "
            f"Epoch={metrics['BestEpoch']} | "
            f"ACC={metrics['ACC']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f} | "
            f"AUC={metrics['AUC']:.4f}"
        )

        if metrics["MCC"] > config_best_fold_mcc:
            config_best_fold_mcc = metrics["MCC"]
            config_best_fold_state = fold_state

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    mean_metrics = {
        metric: float(np.nanmean([x[metric] for x in fold_results]))
        for metric in ["ACC", "MCC", "F1", "Precision", "Recall", "AUC"]
    }

    std_mcc = float(np.nanstd([x["MCC"] for x in fold_results]))
    mean_best_epoch = float(
        np.mean([x["BestEpoch"] for x in fold_results])
    )

    row = {
        "num_cap": num_cap,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "Mean_ACC": mean_metrics["ACC"],
        "Mean_MCC": mean_metrics["MCC"],
        "Std_MCC": std_mcc,
        "Mean_F1": mean_metrics["F1"],
        "Mean_Precision": mean_metrics["Precision"],
        "Mean_Recall": mean_metrics["Recall"],
        "Mean_AUC": mean_metrics["AUC"],
        "Mean_BestEpoch": mean_best_epoch,
    }

    for i, fold_metrics in enumerate(fold_results, start=1):
        row[f"Fold{i}_MCC"] = fold_metrics["MCC"]
        row[f"Fold{i}_BestEpoch"] = fold_metrics["BestEpoch"]

    grid_results.append(row)


    pd.DataFrame(grid_results).to_csv(result_csv, index=False)

    print("\n===== Current configuration: 5-fold average =====")
    print(f"Mean ACC       : {mean_metrics['ACC']:.4f}")
    print(f"Mean MCC       : {mean_metrics['MCC']:.4f}")
    print(f"Std MCC        : {std_mcc:.4f}")
    print(f"Mean F1        : {mean_metrics['F1']:.4f}")
    print(f"Mean Precision : {mean_metrics['Precision']:.4f}")
    print(f"Mean Recall    : {mean_metrics['Recall']:.4f}")
    print(f"Mean AUC       : {mean_metrics['AUC']:.4f}")
    print(f"Mean BestEpoch : {mean_best_epoch:.1f}")


    if mean_metrics["MCC"] > global_best_mean_mcc:
        global_best_mean_mcc = mean_metrics["MCC"]

        global_best_params = {
            "num_cap": num_cap,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "mean_mcc": mean_metrics["MCC"],
            "std_mcc": std_mcc,
            "mean_best_epoch": mean_best_epoch,
        }

        global_best_fold_state = config_best_fold_state
        global_best_fold_mcc = config_best_fold_mcc



        torch.save(global_best_fold_state, best_cv_model_path)

        with open(best_params_txt, "w", encoding="utf-8") as f:
            f.write("Best hyperparameters selected by mean 5-fold MCC\n")
            f.write("================================================\n")
            f.write(f"num_cap = {num_cap}\n")
            f.write(f"learning_rate = {learning_rate}\n")
            f.write(f"batch_size = {batch_size}\n")
            f.write(f"mean_MCC = {mean_metrics['MCC']:.6f}\n")
            f.write(f"std_MCC = {std_mcc:.6f}\n")
            f.write(f"mean_best_epoch = {mean_best_epoch:.2f}\n")
            f.write(f"best_single_fold_MCC = {global_best_fold_mcc:.6f}\n")





results_df = pd.DataFrame(grid_results)
results_df = results_df.sort_values(
    by="Mean_MCC",
    ascending=False,
).reset_index(drop=True)
results_df.to_csv(result_csv, index=False)


print("\n\n============================================================")
print("Grid Search Finished")
print("============================================================")
print("Best parameters selected by mean 5-fold MCC:")
print(f"num_cap       : {global_best_params['num_cap']}")
print(f"learning_rate : {global_best_params['learning_rate']}")
print(f"batch_size    : {global_best_params['batch_size']}")
print(f"Mean MCC      : {global_best_params['mean_mcc']:.4f}")
print(f"Std MCC       : {global_best_params['std_mcc']:.4f}")
print(f"Mean BestEpoch: {global_best_params['mean_best_epoch']:.1f}")
print(f"\nGrid search table saved to: {result_csv}")
print(f"Best parameter file saved to: {best_params_txt}")
print(f"Reference CV checkpoint saved to: {best_cv_model_path}")
print(
    "\nNOTE: The saved checkpoint is only the best CV-fold model for reference. "
    "After selecting the hyperparameters, retrain the model on the entire "
    "training set before evaluating the independent test set."
)
