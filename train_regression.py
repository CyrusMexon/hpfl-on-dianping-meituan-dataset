import os

# Set the maximum number of CPUs for joblib so that it does not complain about “0 physical cores.”
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from DS_data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import copy
import random
from sklearn.cluster import KMeans
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt  # For plotting

# Device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_n = 6
Epoch = 6

# Define a tolerance (in target units) for “accuracy.”
# If the absolute error is less than tolerance, we consider the prediction “correct.”
tolerance = 0.5


def Fedknow(w, weights, know, indice, method):
    # Determine the global number of exercises by taking the maximum rows among clients.
    global_exer_n = max(w[i]["k_difficulty.weight"].shape[0] for i in range(len(w)))
    feature_dim = w[0]["k_difficulty.weight"].shape[1]

    # Aggregate the shared parameter ("k_difficulty.weight") after padding.
    total_weight = sum(weights)
    agg_kd = torch.zeros(
        (global_exer_n, feature_dim), device=w[0]["k_difficulty.weight"].device
    )
    for i in range(len(w)):
        client_weight = w[i]["k_difficulty.weight"]
        r = client_weight.shape[0]
        # Create a padded tensor: rows 0..r-1 from client; remaining rows stay zero.
        padded = torch.zeros((global_exer_n, feature_dim), device=client_weight.device)
        padded[:r, :] = client_weight
        agg_kd += padded * weights[i]
    agg_kd /= total_weight

    # For the student embeddings (which may have heterogeneous sizes) we use clustering.
    student_centers = []
    for i in range(len(w)):
        stud_emb = w[i]["student_emb.weight"].detach().cpu().numpy()
        n_clusters = max(1, int(stud_emb.shape[0] / 5))
        cluster = KMeans(n_clusters=n_clusters, random_state=0)
        cluster.fit(stud_emb)
        student_centers.append(cluster.cluster_centers_)
    student_centers = np.vstack(student_centers)
    n_clusters_final = max(1, int(student_centers.shape[0] / 10))
    cluster = KMeans(n_clusters=n_clusters_final, random_state=0)
    cluster.fit(student_centers)
    student_group = cluster.cluster_centers_

    # Optionally, you can also cluster the aggregated exercise embedding.
    agg_kd_np = agg_kd.detach().cpu().numpy()
    n_clusters_ex = max(1, int(agg_kd_np.shape[0] / 10))
    cluster = KMeans(n_clusters=n_clusters_ex, random_state=0)
    cluster.fit(agg_kd_np)
    question_group = cluster.cluster_centers_

    # Build a new (global) model state dictionary.
    w_avg = copy.deepcopy(w[0])
    # Replace the exercise embedding with the aggregated (padded) one.
    w_avg["k_difficulty.weight"] = agg_kd
    # (You may choose to leave the student embeddings as is, since they’ll be updated via clustering.)
    return w_avg, student_group, question_group, []


def Apply(g_model, local, auc, student_group, question_group, method):
    l_w = copy.deepcopy(local.state_dict())

    if "fed" in method:
        metricstr = "euclidean"
        # Update student embeddings using the cluster centers from Fedknow.
        student_emb = l_w["student_emb.weight"]
        dist = cdist(
            student_emb.detach().cpu().numpy(), student_group, metric=metricstr
        )
        dist[np.isnan(dist)] = 1
        dist = dist / (np.sum(dist, axis=1, keepdims=True))
        dist = torch.FloatTensor(dist)
        centers = dist @ torch.FloatTensor(student_group)
        # Here auc[1] is used as a scalar mixing weight.
        q = torch.FloatTensor([auc[1]]).to(device).view(1, -1)
        l_w["student_emb.weight"] = l_w["student_emb.weight"] * q + centers.to(
            device
        ) * (1 - q)

        # Update exercise (k_difficulty) embeddings similarly.
        question_emb = l_w["k_difficulty.weight"]
        dist = cdist(
            question_emb.detach().cpu().numpy(), question_group, metric=metricstr
        )
        dist[np.isnan(dist)] = 1
        dist = dist / (np.sum(dist, axis=1, keepdims=True))
        dist = torch.FloatTensor(dist)
        centers = dist @ torch.FloatTensor(question_group)
        l_w["k_difficulty.weight"] = l_w["k_difficulty.weight"] * q + centers.to(
            device
        ) * (1 - q)
    local.load_state_dict(l_w)


def total(result):
    pred_all = []
    label_all = []
    for i in range(len(result)):
        pred_all += result[i][1]
        label_all += result[i][2]
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    rmse = np.sqrt(mean_squared_error(label_all, pred_all))
    mae = mean_absolute_error(label_all, pred_all)
    acc = np.mean(np.abs(np.array(label_all) - np.array(pred_all)) < tolerance)
    print(f"Federated RMSE: {rmse:.4f}")
    print(f"Federated MAE: {mae:.4f}")
    print(f"Federated Accuracy: {acc:.4f}")
    return rmse, mae, acc


def train(method):
    seed = 0
    # Read the config file to get all client IDs (distinct cities)
    config = np.loadtxt("config.txt", delimiter=" ", dtype=int)
    client_list = config[:, 0].tolist()
    Nets = []
    random.seed(seed)
    path = "hpfl_data/"
    for client in client_list:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        data_loader = TrainDataLoader(client, path)
        val_loader = ValTestDataLoader(client, path)
        net = Net(data_loader.student_n, data_loader.exer_n, data_loader.knowledge_n)
        net = net.to(device)
        Nets.append(
            [client, data_loader, net, copy.deepcopy(net.state_dict()), val_loader]
        )

    loss_function = nn.MSELoss()
    best_rmse = float("inf")

    # Lists to store global metrics per federated round
    global_rmse_list = []
    global_mae_list = []
    global_acc_list = []

    for i in range(Epoch):
        print(f"\n=== Federated Round {i+1}/{Epoch} ===")
        for index in range(len(Nets)):
            school = Nets[index][0]
            net = Nets[index][2]
            data_loader = Nets[index][1]
            val_loader = Nets[index][4]
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            print(f"\nTraining client {school}")
            net.train()
            for epoch in range(epoch_n):
                data_loader.reset()
                running_loss = 0.0
                while not data_loader.is_end():
                    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = (
                        data_loader.next_batch()
                    )
                    input_stu_ids = input_stu_ids.to(device)
                    input_exer_ids = input_exer_ids.to(device)
                    input_knowledge_embs = input_knowledge_embs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                val_results = validate(net, val_loader)
                current_rmse = val_results[0][0]
                print(
                    f"Epoch {epoch+1} - Loss: {running_loss:.8f}, Val RMSE: {current_rmse:.4f}"
                )
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    Nets[index][3] = copy.deepcopy(net.state_dict())
            net.load_state_dict(Nets[index][3])
            Nets[index][2] = net

        # Federated aggregation
        l_net = [item[3] for item in Nets]
        l_weights = [len(item[1].data) for item in Nets]
        global_model, student_group, question_group, _ = Fedknow(
            l_net, l_weights, [], [], method
        )
        for k in range(len(Nets)):
            Apply(
                global_model,
                Nets[k][2],
                [0.5, 0.5],
                student_group,
                question_group,
                method,
            )

        # Global evaluation across all clients
        print("\nGlobal Evaluation:")
        global_results = []
        for net in Nets:
            val_loader = net[4]
            results = validate(net[2], val_loader)
            global_results.append(results)
        rmse_val, mae_val, acc_val = total(global_results)
        # Save the metrics for plotting
        global_rmse_list.append(rmse_val)
        global_mae_list.append(mae_val)
        global_acc_list.append(acc_val)

    # After training, plot the global metrics over federated rounds.
    if not os.path.exists("plots"):
        os.makedirs("plots")

    rounds = list(range(1, len(global_rmse_list) + 1))

    plt.figure()
    plt.plot(rounds, global_rmse_list, marker="o", label="RMSE")
    plt.xlabel("Federated Round")
    plt.ylabel("RMSE")
    plt.title("Federated RMSE over Rounds")
    plt.legend()
    plt.savefig(os.path.join("plots", "federated_rmse.png"))
    plt.close()

    plt.figure()
    plt.plot(rounds, global_mae_list, marker="o", color="orange", label="MAE")
    plt.xlabel("Federated Round")
    plt.ylabel("MAE")
    plt.title("Federated MAE over Rounds")
    plt.legend()
    plt.savefig(os.path.join("plots", "federated_mae.png"))
    plt.close()

    plt.figure()
    plt.plot(rounds, global_acc_list, marker="o", color="green", label="Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Accuracy over Rounds")
    plt.legend()
    plt.savefig(os.path.join("plots", "federated_accuracy.png"))
    plt.close()

    print("\nPlots saved in the 'plots' directory.")


def validate(model, val_loader):
    model.eval()
    pred_all = []
    label_all = []
    with torch.no_grad():
        val_loader.reset()
        while not val_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, _, _ = (
                val_loader.next_batch()
            )
            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            outputs = model(input_stu_ids, input_exer_ids, input_knowledge_embs)
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            if outputs_np.ndim == 0:
                pred_all.append(outputs_np.item())
                label_all.append(labels_np.item())
            else:
                pred_all.extend(outputs_np.tolist())
                label_all.extend(labels_np.tolist())
    rmse = np.sqrt(mean_squared_error(label_all, pred_all))
    mae = mean_absolute_error(label_all, pred_all)
    acc = np.mean(np.abs(np.array(label_all) - np.array(pred_all)) < tolerance)
    print(f"Validation RMSE: {rmse:.3f}, MAE: {mae:.3f}, Accuracy: {acc:.3f}")
    return [rmse, mae, acc], pred_all, label_all


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_regression.py [device] [epoch_n] [method]")
        print("Example: python train_regression.py cuda:0 50 fedavg")
        sys.exit(1)
    device = torch.device(sys.argv[1])
    epoch_n = int(sys.argv[2])
    method = sys.argv[3]
    Epoch = epoch_n  # Set global epoch parameter
    train(method)
