import numpy as np
import torch
import torch_geometric


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5


def _run_model(model, data):
    return model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        query=data.query,
        batch=data.batch,
    )


def get_recall_n(ranks, y, batch, n):
    in_top_n = (ranks < n) & y.bool()
    per_graph = torch_geometric.utils.scatter(in_top_n.long(), batch, reduce='sum')
    return (per_graph >= 1).sum().item()


EVAL_STATS = {"loss", "recall_1", "recall_3", "recall_5"}
EVAL_MULTI_STATS = {"loss", "precision", "recall", "accuracy"}


@torch.no_grad()
def eval_model(model, dataset, criterion, device=DEVICE):
    total_loss = 0
    total_recall_1 = 0
    total_recall_3 = 0
    total_recall_5 = 0
    total_samples = 0
    model.eval()
    for data in dataset:
        data.to(device)
        out = _run_model(model, data)
        loss = criterion(out, data.y, data.batch)
        total_loss += loss.item() * data.batch_size

        ranks = torch_geometric.utils.group_argsort(out, data.batch, descending=True)
        total_recall_1 += get_recall_n(ranks, data.y, data.batch, 1)
        total_recall_3 += get_recall_n(ranks, data.y, data.batch, 3)
        total_recall_5 += get_recall_n(ranks, data.y, data.batch, 5)
        total_samples += data.batch_size
    return {
        "loss": total_loss / total_samples,
        "recall_1": total_recall_1 / total_samples,
        "recall_3": total_recall_3 / total_samples,
        "recall_5": total_recall_5 / total_samples,
    }


@torch.no_grad()
def eval_multi_model(model, dataset, criterion, device=DEVICE):
    """Evaluate the model in multi-answer mode.

    For each node prediction the model may flag multiple nodes as correct using a
    fixed threshold (THRESHOLD).  Precision, recall, and accuracy are accumulated
    globally across all batches and returned together with the loss.
    """
    total_loss = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_samples = 0
    model.eval()
    for data in dataset:
        data.to(device)
        out = _run_model(model, data)
        loss = criterion(out, data.y, data.batch)
        total_loss += loss.item()

        pred = out >= THRESHOLD
        y_bool = data.y.bool()
        total_tp += (pred & y_bool).sum().item()
        total_fp += (pred & ~y_bool).sum().item()
        total_fn += (~pred & y_bool).sum().item()
        total_tn += (~pred & ~y_bool).sum().item()
        total_samples += data.batch_size

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_nodes = total_tp + total_fp + total_fn + total_tn
    accuracy = (total_tp + total_tn) / total_nodes if total_nodes > 0 else 0.0
    return {
        "loss": total_loss / total_samples,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def train(model, optimizer, dataset, criterion, device=DEVICE):
    total_loss = 0
    total_samples = 0
    model.train()

    for data in dataset:
        data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = _run_model(model, data)
        loss = criterion(out, data.y, data.batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += data.batch_size

    return total_loss / total_samples
