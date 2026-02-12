import numpy as np
import torch
import torch_geometric


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run_model(model, data):
    return model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        query=data.query,
        batch=data.batch,
    )


def get_recall_n(ranks, n):
    return torch.sum(torch.eq(ranks, n)).item()


EVAL_STATS = {"loss", "recall_1", "recall_3", "recall_5"}


@torch.no_grad()
def eval_model(model, dataset, criterion, device=DEVICE):
    total_loss = 0
    total_recall_1 = 0
    total_recall_3 = 0
    total_recall_5 = 0
    model.eval()
    for data in dataset:
        data.to(device)
        out = _run_model(model, data)
        loss = criterion(out, data.y)
        total_loss += loss.item()

        ranks = torch_geometric.utils.group_argsort(out, data.batch) + 1
        answer_ranks = ranks * data.y
        recall_1 = get_recall_n(answer_ranks, 1)
        recall_3 = get_recall_n(answer_ranks, 3)
        recall_5 = get_recall_n(answer_ranks, 5)
        total_recall_1 += recall_1
        total_recall_3 += recall_3
        total_recall_5 += recall_5
    return {
        "loss": total_loss,
        "recall_1": total_recall_1,
        "recall_3": total_recall_3,
        "recall_5": total_recall_5,
    }


def train(model, optimizer, dataset, criterion, device=DEVICE):
    total_loss = 0
    model.train()
    for data in dataset:
        data.to(device)
        optimizer.zero_grad()
        out = _run_model(model, data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss
