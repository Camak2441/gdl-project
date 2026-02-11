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


@torch.no_grad()
def eval_model(model, dataset, criterion, device=DEVICE):
    total_loss = 0
    model.eval()
    for data in dataset:
        data.to(device)
        out = _run_model(model, data)
        loss = criterion(out, data.y)
        # torch_geometric.utils.group_argsort(out, data.qtype)
        total_loss += loss.item()
    return total_loss


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
