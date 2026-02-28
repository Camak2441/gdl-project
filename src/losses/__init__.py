import torch
import torch_geometric


class WeightedLosses(torch.nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        assert len(losses) == len(weights)
        for i, loss in enumerate(losses):
            self.register_module(str(i), loss)
        self.losses = list(zip(losses, weights))

    def forward(self, input, target):
        losses = [loss(input, target) * weight for loss, weight in self.losses]
        losses = torch.stack(losses, dim=-1)
        return torch.sum(losses, dim=-1)


class BatchLoss(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input, target, batch):
        inputs = torch_geometric.utils.unbatch(input, batch)
        targets = torch_geometric.utils.unbatch(target, batch)
        total_loss = 0
        for input, target in zip(inputs, targets):
            total_loss += self.loss(input, target)
        return total_loss


class RecallPrecisionWeightedLoss(torch.nn.Module):
    def __init__(
        self,
        loss,
        recall_weight=0.5,
        reduce="mean",
        total_weight_fn=None,
    ):
        super().__init__()
        self.loss = loss
        self.recall_weight = recall_weight
        self.reduce = reduce
        if reduce not in {"add", "mean"}:
            raise Exception(f"Unknown reduction type {self.reduce}")
        self.total_weights = total_weight_fn

    def forward(self, input, target):
        total_samples = target.shape[0]
        per_loss = self.loss(input, target)
        index = target.to(torch.int32)
        losses = (
            torch.zeros(2)
            .type_as(per_loss)
            .to(input.device)
            .scatter_add_(0, index, per_loss)
        ) / total_samples
        totals = (
            torch.zeros(2, dtype=torch.int32)
            .to(input.device)
            .scatter_add_(0, index, torch.ones_like(index).to(input.device))
        )
        match self.reduce:
            case "add":
                pass
            case "mean":
                total_weight = total_samples
                totals = torch.max(torch.ones_like(totals), totals)
                losses *= total_weight / totals
            case _:
                raise Exception(f"Unknown reduction type {self.reduce}")
        return (1 - self.recall_weight) * losses[0] + self.recall_weight * losses[1]
