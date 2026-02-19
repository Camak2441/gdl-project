import torch
from pathlib import Path
from torch.utils.data import Dataset

from data.query_data import QueryData


class QueriedSceneGraphDataset(Dataset):
    def __init__(self, root: Path):
        self.scene_graphs_dir = Path(root) / "scene_graphs"
        self.questions_dir = Path(root) / "questions"

        questions_files = sorted(self.questions_dir.glob("*.pth"))
        self.questions = []
        queries = []
        ys = {}
        qtypes = []
        self.scan_ids = []
        self.q_idxs = []
        for qf in questions_files:
            q = torch.load(qf, weights_only=False)

            num_qs = q["query"].shape[0]
            queries.append(q["query"])
            ys[q["scanId"]] = q["y"]
            qtypes.append(q["qtype"])
            self.scan_ids.extend([q["scanId"]] * num_qs)
            self.q_idxs.extend(range(q["query"].shape[0]))

        self.queries = torch.cat(queries)
        self.ys = ys
        self.qtypes = torch.cat(qtypes)

        self.scan_index: dict[str, Path] = {
            p.stem: p for p in self.scene_graphs_dir.glob("*.pth")
        }

        self.scans = {
            scan_id: torch.load(self.scan_index[scan_id], weights_only=False)
            for scan_id in self.scan_index
        }

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]

        return QueryData(
            x=self.scans[scan_id].x,
            pos=self.scans[scan_id].pos,
            edge_index=self.scans[scan_id].edge_index,
            edge_attr=self.scans[scan_id].edge_attr,
            query=self.queries[idx],
            y=self.ys[scan_id][self.q_idxs[idx]],
            qtype=self.qtypes[idx],
        )
