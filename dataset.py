import glob
from typing import List, Dict, Any, Iterable
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

class ParquetDataset(IterableDataset):
    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for path in self.paths:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(
                    rg,
                    columns=["input_ids", "attention_mask", "label_index", "n_candidates"],
                )
                cols = table.to_pydict()
                n_rows = len(cols["label_index"])
                for i in range(n_rows):
                    yield {
                        "input_ids": cols["input_ids"][i],            # list of N lists
                        "attention_mask": cols["attention_mask"][i],  # list of N lists
                        "label_index": int(cols["label_index"][i]),
                        "n_candidates": int(cols["n_candidates"][i]),
                    }

def collate_queries(batch, pad_id: int = 1):
    """
    Inputs: list of dicts (one per query)
    Outputs:
      input_ids      : [B, max_N, max_L]
      attention_mask : [B, max_N, max_L]
      candidate_mask : [B, max_N]  (1 for real, 0 for padded candidate slots)
      label_index    : [B]
      Ns             : [B] (actual N per query; optional but handy for metrics)
    """
    B = len(batch)
    Ns = [len(ex["input_ids"]) for ex in batch]
    max_N = max(Ns) if Ns else 0

    # dynamic L (token length) across *all* pairs in this batch
    max_L = 0
    for ex in batch:
        for seq in ex["input_ids"]:
            if seq:
                max_L = max(max_L, len(seq))

    # allocate
    input_ids = torch.full((B, max_N, max_L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_N, max_L), dtype=torch.long)
    candidate_mask = torch.zeros((B, max_N), dtype=torch.bool)
    label_index = torch.empty((B,), dtype=torch.long)

    # fill
    for b, ex in enumerate(batch):
        N = len(ex["input_ids"])
        label_index[b] = ex["label_index"]
        candidate_mask[b, :N] = True

        for n in range(N):
            ids = ex["input_ids"][n]
            att = ex["attention_mask"][n]
            L = len(ids)
            if L > 0:
                input_ids[b, n, :L] = torch.tensor(ids, dtype=torch.long)
                attention_mask[b, n, :L] = torch.tensor(att, dtype=torch.long)

    return {
        "input_ids": input_ids,              # [B, max_N, max_L]
        "attention_mask": attention_mask,    # [B, max_N, max_L]
        "candidate_mask": candidate_mask,    # [B, max_N]
        "label_index": label_index,          # [B]
        "Ns": torch.tensor(Ns, dtype=torch.long),
    }