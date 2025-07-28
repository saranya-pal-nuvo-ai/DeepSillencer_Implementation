import os
import sys
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.DataLoader import preprocess_data, transform_data
from models.model_architecture import ConvNetXtEncoder, TransformerEncoder


def set_seed(seed: int = 111):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def smooth_l1_beta(input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = 'mean'):
    """Smooth L1 (Huber) with explicit beta.
    Matches the piecewise definition used in many works.
    """
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss




def classification_loss_from_probs(y_prob: torch.Tensor, y_true: torch.Tensor):
    """y_prob: (B, 2) probabilities (Softmax outputs)
       y_true: (B,) int64 labels in {0,1}
    """
    # clamp for numerical stability
    y_prob = y_prob.clamp(min=1e-6, max=1.0)
    gathered = y_prob.gather(1, y_true.view(-1, 1)).squeeze(1)
    return -torch.log(gathered).mean()



# def selective_pair_sampling(df, as_idx, alpha1, alpha2):
#     n = len(df)
#     i = torch.randint(low=0, high=n, size=(1,)).item()

#     while i != as_idx:
#         cnt += 1
#         if i != as_idx and alpha1 <= np.abs(df.iloc[i]['label'] - df.iloc[as_idx]['label']) <= alpha2:
#             return as_idx, i
#         else:
#             i = torch.randint(low=0, high=n, size=(1,)).item()


class SelectivePairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sirna_embeddings, tr_features: np.ndarray | None,
                 alpha1: float = 0.05, alpha2: float = 0.20, threshold: float = 0.70):
        self.df = df.reset_index(drop=True)
        self.sirna_embeddings = sirna_embeddings
        self.tr_features = tr_features
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.threshold = threshold
        self.n = len(self.df)

        # ensure labels in [0,1]
        if self.df['label'].max() > 1.0:
            self.df['label'] = self.df['label'] / 100.0

        self.labels = self.df['label'].to_numpy(dtype=np.float32)
        self.cls_labels = (self.labels >= self.threshold).astype(np.int64)

        # sanity check for alignment
        if len(self.sirna_embeddings) != self.n:
            raise ValueError(f"Embeddings length {len(self.sirna_embeddings)} != dataframe rows {self.n}")
        if self.tr_features is not None and len(self.tr_features) != self.n:
            raise ValueError("TR feature rows do not match dataframe rows")

    def __len__(self):
        # Each __getitem__ returns one pair. You can oversample by scaling this if needed.
        return self.n

    def _sample_pair_for_anchor(self, as_idx: int):
        # dynamic random pairing until condition met
        # NOTE: guards against infinite loops by capping trials
        y_as = self.labels[as_idx]
        for _ in range(1000):
            j = torch.randint(low=0, high=self.n, size=(1,)).item()
            if j == as_idx:
                continue
            if self.alpha1 <= abs(self.labels[j] - y_as) <= self.alpha2:
                return as_idx, j
        # fallback: random j if condition failed after many tries
        j = torch.randint(low=0, high=self.n, size=(1,)).item()
        return as_idx, j

    def __getitem__(self, _):
        as_idx = torch.randint(low=0, high=self.n, size=(1,)).item()
        i, j = self._sample_pair_for_anchor(as_idx)

        emb_i = self.sirna_embeddings[i]  # (21, 640)
        emb_j = self.sirna_embeddings[j]  # (21, 640)

        # convert to tensors if not already
        if not torch.is_tensor(emb_i):
            emb_i = torch.tensor(emb_i, dtype=torch.float32)
        if not torch.is_tensor(emb_j):
            emb_j = torch.tensor(emb_j, dtype=torch.float32)

        y1 = torch.tensor(self.labels[i], dtype=torch.float32)
        y2 = torch.tensor(self.labels[j], dtype=torch.float32)
        c1 = torch.tensor(self.cls_labels[i], dtype=torch.long)
        c2 = torch.tensor(self.cls_labels[j], dtype=torch.long)

        if self.tr_features is not None:
            tr1 = torch.tensor(self.tr_features[i], dtype=torch.float32)
            tr2 = torch.tensor(self.tr_features[j], dtype=torch.float32)
        else:
            tr1 = torch.tensor([])
            tr2 = torch.tensor([])

        return emb_i, emb_j, y1, y2, c1, c2, tr1, tr2
    




# def make_a_batch(batch_size, start_idx, sirna_emb, mrna_emb, alpha1, alpha2):
#     batch_list = []
#     for _ in range(batch_size):
#         i, j = selective_pair_sampling(combined_df, start_idx, 0.1, 0.5)
#         batch_list.append( (i,j) )

    

def collate_fn(batch):
    e1, e2, y1, y2, c1, c2, tr1, tr2 = zip(*batch)
    e1 = torch.stack(e1, dim=0)  # (B, 21, 640)
    e2 = torch.stack(e2, dim=0)  # (B, 21, 640)
    y1 = torch.stack(y1, dim=0)
    y2 = torch.stack(y2, dim=0)
    c1 = torch.stack(c1, dim=0)
    c2 = torch.stack(c2, dim=0)

    # TR features may be empty tensors
    if tr1[0].numel() > 0:
        tr1 = torch.stack(tr1, dim=0)
        tr2 = torch.stack(tr2, dim=0)
    else:
        tr1 = None
        tr2 = None

    return e1, e2, y1, y2, c1, c2, tr1, tr2




class DeepSilencer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, nhead: int = 4,
                 dim_ff: int = 128 * 4, dropout: float = 0.1,
                 in_dim: int = 640, tr_dim: int | None = None):
        super().__init__()
        self.transformer = TransformerEncoder(
            in_dim=in_dim,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout
        )
        self.tr_proj = None
        if tr_dim is not None and tr_dim > 0:
            self.tr_proj = nn.Sequential(
                nn.Linear(tr_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        self.convnext = ConvNetXtEncoder(dropout=dropout)

    def forward_once(self, x, tr=None):
        # x: (B, L, in_dim)
        h = self.transformer(x)  # (B, L, d_model)
        if self.tr_proj is not None and tr is not None:
            b = self.tr_proj(tr)           # (B, d_model)
            h = h + b.unsqueeze(1)         # add as bias to each token
        h = h.transpose(1, 2)              # (B, d_model, L)
        y_reg, y_cls = self.convnext(h)
        return y_reg, y_cls

    def forward(self, e1, e2, tr1=None, tr2=None):
        y1_reg, y1_cls = self.forward_once(e1, tr1)
        y2_reg, y2_cls = self.forward_once(e2, tr2)
        return (y1_reg, y1_cls, y2_reg, y2_cls)
    




def train_epoch(model, loader, optimizer, device, beta_reg: float, lambda_cont: float = 2.0):
    model.train()
    total = 0
    loss_meter = 0.0
    loss_cls_meter = 0.0
    loss_reg_meter = 0.0
    loss_cont_meter = 0.0

    for batch in loader:
        e1, e2, y1, y2, c1, c2, tr1, tr2 = batch
        e1 = e1.to(device)
        e2 = e2.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        c1 = c1.to(device)
        # c2 is not used in cls loss (by design), but move to device to keep symmetry
        c2 = c2.to(device)
        if tr1 is not None:
            tr1 = tr1.to(device)
            tr2 = tr2.to(device)

        y1_pred, p1, y2_pred, p2 = model(e1, e2, tr1, tr2)

        # losses
        loss_cls = classification_loss_from_probs(p1, c1)
        loss_reg = smooth_l1_beta(y1_pred, y1, beta_reg) + smooth_l1_beta(y2_pred, y2, beta_reg)
        diff_pred = y1_pred - y2_pred
        diff_true = y1 - y2
        loss_cont = smooth_l1_beta(diff_pred, diff_true, beta_reg)

        loss = loss_cls + loss_reg + lambda_cont * loss_cont

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = e1.size(0)
        total += bs
        loss_meter += loss.item() * bs
        loss_cls_meter += loss_cls.item() * bs
        loss_reg_meter += loss_reg.item() * bs
        loss_cont_meter += loss_cont.item() * bs

    return {
        'loss': loss_meter / total,
        'loss_cls': loss_cls_meter / total,
        'loss_reg': loss_reg_meter / total,
        'loss_cont': loss_cont_meter / total,
    }




def evaluate_epoch(model, loader, device, beta_reg: float, lambda_cont: float = 2.0):
    model.eval()
    total = 0
    loss_meter = 0.0
    loss_cls_meter = 0.0
    loss_reg_meter = 0.0
    loss_cont_meter = 0.0

    with torch.no_grad():
        for batch in loader:
            e1, e2, y1, y2, c1, c2, tr1, tr2 = batch
            e1 = e1.to(device)
            e2 = e2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            if tr1 is not None:
                tr1 = tr1.to(device)
                tr2 = tr2.to(device)

            y1_pred, p1, y2_pred, p2 = model(e1, e2, tr1, tr2)

            loss_cls = classification_loss_from_probs(p1, c1)
            loss_reg = smooth_l1_beta(y1_pred, y1, beta_reg) + smooth_l1_beta(y2_pred, y2, beta_reg)
            diff_pred = y1_pred - y2_pred
            diff_true = y1 - y2
            loss_cont = smooth_l1_beta(diff_pred, diff_true, beta_reg)

            loss = loss_cls + loss_reg + lambda_cont * loss_cont

            bs = e1.size(0)
            total += bs
            loss_meter += loss.item() * bs
            loss_cls_meter += loss_cls.item() * bs
            loss_reg_meter += loss_reg.item() * bs
            loss_cont_meter += loss_cont.item() * bs

    return {
        'loss': loss_meter / total,
        'loss_cls': loss_cls_meter / total,
        'loss_reg': loss_reg_meter / total,
        'loss_cont': loss_cont_meter / total,
    }






if __name__ == '__main__':
    set_seed(111)

    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    base_path = os.getenv('DATA_PATH')
    CACHE_PATH = os.getenv('CACHE_PATH')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load / transform data
    combined_df = transform_data(base_path)  # must contain a 'label' column scaled either to 0-1 or 0-100

    # returns: siRNA_seq, siRNA_embeddings(list[(21,640)]), mRNA_embeddings(list[(59,640)]), bio_features_df (np.array or df)
    siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df = preprocess_data(
        base_path, CACHE_PATH, bio_features_return=True
    )

    # optional TR features (thermodynamic etc.)
    if bio_features_df is not None:
        if isinstance(bio_features_df, pd.DataFrame):
            tr_features = bio_features_df.to_numpy(dtype=np.float32)
        else:
            tr_features = np.asarray(bio_features_df, dtype=np.float32)
        tr_dim = tr_features.shape[1]
    else:
        tr_features = None
        tr_dim = None

    # Dataset / loaders
    alpha1 = 0.05
    alpha2 = 0.20
    threshold = 0.70

    train_ds = SelectivePairDataset(combined_df, siRNA_embeddings, tr_features, alpha1, alpha2, threshold)
    # for simplicity, use the same for eval here; you should split combined_df for real experiments
    val_ds = SelectivePairDataset(combined_df, siRNA_embeddings, tr_features, alpha1, alpha2, threshold)

    batch_size = int(os.getenv('BATCH_SIZE', 64))
    num_workers = int(os.getenv('NUM_WORKERS', 4))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # Model
    model = DeepSilencer(
        d_model=128, num_layers=4, nhead=4, dim_ff=128 * 4, dropout=0.1,
        in_dim=640, tr_dim=(tr_dim if tr_dim is not None else 0)
    ).to(device)

    # Optimizer
    lr = float(os.getenv('LEARNING_RATE', 3e-4))
    weight_decay = float(os.getenv('WEIGHT_DECAY', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss hyperparameters
    # If labels are scaled to 0-1, use beta_reg = 0.024; if 0-100, use 2.4
    beta_reg = float(os.getenv('BETA_REG', 0.024))
    lambda_cont = float(os.getenv('LAMBDA_CONT', 2.0))

    # Training loop
    epochs = int(os.getenv('EPOCHS', 20))
    best_val = float('inf')
    ckpt_dir = Path(os.getenv('CKPT_DIR', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_metrics = train_epoch(model, train_loader, optimizer, device, beta_reg, lambda_cont)
        va_metrics = evaluate_epoch(model, val_loader, device, beta_reg, lambda_cont)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_metrics['loss']:.4f} (cls {tr_metrics['loss_cls']:.4f} reg {tr_metrics['loss_reg']:.4f} cont {tr_metrics['loss_cont']:.4f}) | "
              f"val loss {va_metrics['loss']:.4f} (cls {va_metrics['loss_cls']:.4f} reg {va_metrics['loss_reg']:.4f} cont {va_metrics['loss_cont']:.4f})")

        # save best by total val loss
        if va_metrics['loss'] < best_val:
            best_val = va_metrics['loss']
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_metrics': va_metrics,
                'beta_reg': beta_reg,
                'lambda_cont': lambda_cont,
                'alpha1': alpha1,
                'alpha2': alpha2,
                'threshold': threshold,
            }
            torch.save(ckpt, ckpt_dir / 'deepsilencer_best.pt')

    # quick smoke test of sampler
    for _ in range(5):
        i, j = train_ds._sample_pair_for_anchor(5)
        print('sampled pair:', i, j)
