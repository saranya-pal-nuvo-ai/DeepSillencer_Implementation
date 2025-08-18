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

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

from data.Dataloader import preprocess_data, transform_data
from models.model_architecture import CombinedSilencer


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




class SelectivePairDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 sirna_embeddings,
                 mrna_embeddings,
                 tr_features: np.ndarray | None, 
                 alpha1: float = 0.05, 
                 alpha2: float = 0.20, 
                ):
                #  threshold: float = 0.70):
        
        self.df = df.reset_index(drop=True)
        self.sirna_embeddings = sirna_embeddings
        self.mrna_embeddings  = mrna_embeddings
        self.tr_features = tr_features
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.n = len(self.df)

        self.labels = self.df['label'].to_numpy(dtype=np.float32)
        self.cls_labels = self.df['y'].astype(np.int64)




    def __len__(self):
        # Each __getitem__ returns one pair. You can oversample by scaling this if needed.
        return self.n
    


    def _sample_pair_for_anchor(self, as_idx: int, MAX_LIM=10000):
        y_as = self.labels[as_idx]

        for _ in range(MAX_LIM):
            j = torch.randint(low=0, high=self.n, size=(1,)).item()
            if j == as_idx:
                continue
            if self.alpha1 <= abs(self.labels[j] - y_as) <= self.alpha2:
                return as_idx, j
            
        #   Just in case, the loop runs out and no j is chosen.....
        j = torch.randint(low=0, high=self.n, size=(1,)).item()
        return as_idx, j



    def __getitem__(self, _):
        as_idx = torch.randint(low=0, high=self.n, size=(1,)).item()
        i, j = self._sample_pair_for_anchor(as_idx)

        emb_i = torch.tensor(self.sirna_embeddings[i], dtype=torch.float32)  # (21, 640)
        emb_j = torch.tensor(self.sirna_embeddings[j], dtype=torch.float32)  # (21, 640)
        mrna_i = torch.tensor(self.mrna_embeddings[i], dtype=torch.float32)  # (59,640)
        mrna_j = torch.tensor(self.mrna_embeddings[j], dtype=torch.float32)


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

        return emb_i, emb_j, mrna_i, mrna_j, y1, y2, c1, c2, tr1, tr2      #   All of these returned values are torch.tensors
    


    

def collate_fn(batch):
    e1, e2, m1, m2, y1, y2, c1, c2, tr1, tr2 = zip(*batch)
    
    e1 = torch.stack(e1, dim=0)  # (B, 21, 640)
    e2 = torch.stack(e2, dim=0)  # (B, 21, 640)
    
    m1 = torch.stack(m1, dim=0)  # (B, 59, 640)
    m2 = torch.stack(m2, dim=0)
    
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

    return e1, e2, m1, m2, y1, y2, c1, c2, tr1, tr2





def train_epoch(model, loader, optimizer, device, beta_reg: float, lambda_cont: float = 2.0):
    model.train()
    total = 0
    loss_meter = 0.0
    loss_cls_meter = 0.0
    loss_reg_meter = 0.0
    loss_cont_meter = 0.0

    all_y1 = []
    all_y1_pred = []
    all_y2 = []
    all_y2_pred = []
    all_c1 = []
    all_p1 = []

    for batch in loader:
        e1, e2, m1, m2, y1, y2, c1, c2, tr1, tr2 = batch
        e1 = e1.to(device)
        e2 = e2.to(device)
        m1 = m1.to(device)
        m2 = m2.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        c1 = c1.to(device)
        c2 = c2.to(device)
        if tr1 is not None:
            tr1 = tr1.to(device)
            tr2 = tr2.to(device)

        y1_pred, p1, y2_pred, p2 = model(e1, e2, m1, m2, tr1, tr2)

        
        # loss_cls = classification_loss_from_probs(p1, c1)  # binary classification loss
        # loss_reg = smooth_l1_beta(y1_pred, y1, beta_reg) + smooth_l1_beta(y2_pred, y2, beta_reg)
        # diff_pred = y1_pred - y2_pred
        # diff_true = y1 - y2
        # loss_cont = smooth_l1_beta(diff_pred, diff_true, beta_reg)
        # loss = loss_cls + loss_reg + lambda_cont * loss_cont

        mse = nn.MSELoss()
        loss = mse(y1_pred, y1) + mse(y2_pred, y2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = e1.size(0)
        total += bs
        loss_meter += loss.item() * bs
        # loss_cls_meter += loss_cls.item() * bs
        # loss_reg_meter += loss_reg.item() * bs
        # loss_cont_meter += loss_cont.item() * bs

        
        all_y1.append(y1.detach().cpu().numpy())
        all_y1_pred.append(y1_pred.detach().cpu().numpy())
        all_y2.append(y2.detach().cpu().numpy())
        all_y2_pred.append(y2_pred.detach().cpu().numpy())
        all_c1.append(c1.detach().cpu().numpy())
        all_p1.append(p1.detach().cpu().numpy()[:, 1])


    y1_true = np.concatenate(all_y1)
    y1_pred = np.concatenate(all_y1_pred)
    y2_true = np.concatenate(all_y2)
    y2_pred = np.concatenate(all_y2_pred)
    c1_true = np.concatenate(all_c1)
    p1_pos_prob = np.concatenate(all_p1)

    # Safe metric calculation helper
    def safe_metric(f, x, y):
        try:
            return f(x, y)[0]
        except Exception:
            return np.nan

    pcc1 = safe_metric(pearsonr, y1_true.ravel(), y1_pred.ravel())
    pcc2 = safe_metric(pearsonr, y2_true.ravel(), y2_pred.ravel())
    spcc1 = safe_metric(spearmanr, y1_true.ravel(), y1_pred.ravel())
    spcc2 = safe_metric(spearmanr, y2_true.ravel(), y2_pred.ravel())
    roc_auc = roc_auc_score(c1_true, p1_pos_prob)

    return {
        'loss': loss_meter / total,
        'loss_cls': loss_cls_meter / total,
        'loss_reg': loss_reg_meter / total,
        'loss_cont': loss_cont_meter / total,
        'PCC_y1': pcc1,
        'PCC_y2': pcc2,
        'SPCC_y1': spcc1,
        'SPCC_y2': spcc2,
        'ROC_AUC': roc_auc,
    }






def evaluate_epoch(model, loader, device, beta_reg: float, lambda_cont: float = 2.0):
    model.eval()
    total = 0
    loss_meter = 0.0
    loss_cls_meter = 0.0
    loss_reg_meter = 0.0
    loss_cont_meter = 0.0

    all_y1 = []
    all_y1_pred = []
    all_y2 = []
    all_y2_pred = []
    all_c1 = []
    all_p1 = []


    with torch.no_grad():
        for batch in loader:
            e1, e2, m1, m2, y1, y2, c1, c2, tr1, tr2 = batch
            e1 = e1.to(device)
            e2 = e2.to(device)
            m1 = m1.to(device)
            m2 = m2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            if tr1 is not None:
                tr1 = tr1.to(device)
                tr2 = tr2.to(device)

            y1_pred, p1, y2_pred, p2 = model(e1, e2, m1, m2, tr1, tr2)

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



            all_y1.append(y1.detach().cpu().numpy())
            all_y1_pred.append(y1_pred.detach().cpu().numpy())
            all_y2.append(y2.detach().cpu().numpy())
            all_y2_pred.append(y2_pred.detach().cpu().numpy())
            all_c1.append(c1.detach().cpu().numpy())
            all_p1.append(p1.detach().cpu().numpy()[:, 1]) 

 
    y1_true = np.concatenate(all_y1)
    y1_pred = np.concatenate(all_y1_pred)
    y2_true = np.concatenate(all_y2)
    y2_pred = np.concatenate(all_y2_pred)
    c1_true = np.concatenate(all_c1)
    p1_pos_prob = np.concatenate(all_p1)

    def safe_metric(f, x, y):
        try:
            return f(x, y)[0]
        except Exception:
            return np.nan

    pcc1 = safe_metric(pearsonr, y1_true.ravel(), y1_pred.ravel())
    pcc2 = safe_metric(pearsonr, y2_true.ravel(), y2_pred.ravel())
    spcc1 = safe_metric(spearmanr, y1_true.ravel(), y1_pred.ravel())
    spcc2 = safe_metric(spearmanr, y2_true.ravel(), y2_pred.ravel())
    roc_auc = roc_auc_score(c1_true, p1_pos_prob)

    return {
        'loss': loss_meter / total,
        'loss_cls': loss_cls_meter / total,
        'loss_reg': loss_reg_meter / total,
        'loss_cont': loss_cont_meter / total,
        'PCC_y1': pcc1,
        'PCC_y2': pcc2,
        'SPCC_y1': spcc1,
        'SPCC_y2': spcc2,
        'ROC_AUC': roc_auc,
    }



def train_test_split(df, sirna_embeddings, mrna_embeddings, bio_features_df=None, test_size=0.2, seed=None):
    n = len(df)
    if seed is not None:
        np.random.seed(seed)

    idxs = np.random.permutation(n)
    split = int(n * (1 - test_size))

    tr_idx = idxs[:split].tolist()
    test_idx = idxs[split:].tolist()

    df_train = df.iloc[tr_idx]
    df_test = df.iloc[test_idx]

    sirna_train = [sirna_embeddings[i] for i in tr_idx]
    sirna_test = [sirna_embeddings[i] for i in test_idx]

    mrna_train = [mrna_embeddings[i] for i in tr_idx]
    mrna_test = [mrna_embeddings[i] for i in test_idx]

    if bio_features_df is not None:
        bio_features_array = bio_features_df.to_numpy(dtype=np.float32)
        
        bio_train = bio_features_array[tr_idx]
        tr_dim = bio_train.shape[1]
        
        bio_test = bio_features_array[test_idx]
        test_dm = bio_test.shape[1]
    else:
        bio_train = None
        tr_dim = None
        bio_test = None
        test_dm = None

    return df_train, df_test, sirna_train, sirna_test, mrna_train, mrna_test, bio_train, bio_test, tr_dim, test_dm







if __name__ == '__main__':
    set_seed(111)

    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    base_path = os.getenv('DATA_PATH')
    CACHE_PATH = os.getenv('CACHE_PATH')

    combined_df = transform_data(base_path)  

    siRNA_seq, siRNA_embeddings, mRNA_embeddings, bio_features_df = preprocess_data(
        base_path, CACHE_PATH, bio_features_return=True
    )


    com_df_train, com_df_test, sirna_train, sirna_test, mrna_train, mrna_test, tr_train, tr_test, tr_dim, test_dm = train_test_split(combined_df, siRNA_embeddings, mRNA_embeddings, bio_features_df, test_size=0.2)
    print(com_df_train.shape, com_df_test.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha1 = 0.05
    alpha2 = 0.20
 
    train_ds = SelectivePairDataset(com_df_train, sirna_train, mrna_train,  tr_train, alpha1, alpha2)
    val_ds = SelectivePairDataset(com_df_test, sirna_test, mrna_test, tr_test, alpha1, alpha2)

    batch_size = int(os.getenv('BATCH_SIZE', 64))
    num_workers = int(os.getenv('NUM_WORKERS', 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False)


    model = CombinedSilencer(
        d_model=128, num_layers=4, nhead=4, dim_ff=128 * 4, dropout=0.1,
        sirna_dim=640, mrna_dim=640, prior_dim=(tr_dim or 0)
    ).to(device)



    # def __init__(
    #     self,
    #     d_model: int = 128,
    #     num_layers: int = 4,
    #     nhead: int = 4,
    #     dim_ff: int = 128 * 4,
    #     dropout: float = 0.1,
    #     sirna_dim: int = 640,
    #     mrna_dim: int = 640,
    #     prior_dim: int | None = None,
    # ):

    
    lr = float(os.getenv('LEARNING_RATE', 3e-4))
    weight_decay = float(os.getenv('WEIGHT_DECAY', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    
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
            f"train loss {tr_metrics['loss']:.4f} | "
            f"train PCC_y1 {tr_metrics['PCC_y1']:.4f} SPCC_y1 {tr_metrics['SPCC_y1']:.4f} ROC_AUC {tr_metrics['ROC_AUC']:.4f} | "
            f"val loss {va_metrics['loss']:.4f} | "
            f"val PCC_y1 {va_metrics['PCC_y1']:.4f} SPCC_y1 {va_metrics['SPCC_y1']:.4f} ROC_AUC {va_metrics['ROC_AUC']:.4f}")


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
        }
            
        torch.save(ckpt, ckpt_dir / 'combined_best.pt')
