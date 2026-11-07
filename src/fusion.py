import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ["B", "CD4 T", "CD8 T", "Mono", "NK"]


class FusionMLP(nn.Module):
    """
    Two encoders (protein, RNA) whose outputs are concatenated and classified.
    """

    def __init__(self, n_protein, n_rna, hidden=128, dropout=0.3, n_classes=5,
                 use_protein=True, use_rna=True):
        super().__init__()
        self.use_protein, self.use_rna = use_protein, use_rna
        width = 0
        if use_protein:
            self.protein_enc = nn.Sequential(
                nn.Linear(n_protein, hidden), 
                nn.ReLU(), 
                nn.Dropout(dropout))
            width += hidden
        if use_rna:
            self.rna_enc = nn.Sequential(
                nn.Dropout(0.2), 
                nn.Linear(n_rna, hidden), 
                nn.ReLU(), 
                nn.Dropout(dropout))
            width += hidden
        self.head = nn.Sequential(
            nn.Linear(width, hidden), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden, n_classes))

    def forward(self, protein, rna):
        parts = []
        if self.use_protein:
            parts.append(self.protein_enc(protein))
        if self.use_rna:
            parts.append(self.rna_enc(rna))
        return self.head(torch.cat(parts, dim=1))


def fit_predict_mlp(prot_tr, rna_tr, y_tr, prot_te_list, rna_te_list, epochs=30, lr=1e-3,
                    weight_decay=1e-3, batch_size=128, seed=0, class_weight=True, **model_kwargs):
    """
    Train a FusionMLP and return class probabilities for each (protein, rna) test pair.
    Scalers are fit on the training data only. y_tr is an int array 0-4.
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    ps, rs = StandardScaler().fit(prot_tr), StandardScaler().fit(rna_tr)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xp, Xr, y = t(ps.transform(prot_tr)), t(rs.transform(rna_tr)), torch.tensor(y_tr)

    model = FusionMLP(Xp.shape[1], Xr.shape[1], **model_kwargs)
    counts = np.bincount(y_tr, minlength=5)
    weight = torch.tensor(len(y_tr) / (5 * counts), dtype=torch.float32) if class_weight else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=epochs * ((len(y) + batch_size - 1) // batch_size))

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(y))
        for i in range(0, len(y), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xp[idx], Xr[idx]), y[idx]).backward()
            opt.step()
            sched.step()

    model.eval()
    outputs = []
    with torch.no_grad():
        for p, r in zip(prot_te_list, rna_te_list):
            logits = model(t(ps.transform(p)), t(rs.transform(r)))
            outputs.append(torch.softmax(logits, dim=1).numpy())
    return outputs


def fit_predict_lr(x_tr, y_tr, x_te_list, C=0.1):
    """
    Scaled logistic regression; returns class probabilities for each test matrix.
    """

    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_tr), y_tr)
    return [model.predict_proba(scaler.transform(x)) for x in x_te_list]
