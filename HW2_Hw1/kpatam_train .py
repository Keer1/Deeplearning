
import argparse, json, os, re, random, pickle, math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"

def build_vocab(train_label_json: str, min_freq: int = 5) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(train_label_json, "r") as f:
        ann = json.load(f)

    def tokenize(s: str) -> List[str]:
   
        return re.sub(r"[.!,;?\t\r\n]", " ", s.lower()).split()

    cnt = Counter()
    for item in ann:
        for cap in item["caption"]:
            cnt.update(tokenize(cap))

   
    itos = [PAD, BOS, EOS, UNK]
    for w, c in cnt.items():
        if c >= min_freq and w not in itos:
            itos.append(w)

    stoi = {w: i for i, w in enumerate(itos)}
    return dict(enumerate(itos)), stoi

def encode_sentence(s: str, stoi: Dict[str, int]) -> List[int]:
    toks = re.sub(r"[.!,;?\t\r\n]", " ", s.lower()).split()
    ids = [stoi.get(tok, stoi[UNK]) for tok in toks]
    return [stoi[BOS]] + ids + [stoi[EOS]]

def decode_ids(ids: List[int], itos: Dict[int, str]) -> str:
    words = []
    for i in ids:
        w = itos.get(i, UNK)
        if w == EOS:
            break
        if w not in (PAD, BOS):
            words.append("something" if w == UNK else w)
    return " ".join(words)


class TrainVideoCaptionSet(Dataset):
    
    def __init__(self, features_dir: str, labels_json: str, stoi: Dict[str, int]):
        super().__init__()
        with open(labels_json, "r") as f:
            ann = json.load(f)
        self.stoi = stoi
        self.samples = []  # list of (video_id, caption_ids)
        for item in ann:
            vid = item["id"]
            for cap in item["caption"]:
                self.samples.append((vid, encode_sentence(cap, stoi)))
        self.features_dir = features_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid, cap = self.samples[idx]
        feat_path = os.path.join(self.features_dir, f"{vid}.npy")
        feat = np.load(feat_path)  # [T, 4096]
        feat = torch.tensor(feat, dtype=torch.float32)
        cap = torch.tensor(cap, dtype=torch.long)
        return vid, feat, cap

def train_collate(batch):
    
    vids, feats, caps = zip(*batch)
    
    T = max(f.shape[0] for f in feats)
    Fdim = feats[0].shape[1]
    bsz = len(feats)
    feat_tensor = torch.zeros(bsz, T, Fdim, dtype=torch.float32)
    vid_lens = []
    for i, f in enumerate(feats):
        t = f.shape[0]
        feat_tensor[i, :t] = f
        vid_lens.append(t)

   
    L = max(c.shape[0] for c in caps)
    cap_tensor = torch.full((bsz, L), fill_value=0, dtype=torch.long)  
    cap_lens = []
    for i, c in enumerate(caps):
        l = c.shape[0]
        cap_tensor[i, :l] = c
        cap_lens.append(l)

    return vids, feat_tensor, torch.tensor(vid_lens, dtype=torch.long), cap_tensor, torch.tensor(cap_lens, dtype=torch.long)

class TestVideoSet(Dataset):
    """
    For inference. Expects a feat_dir with <video_id>.npy files.
    """
    def __init__(self, feat_dir: str):
        super().__init__()
        self.items = []
        for fn in sorted(os.listdir(feat_dir)):
            if fn.endswith(".npy"):
                vid = os.path.splitext(fn)[0]
                self.items.append((vid, os.path.join(feat_dir, fn)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        vid, path = self.items[idx]
        feat = np.load(path)
        feat = torch.tensor(feat, dtype=torch.float32)
        return vid, feat

def test_collate(batch):
    vids, feats = zip(*batch)
    T = max(f.shape[0] for f in feats)
    Fdim = feats[0].shape[1]
    bsz = len(feats)
    feat_tensor = torch.zeros(bsz, T, Fdim, dtype=torch.float32)
    vid_lens = []
    for i, f in enumerate(feats):
        t = f.shape[0]
        feat_tensor[i, :t] = f
        vid_lens.append(t)
    return vids, feat_tensor, torch.tensor(vid_lens, dtype=torch.long)


class Encoder(nn.Module):
    def __init__(self, in_dim=4096, proj_dim=512, hid=512, pdrop=0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.Tanh(),
            nn.Dropout(pdrop),
        )
        self.rnn = nn.LSTM(input_size=proj_dim, hidden_size=hid, num_layers=1, batch_first=True)

    def forward(self, x, lens):
        # x: [B, T, 4096]
        B, T, D = x.shape
        x = x.view(B * T, D)
        x = self.proj(x)
        x = x.view(B, T, -1)
        
        h, (h_n, c_n) = self.rnn(x)              
        return h, (h_n, c_n)                     

class AdditiveAttention(nn.Module):
    """Bahdanau-style: score = v^T tanh(W_h h_enc + W_s s_dec)."""
    def __init__(self, hid):
        super().__init__()
        self.W_h = nn.Linear(hid, hid, bias=False)
        self.W_s = nn.Linear(hid, hid, bias=False)
        self.v   = nn.Linear(hid, 1, bias=False)

    def forward(self, enc_seq, enc_mask, dec_state):
        
        B, T, H = enc_seq.shape
        
        s = self.W_s(dec_state).unsqueeze(1).expand(B, T, H)
        h = self.W_h(enc_seq)
        e = self.v(torch.tanh(h + s)).squeeze(-1)    
       
        e = e.masked_fill(enc_mask == 0, -1e9)
        a = F.softmax(e, dim=1)                      
        ctx = torch.bmm(a.unsqueeze(1), enc_seq).squeeze(1)  
        return ctx, a

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid=512, pdrop=0.35):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim + hid, hidden_size=hid, num_layers=1, batch_first=True)
        self.att = AdditiveAttention(hid)
        self.drop = nn.Dropout(pdrop)
        self.out = nn.Linear(hid, vocab_size)

    def forward_train(self, enc_seq, enc_mask, init_state, tgt, tf_ratio: float):
        
        B, L = tgt.shape
        H = enc_seq.shape[-1]
        (h_t, c_t) = init_state                         # each [1,B,H]
        h_t, c_t = h_t.transpose(0,1).contiguous(), c_t.transpose(0,1).contiguous()  # [B,1,H]
        h_t, c_t = h_t.squeeze(1), c_t.squeeze(1)       # [B,H]

        logits = []
        y_tm1 = tgt[:, 0]                               # start with BOS

        for t in range(1, L):
            # decide input token
            if t == 1 or random.random() < tf_ratio:
                emb_t = self.emb(y_tm1)                 # [B,E]
            else:
                # use previous model prediction
                prev_logits = logits[-1]                # [B,V]
                y_hat = prev_logits.argmax(dim=-1)
                emb_t = self.emb(y_hat)

            ctx, _ = self.att(enc_seq, enc_mask, h_t)   # [B,H]
            rnn_in = torch.cat([emb_t, ctx], dim=-1).unsqueeze(1)  # [B,1,E+H]
            out_t, (h1, c1) = self.rnn(rnn_in, (h_t.unsqueeze(0), c_t.unsqueeze(0)))
            h_t, c_t = h1.squeeze(0), c1.squeeze(0)
            logit_t = self.out(self.drop(out_t.squeeze(1)))        # [B,V]
            logits.append(logit_t)

            y_tm1 = tgt[:, t]                            # teacher token for next step (if used)

        return torch.stack(logits, dim=1)                # [B,L-1,V]

    def greedy_decode(self, enc_seq, enc_mask, init_state, bos_id: int, eos_id: int, max_len: int = 28):
        B = enc_seq.shape[0]
        (h_t, c_t) = init_state
        h_t, c_t = h_t.transpose(0,1).contiguous().squeeze(1), c_t.transpose(0,1).contiguous().squeeze(1)

        y_tm1 = torch.full((B,), bos_id, dtype=torch.long, device=enc_seq.device)
        outputs = []

        for _ in range(max_len-1):
            emb_t = self.emb(y_tm1)                      # [B,E]
            ctx, _ = self.att(enc_seq, enc_mask, h_t)    # [B,H]
            rnn_in = torch.cat([emb_t, ctx], dim=-1).unsqueeze(1)
            out_t, (h1, c1) = self.rnn(rnn_in, (h_t.unsqueeze(0), c_t.unsqueeze(0)))
            h_t, c_t = h1.squeeze(0), c1.squeeze(0)
            logit_t = self.out(out_t.squeeze(1))         # [B,V]
            y_hat = logit_t.argmax(dim=-1)               # [B]
            outputs.append(y_hat)
            y_tm1 = y_hat
        # outputs: list of [B] length max_len-1
        return torch.stack(outputs, dim=1)               # [B, max_len-1]

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, in_dim=4096, proj_dim=512, hid=512, emb_dim=512, pdrop=0.35):
        super().__init__()
        self.encoder = Encoder(in_dim, proj_dim, hid, pdrop)
        self.decoder = Decoder(vocab_size, emb_dim, hid, pdrop)

    def forward_train(self, feats, vid_lens, tgt, tf_ratio):
        # feats: [B,T,4096], tgt: [B,L]
        enc_seq, (h_n, c_n) = self.encoder(feats, vid_lens)
        # build mask: 1 for valid time steps
        B, T, _ = enc_seq.shape
        mask = torch.arange(T, device=feats.device).unsqueeze(0).expand(B, T) < vid_lens.unsqueeze(1)
        logits = self.decoder.forward_train(enc_seq, mask, (h_n, c_n), tgt, tf_ratio)
        return logits

    def forward_infer(self, feats, vid_lens, bos_id, eos_id, max_len=28):
        enc_seq, (h_n, c_n) = self.encoder(feats, vid_lens)
        B, T, _ = enc_seq.shape
        mask = torch.arange(T, device=feats.device).unsqueeze(0).expand(B, T) < vid_lens.unsqueeze(1)
        pred_ids = self.decoder.greedy_decode(enc_seq, mask, (h_n, c_n), bos_id, eos_id, max_len=max_len)
        return pred_ids

# -------------------------------
# Training / Evaluation Utilities
# -------------------------------
def scheduled_teacher_forcing(step: int, total_steps: int, floor: float = 0.3, ceil: float = 0.95):
    """
    Smooth schedule from ceil down to floor over total_steps using a cosine decay.
    """
    if total_steps <= 0:
        return ceil
    progress = min(step / total_steps, 1.0)
    # cosine decay from ceil -> floor
    return floor + 0.5 * (ceil - floor) * (1 + math.cos(math.pi * progress))

def token_cross_entropy(logits, targets, pad_idx: int):
    """
    logits: [B, L-1, V]
    targets: [B, L] (we compare against targets[:,1:])
    """
    B, Lm1, V = logits.shape
    tgt = targets[:, 1:]                              # drop BOS
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=pad_idx, reduction="sum")
    # average over non-PAD positions actually compared
    valid = (tgt != pad_idx).sum().clamp(min=1)
    return loss / valid

# -------------------------------
# Training Loop
# -------------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # vocab
    itos, stoi = build_vocab(args.train_labels, min_freq=args.min_freq)
    with open(os.path.join(args.save_dir, "i2w_kpatam.pkl"), "wb") as f:
        pickle.dump(itos, f)
    with open(os.path.join(args.save_dir, "w2i_kpatam.pkl"), "wb") as f:
        pickle.dump(stoi, f)

    # datasets
    train_ds = TrainVideoCaptionSet(
        features_dir=args.train_feats,
        labels_json=args.train_labels,
        stoi=stoi
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate
    )

    # model
    vocab_size = len(itos)
    model = Seq2Seq(
        vocab_size=vocab_size,
        in_dim=4096,
        proj_dim=args.proj_dim,
        hid=args.hidden_size,
        emb_dim=args.emb_dim,
        pdrop=args.dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    global_step = 0
    total_steps = args.epochs * max(1, len(train_loader))

    loss_curve = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for vids, feats, vid_lens, caps, cap_lens in train_loader:
            global_step += 1
            feats = feats.to(device)
            vid_lens = vid_lens.to(device)
            caps = caps.to(device)

            tf_ratio = scheduled_teacher_forcing(global_step, total_steps, floor=0.35, ceil=0.95)

            opt.zero_grad()
            logits = model.forward_train(feats, vid_lens, caps, tf_ratio)   # [B,L-1,V]
            loss = token_cross_entropy(logits, caps, pad_idx=0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        loss_curve.append(epoch_loss)
        print(f"[Epoch {epoch:03d}] train_loss={epoch_loss:.4f}  tf_ratio~{tf_ratio:.3f}")

        # checkpointing
        final_path = os.path.join(args.save_dir, "kpatam_seq2seq_last.pth")
        torch.save({"model": model.state_dict(), "itos": itos, "stoi": stoi}, final_path)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, "kpatam_seq2seq_best.pth")
            torch.save({"model": model.state_dict(), "itos": itos, "stoi": stoi}, best_path)
            print(f"  ↳ new best saved to {best_path} (loss {best_loss:.4f})")

    # write loss curve
    with open(os.path.join(args.save_dir, "kpatam_train_loss.txt"), "w") as f:
        for v in loss_curve:
            f.write(f"{v:.6f}\n")
    print("Training complete.")

# -------------------------------
# Inference
# -------------------------------
def infer(args):
    # load vocab + model
    ckpt = torch.load(args.model, map_location=device)
    itos, stoi = ckpt["itos"], ckpt["stoi"]
    vocab_size = len(itos)

    model = Seq2Seq(
        vocab_size=vocab_size,
        in_dim=4096,
        proj_dim=args.proj_dim,
        hid=args.hidden_size,
        emb_dim=args.emb_dim,
        pdrop=args.dropout
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = TestVideoSet(args.test_feats)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_collate
    )

    bos_id, eos_id = stoi[BOS], stoi[EOS]

    results = []
    with torch.no_grad():
        for vids, feats, vid_lens in test_loader:
            feats = feats.to(device)
            vid_lens = vid_lens.to(device)

            pred_ids = model.forward_infer(feats, vid_lens, bos_id=bos_id, eos_id=eos_id, max_len=args.max_len)
            # pred_ids: [B, max_len-1]
            for vid, ids in zip(vids, pred_ids.tolist()):
                caption = decode_ids(ids, itos)
                results.append((vid, caption))

    # write predictions
    with open(args.out, "w") as f:
        for vid, cap in results:
            f.write(f"{vid},{cap}\n")
    print(f"Wrote predictions → {args.out}")

# -------------------------------
# CLI
# -------------------------------
def get_argparser():
    p = argparse.ArgumentParser(description="kpatam HW2 Video Captioning (Seq2Seq + Attention)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared model hyperparams
    def add_model_args(q):
        q.add_argument("--proj_dim", type=int, default=512)
        q.add_argument("--hidden_size", type=int, default=512)
        q.add_argument("--emb_dim", type=int, default=512)
        q.add_argument("--dropout", type=float, default=0.35)

    # TRAIN
    t = sub.add_parser("train")
    t.add_argument("--train_labels", type=str, default=os.path.join("data", "training_label.json"))
    t.add_argument("--train_feats",  type=str, default=os.path.join("data", "training_data", "feat"))
    t.add_argument("--save_dir",     type=str, default="models")
    t.add_argument("--epochs",       type=int, default=30)
    t.add_argument("--batch_size",   type=int, default=64)
    t.add_argument("--num_workers",  type=int, default=2)
    t.add_argument("--lr",           type=float, default=1e-3)
    t.add_argument("--min_freq",     type=int, default=5)
    t.add_argument("--seed",         type=int, default=1337)
    add_model_args(t)

    # TEST
    te = sub.add_parser("test")
    te.add_argument("--model",       type=str, default=os.path.join("models", "kpatam_seq2seq_best.pth"))
    te.add_argument("--test_feats",  type=str, default=os.path.join("data", "testing_data", "feat"))
    te.add_argument("--out",         type=str, default="preds.txt")
    te.add_argument("--batch_size",  type=int, default=32)
    te.add_argument("--num_workers", type=int, default=0)
    te.add_argument("--max_len",     type=int, default=28)
    add_model_args(te)

    return p

def main():
    args = get_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "test":
        infer(args)

if __name__ == "__main__":
    main()
