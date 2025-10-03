#!/usr/bin/env python3
# test_seq2seq.py
# Usage: python3 test_seq2seq.py <data_dir> <output_file>
#   <data_dir>    e.g., data/testing_data  (also accepts "testing_data" and auto-prefixes "data/")
#   <output_file> e.g., testset_output.txt
#
# This script:
#  - loads models/kpatam_seq2seq_best.pth (state_dict checkpoint)
#  - loads vocab from models/i2w_kpatam.pkl (fallbacks supported)
#  - runs greedy inference on <data_dir>/feat/*.npy (shape [T,4096])
#  - writes "<video_id>,<caption>" lines to <output_file>
#  - if testing_label.json exists next to <data_dir>, prints Average BLEU

import os, sys, json, pickle, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"

# ---------------------------
# Data
# ---------------------------
class FeatOnlyTestSet(Dataset):
    def __init__(self, feat_dir: str):
        self.items = []
        for fn in sorted(os.listdir(feat_dir)):
            if fn.endswith(".npy"):
                vid = os.path.splitext(fn)[0]
                self.items.append((vid, os.path.join(feat_dir, fn)))
        if not self.items:
            raise FileNotFoundError(f"No .npy features found in {feat_dir}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        vid, path = self.items[i]
        arr = np.load(path)  # [T, 4096]
        return vid, torch.tensor(arr, dtype=torch.float32)

def collate_feats(batch):
    vids, feats = zip(*batch)
    T = max(x.shape[0] for x in feats)
    F = feats[0].shape[1]
    B = len(feats)
    out = torch.zeros(B, T, F, dtype=torch.float32)
    for i, x in enumerate(feats):
        out[i, :x.shape[0]] = x
    return list(vids), out

# ---------------------------
# Minimal model (matches trainer hyperparams)
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim=4096, proj_dim=512, hid=512, pdrop=0.35):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.Tanh(), nn.Dropout(pdrop))
        self.rnn = nn.LSTM(input_size=proj_dim, hidden_size=hid, num_layers=1, batch_first=True)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B*T, D)
        x = self.proj(x).view(B, T, -1)
        h, (h_n, c_n) = self.rnn(x)
        return h, (h_n, c_n)  # h:[B,T,H]

class AdditiveAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W_h = nn.Linear(hid, hid, bias=False)
        self.W_s = nn.Linear(hid, hid, bias=False)
        self.v   = nn.Linear(hid, 1,  bias=False)

    def forward(self, enc_seq, enc_mask, dec_state):
        B, T, H = enc_seq.shape
        s = self.W_s(dec_state).unsqueeze(1).expand(B, T, H)
        h = self.W_h(enc_seq)
        e = self.v(torch.tanh(h + s)).squeeze(-1)       # [B,T]
        e = e.masked_fill(enc_mask == 0, -1e9)
        a = torch.softmax(e, dim=1)                     # [B,T]
        ctx = torch.bmm(a.unsqueeze(1), enc_seq).squeeze(1)
        return ctx

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid=512, pdrop=0.35):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim + hid, hidden_size=hid, num_layers=1, batch_first=True)
        self.att = AdditiveAttention(hid)
        self.drop = nn.Dropout(pdrop)
        self.out = nn.Linear(hid, vocab_size)

    def greedy_decode(self, enc_seq, enc_mask, init_state, bos_id, eos_id, max_len=28):
        (h_t, c_t) = init_state
        h_t, c_t = h_t.transpose(0,1).contiguous().squeeze(1), c_t.transpose(0,1).contiguous().squeeze(1)
        y_tm1 = torch.full((enc_seq.shape[0],), bos_id, dtype=torch.long, device=enc_seq.device)
        outputs = []
        for _ in range(max_len-1):
            emb_t = self.emb(y_tm1)
            ctx   = self.att(enc_seq, enc_mask, h_t)
            rnn_in = torch.cat([emb_t, ctx], dim=-1).unsqueeze(1)
            out_t, (h1, c1) = self.rnn(rnn_in, (h_t.unsqueeze(0), c_t.unsqueeze(0)))
            h_t, c_t = h1.squeeze(0), c1.squeeze(0)
            logits = self.out(self.drop(out_t.squeeze(1)))
            y_hat  = logits.argmax(dim=-1)
            outputs.append(y_hat)
            y_tm1 = y_hat
        return torch.stack(outputs, dim=1)  # [B, max_len-1]

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, in_dim=4096, proj_dim=512, hid=512, emb_dim=512, pdrop=0.35):
        super().__init__()
        self.encoder = Encoder(in_dim, proj_dim, hid, pdrop)
        self.decoder = Decoder(vocab_size, emb_dim, hid, pdrop)

    def forward_infer(self, feats, bos_id=1, eos_id=2, max_len=28):
        B, T, _ = feats.shape
        enc_seq, (h_n, c_n) = self.encoder(feats)
        mask = torch.arange(T, device=feats.device).unsqueeze(0).expand(B, T) < T
        return self.decoder.greedy_decode(enc_seq, mask, (h_n, c_n), bos_id, eos_id, max_len)

# ---------------------------
# Vocab & BLEU helpers
# ---------------------------
def load_i2w():
    """Try common paths for index→word mapping."""
    candidates = [
        os.path.join("models", "i2w_kpatam.pkl"),
        "i2w_kpatam.pkl",
        "indexToWord_mapping.pickle",
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                return {i: w for i, w in enumerate(obj)}
            return obj  # dict[int->str]
    # Fallback: minimal vocab with just specials (works, but all UNK -> "something")
    return {0: PAD, 1: BOS, 2: EOS, 3: UNK}

def detok(ids, i2w):
    words = []
    for t in ids:
        w = i2w.get(int(t), UNK)
        if w == EOS: break
        if w not in (BOS, PAD):
            words.append("something" if w == UNK else w)
    return " ".join(words)

def maybe_bleu(labels_path, pred_path):
    try:
        if os.path.isfile(labels_path) and os.path.isfile("bleu_eval.py"):
            from bleu_eval import BLEU
            gold = json.load(open(labels_path, "r"))
            preds = {}
            with open(pred_path, "r") as f:
                for line in f:
                    k = line.find(",")
                    vid, cap = line[:k], line[k+1:].rstrip("\n")
                    preds[vid] = cap
            scores = []
            for item in gold:
                refs = [c.rstrip(".") for c in item["caption"]]
                hyp  = preds.get(item["id"], "")
                scores.append(BLEU(hyp, refs, True))
            avg = sum(scores) / max(1, len(scores))
            print(f"Average BLEU: {avg:.4f}")
    except Exception as e:
        # BLEU is optional; do not fail testing if it errors
        print(f"(BLEU skipped: {e})")

# ---------------------------
# Main
# ---------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 test_seq2seq.py <data_dir> <output_file>")
        sys.exit(1)

    data_dir = sys.argv[1]
    out_file = sys.argv[2]

    # Accept "testing_data" or full "data/testing_data"
    if not os.path.isdir(data_dir) and os.path.isdir(os.path.join("data", data_dir)):
        data_dir = os.path.join("data", data_dir)

    feat_dir  = os.path.join(data_dir, "feat")
    model_ckp = os.path.join("models", "kpatam_seq2seq_best.pth")

    if not os.path.isdir(feat_dir):
        print(f"ERROR: feat directory not found: {feat_dir}")
        sys.exit(2)
    if not os.path.isfile(model_ckp):
        print(f"ERROR: model checkpoint not found: {model_ckp}")
        print("Train first so the checkpoint exists.")
        sys.exit(3)

    # Load checkpoint (state_dict) & vocab
    ckpt = torch.load(model_ckp, map_location=DEVICE)
    itos = ckpt.get("itos", None)
    i2w  = ({int(k): v for k, v in itos.items()} if isinstance(itos, dict)
            else {i: w for i, w in enumerate(itos)} ) if itos is not None else load_i2w()
    vocab_size = len(i2w)

    # Rebuild model and load weights
    model = Seq2Seq(vocab_size=vocab_size).to(DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        print("ERROR: Unexpected checkpoint format (no 'model' in checkpoint).")
        sys.exit(4)
    model.eval()

    # Loader
    ds = FeatOnlyTestSet(feat_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_feats)

    # Inference
    results = []
    with torch.no_grad():
        for vids, feats in dl:
            feats = feats.to(DEVICE)
            pred_ids = model.forward_infer(feats, bos_id=1, eos_id=2, max_len=28)  # PAD=0,BOS=1,EOS=2,UNK=3
            for vid, row in zip(vids, pred_ids.tolist()):
                cap = detok(row, i2w)
                results.append((vid, cap))

    with open(out_file, "w") as f:
        for vid, cap in results:
            f.write(f"{vid},{cap}\n")
    print(f"Wrote predictions → {out_file}")

    # Optional BLEU: parent of data_dir + testing_label.json
    labels_path = os.path.join(os.path.dirname(data_dir.rstrip("/")), "testing_label.json")
    maybe_bleu(labels_path, out_file)

if __name__ == "__main__":
    main()
