"""SigLIP(frozen, cached) -> MLP connector -> Qwen3 (frozen; LoRA in stages 2-3).

Sequence layout per example:  [pre-template] [visual tokens] [\\n question] [assistant prompt] [target + <|im_end|>]
Visual tokens are inserted INSIDE the user message, before the question, preserving Qwen's chat structure.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import lora as L

IMG, QPH = "<<IMG>>", "<<Q>>"
CAPS = {"question": 80, "caption": 64, "answer": 16}


class Connector(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_out
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


class VLM(nn.Module):
    def __init__(self, qwen, tok, feat_dim, n_tokens=196, backbone_dtype=torch.float16):
        super().__init__()
        assert n_tokens in (196, 49)
        self.qwen, self.tok, self.n_tokens = qwen, tok, n_tokens
        for p in self.qwen.parameters():
            p.requires_grad_(False)                                  # original backbone frozen throughout
        self.connector = Connector(feat_dim, qwen.config.hidden_size)
        self.register_buffer("vis_scale", torch.tensor(1.0))
        self.lora_stats = None
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")
        txt = tok.apply_chat_template([{"role": "user", "content": f"{IMG}\n{QPH}"}], add_generation_prompt=True,
                                      tokenize=False, enable_thinking=False)
        a, rest = txt.split(IMG); b, c = rest.split(QPH)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.pre, self.mid, self.post = enc(a), enc(b), enc(c)

    # ---------------------------------------------------------- visual tokens
    def visual(self, feats):
        x = feats.float()
        if self.n_tokens == 49:                                      # average each 2x2 block of the 14x14 grid
            B, _, D = x.shape
            x = x.view(B, 14, 14, D).permute(0, 3, 1, 2)
            x = F.avg_pool2d(x, 2).flatten(2).transpose(1, 2)
        return self.connector(x) * self.vis_scale                    # [B, n_tokens, H]

    @torch.no_grad()
    def calibrate(self, feats, examples):
        """Rule (training data only): vis_scale = median ||text token embedding|| / median ||connector token output||
        (connector at initialisation, scale 1).  Fixed afterwards."""
        self.vis_scale.fill_(1.0)
        emb = self.qwen.get_input_embeddings()
        ids = sorted({t for e in examples for t in self.tok(e["question"], add_special_tokens=False)["input_ids"]
                      + self.tok(e["target"], add_special_tokens=False)["input_ids"]})
        tn = emb(torch.tensor(ids, device=emb.weight.device)).float().norm(dim=-1).median().item()
        cn = self.visual(feats.to(emb.weight.device)).norm(dim=-1).median().item()
        self.vis_scale.fill_(tn / max(cn, 1e-8))
        return {"rule": "median text-embedding norm / median connector-output norm on training batch",
                "text_norm": tn, "connector_norm": cn, "scale": float(self.vis_scale)}

    # ---------------------------------------------------------- batching
    def _ex_ids(self, e, with_target=True, include_eos=True):
        q = self.tok(e["question"], add_special_tokens=False)["input_ids"][:CAPS["question"]]
        t, eos = [], []
        if with_target:
            cap = CAPS["caption"] if e.get("kind") == "caption" else CAPS["answer"]
            t = self.tok(e["target"], add_special_tokens=False)["input_ids"][:cap]
            eos = [self.eos] if include_eos else []
        return q, t, eos

    def build(self, examples, feats, use_image=True, with_target=True, include_eos=True, left_pad=False):
        """examples: list of dict(question, target[, kind]); feats: [B,196,D] fp16 or None.
        Returns inputs_embeds, attention_mask, labels, position_ids (all rebuilt for the actual lengths)."""
        dev = self.qwen.get_input_embeddings().weight.device
        emb = self.qwen.get_input_embeddings()
        dt = emb.weight.dtype
        vis = self.visual(feats.to(dev)) if (use_image and feats is not None) else None
        seqs, labs = [], []
        for i, e in enumerate(examples):
            q, t, eos = self._ex_ids(e, with_target, include_eos)
            pre = emb(torch.tensor(self.pre, device=dev))
            parts = [pre]; labels = [-100] * len(self.pre)
            if vis is not None:
                parts.append(vis[i].to(dt)); labels += [-100] * vis.shape[1]
            rest = self.mid + q + self.post
            parts.append(emb(torch.tensor(rest, device=dev))); labels += [-100] * len(rest)
            if t or eos:
                parts.append(emb(torch.tensor(t + eos, device=dev))); labels += t + eos
            seqs.append(torch.cat(parts)); labs.append(labels)
        T = max(len(l) for l in labs); B = len(seqs); H = seqs[0].shape[-1]
        X = torch.zeros(B, T, H, dtype=dt, device=dev); M = torch.zeros(B, T, dtype=torch.long, device=dev)
        Y = torch.full((B, T), -100, dtype=torch.long, device=dev)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            n = len(l); sl = slice(T - n, T) if left_pad else slice(0, n)
            X[i, sl] = s; M[i, sl] = 1; Y[i, sl] = torch.tensor(l, device=dev)
        pos = (M.cumsum(-1) - 1).clamp(min=0)
        return X, M, Y, pos

    # ---------------------------------------------------------- losses / likelihoods
    def example_nll(self, examples, feats, use_image=True, include_eos=True):
        """Per-example mean token NLL of the target (teacher forced). include_eos=False for answer NLL."""
        X, M, Y, pos = self.build(examples, feats, use_image, True, include_eos)
        logits = self.qwen(inputs_embeds=X, attention_mask=M, position_ids=pos, use_cache=False).logits[:, :-1]
        tgt = Y[:, 1:]; m = tgt != -100
        lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.clamp(min=0).reshape(-1), reduction="none").view_as(tgt)
        return -(lp * m).sum(1) / m.sum(1).clamp(min=1)

    @torch.no_grad()
    def generate(self, examples, feats, use_image=True, max_new=16):
        X, M, _, pos = self.build(examples, feats, use_image, with_target=False, left_pad=True)
        out = self.qwen.generate(inputs_embeds=X, attention_mask=M, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    # ---------------------------------------------------------- bookkeeping
    def add_lora(self, r=16, alpha=32, dropout=0.05, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
        self.lora_stats = L.inject_lora(self.qwen, list(targets), r, alpha, dropout, mode="lora", enabled=False)
        for p in self.connector.parameters():
            p.requires_grad_(True)

    def set_stage(self, stage):
        L.set_lora_enabled(self.qwen, stage >= 2)
        for p in self.connector.parameters():
            p.requires_grad_(True)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_count(self):
        return sum(p.numel() for p in self.trainable())

    def verify_frozen(self):
        for n, p in self.qwen.named_parameters():
            if p.requires_grad:
                assert (".A" in n[-3:] or ".B" in n[-3:] or n.endswith(".A") or n.endswith(".B")), f"backbone param trainable: {n}"
        assert all(p.requires_grad for p in self.connector.parameters())
        return True

    def lora_state(self):
        return L.raw_adapter_state(self.qwen)

    def load_lora_state(self, st):
        L.load_raw_adapter_state(self.qwen, st)
