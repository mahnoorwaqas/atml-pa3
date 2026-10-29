"""LoRA and OPLoRA implemented from scratch (used by Task 1 and Task 2).

LoRA:   W = W0 + (alpha/r) B A,                 A in R^{r x d_in} (random), B in R^{d_out x r} (zero)
OPLoRA: W = W0 + (alpha/r) B~ A~,  B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T
        with W0 = U S V^T and (U_k, V_k) the leading-k singular bases (frozen, computed once in FP32).
Effective factors are used (no dense d x d projection matrices are ever formed).
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS_ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha, dropout=0.05, mode="lora", Uk=None, Vk=None):
        super().__init__()
        assert mode in ("lora", "oplora")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale, self.mode = r, alpha, alpha / r, mode
        d_out, d_in = base.weight.shape
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))          # random A, zero B (standard LoRA init)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True
        if mode == "oplora":
            self.register_buffer("Uk", Uk.float().contiguous())    # [d_out, k]  (frozen)
            self.register_buffer("Vk", Vk.float().contiguous())    # [d_in,  k]  (frozen)
        else:
            self.Uk = self.Vk = None

    def effective_factors(self):
        """Return (B_eff, A_eff) in FP32; projected factors for OPLoRA."""
        if self.mode == "oplora":
            Bt = self.B - self.Uk @ (self.Uk.T @ self.B)
            At = self.A - (self.A @ self.Vk) @ self.Vk.T
            return Bt, At
        return self.B, self.A

    def delta_weight(self):
        Be, Ae = self.effective_factors()
        return self.scale * (Be @ Ae)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        Be, Ae = self.effective_factors()
        h = F.linear(self.drop(x).float(), Ae)                     # [.., r]
        return y + (self.scale * F.linear(h, Be)).to(y.dtype)


def leading_singular_bases(W, k):
    """Leading-k singular bases of W in FP32 (computed once)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(k, S.numel())
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


def inject_lora(model, targets, r, alpha, dropout=0.05, mode="lora", k=128, enabled=True):
    """Replace target nn.Linear modules by LoRALinear. Freezes everything else.
    Returns dict with overhead stats (SVD preprocessing time, basis storage bytes)."""
    for p in model.parameters():
        p.requires_grad_(False)
    stats = {"svd_time_s": 0.0, "basis_bytes": 0, "n_modules": 0}
    named = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for name, mod in named:
        Uk = Vk = None
        if mode == "oplora":
            t = time.perf_counter()
            Uk, Vk = leading_singular_bases(mod.weight.data, k)
            if mod.weight.is_cuda: torch.cuda.synchronize()
            stats["svd_time_s"] += time.perf_counter() - t
            stats["basis_bytes"] += (Uk.numel() + Vk.numel()) * 4
        new = LoRALinear(mod, r, alpha, dropout, mode, Uk, Vk)
        new.enabled = enabled
        for p in (new.A, new.B):
            p.requires_grad_(enabled)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.split(".")[-1], new)
        stats["n_modules"] += 1
    return stats


def set_lora_enabled(model, flag: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = flag
            m.A.requires_grad_(flag); m.B.requires_grad_(flag)


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state(model, export_projected=True):
    """State to save. For OPLoRA, export the *projected* factors (what inference uses)."""
    st = {}
    for n, m in lora_modules(model).items():
        Be, Ae = m.effective_factors() if export_projected else (m.B, m.A)
        st[n] = {"B": Be.detach().cpu().clone(), "A": Ae.detach().cpu().clone(), "scale": m.scale}
    return st


def raw_adapter_state(model):
    """Unprojected trainable factors (for resuming training)."""
    return {n: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for n, m in lora_modules(model).items()}


def load_raw_adapter_state(model, st):
    for n, m in lora_modules(model).items():
        m.A.data.copy_(st[n]["A"]); m.B.data.copy_(st[n]["B"])


@torch.no_grad()
def merge_adapters_into(fresh_model, adapter_st):
    """Merge exported (projected) factors into a *fresh* backbone copy in place."""
    mods = dict(fresh_model.named_modules())
    for n, f in adapter_st.items():
        lin = mods[n]
        dW = f["scale"] * (f["B"].float() @ f["A"].float())
        lin.weight.data = (lin.weight.data.float() + dW.to(lin.weight.device)).to(lin.weight.dtype)
    return fresh_model


def adapter_param_count(model):
    return sum(m.A.numel() + m.B.numel() for m in lora_modules(model).values())
