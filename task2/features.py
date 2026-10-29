"""Frozen SigLIP feature caching: 196 final patch features (14x14 order), final LayerNorm applied exactly once.

HF SiglipVisionModel.last_hidden_state is ALREADY post-layernormed (post_layernorm is inside the vision transformer),
so we use it as-is and never normalise again.  Features are cached in FP16; the encoder is then freed from GPU."""
import json, os, time
import numpy as np
import torch

from common.utils import get_device, save_json, peak_mem_gb, reset_peak_mem


class FeatureStore:
    def __init__(self, root):
        self.root = root
        self.index = json.load(open(os.path.join(root, "index.json")))
        n, d = self.index["n"], self.index["dim"]
        self.mm = np.memmap(os.path.join(root, "feats.f16"), dtype=np.float16, mode="r", shape=(n, 196, d))
        self.pos = {str(k): i for i, k in enumerate(self.index["ids"])}
        self.dim = d

    def get(self, ids):
        return torch.from_numpy(np.stack([self.mm[self.pos[str(i)]] for i in ids]).astype(np.float16))


@torch.no_grad()
def cache_features(image_paths: dict, out_dir, siglip=None, siglip_name="google/siglip-base-patch16-224", bs=64, device=None):
    """image_paths: {image_id: path}. Returns timing/memory dict (reported separately from training)."""
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel
    device = device or get_device(); os.makedirs(out_dir, exist_ok=True)
    proc = SiglipImageProcessor(size={"height": 224, "width": 224}) if siglip is not None else SiglipImageProcessor.from_pretrained(siglip_name)
    enc = siglip or SiglipVisionModel.from_pretrained(siglip_name, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    enc = enc.to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)                       # always frozen
    ids = sorted(image_paths, key=str); d = enc.config.hidden_size
    arr = np.memmap(os.path.join(out_dir, "feats.f16"), dtype=np.float16, mode="w+", shape=(len(ids), 196, d))
    reset_peak_mem(); t = time.perf_counter()
    for s in range(0, len(ids), bs):
        imgs = [Image.open(image_paths[i]).convert("RGB") for i in ids[s:s + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, enc.dtype)
        h = enc(pixel_values=px).last_hidden_state
        assert h.shape[1:] == (196, d), h.shape
        arr[s:s + len(imgs)] = h.float().cpu().numpy().astype(np.float16)
    arr.flush()
    if device.type == "cuda": torch.cuda.synchronize()
    info = {"n": len(ids), "dim": d, "ids": ids, "feature_cache_time_s": time.perf_counter() - t, "feature_cache_peak_mem_gb": peak_mem_gb()}
    json.dump(info, open(os.path.join(out_dir, "index.json"), "w"))
    del enc
    if device.type == "cuda": torch.cuda.empty_cache()                       # remove encoder from GPU before Qwen training
    return {k: v for k, v in info.items() if k != "ids"}
