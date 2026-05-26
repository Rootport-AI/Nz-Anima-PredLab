from __future__ import annotations

from typing import Any


def natten_status() -> dict[str, Any]:
    try:
        import natten  # type: ignore

        return {
            "available": True,
            "version": getattr(natten, "__version__", "unknown"),
            "reason": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "reason": str(exc),
        }


def local_attention_2d_torch(q, k, v, height: int, width: int, window: int, dilation: int):
    import math

    import torch
    import torch.nn.functional as F

    batch, tokens, heads, head_dim = q.shape
    expected_tokens = height * width
    if tokens != expected_tokens:
        raise ValueError(f"expected {expected_tokens} spatial tokens, got {tokens}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v shapes must match for local self-attention: {q.shape}, {k.shape}, {v.shape}")

    window = int(window)
    if window % 2 == 0:
        window += 1
    dilation = max(1, int(dilation))
    padding = dilation * (window // 2)
    patch_count = window * window

    query = q.permute(0, 2, 1, 3).contiguous()
    k_image = k.permute(0, 2, 3, 1).contiguous().view(batch, heads * head_dim, height, width)
    v_image = v.permute(0, 2, 3, 1).contiguous().view(batch, heads * head_dim, height, width)

    k_patches = F.unfold(k_image, kernel_size=window, dilation=dilation, padding=padding)
    v_patches = F.unfold(v_image, kernel_size=window, dilation=dilation, padding=padding)
    if k_patches.shape[-1] != tokens:
        raise ValueError(f"local attention produced {k_patches.shape[-1]} patches for {tokens} tokens")

    k_patches = k_patches.view(batch, heads, head_dim, patch_count, tokens)
    v_patches = v_patches.view(batch, heads, head_dim, patch_count, tokens)
    k_patches = k_patches.permute(0, 1, 4, 3, 2).contiguous()
    v_patches = v_patches.permute(0, 1, 4, 3, 2).contiguous()

    scores = torch.einsum("bhnd,bhnkd->bhnk", query, k_patches)
    scores = scores * (1.0 / math.sqrt(head_dim))
    valid = _local_valid_mask(height, width, window, dilation, padding, q.device)
    scores = scores.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    output = torch.einsum("bhnk,bhnkd->bhnd", weights, v_patches)
    return output.permute(0, 2, 1, 3).contiguous().view(batch, tokens, heads * head_dim)


def local_attention_2d_natten(q, k, v, height: int, width: int, window: int, dilation: int):
    from natten import na2d  # type: ignore

    batch, tokens, heads, head_dim = q.shape
    expected_tokens = height * width
    if tokens != expected_tokens:
        raise ValueError(f"expected {expected_tokens} spatial tokens, got {tokens}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v shapes must match for NATTEN self-attention: {q.shape}, {k.shape}, {v.shape}")

    window = int(window)
    if window % 2 == 0:
        window += 1
    dilation = max(1, int(dilation))

    q_2d = q.contiguous().view(batch, height, width, heads, head_dim)
    k_2d = k.contiguous().view(batch, height, width, heads, head_dim)
    v_2d = v.contiguous().view(batch, height, width, heads, head_dim)
    output = na2d(
        q_2d,
        k_2d,
        v_2d,
        kernel_size=(window, window),
        dilation=(dilation, dilation),
    )
    return output.contiguous().view(batch, tokens, heads * head_dim)


def _local_valid_mask(height: int, width: int, window: int, dilation: int, padding: int, device):
    import torch
    import torch.nn.functional as F

    ones = torch.ones(1, 1, height, width, device=device)
    valid = F.unfold(ones, kernel_size=window, dilation=dilation, padding=padding)
    valid = valid.view(1, 1, window * window, height * width)
    return valid.permute(0, 1, 3, 2).to(dtype=torch.bool)
