from __future__ import annotations

from typing import Any


class FastChebyshevForecaster:
    def __init__(self, m: int, lam: float, steps: int):
        self.m = max(1, int(m))
        self.k = max(self.m + 2, 8)
        self.lam = max(0.0, float(lam))
        self.steps = max(1, int(steps))
        self.h_buf: list[Any] = []
        self.t_buf: list[float] = []
        self.time_buf: list[int] = []
        self.shape = None
        self.dtype = None
        self.device = None

    def reset(self) -> None:
        self.h_buf.clear()
        self.t_buf.clear()
        self.time_buf.clear()
        self.shape = None
        self.dtype = None
        self.device = None

    def update(self, cnt: int, h: Any) -> None:
        shape = getattr(h, "shape", None)
        dtype = getattr(h, "dtype", None)
        device = getattr(h, "device", None)
        if self.shape is not None and (shape != self.shape or dtype != self.dtype or device != self.device):
            self.reset()

        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.h_buf.append(h.detach().reshape(-1))
        self.t_buf.append(self._tau(cnt))
        self.time_buf.append(int(cnt))
        if len(self.h_buf) > self.k:
            self.h_buf.pop(0)
            self.t_buf.pop(0)
            self.time_buf.pop(0)

    def ready(self) -> bool:
        return bool(self.h_buf and self.shape is not None)

    def compatible(self, h: Any) -> bool:
        return (
            self.shape is None
            or (
                getattr(h, "shape", None) == self.shape
                and getattr(h, "dtype", None) == self.dtype
                and getattr(h, "device", None) == self.device
            )
        )

    def predict(self, cnt: int, w: float):
        import torch

        if not self.ready():
            raise RuntimeError("Spectrum forecaster has no history")

        device = self.h_buf[-1].device
        h = torch.stack(self.h_buf, dim=0).to(torch.float32)
        t = torch.tensor(self.t_buf, dtype=torch.float32, device=device)
        x = self._design(t)
        lam_i = self.lam * torch.eye(self.m + 1, device=device, dtype=torch.float32)
        xtx = x.T @ x + lam_i

        try:
            chol = torch.linalg.cholesky(xtx)
        except RuntimeError:
            jitter = 1e-5 * xtx.diag().mean()
            chol = torch.linalg.cholesky(
                xtx + jitter * torch.eye(self.m + 1, device=device, dtype=torch.float32)
            )

        coef = torch.cholesky_solve(x.T @ h, chol)
        tau_star = torch.tensor([self._tau(cnt)], dtype=torch.float32, device=device)
        pred_cheb = (self._design(tau_star) @ coef).squeeze(0)

        if len(self.h_buf) >= 2:
            h_i = self.h_buf[-1].to(torch.float32)
            h_im1 = self.h_buf[-2].to(torch.float32)
            t_i = self.time_buf[-1]
            t_im1 = self.time_buf[-2]
            dt = t_i - t_im1
            scale = (int(cnt) - t_i) / dt if abs(dt) > 1e-8 else 1.0
            pred_taylor = h_i + scale * (h_i - h_im1)
        else:
            pred_taylor = self.h_buf[-1].to(torch.float32)

        weight = max(0.0, min(1.0, float(w)))
        result = (1.0 - weight) * pred_taylor + weight * pred_cheb
        return torch.clamp(result, -10.0, 10.0).to(self.dtype).view(self.shape)

    def _tau(self, cnt: int) -> float:
        return (float(cnt) / float(self.steps)) * 2.0 - 1.0

    def _design(self, taus: Any):
        import torch

        taus = taus.reshape(-1, 1)
        terms = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.m > 0:
            terms.append(taus)
            for _ in range(2, self.m + 1):
                terms.append(2 * taus * terms[-1] - terms[-2])
        return torch.cat(terms[: self.m + 1], dim=1)
