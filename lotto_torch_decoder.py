"""Lotto649 v0.2 Phase 1: differentiable weighted k-subset decoder.

For unordered S of k distinct numbers in 1..n:
    log P(S|logits) = sum(logits[i-1] for i in S) - log Z
    Z = sum_{|T|=k} exp(sum(logits[j-1] for j in T)).

Numerically stable log-space DP evaluates Z in O(batch*n*k), while
PyTorch autograd differentiates the exact log likelihood. No training or
lottery predictions are performed by this module.
"""
from __future__ import annotations

import argparse
from math import comb, log

import torch
from torch import Tensor, nn

VERSION = "0.1.0"


def log_partition(logits: Tensor, k: int = 6) -> Tensor:
    """Return log elementary symmetric polynomial e_k(exp(logits)).

    Args:
        logits: finite float tensor [n] or [batch, n].
        k: exact number of unique selected numbers, 0 <= k <= n.
    Returns:
        Scalar for [n] input, or a length-batch tensor for [batch,n].

    Complexity: O(batch * n * k) operations and O(batch*k) DP state.
    """
    if not torch.is_tensor(logits) or logits.ndim not in (1, 2):
        raise ValueError("logits must have shape [n] or [batch,n]")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    squeeze = logits.ndim == 1
    values = logits.unsqueeze(0) if squeeze else logits
    batch, n = values.shape
    if n < 1 or batch < 1 or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("invalid shape or k")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("logits must be finite")

    zero = values.new_zeros(batch)
    impossible = values.new_full((batch,), -torch.inf)
    dp = [zero] + [impossible] * k
    for i in range(n):
        # Descending cardinality avoids reusing this same item's contribution.
        for j in range(min(i + 1, k), 0, -1):
            dp[j] = torch.logaddexp(dp[j], dp[j - 1] + values[:, i])
    result = dp[k]
    return result[0] if squeeze else result


class FixedCardinalityDecoder(nn.Module):
    """Differentiable exact normalized distribution over n-choose-k subsets.

    Numbers are 1-based and unordered; input validates distinctness.
    Forward returns a per-example log probability (not an averaged loss).
    """

    def __init__(self, n: int = 49, k: int = 6):
        super().__init__()
        if not isinstance(n, int) or not isinstance(k, int) or n < 1 or k < 1 or k > n:
            raise ValueError("require 1 <= k <= n")
        self.n, self.k = n, k

    def forward(self, logits: Tensor, numbers: Tensor) -> Tensor:
        if not torch.is_tensor(logits) or logits.ndim not in (1, 2):
            raise ValueError("logits must be [n] or [batch, n]")
        single = logits.ndim == 1
        batched = logits.unsqueeze(0) if single else logits
        if batched.shape[1] != self.n:
            raise ValueError(f"expected {self.n} logits per example")
        if not torch.is_tensor(numbers):
            raise ValueError("numbers must be a tensor of integer ids")
        if numbers.ndim == 1:
            numbers = numbers.unsqueeze(0)
        if numbers.shape != (batched.shape[0], self.k):
            raise ValueError(f"numbers must have shape [batch,{self.k}]")
        if numbers.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("numbers must have an integer dtype")
        if numbers.device != batched.device:
            raise ValueError("numbers and logits must use the same device")
        if not bool(((numbers >= 1) & (numbers <= self.n)).all()):
            raise ValueError(f"numbers must be within 1..{self.n}")
        sorted_numbers = torch.sort(numbers.long(), dim=-1).values
        if self.k > 1 and not bool((sorted_numbers[:, 1:] > sorted_numbers[:, :-1]).all()):
            raise ValueError("numbers must be unique")
        picked = batched.gather(1, numbers.long() - 1).sum(dim=1)
        logp = picked - log_partition(batched, self.k)
        return logp[0] if single else logp


def main():
    parser = argparse.ArgumentParser(description="Phase 1 PyTorch decoder self-test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    x = torch.zeros(49, dtype=torch.float64, requires_grad=True, device=device)
    s = torch.tensor([3, 11, 18, 27, 35, 46], device=device)
    d = FixedCardinalityDecoder()
    lp = d(x, s)
    expected = -log(comb(49, 6))
    assert abs(lp.item() - expected) < 1e-11
    lp.backward()
    expected_grad = torch.full((49,), -6 / 49, dtype=torch.float64, device=device)
    expected_grad[s - 1] += 1
    assert torch.allclose(x.grad, expected_grad, atol=1e-11, rtol=1e-11)
    print(f"Lotto649 Torch Decoder v{VERSION} | torch={torch.__version__} | {device}")
    print(f"[PASS] fair log P={lp.item():.10f}")
    print(f"[PASS] exact gradient: max absolute error={(x.grad-expected_grad).abs().max().item():.3g}")
    print("[INFO] Phase 1 mathematics only; no neural training or ticket selection")


if __name__ == "__main__":
    main()
