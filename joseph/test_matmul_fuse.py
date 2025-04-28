import torch
import triton
import triton.language as tl


@triton.jit
def fused_gemm_kernel(a_ptr, b_ptr, e_ptr, f_ptr,
                      M, N, K, P,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_en, stride_ep,
                      stride_fm, stride_fp,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_P: tl.constexpr,
                      BLOCK_K: tl.constexpr):

    pid_m = tl.program_id(0)          # tile index along M
    pid_p = tl.program_id(1)          # tile index along P

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)   # [BLOCK_P]

    f_tile_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_p[None, :] * stride_fp
    acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

    # iterate over K-N dimension in BLOCK_K / BLOCK_N chunks
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)                      

            # A tile:  [BLOCK_M, BLOCK_K]
            a_tile_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_tile = tl.load(a_tile_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

            # B tile:  [BLOCK_K, BLOCK_N]
            b_tile_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_tile = tl.load(b_tile_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))

            # Compute C_partial = A_tile @ B_tile   -> shape [BLOCK_M, BLOCK_N]
            # c_partial = tl.dot(a_tile, b_tile, out_dtype=tl.float32, allow_tf32=False)
            c_acc += tl.dot(a_tile, b_tile, allow_tf32=False)


        e_tile_ptrs = e_ptr + (offs_n[:, None] * stride_en) + (offs_p[None, :] * stride_ep)
        e_tile = tl.load(
            e_tile_ptrs, mask=(offs_n[:, None] < N) & (offs_p[None, :] < P),
        ).to(tl.float32)

        acc += tl.dot(c_acc, e_tile, allow_tf32=False)

    # store final F tile
    tl.store(f_tile_ptrs,
            acc.to(tl.float32),
            mask=(offs_m[:, None] < M) & (offs_p[None, :] < P))


def fused_mm(a: torch.Tensor, b: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper around the Triton kernel.
    All inputs must be on the same CUDA device and fp16/fp32 compatible.
    """
    assert a.dim() == b.dim() == e.dim() == 2
    M, K = a.shape
    _K, N = b.shape
    _N, P = e.shape
    assert K == _K and N == _N, "shape mismatch"

    # output
    f = torch.empty((M, P), device=a.device, dtype=torch.float32)

    # choose tile sizes (heuristics – feel free to tweak)
    BLOCK_M = 64
    BLOCK_K = 32
    BLOCK_N = 64
    BLOCK_P = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(P, BLOCK_P))

    fused_gemm_kernel[grid](
        a, b, e, f,
        M, N, K, P,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        e.stride(0), e.stride(1),
        f.stride(0), f.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
        BLOCK_K=BLOCK_K,
    )
    return f


# ------------------------ quick correctness / speed check ---------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    M, K, N, P = 4096, 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    e = torch.randn(N, P, device=device, dtype=torch.float32)

    f_triton = fused_mm(a, b, e)

    with torch.no_grad():
        f_ref = (a @ b).to(torch.float32) @ e
        f_ref = f_ref.to(torch.float32)

    max_err = (f_triton - f_ref).abs().max()
    print(f"max |Δ| between fused Triton and reference: {max_err.item():.3e}")

    # benchmark
    import time
    torch.cuda.synchronize()
    for idx in range(0):
        print(f"iteration {idx}")
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        e = torch.randn(N, P, device=device, dtype=torch.float32)
        start = time.time()
        f_triton = fused_mm(a, b, e)
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        print(f"fused_mm took {elapsed:.3f} seconds")
