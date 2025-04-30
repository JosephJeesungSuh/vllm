# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import List, Tuple

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op  # type: ignore
from vllm.triton_utils import HAS_TRITON  # type: ignore
from vllm.lora.ops.triton_ops.utils import (
    _get_lora_a_ptr,
    _get_lora_b_ptr,
)
from vllm.lora.ops.triton_ops.kernel_utils import mm_k


@triton.jit
def _lora_shrink_expand_kernel(
    # Pointers ----------------------------------------------------------------
    x_ptr,
    la_ptr,
    lb_ptr,
    # intermediate_ptr,
    # dummy_ptr,
    y_ptr,
    # Scalars / sizes ---------------------------------------------------------
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    O_hidden_max: tl.constexpr,
    # Mapping tensors ---------------------------------------------------------
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_tensor,
    # Strides (may be tensor or scalar) --------------------------------------
    x_s0, x_s1,
    la_s0, la_s1, la_s2,
    lb_s0, lb_s1, lb_s2,
    # intermediate_s0, intermediate_s1, intermediate_s2,
    y_s0, y_s1,
    hidden_sizes_tensor,
    # Meta --------------------------------------------------------------------
    SCALE: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_O: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    NUM_SLICES: tl.constexpr,
):
    """
    Compute Y = (X @ A) * scale @ B (or +=, depending on ADD_INPUTS).
    This CTA is responsible for BLOCK_M tokens, BLOCK_N rank, and BLOCK_O output hidden size.
    """

    cta_mnko = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    cta_m_num = tl.cdiv(M, BLOCK_M) # number of CTAs in M (token) dimension
    cta_n_num = tl.cdiv(N, BLOCK_N) # number of CTAs in N (lora rank) dimension
    cta_o_num = tl.cdiv(O_hidden_max, BLOCK_O) # number of CTAs in O (hidden) dimension

    pid_o = cta_mnko % cta_o_num
    pid_k = (cta_mnko // cta_o_num) % SPLIT_K
    pid_m = (cta_mnko // (cta_o_num * SPLIT_K)) % cta_m_num
    pid_n = (cta_mnko // (cta_o_num * SPLIT_K * cta_m_num)) % cta_n_num

    cta_m_offset = pid_m * BLOCK_M
    cta_n_offset = pid_n * BLOCK_N
    cta_o_offset = pid_o * BLOCK_O
    offset_cm = tl.arange(0, BLOCK_M) + cta_m_offset
    offset_cn = tl.arange(0, BLOCK_N) + cta_n_offset
    offset_co = tl.arange(0, BLOCK_O) + cta_o_offset

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    if cta_m_offset >= lora_m_size:
        return
    if cta_n_offset >= N:
        return
    if cta_o_offset >= O_hidden_max:
        return

    tok_start = tl.load(lora_token_start_loc + lora_idx)
    token_ptr = token_indices_sorted_by_lora_ids + tok_start + cta_m_offset
    cta_m_len = tl.minimum(BLOCK_M, lora_m_size - cta_m_offset)
    row_mask = tl.arange(0, BLOCK_M) < cta_m_len
    ram = tl.load(token_ptr + tl.arange(0, BLOCK_M) % cta_m_len) # random access of token indices

    if NUM_SLICES == 1:
        la_base = la_ptr
        lb_base = lb_ptr
        slice_start = slice_start_tensor
    else:
        la_base = tl.load(la_ptr + slice_id).to(tl.pointer_type(x_ptr.dtype.element_ty))
        lb_base = tl.load(lb_ptr + slice_id).to(tl.pointer_type(x_ptr.dtype.element_ty))
        slice_start = tl.load(slice_start_tensor + slice_id)

    rbn = tl.max_contiguous(tl.multiple_of(offset_cn % N, BLOCK_N), BLOCK_N) # must be mod N to prevent segfault
    offset_k = tl.arange(0, BLOCK_K) + pid_k * BLOCK_K
    
    accumulator = mm_k(
        x_ptr + ram[:, None] * x_s0 + offset_k[None, :] * x_s1,
        la_base + la_s0 * lora_idx + la_s1 * rbn[None, :] + la_s2 * offset_k[:, None],
        x_s1,
        la_s2,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        la_ptr.dtype.element_ty,
    ) # this will result in BLOCK_M x BLOCK_N matrix
    accumulator *= SCALE
    
    if SAME_STRIDE:
        cur_lora_d0_stride = lb_s0
        cur_lora_d1_stride = lb_s1
        cur_lora_d2_stride = lb_s2
        output_dim = K
    else:
        cur_lora_d0_stride = tl.load(lb_s0 + slice_id)
        cur_lora_d1_stride = tl.load(lb_s1 + slice_id)
        cur_lora_d2_stride = tl.load(lb_s2 + slice_id)
        output_dim = tl.load(hidden_sizes_tensor + slice_id)

    if NUM_SLICES == 1:
        slice_start_loc = slice_start_tensor
    else:
        slice_start_loc = tl.load(slice_start_tensor + slice_id)

    offset_o = tl.arange(0, BLOCK_O) + pid_o * BLOCK_O
    rbo = tl.max_contiguous(tl.multiple_of(offset_co % output_dim, BLOCK_O), BLOCK_O)
    b_ptr = (
        lb_base + cur_lora_d0_stride * lora_idx
        + rbo[None, :] * cur_lora_d1_stride
        + rbn[:, None] * cur_lora_d2_stride
    )
    
    b_tile = tl.load(
        b_ptr,
        mask=(offset_co[None, :] < output_dim) & (offset_cn[:, None] < N),
        other=0.0,
    )

    output = tl.dot(accumulator.to(b_tile.dtype), b_tile)
    output_ptr = y_ptr + ram[:, None] * y_s0 + (offset_co[None, :] + slice_start_loc) * y_s1
    output_mask = (offset_cm[:, None] < lora_m_size) & (offset_co[None, :] < output_dim)

    if ADD_INPUTS:
        tiled_out = tl.load(output_ptr, mask=output_mask)
        output += tiled_out
    tl.store(output_ptr, output, mask=output_mask)



@torch.inference_mode()
def _lora_shrink_expand_fused(
    inputs: torch.Tensor,                  
    lora_a_weights: List[torch.Tensor],    
    scaling: float,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:

    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        # None of the inputs require LoRA.
        return

    # dimension and type sanity checks
    assert len(lora_a_weights) == len(lora_b_weights)
    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert lora_a_weights[0].size(-2) == lora_b_weights[0].size(-1)
    assert len(lora_b_weights) * lora_b_weights[0].size(-2) == output_tensor.size(1)

    M = inputs.size(0) # num_tokens
    N, K = lora_a_weights[0].shape[-2:] # max rank, hidden size
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)
    (lora_a_ptr_tensor, lora_a_strides_d0, lora_a_strides_d1, lora_a_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    (slice_start_tensor, lora_b_ptr_tensor, lora_b_strides_d0, lora_b_strides_d1, lora_b_strides_d2,
     hidden_sizes_tensor, same_stride, MAX_O) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)

    # metadata sanity check
    assert token_lora_mapping.size(0) == M
    assert M == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_ids.size(0) + 1 == lora_token_start_loc.size(0)

    # Triton kernel configs
    BLOCK_M = 64 # num tokens
    BLOCK_N = 16 # related to lora rank
    BLOCK_K = 128 # breaking up hidden size (shrink stage)
    SPLIT_K = 1
    BLOCK_O = 64 # breaking up hidden size (expand stage)
    EVEN_K = K % BLOCK_K == 0
    NUM_WARPS = 4
    NUM_CTAS = 1
    NUM_STAGES = 2
    MAX_NREG = None
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * SPLIT_K * triton.cdiv(MAX_O, BLOCK_O),
        NUM_SLICES,
        MAX_LORAS,
    )

    # intermediate_tensor = torch.zeros(
    #     (NUM_SLICES, M, N), dtype=torch.float32, device=inputs.device
    # )
    # dummy_tensor = torch.zeros(
    #     (BLOCK_M, BLOCK_N), dtype=torch.float32, device=inputs.device
    # )
    # assert intermediate_tensor.is_contiguous()
    # assert dummy_tensor.is_contiguous()

    # import os
    # if os.environ.get("DEBUG_LORA", "0") == "1":
    #     from remote_pdb import RemotePdb
    #     RemotePdb('0.0.0.0', 4444).set_trace()   

    _lora_shrink_expand_kernel[grid](
        inputs,
        lora_a_ptr_tensor,
        lora_b_ptr_tensor,
        # intermediate_tensor,
        # dummy_tensor,
        output_tensor,
        M,
        N,
        K,
        MAX_O,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_a_strides_d0,
        lora_a_strides_d1,
        lora_a_strides_d2,
        lora_b_strides_d0,
        lora_b_strides_d1,
        lora_b_strides_d2,
        # intermediate_tensor.stride(0),
        # intermediate_tensor.stride(1),
        # intermediate_tensor.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        scaling,
        same_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SPLIT_K,
        BLOCK_O,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        NUM_SLICES,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
        maxnreg=MAX_NREG,
    )

    # from remote_pdb import RemotePdb
    # RemotePdb('0.0.0.0', 4444).set_trace()


def _lora_shrink_expand_fused_fake(*args, **kwargs) -> None:
    """CPU/Meta fake implementation - does nothing (for export)."""
    return

try:
    direct_register_custom_op(
        op_name="lora_shrink_expand_fused",
        op_func=_lora_shrink_expand_fused,
        mutates_args=["output_tensor"],
        fake_impl=_lora_shrink_expand_fused_fake,
    )
    lora_shrink_expand_fused = torch.ops.vllm.lora_shrink_expand_fused

except AttributeError:
    lora_shrink_expand_fused = _lora_shrink_expand_fused
