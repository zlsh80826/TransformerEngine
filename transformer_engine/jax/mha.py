# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from functools import partial
import jax
import jax.numpy as jnp

from .cpp_extensions import self_mha_fwd, self_mha_bwd
# from .cpp_extensions import te_cross_fmha_fwd, te_cross_fmha_bwd
from .sharding import ShardingType
# from .sharding import get_elementwise_sharding_meta
# from .sharding import get_dot_sharding_meta, get_fp8_meta_sharding_meta
# from .sharding import is_dp_enabled, is_tp_enabled, merge_axis_resources
from .sharding import is_dp_enabled
# from .sharding import xmap_runner

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


def self_mha(qkv: jnp.ndarray,
             bias: jnp.ndarray,
             q_seqlen: jnp.ndarray,
             kv_seqlen: jnp.ndarray,
             seed: int,
             scaling_factor: float,
             dropout_probability: float,
             is_causal_masking: bool,
             sharding_type: ShardingType = ShardingType.SINGLE,
             dp_dim_index: int = 0):
    """
    Multi-head attention wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "MHA does not support row-split tensor parallelism currently."

    assert dp_dim_index is not None  # TODO(rewang): remove this

    if sharding_type is ShardingType.SINGLE:
        output = _self_mha(qkv,
                           bias,
                           q_seqlen,
                           kv_seqlen,
                           seed=seed,
                           scaling_factor=scaling_factor,
                           dropout_probability=dropout_probability,
                           is_causal_masking=is_causal_masking,
                           sharding_type=sharding_type,
                           dp_axis_name="")
    else:
        raise NotImplementedError

    return output


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _self_mha(qkv: jnp.ndarray, bias: jnp.ndarray, q_seqlen: jnp.ndarray,
              kv_seqlen: jnp.ndarray, seed: int, scaling_factor: float,
              dropout_probability: float, is_causal_masking: bool,
              sharding_type: ShardingType, dp_axis_name: str):
    output, _ = _self_mha_fwd(qkv, bias, q_seqlen, kv_seqlen, seed,
                              scaling_factor, dropout_probability,
                              is_causal_masking, sharding_type, dp_axis_name)
    return output


def _self_mha_fwd(
    qkv,
    bias,
    q_seqlen,
    kv_seqlen,
    seed,
    scaling_factor,
    dropout_probability,
    is_causal_masking,
    sharding_type,  # pylint: disable=unused-argument
    dp_axis_name  # pylint: disable=unused-argument
):
    output, softmax_output = self_mha_fwd(qkv, bias, q_seqlen, kv_seqlen, seed,
                                          scaling_factor, dropout_probability,
                                          is_causal_masking)
    return output, (softmax_output, qkv, q_seqlen, kv_seqlen)


def _self_mha_bwd(
        seed,  # pylint: disable=unused-argument
        scaling_factor,
        dropout_probability,
        is_causal_masking,
        sharding_type,
        dp_axis_name,
        ctx,
        grad):
    softmax_output, qkv, q_seqlen, kv_seqlen = ctx

    doutput = grad

    grad_qkv, grad_softmax = self_mha_bwd(
        qkv,
        softmax_output,
        doutput,
        q_seqlen,
        kv_seqlen,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_causal_masking=is_causal_masking)

    _, max_seqlen, nqkv, num_head, _ = qkv.shape
    assert nqkv == 3

    grad_beta = jnp.sum(grad_softmax.astype(jnp.float32) / scaling_factor,
                        axis=0,
                        keepdims=True,
                        dtype=jnp.float32).astype(grad.dtype)
    grad_beta = _reshape_softmax(grad_beta, 1, max_seqlen, num_head)

    # TODO(rewang): is it needed?
    if is_dp_enabled(sharding_type.value[0]):
        grad_qkv = jax.lax.psum(grad_qkv, dp_axis_name)
        grad_beta = jax.lax.psum(grad_beta, dp_axis_name)

    return grad_qkv, grad_beta, None, None


_self_mha.defvjp(_self_mha_fwd, _self_mha_bwd)


# TODO(rewang): refactor
def _reshape_softmax(S, b, s, h, warps_m=1, warps_n=4):
    # This should not expose to public
    m = s if s == 128 else 16
    n = s
    m_per_cta = warps_m * 16
    n_per_cta = warps_n * 16
    mmas_m = m // m_per_cta
    mmas_n = n // n_per_cta
    loops = s // (mmas_m * m_per_cta)
    assert (loops == 1 and s == 128) or (loops == 16 and s == 256) or (
        loops == 32 and s == 512) or (loops == 24 and s == 384), "no.."
    quads = 8
    lohi = 2
    lr = 2
    vals = 2

    magic_shape = (b, h, loops, mmas_m, mmas_n, warps_n, warps_m, quads, 4,
                   lohi, lr, vals)

    magic_s = jnp.reshape(S, magic_shape)
    transposed_s = jnp.transpose(magic_s,
                                 [0, 1, 2, 3, 6, 9, 7, 4, 5, 10, 8, 11])
    ret_s = jnp.reshape(transposed_s, (b, h, s, s))
    return ret_s
