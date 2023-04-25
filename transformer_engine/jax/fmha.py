# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from functools import partial
import jax
import jax.numpy as jnp

from .cpp_extensions import cross_fmha_fwd, cross_fmha_bwd
from .cpp_extensions import self_fmha_fwd, self_fmha_bwd
from .sharding import get_fmha_sharding_meta
from .sharding import ShardingType
from .sharding import xmap_runner

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


def self_fmha(qkv: jnp.ndarray,
              bias: jnp.ndarray,
              mask: jnp.ndarray,
              seed: int,
              scaling_factor: float,
              dropout_probability: float,
              is_causal_masking: bool,
              sharding_type: ShardingType = ShardingType.SINGLE):
    """
    Self multi-head attention wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "FMHA does not support row-split tensor parallelism currently."

    if sharding_type is ShardingType.SINGLE:
        output = _self_fmha(qkv,
                            bias,
                            mask,
                            seed=seed,
                            scaling_factor=scaling_factor,
                            dropout_probability=dropout_probability,
                            is_causal_masking=is_causal_masking)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        inputs = [qkv, bias, mask]
        batch, seqlen, _, num_head, head_dim = qkv.shape
        output_shape = [batch, seqlen, num_head, head_dim]
        sharding_meta = get_fmha_sharding_meta(sharding_type, [x.shape for x in inputs],
                                               [output_shape],
                                               dp_dims=([0, None, 0], [0]),
                                               tp_dims=([3, 1, None], [2]),
                                               dp_axis_name=dp_axis_name,
                                               tp_axis_name=tp_axis_name)

        inputs_ = tuple(
            jnp.reshape(x, new_shape) for x, new_shape in zip(inputs, sharding_meta.input_shapes))

        partial_self_fmha = partial(_self_fmha,
                                    seed=seed,
                                    scaling_factor=scaling_factor,
                                    dropout_probability=dropout_probability,
                                    is_causal_masking=is_causal_masking)

        output_ = xmap_runner(partial_self_fmha, sharding_meta.in_axes, sharding_meta.out_axes[0],
                              sharding_meta.axis_resources, inputs_)

        output = jnp.reshape(output_, sharding_meta.output_shapes[0])

    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _self_fmha(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray, seed: int,
               scaling_factor: float, dropout_probability: float, is_causal_masking: bool):
    output, _ = _self_fmha_fwd(qkv, bias, mask, seed, scaling_factor, dropout_probability,
                               is_causal_masking)
    return output


def _self_fmha_fwd(qkv, bias, mask, seed, scaling_factor, dropout_probability, is_causal_masking):

    q_seqlen = jnp.sum(mask[:, :, :, 0] == 0, axis=(-1, -2), dtype=jnp.int32)
    q_cu_seqlen = jnp.cumsum(q_seqlen)
    q_cu_seqlen = jnp.hstack((0, q_cu_seqlen))

    kv_cu_seqlen = q_cu_seqlen

    output, softmax_aux = self_fmha_fwd(qkv, bias, q_cu_seqlen, kv_cu_seqlen, seed, scaling_factor,
                                        dropout_probability, is_causal_masking)
    return output, (softmax_aux, qkv, q_cu_seqlen, kv_cu_seqlen)


def _self_fmha_bwd(
        seed,    # pylint: disable=unused-argument
        scaling_factor,
        dropout_probability,
        is_causal_masking,
        ctx,
        grad):
    softmax_aux, qkv, q_seqlen, kv_seqlen = ctx

    doutput = grad

    grad_qkv, _, grad_bias = self_fmha_bwd(qkv,
                                           softmax_aux,
                                           doutput,
                                           q_seqlen,
                                           kv_seqlen,
                                           scaling_factor=scaling_factor,
                                           dropout_probability=dropout_probability,
                                           is_causal_masking=is_causal_masking)

    return grad_qkv, grad_bias, None


_self_fmha.defvjp(_self_fmha_fwd, _self_fmha_bwd)


def cross_fmha(q: jnp.ndarray,
               kv: jnp.ndarray,
               mask: jnp.ndarray,
               seed: int,
               scaling_factor: float,
               dropout_probability: float,
               is_causal_masking: bool,
               sharding_type: ShardingType = ShardingType.SINGLE):
    """
    Cross multi-head attention wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "FMHA does not support row-split tensor parallelism currently."

    if sharding_type is ShardingType.SINGLE:
        output = _cross_fmha(q,
                             kv,
                             mask,
                             seed=seed,
                             scaling_factor=scaling_factor,
                             dropout_probability=dropout_probability,
                             is_causal_masking=is_causal_masking)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        inputs = [q, kv, mask]
        output_shape = q.shape
        sharding_meta = get_fmha_sharding_meta(sharding_type, [x.shape for x in inputs],
                                               [output_shape],
                                               dp_dims=([0, 0, 0], [0]),
                                               tp_dims=([2, 3, None], [2]),
                                               dp_axis_name=dp_axis_name,
                                               tp_axis_name=tp_axis_name)

        inputs_ = tuple(
            jnp.reshape(x, new_shape) for x, new_shape in zip(inputs, sharding_meta.input_shapes))

        partial_cross_fmha = partial(_cross_fmha,
                                     seed=seed,
                                     scaling_factor=scaling_factor,
                                     dropout_probability=dropout_probability,
                                     is_causal_masking=is_causal_masking)

        output_ = xmap_runner(partial_cross_fmha, sharding_meta.in_axes, sharding_meta.out_axes[0],
                              sharding_meta.axis_resources, inputs_)

        output = jnp.reshape(output_, sharding_meta.output_shapes[0])

    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _cross_fmha(q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, seed: int,
                scaling_factor: float, dropout_probability: float, is_causal_masking: bool):

    output, _ = _cross_fmha_fwd(q, kv, mask, seed, scaling_factor, dropout_probability,
                                is_causal_masking)
    return output


def _cross_fmha_fwd(q, kv, mask, seed, scaling_factor, dropout_probability, is_causal_masking):

    q_seqlen = jnp.sum(mask[:, :, :, 0] == 0, axis=(-1, -2), dtype=jnp.int32)
    q_cu_seqlen = jnp.cumsum(q_seqlen)
    q_cu_seqlen = jnp.hstack((0, q_cu_seqlen))

    kv_seqlen = jnp.sum(mask[:, :, 0, :] == 0, axis=(-1, -2), dtype=jnp.int32)
    kv_cu_seqlen = jnp.cumsum(kv_seqlen)
    kv_cu_seqlen = jnp.hstack((0, kv_cu_seqlen))

    output, softmax_aux = cross_fmha_fwd(q, kv, q_cu_seqlen, kv_cu_seqlen, seed, scaling_factor,
                                         dropout_probability, is_causal_masking)
    return output, (softmax_aux, q, kv, q_cu_seqlen, kv_cu_seqlen)


def _cross_fmha_bwd(
        seed,    # pylint: disable=unused-argument
        scaling_factor,
        dropout_probability,
        is_causal_masking,
        ctx,
        grad):
    softmax_aux, q, kv, q_seqlen, kv_seqlen = ctx

    doutput = grad

    # TODO(rewang): remove dsoftmax for cross_fmha
    grad_q, grad_kv, _ = cross_fmha_bwd(q,
                                        kv,
                                        softmax_aux,
                                        doutput,
                                        q_seqlen,
                                        kv_seqlen,
                                        scaling_factor=scaling_factor,
                                        dropout_probability=dropout_probability,
                                        is_causal_masking=is_causal_masking)

    return grad_q, grad_kv, None


_cross_fmha.defvjp(_cross_fmha_fwd, _cross_fmha_bwd)
