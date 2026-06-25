import tilelang 
import tilelang.language as T 
import torch 
import torch.nn.functional as F 
from tilelang.autotuner import *
import itertools
import argparse
from functools import partial

def get_configs():
    iter_params = dict( block_M= [64], block_N=[64], num_stages=[1], threads=[128])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def Attention(
    batch: int, 
    heads: int, 
    seq_len: int, 
    dim: int, 
    is_causal: bool, 
    block_M: int = 64, 
    block_N: int = 64, 
    num_stages: int = 1, 
    threads :int = 128
):
    scale = (1/dim)**0.5*1.44269504
    shape = (batch, seq_len, heads, dim)
    dtype = T.float16
    acc_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype), #type: ignore
        K: T.Tensor(shape, dtype), #type: ignore
        V: T.Tensor(shape, dtype), #type: ignore
        O: T.Tensor(shape, dtype)  #type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads , batch, threads = threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), dtype)
            K_shared = T.alloc_shared((block_M, dim), dtype)
            V_shared = T.alloc_shared((block_M, dim), dtype)
            O_shared = T.alloc_shared((block_M, dim), dtype)

            acc_scores = T.alloc_fragment((block_M, block_N), acc_dtype)   # [block_M, block_N] — attention score tile
            acc_scores_cast = T.alloc_fragment((block_M, block_N), dtype)  # fp16 cast for gemm into output
            acc_output = T.alloc_fragment((block_M, dim), acc_dtype)
            scores_max = T.alloc_fragment([block_M], acc_dtype)
            scores_max_prev = T.alloc_fragment([block_M], acc_dtype)
            scores_scale = T.alloc_fragment([block_M], acc_dtype)
            scores_sum = T.alloc_fragment([block_M], acc_dtype)
            logsum = T.alloc_fragment([block_M], acc_dtype)
            
            T.copy(Q[bz, bx*block_M:(bx+1)*block_M, by, :], Q_shared)
            T.fill(acc_output, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(acc_dtype))  # must start at -inf, not +inf

            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N)  # causal: only attend to positions <= current
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )
            
            for k in T.Pipelined(loop_range, num_stages = num_stages):
                T.copy(K[bz, k*block_N:(k+1)*block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_scores[i,j] = T.if_then_else((bx*block_M+i)>=(k*block_N+j), 0, -T.infinity(acc_scores.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_scores[i, j] = T.if_then_else((k*block_N+j)>=seq_len, -T.infinity(acc_scores.dtype),0 )
                T.gemm(Q_shared, K_shared, acc_scores, transpose_B= True, policy= T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_scores, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_scores[i, j] = T.exp2(acc_scores[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_scores, scores_sum, dim = 1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_scores, acc_scores_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_output[i, j] *= scores_scale[i]
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_scores_cast, V_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_output[i, j] /= logsum[i]
            T.copy(acc_output, O_shared)
            T.copy(O_shared, O[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main        