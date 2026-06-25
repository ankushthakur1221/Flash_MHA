from tilelang.transform import pass_config
import tilelang
import tilelang.language as T 
from tilelang.autotuner import *
import torch 
import torch.nn as nn 
import itertools

def get_config():
    iter_params = dict(
        block_M = [64, 128],
        block_N = [64, 128],
        num_stages = [1],
        threads = [128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@tilelang.autotune(configs= get_config(), warmup =5, rep=10)
@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattention_fwd(
    batch: int,
    seq_len:int,
    heads:int,
    head_dim: int,
    is_causal:bool,
    block_M:int = 64,
    block_N:int = 64,
    num_stages:int = 1,
    num_threads:int= 128
):
    shape = [batch, seq_len, heads, head_dim]
    scale = (1/head_dim)**0.5*1.44269504
    dtype = T.float16
    acc_dtype = T.float32

    @T.prim_func
    def flash_forward(
        Q: T.Tensor(shape, dtype), #type: ignore
        K: T.Tensor(shape, dtype), #type: ignore
        V: T.Tensor(shape, dtype), #type: ignore
        O:T.Tensor(shape, dtype), #type: ignore
        Lse : T.Tensor ([batch, seq_len, heads], acc_dtype) #type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads = num_threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, head_dim], dtype)
            K_shared = T.alloc_shared([block_N, head_dim], dtype)
            V_shared = T.alloc_shared([block_N, head_dim], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], acc_dtype)
            acc_cast_s = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, head_dim], acc_dtype)

            acc_s_max = T.alloc_fragment([block_M,], acc_dtype)
            acc_s_max_prev = T.alloc_fragment([block_M,], acc_dtype)
            acc_sum = T.alloc_fragment([block_M,], acc_dtype)
            acc_logsum = T.alloc_fragment([block_M,], acc_dtype)
            acc_score_scale = T.alloc_fragment([block_M,], acc_dtype)

            T.copy(Q[bz, bx*block_M:(bx+1)*block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(acc_logsum, 0)
            T.fill(acc_s_max, -T.infinity(acc_dtype))

            loop_range = T.ceildiv((bx+1)*block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages = num_stages):
                T.copy(K[bz, k*block_N:(k+1)*block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((bx*block_M+i)>=(k*block_N+j), 0, -T.infinity(acc_dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((k*block_N+j)>=seq_len, -T.infinity(acc_dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B = True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k*block_N:(k+1)*block_N, by, :], V_shared)
                T.copy(acc_s_max, acc_s_max_prev)
                T.reduce_max(acc_s, acc_s_max, dim = 1 , clear = False)
                for i in T.Parallel(block_M):
                    acc_s_max[i] = T.max(acc_s_max[i], acc_s_max_prev[i])
                for i in T.Parallel(block_M):
                    acc_score_scale[i] = T.exp2(acc_s_max_prev[i]*scale - acc_s_max[i]*scale)
                for i, j in T.Parallel(block_M, head_dim):
                    acc_o[i, j] = acc_o[i, j]*acc_score_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j]*scale - acc_s_max[i]*scale)
                T.copy(acc_s, acc_cast_s)
                T.gemm(acc_cast_s, V_shared, acc_o, policy = T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, acc_sum, dim = 1)
                for i in T.Parallel(block_M):
                    acc_logsum[i]= acc_logsum[i]*acc_score_scale[i]+acc_sum[i]
            for i, j in T.Parallel(block_M, head_dim):
                acc_o[i, j] /= acc_logsum[i]
            T.copy(acc_o, O[bz, bx*block_M:(bx+1)*block_M, by, :])
            for i in T.Parallel(block_M):
                acc_logsum[i] = T.log2(acc_logsum[i]) + acc_s_max[i]*scale
            T.copy(acc_logsum, Lse[bz, by, bx*block_M:(bx+1)*block_M] )
    return flash_forward



@tilelang.autotune(configs= get_config(), warmup =5, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def FlashAtten_bwd_pre(
    batch:int,
    seq_len:int, 
    heads:int,
    head_dim:int
):
    shape = [batch, seq_len, heads, head_dim]
    dtype = T.float16
    acc_dtype = T.float32
    block_K = 32
    @T.prim_func
    def flsh_bwd_pre(
        O: T.Tensor(shape, dtype), #type: ignore
        dO: T.Tensor(shape, dtype), #type: ignore
        Delta: T.Tensor((batch, heads, seq_len)) #type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_K), batch, threads = 64) as (bx, by, bz):
            O_acc = T.alloc_fragment([block_K, block_K], dtype)
            dO_acc = T.alloc_fragment([block_K, block_K], dtype)
            acc = T.alloc_fragment([block_K, block_K], acc_dtype)
            acc_Delta = T.alloc_fragment([block_K,], acc_dtype)

            T.clear(acc)
            for k in range(T.ceildiv(head_dim, block_K)):
                T.copy(O[bz, by*block_K:(by+1)*block_K, bx, k*block_K:(k+1)*block_K], O_acc)
                T.copy(dO[bz, by*block_K:(by+1)*block_K, bx, k*block_K:(k+1)*block_K], dO_acc)
                for i, j in T.Parallel(block_K, block_K):
                    acc[i, j] = O_acc[i, j]*dO_acc[i,j]
            T.reduce_sum(acc, acc_Delta, dim = 1)
            T.copy(acc_Delta, Delta[bz, bx, by*block_K:(by+1)*block_K ])
    return flsh_bwd_pre


def make_dq_layout(dQ):
    return T.Layout(dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])

@tilelang.autotune(configs= get_config(), warmup =5, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def FlashAtten_bwd_post(
    batch:int,
    heads:int,
    seq_len:int,
    head_dim:int
):
    shape = [batch, heads, seq_len, head_dim]
    acc_dtype = T.float32
    dtype = T.float16
    block_K = 64

    @T.prim_func
    def attn_bwd_post(
        dQ: T.Tensor(shape, acc_dtype), #type: ignore
        dQ_out: T.Tensor(shape, dtype) #type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_K), heads, batch, threads = 128) as (bx, by , bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * block_K : (bx + 1) * block_K, by, :],
                dQ_out[bz, bx * block_K : (bx + 1) * block_K, by, :],
            )
    return attn_bwd_post

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def FlashAttention_bwd(
    batch:int,
    seq_len: int, 
    heads: int, 
    head_dim: int,
    is_causal: bool, 
    block_M: int, 
    block_N: int
):
    sm_scale = (1/head_dim)**0.5
    scale = (1/head_dim)**0.5*1.44269504 
    shape = [batch, seq_len, heads, head_dim]
    shape2 = [batch, heads, seq_len]
    acc_dtype = T.float32
    dtype = T.float16

    @T.prim_func
    def flash_atten_bwd(
        Q: T.Tensor(shape, dtype), # type: ignore
        K: T.Tensor(shape, dtype), # type: ignore
        V: T.Tensor(shape, dtype), # type: ignore
        dO: T.Tensor(shape, dtype), # type: ignore
        
        Lse: T.Tensor(shape2, acc_dtype), # type: ignore
        Delta: T.Tensor(shape2, acc_dtype), # type: ignore

        dQ: T.Tensor(shape, acc_dtype), # type: ignore
        dK: T.Tensor(shape, dtype), # type: ignore
        dV: T.Tensor(shape, dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads = 64) as (bx, by, bz):

            Q_shared = T.alloc_shared([block_M, head_dim], dtype)
            K_shared = T.alloc_shared([block_M, head_dim], dtype)
            V_shared = T.alloc_shared([block_M, head_dim], dtype)
            dsT_shared =  T.alloc_shared([block_M, block_N], dtype)
            
            dsT = T.alloc_fragment([block_M, block_N], acc_dtype)
            qkt = T.alloc_fragment([block_M, block_N], acc_dtype)

            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)

            lse_shared = T.alloc_shared([block_N,], acc_dtype)
            Delta_shared = T.alloc_shared([block_N,], acc_dtype)

            do = T.alloc_fragment([block_N, head_dim], dtype)
            dv = T.alloc_fragment([block_M, head_dim], acc_dtype)
            dk = T.alloc_fragment([block_M, head_dim], acc_dtype)
            dq = T.alloc_fragment([block_M, head_dim], acc_dtype)

            dv_shared = T.alloc_shared([block_M, head_dim], dtype)
            dk_shared = T.alloc_shared([block_M, head_dim], dtype)

            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                }
            )

            T.copy(V[bz, by*block_M:(by+1)*block_M, bx, :], V_shared)
            T.copy(K[bz, by*block_M:(by+1)*block_M, bx, :], K_shared)
            T.clear(dv)
            T.clear(dk)

            loop_start = T.floordiv(by*block_M, block_N) if is_causal else 0
            loop_end = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_start, loop_end, num_stages = 2):
                T.copy(Q[bz, k*block_N:(k+1)*block_N, bx, :], Q_shared)
                T.clear(qkt)
                T.gemm(K_shared, Q_shared, qkt, transpose_B = True, policy = T.GemmWarpPolicy.FullRow)
                
                T.copy(Lse[bz, bx, k*block_N:(k+1)*block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkt[i, j] = T.exp2(qkt[i, j]*scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M,block_N):
                        qkt[i, j] = T.if_then_else((by*block_M+i)<=(k*block_N+j), qkt[i, j], 0)

                T.copy(dO[bz, k*block_N:(k+1)*block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B= True, policy = T.GemmWarpPolicy.FullRow)
                T.copy(qkt, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy = T.GemmWarpPolicy.FullRow )

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], Delta_shared)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkt[i, j] * (dsT[i, j] - Delta_shared[j]) * sm_scale
                T.gemm(dsT_cast, Q_shared, dk, policy=T.GemmWarpPolicy.FullRow)
                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, head_dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
                T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])
    return flash_atten_bwd

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        Batch, N_ctx, Head, D_head = q.shape
        block_M = 64
        block_N = 64 if D_head <= 128 else 32
        o, lse = flashattention_fwd(Batch, Head, N_ctx, D_head, causal, block_M, block_N)(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o       
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 64
        block_N = 64 if D_HEAD <= 64 else 32
        kernel_prep = FlashAtten_bwd_pre(BATCH, H, N_CTX, D_HEAD)
        kernel_post = FlashAtten_bwd_post(BATCH, H, N_CTX, D_HEAD)
        delta = kernel_prep(o, do)
        kernel = FlashAttention_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M, block_N)
        shape = [BATCH, N_CTX, H, D_HEAD]
        dq = torch.zeros(shape, dtype=torch.float32, device=q.device)
        dk = torch.empty(shape, dtype=torch.float16, device=q.device)
        dv = torch.empty(shape, dtype=torch.float16, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = kernel_post(dq)
        return dq, dk, dv, None
    


        














