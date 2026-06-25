import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_cuda():
    if not torch.cuda.is_available():
        print("=" * 60)
        print("WARNING: CUDA is not available on this system.")
        print("Since TileLang compiles high-performance GPU kernels to CUDA/Triton,")
        print("you must run these tests on a machine with a CUDA-capable GPU.")
        print("=" * 60)
        return False
    return True

def run_tests():
    print("CUDA is available! Running TileLang kernel correctness tests...")
    
    # ----------------------------------------------------
    # Test 1: Forward-only Attention Kernel (kernel.py)
    # ----------------------------------------------------
    try:
        from kernel import Attention
        print("\n[Test 1] Compiling and verifying forward-only Attention kernel...")
        
        batch = 2
        heads = 4
        seq_len = 128
        dim = 64
        is_causal = True
        
        # TileLang JIT compiles the decorator-wrapped function when called
        attention_kernel = Attention(batch, heads, seq_len, dim, is_causal)
        
        q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16, device="cuda")
        
        # Run custom TileLang kernel
        # Argument index 3 is output O, which is auto-allocated and returned
        out_tilelang = attention_kernel(q, k, v)
        
        # Run PyTorch reference
        # PyTorch F.scaled_dot_product_attention expects [batch, heads, seq_len, dim]
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=is_causal
        )
        # Transpose reference back to [batch, seq_len, heads, dim] to match TileLang
        out_ref = out_ref.transpose(1, 2)
        
        # Check tolerance
        torch.testing.assert_close(out_tilelang, out_ref, atol=1e-2, rtol=1e-2)
        print("-> [Test 1] Forward-only Attention JIT Kernel matches PyTorch reference! ✅")
        
    except Exception as e:
        print(f"-> [Test 1] Failed with error: {e}")
        import traceback
        traceback.print_exc()

    # ----------------------------------------------------
    # Test 2: Full Autograd MultiHeadAttention (Mla.py)
    # ----------------------------------------------------
    try:
        from Mla import MultiHeadAttention
        print("\n[Test 2] Compiling and verifying Autograd MultiHeadAttention module (forward & backward)...")
        
        embed_dim = 128
        num_heads = 4
        head_dim = 32  # 128 / 4 = 32
        batch = 2
        seq_len = 64
        
        # Model & Inputs
        mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim).cuda().half()
        
        # Enforce deterministic weights for exact matching of projections
        nn.init.ones_(mha.Q_proj.weight)
        nn.init.ones_(mha.K_proj.weight)
        nn.init.ones_(mha.V_proj.weight)
        nn.init.ones_(mha.O_proj.weight)
        if mha.Q_proj.bias is not None:
            nn.init.zeros_(mha.Q_proj.bias)
            nn.init.zeros_(mha.K_proj.bias)
            nn.init.zeros_(mha.V_proj.bias)
            nn.init.zeros_(mha.O_proj.bias)

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float16, device="cuda", requires_grad=True)
        
        # Forward pass
        out_tilelang = mha(x, x, x, is_causal=True)
        
        # Reference implementation using standard PyTorch operations
        q = mha.Q_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = mha.K_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = mha.V_proj(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        
        ref_attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref_attn_out = ref_attn_out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        out_ref = mha.O_proj(ref_attn_out)
        
        # Verify forward correctness
        torch.testing.assert_close(out_tilelang, out_ref, atol=1e-2, rtol=1e-2)
        print("-> [Test 2] Forward pass matches PyTorch reference! ✅")
        
        # Backward pass
        loss_tilelang = out_tilelang.sum()
        loss_tilelang.backward(retain_graph=True)
        grad_tilelang = x.grad.clone()
        x.grad.zero_()
        
        loss_ref = out_ref.sum()
        loss_ref.backward()
        grad_ref = x.grad.clone()
        
        # Verify backward correctness
        torch.testing.assert_close(grad_tilelang, grad_ref, atol=1e-2, rtol=1e-2)
        print("-> [Test 2] Backward pass matches PyTorch reference! ✅")
        
    except Exception as e:
        print(f"-> [Test 2] Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if check_cuda():
        run_tests()
    else:
        sys.exit(0)
