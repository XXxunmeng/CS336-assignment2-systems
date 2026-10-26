import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
                                stride_qb, stride_qq, stride_qd,
                                stride_kb, stride_kk, stride_kd,
                                stride_vb, stride_vk, stride_vd,
                                stride_ob, stride_oq, stride_od,
                                stride_lb, stride_lq,
                                N_queries, N_keys, scale,
                                D: tl.constexpr, Q_TILE_SIZE: tl.constexpr,
                                K_TILE_SIZE: tl.constexpr, is_causal: tl.constexpr
                                ):
    query_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # (Q_TILE_SIZE, D)
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_i = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    o_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for start_idx in range(0, N_keys, K_TILE_SIZE):

        k = tl.load(K_block_ptr, boundary_check=(0,1), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option='zero')

        s = tl.dot(q_i, tl.trans(k)) * scale
        if is_causal:
            q_idx = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = start_idx + tl.arange(0, K_TILE_SIZE)
            q_mask = q_idx < N_queries
            k_mask = k_idx < N_keys
            valid_mask = q_mask[:, None] & k_mask[None, :]
            mask = valid_mask & (q_idx[:, None] >= k_idx[None, :])
            s = tl.where(mask, s, s-1e6)

        m_i_now = tl.maximum(m_i, tl.max(s, axis=1))
        p_ij = tl.exp(s - m_i_now[:, None])
        alpha = tl.exp(m_i - m_i_now)
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(p_ij.to(v.dtype), v)


        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        m_i = m_i_now

    o_i = o_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@triton.jit
def _flash_attention_bwd_dkdv_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, 
                                    dO_ptr, D_ptr, dK_ptr, dV_ptr,
                                    stride_qb, stride_qq, stride_qd,
                                    stride_kb, stride_kk, stride_kd,
                                    stride_vb, stride_vk, stride_vd,
                                    stride_ob, stride_oq, stride_od,
                                    stride_lb, stride_lq,
                                    stride_dob, stride_doq, stride_dod,
                                    stride_db, stride_dq, 
                                    stride_dkb, stride_dkk, stride_dkd,
                                    stride_dvb, stride_dvk, stride_dvd,
                                    N_queries, N_keys, scale, D: tl.constexpr,
                                    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
                                    is_causal: tl.constexpr):
    key_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_queries, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_queries,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_db,
        shape=(N_queries,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_dkb,
        shape=(N_keys, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_dvb,
        shape=(N_keys, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    
    k_idx = key_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    k_mask = k_idx < N_keys

    dk_acc = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dv_acc = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    for start_idx in range(0, N_queries, Q_TILE_SIZE):
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        q_idx = start_idx + tl.arange(0, Q_TILE_SIZE)
        q_mask = q_idx < N_queries

        s = tl.dot(q, tl.trans(k)) * scale

        valid_mask = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            valid_mask = valid_mask & causal_mask

        s = tl.where(valid_mask, s, s - 1.0e6)

        p = tl.exp(s - l[:, None])
        p = tl.where(valid_mask, p, 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do.dtype), do)
        
        d_p = tl.dot(do, tl.trans(v))
        d_s = p * (d_p - D_block[:, None]) * scale
        dk_acc += tl.dot(tl.trans(d_s).to(q.dtype), q)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dk_acc.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def _flash_attention_bwd_dq_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, 
                                    dO_ptr, D_ptr, dQ_ptr,
                                    stride_qb, stride_qq, stride_qd,
                                    stride_kb, stride_kk, stride_kd,
                                    stride_vb, stride_vk, stride_vd,
                                    stride_ob, stride_oq, stride_od,
                                    stride_lb, stride_lq,
                                    stride_dob, stride_doq, stride_dod,
                                    stride_db, stride_dq, 
                                    stride_dqb, stride_dqq, stride_dqd,
                                    N_queries, N_keys, scale, D: tl.constexpr,
                                    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
                                    is_causal: tl.constexpr):
    query_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_queries, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_db,
        shape=(N_queries,),
        strides=(stride_dq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_index * stride_dqb,
        shape=(N_queries, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    do = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
    l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    
    q_idx = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    q_mask = q_idx < N_queries

    dq_acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for start_idx in range(0, N_keys, K_TILE_SIZE):
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        k_idx = start_idx + tl.arange(0, K_TILE_SIZE)
        k_mask = k_idx < N_keys

        s = tl.dot(q, tl.trans(k)) * scale

        valid_mask = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            valid_mask = valid_mask & causal_mask

        s = tl.where(valid_mask, s, s - 1.0e6)

        p = tl.exp(s - l[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        d_p = tl.dot(do, tl.trans(v))
        d_s = p * (d_p - D_block[:, None]) * scale
        dq_acc += tl.dot(d_s.to(k.dtype), k)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dq_acc.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

class FlashAttention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        N_k = K.shape[-2]
        B_q, B_k = 64, 64
        output = torch.zeros_like(Q)
        L = torch.zeros(B, N_q, device=Q.device, dtype=Q.dtype)
        scale = 1 / (d ** 0.5)

        _flash_attention_fwd_kernel[(triton.cdiv(N_q, B_q), B)](Q, K, V, output, L, 
                                    Q.stride(0), Q.stride(1), Q.stride(2),
                                    K.stride(0), K.stride(1), K.stride(2),
                                    V.stride(0), V.stride(1), V.stride(2),
                                    output.stride(0), output.stride(1), output.stride(2),
                                    L.stride(0), L.stride(1),
                                    N_q, N_k, scale, D=d, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k, is_causal=is_causal)
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output


    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        d_Q = torch.zeros_like(Q)
        d_K = torch.zeros_like(K)
        d_V = torch.zeros_like(V)

        B, N_q, hidden = Q.shape
        N_k = K.shape[-2]
        B_q, B_k = 64, 64
        scale = 1 / (hidden ** 0.5)

        grid_kv = (triton.cdiv(N_k, B_k), B)
        grid_q = (triton.cdiv(N_q, B_q), B)

        D = torch.sum(O * dO, dim=-1).float()

        _flash_attention_bwd_dkdv_kernel[grid_kv](
            Q, K, V, O, L, dO, D, d_K, d_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            d_K.stride(0), d_K.stride(1), d_K.stride(2),
            d_V.stride(0), d_V.stride(1), d_V.stride(2),
            N_q, N_k, scale, D=hidden, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k, is_causal=is_causal
        )

        _flash_attention_bwd_dq_kernel[grid_q](
            Q, K, V, O, L, dO, D, d_Q,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            d_Q.stride(0), d_Q.stride(1), d_Q.stride(2),
            N_q, N_k, scale, D=hidden, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k, is_causal=is_causal
        )

        return d_Q, d_K, d_V, None