import torch

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q, K, V: (batch_size, seq_len, hidden)
        B, N_q, d = Q.shape
        N_k = K.shape[-2]
        B_q, B_k = 64, 64
        # output: (batch_size, seq_len, hidden)
        output = torch.zeros_like(Q)
        # L: (batch_size, seq_len) - logsumexp
        L = torch.zeros(B, N_q, device=Q.device, dtype=Q.dtype)
        scale = 1 / (d ** 0.5)

        # 将QKV分块
        for i in range(0, N_q, B_q):
            end_idx = min(N_q, i + B_q)
            # q: (batch_size, B_q, hidden)
            q = Q[..., i:end_idx, :]
            q_len = end_idx - i
            # m_i: (batch_size, B_q) - 每个小块中softmax最大值
            m_i = torch.full((B, q_len), fill_value=float('-inf'), device=Q.device, dtype=torch.float32)
            # l_ij: (batch_size, B_q) - softmax分母，即指数和
            l_ij = torch.zeros((B, q_len), device=Q.device, dtype=torch.float32)
            # o: (batch_size, B_q, hidden_size) - 小块的输出结果
            o = torch.zeros((B, q_len, d), device=Q.device, dtype=torch.float32)

            for j in range(0, N_k, B_k):
                if is_causal and j >= end_idx:
                    break
                end_idx_k = min(j + B_k, N_k)
                # k/v: (batch_size, B_k, hidden_size)
                k = K[..., j:end_idx_k, :]
                v = V[..., j:end_idx_k, :]
                # s_ij: (batch_size, B_q, B_k)
                s_ij = q @ k.transpose(-1, -2)
                s_ij = s_ij * scale
                if is_causal:
                    # 加掩码
                    q_idx = torch.arange(i, end_idx, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(j, end_idx_k, device=Q.device).unsqueeze(0)
                    mask = q_idx >= k_idx
                    s_ij = s_ij.masked_fill(~mask, float('-inf'))
                # 找最大值
                m_i_now = torch.max(s_ij.max(dim=-1).values, m_i)
                # 分子
                p_ij = torch.exp(s_ij - m_i_now.unsqueeze(-1))
                # 分母将之前的结果按倍数修改后加上新的结果
                l_ij = torch.exp(m_i - m_i_now) * l_ij + torch.sum(p_ij, dim=-1)
                # 结果分子，同样按照倍数修改结果后加上新的结果
                o = torch.exp(m_i - m_i_now).unsqueeze(-1) * o + p_ij @ v
                m_i = m_i_now
            # 将结果除以最终的分母
            o = o / l_ij.unsqueeze(-1)
            output[..., i:end_idx, :] = o.to(Q.dtype)
            # L = log sum exp(s) - softmax的分母（未减最大值版）
            L[..., i:end_idx] = m_i + torch.log(l_ij)
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, output = ctx.saved_tensors
        is_causal = ctx.is_causal
        d_Q = torch.zeros_like(Q)
        d_K = torch.zeros_like(K)
        d_V = torch.zeros_like(V)

        B, N_q, hidden = Q.shape
        N_k = K.shape[-2]
        B_q, B_k = 64, 64
        scale = 1 / (hidden ** 0.5)
        D = torch.sum(output * dO, dim=-1)

        for i in range(0, N_q, B_q):
            end_idx = min(i + B_q, N_q)
            q = Q[:, i:end_idx, :]
            do_i = dO[:, i:end_idx, :]
            # (batch_size, B_q)
            l_i = L[:, i:end_idx]
            D_i = D[:, i:end_idx]

            for j in range(0, N_k, B_k):
                if is_causal and j >= end_idx:
                    break
                end_idx_k = min(j + B_k, N_k)
                # (batch_size, B_k, hidden)
                k = K[:, j:end_idx_k, :]
                v = V[:, j:end_idx_k, :]
                s_ij = q @ k.transpose(-2, -1)
                s_ij = s_ij * scale

                if is_causal:
                    q_idx = torch.arange(i, end_idx, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(j, end_idx_k, device=Q.device).unsqueeze(0)
                    mask = q_idx >= k_idx
                    s_ij = s_ij.masked_fill(~mask, float('-inf'))

                # (batch_size, B_q, B_k)
                score_ij = torch.exp(s_ij - l_i.unsqueeze(-1))
                # (batch_size, B_k, hidden)
                d_v_tmp = score_ij.transpose(-2, -1) @ do_i
                d_V[:, j:end_idx_k, :] += d_v_tmp

                d_p_ij = do_i @ v.transpose(-2, -1)
                # (batch_size, B_q, B_k)
                d_s_ij = score_ij * (d_p_ij - D_i.unsqueeze(-1))

                # (batch_size, B_q, hidden)
                d_q = d_s_ij @ k * scale
                d_Q[:, i:end_idx, :] += d_q

                # (batch_size, B_k, hidden)
                d_k = d_s_ij.transpose(-2, -1) @ q * scale
                d_K[:, j:end_idx_k, :] += d_k

        return d_Q, d_K, d_V, None