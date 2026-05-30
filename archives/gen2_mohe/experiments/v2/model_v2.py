"""
RINASeqModel V2 —  adiabatic elimination 序列模型。

核心流程:
  for t in range(seq_len):
      1. gate(h, x_t) → h_ssm  [per-step, M=batch]
      2. buffer h_ssm
      3. 每 K 步: batch attractor on 全 buffer [M = batch * K]
      4. attractor 修正后的状态回写 → h, logits

效果: gate 仍是 M=batch 窄 GEMM，attractor 变为 M=batch*K 宽 GEMM。
"""
import torch
import torch.nn as nn
from cell_v2 import CANNSSMCellV2


class RINASeqModelV2(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=4096,
                 beta=1.0, n_slots=4096, attract_every=4,
                 pattern_rank=0, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = CANNSSMCellV2(
            d_model, n_patterns=n_patterns, beta=beta,
            attract_every=attract_every,
            pattern_rank=pattern_rank, n_heads=n_heads,
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))
        self.slot_proj = nn.Linear(d_model, d_model)

    def slot_write(self, key_id, value_id):
        with torch.no_grad():
            ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x):
        """Adiabatic elimination 序列前向。

        Args:
            x: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits_list = []

        buffer_h_ssm = []
        buffer_combined = []
        corrected_states = {}

        t = 0
        while t < seq_len - 1:
            # Step 1: gate (per-step, M=batch)
            h_ssm, combined = self.cell.forward_gate(h, emb[:, t, :])
            buffer_h_ssm.append(h_ssm)
            buffer_combined.append(combined)

            should_attract = ((t + 1) % self.attract_every == 0) or (t == seq_len - 2)

            if should_attract and len(buffer_h_ssm) > 0:
                n_buf = len(buffer_h_ssm)

                # Step 2: batch attractor (M=batch*n_buf)
                h_stacked = torch.stack(buffer_h_ssm, dim=0).reshape(n_buf * bsz, dm)
                c_stacked = torch.stack(buffer_combined, dim=0).reshape(n_buf * bsz, dm * 2)

                h_corrected = self.cell.forward_batch_attractor(h_stacked, c_stacked)
                h_corrected = h_corrected.reshape(n_buf, bsz, dm)

                # Step 3: distribute corrected states to output slots
                for i in range(n_buf):
                    step_idx = t - n_buf + 1 + i
                    corrected_states[step_idx] = h_corrected[i]

                h = self.cell.norm(h_corrected[-1])
                buffer_h_ssm.clear()
                buffer_combined.clear()

                t += 1
            else:
                corrected_states[t] = h_ssm
                h = self.cell.norm(h_ssm)
                t += 1

        # Step 4: compute logits from corrected states
        for step in range(seq_len - 1):
            hn = self.state_norm(corrected_states[step])
            logits_list.append(self.head(hn))

        # Last position: slot injection
        i_ext = self.slot_table[x[:, -1]]
        h_last = h + i_ext
        h_last, combined_last = self.cell.forward_gate(h_last, emb[:, -1, :])
        h_last = self.cell.forward_batch_attractor(
            h_last, combined_last
        )
        h_last = self.cell.norm(h_last)
        hn = self.state_norm(h_last)
        logits_list.append(self.head(hn))

        return torch.stack(logits_list, dim=1)
