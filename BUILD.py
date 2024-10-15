#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BUILD.py
@Author     ：Heywecome
@Date       ：2024/10/15 08:57 
@Description：For
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class WhitenLayer(nn.Module):
    """Single Parametric Whitening Layer"""

    def __init__(self, input_size, output_size, dropout=0.0):
        super(WhitenLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)  # (768, 300)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2 * input_size),
            nn.ReLU(),
            nn.Linear(2 * input_size, output_size),
            nn.Dropout(dropout),
        )
        self.net.apply(self._init_weights)  # Apply the initialization method to all modules in the network

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming/He initialization suitable for ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)


# Assuming you have the above classes PWLayer and MoEAdaptorLayer defined here
class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)  # (768, 300)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # 300, 16
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(300, 300)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out  # 2048, 50, 16


# Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor with top-4 gating"""
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps
        self.top_k = int(n_exps / 2)
        self.noisy_gating = noise
        self.whiten_layer = WhitenLayer(layers[0], layers[1])
        self.experts = nn.ModuleList([Expert(layers[1], layers[1], dropout) for i in range(n_exps)])  # (768, 300)
        self.w_gate = nn.Parameter(torch.zeros(layers[1], n_exps), requires_grad=True)  # (768, 8)
        self.w_noise = nn.Parameter(torch.zeros(layers[1], n_exps), requires_grad=True)  # (768 ,8)

        self.attention_listener = MultiHeadAttention(300, 4, 16, 0.1)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate # (2048, 50, 300) @ (300, 8)
        # 为了负载平衡，从门控的线性层向 logits 激活函数添加标准正态噪声
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits).to(x.device) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Router
        # 通过仅保留沿最后一个维度进行比较的前 k 大的值，来获得稀疏门控的输出。
        # 用负无穷值填充其余部分，在使用 softmax 激活函数。负无穷会被映射至零，而最大的前两个值会更加突出，且和为 1。
        # 要求和为 1 是为了对专家输出的内容进行加权。
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # full_like clones a tensor and fills it with a specified value (like infinity) for masking or calculations.
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        gating_output = F.softmax(sparse_logits, dim=-1)  # [2048, 50, 8]

        # 创建稀疏化的混合专家模块
        # 在获得门控网络的输出结果之后，对于给定的 token，将前 k 个值选择性地与来自相应的前 k 个专家的输出相乘。
        # 这种选择性乘法的结果是一个加权和，该加权和构成 SparseMoe 模块的输出。
        # 这个过程的关键和难点是避免不必要的乘法运算，只为前 k 名专家进行正向转播。
        # 为每个专家执行前向传播将破坏使用稀疏 MoE 的目的，因为这个过程将不再是稀疏的。
        final_output = torch.zeros_like(x)  # (2048, 50, 300)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))  # [102400, 300]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # [102400, 8]

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create the mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)  # (2048, 50)
            flat_mask = expert_mask.view(-1)  # (102400,)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]  # (102400, 300)
                expert_output = expert(expert_input)  # (102400, 300)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # (102400, 1)
                weighted_output = expert_output * gating_scores  # (102400, 300)

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output   # (2048, 50, 300)

    def forward(self, x):
        # Step 1: convert the x to the shape of (2048, 50, 300)
        x = self.whiten_layer(x)

        # Step 2: add Listener
        x = self.attention_listener(x)

        gates = self.noisy_top_k_gating(x, self.training)  # (2048, 50, 300)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)] {list:8}
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # 2048,50,8,300
        multiple_outputs = gates.unsqueeze(-2) * expert_outputs  # (2048, 50, 1, 300) * (2048,50,8,300) -> (2048, 50, 8, 300)
        return multiple_outputs.sum(dim=-2)  # (2048, 50, 300)


class BUILD(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores