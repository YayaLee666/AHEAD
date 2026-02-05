import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from tqdm import tqdm


class Memory(nn.Module):
    """
    A Global Memory Bank module.
    """
    def __init__(self, n_nodes, memory_dimension, device):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device

        # `memory` stores the evolving memory vector for each node.
        # `last_update` stores the timestamp of the last interaction for each node.
        self.memory = nn.Parameter(torch.zeros((n_nodes, memory_dimension), dtype=torch.float32, device=device), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(n_nodes, dtype=torch.float32, device=device), requires_grad=False)
    
    def get_memory(self, node_ids):
        return self.memory[node_ids, :].detach().clone()

    def set_memory(self, node_ids, memory_vectors):
        self.memory[node_ids, :] = memory_vectors

    def get_last_update(self, node_ids):
        return self.last_update[node_ids]

    def update_state(self, nodes, memory, timestamps):
        self.memory[nodes] = memory.detach()
        self.last_update[nodes] = timestamps.detach()

    def reset_memory(self):
        self.memory.data.fill_(0)
        self.last_update.data.fill_(0)

    def detach_memory(self):
        self.memory.detach_()

class MessageFunction(nn.Module):
    """
    Computes messages from raw interaction info.
    """
    def __init__(self, raw_message_dim, message_dim):
        super(MessageFunction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dim, raw_message_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dim // 2, message_dim)
        )
    
    def forward(self, raw_messages):
        return self.mlp(raw_messages)



class MemoryUpdater(nn.Module):
    """
    Updates node memory using a GRU.
    """
    def __init__(self, memory, message_dim, memory_dim):
        super(MemoryUpdater, self).__init__()
        self.memory = memory
        self.gru = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)
    # def forward(self, node_ids, messages):
    #     # Fetches the current memory (hidden state) and updates it with messages.
    #     current_memory = self.memory.get_memory(node_ids)
    #     updated_memory = self.gru(messages, current_memory)
    #     return updated_memory
    def forward(self, node_ids, messages):
        current_memory = self.memory.get_memory(node_ids).detach().clone()
        updated_memory = self.gru(messages, current_memory)
        return updated_memory


    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_tokens: int, memory_dim: int, message_dim: int, device: str = 'cpu',
                 num_layers: int = 2, dropout: float = 0.1,
                 # --- NEW: Hyperparameters for different loss types ---
                 loss_type: str = 'slade',
                 temperature: float = 0.1,
                 div_reg_factor: float = 0.1): # Keep div_reg_factor for slade loss
        
        super(UnsupervisedGraphMixer, self).__init__()
        self.device = device
        self.neighbor_sampler = neighbor_sampler
        self.num_nodes = node_raw_features.shape[0]
        self.loss_type = loss_type
        self.div_reg_factor = div_reg_factor

        # --- Core Encoder (The original GraphMixer) ---
        # It's used as the memory generator in SLADE/InfoNCE context
        self.encoder = GraphMixer(
            node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim, num_tokens=num_tokens,
            num_layers=num_layers, dropout=dropout, device=device)
        
        # Adjust its output layer to match memory_dim
        original_out_dim = self.encoder.output_layer.in_features
        self.encoder.output_layer = nn.Linear(in_features=original_out_dim, out_features=memory_dim, bias=True).to(device)

        # --- Components for stateful losses (SLADE & InfoNCE) ---
        self.memory = Memory(n_nodes=self.num_nodes, memory_dimension=memory_dim, device=device)
        self.time_encoder = TimeEncoder(time_dim=memory_dim, parameter_requires_grad=True)
        raw_message_dim = memory_dim + memory_dim
        self.message_function = MessageFunction(raw_message_dim, message_dim)
        self.memory_updater = MemoryUpdater(self.memory, message_dim, memory_dim)

        # --- InfoNCE specific parameter ---
        if self.loss_type == 'infonce':
            self.temperature = temperature

    def reset_states(self):
        """Resets all persistent states of the model."""
        self.memory.reset_memory()

    def update_states(self, src_nodes_np, dst_nodes_np, timestamps_np):
        """
        STATEFUL method: Updates the model's persistent memory.
        This is the ONLY method that should modify self.memory persistently.
        """
        with torch.no_grad():
            src_nodes = torch.from_numpy(src_nodes_np).long().to(self.device)
            dst_nodes = torch.from_numpy(dst_nodes_np).long().to(self.device)
            timestamps = torch.from_numpy(timestamps_np).float().to(self.device)
            
            # We need the previous state of the other node to compute the message
            prev_src_m = self.memory.get_memory(src_nodes)
            prev_dst_m = self.memory.get_memory(dst_nodes)
            
            # Compute updated memory vectors
            upd_src_m = self._compute_updated_memory(src_nodes, prev_dst_m, timestamps)
            upd_dst_m = self._compute_updated_memory(dst_nodes, prev_src_m, timestamps)

            # Persist the new memory state
            self.memory.update_state(src_nodes, upd_src_m, timestamps)
            self.memory.update_state(dst_nodes, upd_dst_m, timestamps)
            
    def _compute_updated_memory(self, nodes, other_nodes_mem, timestamps):
        """
        A clean helper function to compute the updated memory for a set of nodes
        without any side effects on the persistent memory state.
        
        :param nodes: Tensor, the nodes whose memory is to be updated.
        :param other_nodes_mem: Tensor, the memory of the other nodes in the interaction.
        :param timestamps: Tensor, the timestamps of the interactions.
        :return: Tensor, the new (not yet persisted) memory vectors for the nodes.
        """
        # Get the time difference from the last update
        time_diffs = timestamps - self.memory.get_last_update(nodes)
        
        # Encode the time difference
        time_enc = self.time_encoder(time_diffs.unsqueeze(1)).squeeze(1)
        
        # Create the raw message by concatenating the other node's memory and the time encoding
        raw_messages = torch.cat([other_nodes_mem, time_enc], dim=1)
        
        # Pass the raw message through the message function (MLP)
        messages = self.message_function(raw_messages)
        
        # Update the memory using the GRU updater
        updated_memory = self.memory_updater(nodes, messages)
        
        return updated_memory

    def _compute_components(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap):
        """Helper to compute prev, updated, and generated memory vectors."""
        src_nodes = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_nodes = torch.from_numpy(dst_node_ids).long().to(self.device)
        timestamps = torch.from_numpy(node_interact_times).float().to(self.device)
        
        # Get memory state BEFORE the update
        prev_src_m = self.memory.get_memory(src_nodes)
        prev_dst_m = self.memory.get_memory(dst_nodes)

        # Simulate the update to get updated_memory
        src_time_diffs = timestamps - self.memory.get_last_update(src_nodes)
        src_time_enc = self.time_encoder(src_time_diffs.unsqueeze(1)).squeeze(1)
        raw_src_msg = torch.cat([prev_dst_m.detach(), src_time_enc], dim=1)
        src_msg = self.message_function(raw_src_msg)
        upd_src_m = self.memory_updater(src_nodes, src_msg)
        
        dst_time_diffs = timestamps - self.memory.get_last_update(dst_nodes)
        dst_time_enc = self.time_encoder(dst_time_diffs.unsqueeze(1)).squeeze(1)
        raw_dst_msg = torch.cat([prev_src_m.detach(), dst_time_enc], dim=1)
        dst_msg = self.message_function(raw_dst_msg)
        upd_dst_m = self.memory_updater(dst_nodes, dst_msg)

        # Get generated_memory from the encoder
        gen_src_m = self.encoder.compute_node_temporal_embeddings(
            src_node_ids, node_interact_times, num_neighbors, time_gap)
        gen_dst_m = self.encoder.compute_node_temporal_embeddings(
            dst_node_ids, node_interact_times, num_neighbors, time_gap)
            
        return prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m, src_nodes, dst_nodes, timestamps

    def compute_unsupervised_loss(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
        """Main training loss function, supports 'slade' and 'infonce'."""
        
        # 1. Get all necessary memory vectors
        prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m, src_nodes, dst_nodes, times = \
            self._compute_components(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap)

        # 2. Dispatch to the correct loss function
        if self.loss_type == 'slade':
            loss = self._compute_slade_loss(prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m)
        elif self.loss_type == 'infonce':
            loss = self._compute_infonce_loss(prev_src_m, upd_src_m, gen_src_m)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # 3. Update persistent memory state for the next batch in the epoch
        self.memory.update_state(src_nodes, upd_src_m, times)
        self.memory.update_state(dst_nodes, upd_dst_m, times)
        
        return loss

    def _compute_slade_loss(self, prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m):
        """Computes the SLADE loss (e.g., with divergence regularizer)."""
        # (This is your existing SLADE loss logic)
        pass

    def _compute_infonce_loss(self, prev_src_m, upd_src_m, gen_src_m):
        """Computes the InfoNCE loss for both Drift (Temporal) and Recovery contrast."""
        upd_m_norm = F.normalize(upd_src_m, p=2, dim=1)
        prev_m_norm = F.normalize(prev_src_m, p=2, dim=1)
        gen_m_norm = F.normalize(gen_src_m, p=2, dim=1)
        
        # --- Drift / Temporal Contrastive Loss ---
        drift_logits = torch.mm(upd_m_norm, upd_m_norm.t()) / self.temperature
        drift_positives = torch.sum(upd_m_norm * prev_m_norm, dim=1) / self.temperature
        eye_mask = torch.eye(drift_logits.shape[0], dtype=torch.bool, device=self.device)
        drift_logits[eye_mask] = drift_positives
        labels = torch.arange(drift_logits.shape[0]).long().to(self.device)
        loss_drift = F.cross_entropy(drift_logits, labels)

        # --- Recovery Contrastive Loss ---
        recovery_logits = torch.mm(gen_m_norm, gen_m_norm.t()) / self.temperature
        recovery_positives = torch.sum(gen_m_norm * upd_m_norm, dim=1) / self.temperature
        recovery_logits[eye_mask] = recovery_positives
        loss_recovery = F.cross_entropy(recovery_logits, labels)

        return loss_drift + loss_recovery

    # def compute_anomaly_score(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
    #     """Computes anomaly score, consistent with the chosen loss, and updates state."""
    #     with torch.no_grad():
    #         # 1. Compute all memory components (needed for both scoring methods)
    #         prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m, src_nodes, dst_nodes, times = \
    #             self._compute_components(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap)

    #         # 2. Calculate score based on loss_type
    #         if self.loss_type == 'slade':
    #             cos = nn.CosineSimilarity(dim=1)
    #             drift_score = 1 - cos(upd_src_m, prev_src_m)
    #             recovery_score = 1 - cos(gen_src_m, upd_src_m)
    #             anomaly_score = (drift_score + recovery_score) / 2
            
    #         elif self.loss_type == 'infonce':
    #             upd_m_norm = F.normalize(upd_src_m, p=2, dim=1)
    #             prev_m_norm = F.normalize(prev_src_m, p=2, dim=1)
    #             gen_m_norm = F.normalize(gen_src_m, p=2, dim=1)

    #             # Simplified anomaly score: negative similarity to positive samples.
    #             # Lower similarity (higher score) means more anomalous.
    #             drift_score = 1 - torch.sum(upd_m_norm * prev_m_norm, dim=1)
    #             recovery_score = 1 - torch.sum(gen_m_norm * upd_m_norm, dim=1)
    #             anomaly_score = drift_score + recovery_score
            
    #         else:
    #              raise ValueError(f"Unknown loss_type for scoring: {self.loss_type}")
        
    #         # 3. Update persistent memory state
    #         self.memory.update_state(src_nodes, upd_src_m, times)
    #         self.memory.update_state(dst_nodes, upd_dst_m, times)
            
    #         return anomaly_score

    def compute_anomaly_score(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
        """计算异常分数。"""
        with torch.no_grad():
            prev_src_m, upd_src_m, gen_src_m, _, _, _, src_nodes, dst_nodes, times = \
                self._compute_components(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap)

            if self.loss_type == 'infonce':
                # 异常分数是 per-sample loss
                anomaly_score = self._get_infonce_loss_per_sample(prev_src_m, upd_src_m, gen_src_m)
            else:
                raise ValueError(f"Anomaly scoring for loss_type '{self.loss_type}' is not implemented.")
            
            # 在评估时也需要更新记忆状态，以确保批次间的连续性
            self.memory.update_state(src_nodes, upd_src_m, times)
            # 简化：评估时只更新源节点的记忆
            # self.memory.update_state(dst_nodes, upd_dst_m, times)
            
            return anomaly_score

    def _get_infonce_loss_per_sample(self, prev_m, upd_m, gen_m):
        """辅助函数：计算每个样本的 InfoNCE 损失作为异常分数。"""
        upd_m_norm = nn.functional.normalize(upd_m, p=2, dim=1)
        prev_m_norm = nn.functional.normalize(prev_m, p=2, dim=1)
        gen_m_norm = nn.functional.normalize(gen_m, p=2, dim=1)

        # Drift score
        drift_pos_sim = torch.sum(upd_m_norm * prev_m_norm, dim=1)
        drift_neg_sim_matrix = torch.mm(upd_m_norm, upd_m_norm.t())
        drift_log_sum_exp = torch.logsumexp(drift_neg_sim_matrix / self.temperature, dim=1)
        drift_score = - (drift_pos_sim / self.temperature - drift_log_sum_exp)

        # Recovery score
        recovery_pos_sim = torch.sum(gen_m_norm * upd_m_norm, dim=1)
        recovery_neg_sim_matrix = torch.mm(gen_m_norm, gen_m_norm.t())
        recovery_log_sum_exp = torch.logsumexp(recovery_neg_sim_matrix / self.temperature, dim=1)
        recovery_score = - (recovery_pos_sim / self.temperature - recovery_log_sum_exp)

        return drift_score + recovery_score




    def reset_states(self):
        """Resets all persistent states of the model."""
        if hasattr(self, 'memory'):
            self.memory.reset_memory()

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self, 'encoder'):
            self.encoder.set_neighbor_sampler(neighbor_sampler)