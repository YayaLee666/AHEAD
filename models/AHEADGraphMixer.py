# models/AHEADGraphMixer.py

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch.nn.functional as F 
from .Memorymodel import Memory, MessageFunction, MemoryUpdater
from .PartitionedGraphMixer import PartitionedGraphMixer
from .modules import TimeEncoder
import pandas as pd


class IndicatorCache:
    def __init__(self, num_nodes: int, device: str = 'cpu'):
        self.num_nodes = num_nodes
        self.device = device
        
        self.indicator_cache = np.zeros((num_nodes + 1, 2), dtype=np.float32)
        
        self.epoch_indicators = []

    def add(self, node_ids: np.ndarray, grad_norms: np.ndarray, residuals: np.ndarray):
        self.epoch_indicators.extend(zip(node_ids.tolist(), grad_norms.tolist(), residuals.tolist()))

    def update_global_cache(self):
        if not self.epoch_indicators:
            return
        for node_id, grad_norm, score in self.epoch_indicators:
            self.indicator_cache[node_id] = [grad_norm, score]

        self.clear()

    def get_indicators_for_batch(self, node_ids: np.ndarray) -> np.ndarray:
        return self.indicator_cache[node_ids]

    def clear(self):
        self.epoch_indicators = []

# =====================================================================================
#  Main Model
# =====================================================================================
class AHEADGraphMixer(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, num_tokens: int, memory_dim: int, message_dim: int, 
                 dataset_name: str, 
                 run_idx: int,     
                 device: str = 'cpu',
                 num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1,
                 loss_type: str = 'infonce',
                 temperature: float = 0.1,
                 time_gap: int = 2000,
                 num_hard_negatives: int = 0,
                 lambda_thresh_loss: float = 0.5,
                 **kwargs): 
        
        super(AHEADGraphMixer, self).__init__()
        self.device = device
        self.neighbor_sampler = neighbor_sampler
        self.num_nodes = node_raw_features.shape[0]
        self.loss_type = loss_type
        self.memory_dim = memory_dim
        self.temperature = temperature
        self.time_gap = time_gap
        self.num_hard_negatives = num_hard_negatives

        self._analysis_log_path = None
        self._analysis_log_initialized = False

        self.indicator_weights_logits = nn.Parameter(torch.zeros(2)) 

        self.lambda_thresh_loss = lambda_thresh_loss

        self.indicator_cache = IndicatorCache(
            num_nodes=self.num_nodes, 
            device=self.device
        )

        self.auroc_history = []
        self.auroc_history_size = 10 

        self.memory_generator = PartitionedGraphMixer(
            node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim,
            num_tokens=num_tokens, num_layers=num_layers,
            token_dim_expansion_factor=token_dim_expansion_factor,
            channel_dim_expansion_factor=channel_dim_expansion_factor,
            dropout=dropout, device=device)
        
        original_out_dim = self.memory_generator.output_layer.in_features
        self.memory_generator.output_layer = nn.Linear(in_features=original_out_dim, out_features=memory_dim, bias=True).to(device)

        self.memory = Memory(n_nodes=self.num_nodes, memory_dimension=memory_dim, device=device)
        self.time_encoder = TimeEncoder(time_dim=memory_dim, parameter_requires_grad=True)
        raw_message_dim = memory_dim + memory_dim
        self.message_function = MessageFunction(raw_message_dim, message_dim)
        self.memory_updater = MemoryUpdater(self.memory, message_dim, memory_dim)

        self._analysis_log_path = None
        self._analysis_log_initialized = False

        self.degree_calib_k = int(kwargs.get("degree_calib_k", 3))
        # hard_scale: score <- score * scale  
        self.degree_calib_scale = float(kwargs.get("degree_calib_scale", 0.9))
        for param in self.memory_generator.parameters():
            param.requires_grad = True  

        self.beta_logit = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.beta_quantile = float(kwargs.get("beta_quantile", 0.9))



    def _get_infonce_loss_for_pseudo_grads(self, upd_src_m, prev_src_m, gen_src_m):
        upd_m_norm, prev_m_norm, gen_m_norm = [F.normalize(x, p=2, dim=1) for x in [upd_src_m, prev_src_m, gen_src_m]]
        
        loss_tem = self._calculate_hnm_infonce_per_sample(
            anchor_embeds=upd_m_norm, positive_embeds=prev_m_norm, all_candidates=upd_m_norm)
        loss_str = self._calculate_hnm_infonce_per_sample(
            anchor_embeds=gen_m_norm, positive_embeds=upd_m_norm, all_candidates=gen_m_norm)
        
        return loss_tem + loss_str


    def _compute_components(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap):
        src_nodes_torch = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_nodes_torch = torch.from_numpy(dst_node_ids).long().to(self.device)
        edge_times_torch = torch.from_numpy(node_interact_times).float().to(self.device)

        prev_src_memory = self.memory.get_memory(src_nodes_torch)
        prev_dst_memory = self.memory.get_memory(dst_nodes_torch)

        src_time_diffs = edge_times_torch - self.memory.get_last_update(src_nodes_torch)
        src_time_enc = self.time_encoder(src_time_diffs.unsqueeze(1)).squeeze(1)
        raw_src_messages = torch.cat([prev_dst_memory.detach(), src_time_enc], dim=1)
        src_messages = self.message_function(raw_src_messages)
        updated_src_memory = self.memory_updater(src_nodes_torch, src_messages)
        
        dst_time_diffs = edge_times_torch - self.memory.get_last_update(dst_nodes_torch)
        dst_time_enc = self.time_encoder(dst_time_diffs.unsqueeze(1)).squeeze(1)
        raw_dst_messages = torch.cat([prev_src_memory.detach(), dst_time_enc], dim=1)
        dst_messages = self.message_function(raw_dst_messages)
        updated_dst_memory = self.memory_updater(dst_nodes_torch, dst_messages)
        
        generated_src_memory = self.memory_generator.compute_node_temporal_embeddings(
            node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors, time_gap=time_gap)
        generated_dst_memory = self.memory_generator.compute_node_temporal_embeddings(
            node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors, time_gap=time_gap)
        
        return (prev_src_memory, updated_src_memory, generated_src_memory,
                prev_dst_memory, updated_dst_memory, generated_dst_memory,
                src_nodes_torch, dst_nodes_torch, edge_times_torch)


    def _calculate_hnm_infonce_per_sample(self, anchor_embeds, positive_embeds, all_candidates):

        logits = torch.mm(anchor_embeds, all_candidates.t())
        positive_logits = torch.sum(anchor_embeds * positive_embeds, dim=1)
        use_hnm = self.num_hard_negatives > 0 and self.num_hard_negatives < len(anchor_embeds) - 1
        if use_hnm:
            eye_mask = torch.eye(len(anchor_embeds), dtype=torch.bool, device=self.device)
            logits.masked_fill_(eye_mask, -1e9)
            hard_negative_logits, _ = torch.topk(logits, self.num_hard_negatives, dim=1)
            final_logits = torch.cat([positive_logits.unsqueeze(1), hard_negative_logits], dim=1)
        else:
            final_logits = logits 

        scaled_pos_logits = positive_logits / self.temperature
        log_sum_exp_term = torch.logsumexp(final_logits / self.temperature, dim=1)
        
        return - (scaled_pos_logits - log_sum_exp_term)


    def _calculate_hnm_infonce_loss(self, anchor_embeds, positive_embeds, all_candidates):
        logits = torch.mm(anchor_embeds, all_candidates.t())
        positive_logits = torch.sum(anchor_embeds * positive_embeds, dim=1)
        
        if anchor_embeds.shape == all_candidates.shape:
            eye_mask = torch.eye(len(anchor_embeds), dtype=torch.bool, device=self.device)
            logits.masked_fill_(eye_mask, -float('inf'))
            
        use_hnm = self.num_hard_negatives > 0 and self.num_hard_negatives < all_candidates.shape[0]
        if use_hnm:
            hard_negative_logits, _ = torch.topk(logits, self.num_hard_negatives, dim=1)
            final_logits = torch.cat([positive_logits.unsqueeze(1), hard_negative_logits], dim=1)
            labels = torch.zeros(len(anchor_embeds), dtype=torch.long, device=self.device)
        else: 
            if anchor_embeds.shape == all_candidates.shape:
                eye_mask = torch.eye(len(anchor_embeds), dtype=torch.bool, device=self.device)
                logits[eye_mask] = positive_logits
                final_logits = logits
                labels = torch.arange(len(anchor_embeds), dtype=torch.long, device=self.device)
            else:
                final_logits = torch.cat([positive_logits.unsqueeze(1), logits], dim=1)
                labels = torch.zeros(len(anchor_embeds), dtype=torch.long, device=self.device)

        return F.cross_entropy(final_logits / self.temperature, labels)
    
    def _compute_infonce_loss(self, prev_src_m, upd_src_m, gen_src_m):
        upd_m_norm, prev_m_norm, gen_m_norm = [F.normalize(x, p=2, dim=1) for x in [upd_src_m, prev_src_m, gen_src_m]]
        
        loss_tem = self._calculate_hnm_infonce_loss(
            anchor_embeds=upd_m_norm, positive_embeds=prev_m_norm, all_candidates=upd_m_norm)
        loss_str = self._calculate_hnm_infonce_loss(
            anchor_embeds=gen_m_norm, positive_embeds=upd_m_norm, all_candidates=gen_m_norm)
        
        return loss_tem + loss_str

    
    def find_best_ag_threshold(self, scores: torch.Tensor):
        n_total = len(scores)
        if n_total < 10:  
            return torch.median(scores)
        sorted_scores, _ = torch.sort(scores, descending=True)
    
        max_k = int(n_total * 0.1) + 1 
        k = torch.arange(1, max_k, device=scores.device, dtype=torch.float32)

        sum_scores = torch.cumsum(sorted_scores, dim=0)
        sum_sq_scores = torch.cumsum(sorted_scores**2, dim=0)
        n1 = k
        mu1 = sum_scores[:max_k-1] / n1
        var1 = (sum_sq_scores[:max_k-1] / n1) - mu1**2
        n0 = n_total - n1
        mu0 = (sum_scores[-1] - sum_scores[:max_k-1]) / n0
        var0 = ((sum_sq_scores[-1] - sum_sq_scores[:max_k-1]) / n0) - mu0**2
        w1 = n1 / n_total
        w0 = n0 / n_total
        numerator = w0 * w1 * (mu1 - mu0)**2
        denominator = w0 * var0 + w1 * var1 + 1e-8
        ag_values = numerator / denominator
        best_k_idx = torch.argmax(ag_values)
        best_threshold = sorted_scores[best_k_idx]
        
        return best_threshold

    def _compute_infonce_loss_per_sample(self, prev_src_m, upd_src_m, gen_src_m):

        upd_m_norm = F.normalize(upd_src_m, p=2, dim=1)
        prev_m_norm = F.normalize(prev_src_m, p=2, dim=1)
        gen_m_norm  = F.normalize(gen_src_m, p=2, dim=1)

        loss_tem = self._calculate_hnm_infonce_per_sample(
            anchor_embeds=upd_m_norm, positive_embeds=prev_m_norm, all_candidates=upd_m_norm
        )
        loss_str = self._calculate_hnm_infonce_per_sample(
            anchor_embeds=gen_m_norm, positive_embeds=upd_m_norm, all_candidates=gen_m_norm
        )
        return loss_tem + loss_str  # [B]

    def compute_and_cache_indicators(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
        src_nodes_torch = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_nodes_torch = torch.from_numpy(dst_node_ids).long().to(self.device)
        prev_src_memory = self.memory.get_memory(src_nodes_torch).detach()
        prev_dst_memory = self.memory.get_memory(dst_nodes_torch).detach()
        src_time_diffs = torch.from_numpy(node_interact_times).float().to(self.device) - self.memory.get_last_update(src_nodes_torch)
        src_time_enc = self.time_encoder(src_time_diffs.unsqueeze(1)).squeeze(1)
        raw_src_messages = torch.cat([prev_dst_memory, src_time_enc], dim=1)
        src_messages = self.message_function(raw_src_messages)
        updated_src_memory = self.memory_updater(src_nodes_torch, src_messages).requires_grad_()

        generated_src_memory = self.memory_generator.compute_node_temporal_embeddings(
            node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors, time_gap=time_gap)
        updated_src_memory_grad = updated_src_memory.clone().requires_grad_()
        losses_for_grad = self._get_infonce_loss_for_pseudo_grads(
            updated_src_memory_grad, prev_src_memory.detach(), generated_src_memory.detach()
        )
        grads = torch.autograd.grad(outputs=losses_for_grad, inputs=updated_src_memory_grad, grad_outputs=torch.ones_like(losses_for_grad))[0]       
        grad_norms = torch.linalg.norm(grads, dim=1).detach().cpu().numpy()

        with torch.no_grad():
            residuals = self._get_infonce_loss_for_pseudo_grads(
                updated_src_memory, prev_src_memory, generated_src_memory
            ).cpu().numpy()

        self.indicator_cache.add(node_ids=src_node_ids, grad_norms=grad_norms, 
                                 residuals=residuals)

        return grad_norms, residuals


    def compute_unsupervised_loss(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap,                                  
                                epoch: int, **kwargs):

        last_indicators = self.indicator_cache.get_indicators_for_batch(src_node_ids)
        last_grad_norms = last_indicators[:, 0]
        last_residuals = last_indicators[:, 1]
        
        grad_min, grad_max = np.min(last_grad_norms), np.max(last_grad_norms)
        norm_grad = (last_grad_norms - grad_min) / (grad_max - grad_min + 1e-8)
        
        residual_min, residual_max = np.min(last_residuals), np.max(last_residuals)
        norm_residual = (last_residuals - residual_min) / (residual_max - residual_min + 1e-8)

        indicator_weights = F.softmax(self.indicator_weights_logits, dim=0)
        w_grad = indicator_weights[0]
        w_residual = indicator_weights[1]

        norm_grad_tensor = torch.from_numpy(norm_grad).to(self.device).float()
        norm_residual_tensor = torch.from_numpy(norm_residual).to(self.device).float()

        phi = w_grad * norm_grad_tensor + w_residual * norm_residual_tensor

        with torch.no_grad():
            xi_star = self.find_best_ag_threshold(phi.detach())
        pi = (phi <= xi_star) 

        final_mask = pi.detach().cpu().numpy().astype(bool)
        self.memory_generator.grad_mask_dict = dict(zip(src_node_ids.tolist(), final_mask.tolist()))
        prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m, src_nodes, dst_nodes, times = \
            self._compute_components(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap)
        L_main_vec = self._compute_infonce_loss_per_sample(prev_src_m, upd_src_m, gen_src_m)  # [B], torch.Tensor
        eps = 1e-8
        q_beta = torch.quantile(L_main_vec.detach(), q=self.beta_quantile)  # scalar
        soft_pseudo_labels = torch.clamp(L_main_vec.detach() / (q_beta + eps), max=1.0)  # [B], in [0,1]
        phi_prob = torch.clamp(phi, min=eps, max=1.0 - eps)  # [B], already ~[0,1]
        loss_thresh = F.binary_cross_entropy(phi_prob, soft_pseudo_labels, reduction='mean')

        if self.loss_type == 'infonce':
            main_loss = self._compute_infonce_loss(prev_src_m, upd_src_m, gen_src_m)
        else:
            raise NotImplementedError(f"Unknown '{self.loss_type}' ")

        final_loss = main_loss + self.lambda_thresh_loss * loss_thresh

        self.memory.update_state(src_nodes, upd_src_m, times)
        self.memory.update_state(dst_nodes, upd_dst_m, times)

        return final_loss

    def _calibrate_anomaly_scores_by_degree(self, scores, degrees):
        """
        scores: torch.Tensor [B] or np.ndarray [B]
        degrees: np.ndarray [B] (or torch.Tensor [B])
        return: same type as scores
        """
        # --- degrees -> torch mask on correct device ---
        if isinstance(degrees, np.ndarray):
            low_mask = torch.from_numpy(degrees < self.degree_calib_k).to(self.device)
        elif isinstance(degrees, torch.Tensor):
            low_mask = (degrees < self.degree_calib_k).to(self.device)
        else:
            degrees = np.asarray(degrees)
            low_mask = torch.from_numpy(degrees < self.degree_calib_k).to(self.device)

        if isinstance(scores, torch.Tensor):
            # keep dtype/device; avoid in-place on shared tensor
            out = scores.clone()
            if low_mask.any():
                out[low_mask] = out[low_mask] * float(self.degree_calib_scale)
            return out

        scores = np.asarray(scores, dtype=np.float32).copy()
        low_np = low_mask.detach().cpu().numpy().astype(bool)
        if np.any(low_np):
            scores[low_np] *= float(self.degree_calib_scale)
        return scores

    def _get_temporal_degree_from_sampler(self, node_ids, node_interact_times, num_neighbors, time_gap):
        import numpy as np

        fn = self.neighbor_sampler.get_historical_neighbors
        neighbors = None

        try:
            out = fn(node_ids, node_interact_times, num_neighbors=num_neighbors, time_gap=time_gap)
            neighbors = out[0] if isinstance(out, (tuple, list)) else out
        except TypeError:
            pass

        if neighbors is None:
            try:
                out = fn(node_ids, node_interact_times, num_neighbors=num_neighbors)
                neighbors = out[0] if isinstance(out, (tuple, list)) else out
            except TypeError:
                pass

        if neighbors is None:
            try:
                out = fn(node_ids, node_interact_times, num_neighbors, time_gap)
                neighbors = out[0] if isinstance(out, (tuple, list)) else out
            except TypeError:
                pass

        if neighbors is None:
            out = fn(node_ids, node_interact_times, num_neighbors)
            neighbors = out[0] if isinstance(out, (tuple, list)) else out

        neighbors = np.asarray(neighbors)
        deg = (neighbors > 0).sum(axis=1).astype(np.int32)
        return deg

    def compute_anomaly_score(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
        with torch.no_grad():
            prev_src_m, upd_src_m, gen_src_m, prev_dst_m, upd_dst_m, gen_dst_m, src_nodes, dst_nodes, times = \
                self._compute_components(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, time_gap)
            # anomaly_scores = self._get_infonce_loss_for_pseudo_grads(upd_src_m, prev_src_m, gen_src_m)
            anomaly_scores = self._compute_infonce_loss_per_sample(prev_src_m,upd_src_m,  gen_src_m)
            self.memory.update_state(src_nodes, upd_src_m, times)
            self.memory.update_state(dst_nodes, upd_dst_m, times)

        degrees = self._get_temporal_degree_from_sampler(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=num_neighbors,
            time_gap=time_gap
        )
        anomaly_scores = self._calibrate_anomaly_scores_by_degree(anomaly_scores, degrees)
        
        return anomaly_scores
    
    def reset_states(self):
        if hasattr(self, 'memory') and hasattr(self.memory, 'reset_memory'):
            self.memory.reset_memory()


    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self.memory_generator, 'set_neighbor_sampler'):
            self.memory_generator.set_neighbor_sampler(neighbor_sampler)
