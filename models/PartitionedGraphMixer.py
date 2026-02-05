import torch
import torch.nn as nn
import numpy as np
import os
from models.modules import TimeEncoder
from models.GraphMixer import MLPMixer
import pickle

class PartitionedGraphMixer(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
                 time_feat_dim, num_tokens, num_layers=2,
                 token_dim_expansion_factor=0.5, channel_dim_expansion_factor=4.0,
                 dropout=0.1, device='cpu', grad_split_percentile=99, grad_threshold=1.0,
                 mask_save_dir=None):
        super().__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.device = device
        self.grad_split_percentile = grad_split_percentile
        self.grad_threshold = grad_threshold
        self.mask_save_dir = mask_save_dir or "./saved_masks"

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_raw_features.shape[1] + time_feat_dim, 64)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=num_tokens, num_channels=64,
                     token_dim_expansion_factor=token_dim_expansion_factor,
                     channel_dim_expansion_factor=channel_dim_expansion_factor,
                     dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(64 + self.node_raw_features.shape[1], self.node_raw_features.shape[1])

        self.grad_mask = None
        self.grad_mask_node_ids = None
        self.grad_mask_dict = None

    def compute_node_temporal_embeddings(self, node_ids, node_interact_times, num_neighbors, time_gap):
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        edge_feats = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        time_feats = self.time_encoder(torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        time_feats[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

        combined_features = torch.cat([edge_feats, time_feats], dim=-1)
        combined_features = self.projection_layer(combined_features)

        if self.grad_mask_dict is not None:
            grad_mask_tensor = torch.tensor(
                [self.grad_mask_dict.get(int(nid), False) for nid in node_ids],
                dtype=torch.bool,
                device=self.device
            )
            combined_features[grad_mask_tensor] = 0.0  # Dropout-style masking

        for mixer in self.mlp_mixers:
            combined_features = mixer(combined_features)

        agg = torch.mean(combined_features, dim=1)

        gap_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=time_gap)
        node_feats = self.node_raw_features[torch.from_numpy(gap_ids)]
        valid_mask = torch.from_numpy((gap_ids > 0).astype(np.float32))
        valid_mask[valid_mask == 0] = -1e10
        scores = torch.softmax(valid_mask, dim=1).to(self.device)
        agg_node = torch.mean(node_feats * scores.unsqueeze(-1), dim=1)

        final_output = self.output_layer(torch.cat([agg, agg_node + self.node_raw_features[torch.from_numpy(node_ids)]], dim=1))
        return final_output

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors=20, time_gap=2000):
        src_embeddings = self.compute_node_temporal_embeddings(src_node_ids, node_interact_times, num_neighbors, time_gap)
        dst_embeddings = self.compute_node_temporal_embeddings(dst_node_ids, node_interact_times, num_neighbors, time_gap)
        return src_embeddings, dst_embeddings

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()



# # decouple
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from models.modules import TimeEncoder
# from models.GraphMixer import MLPMixer
# import pickle

# class PartitionedGraphMixer(nn.Module):
#     def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
#                  time_feat_dim, num_tokens, num_layers=2,
#                  token_dim_expansion_factor=0.5, channel_dim_expansion_factor=4.0,
#                  dropout=0.1, device='cpu', grad_split_percentile=99, grad_threshold=1.0,
#                  mask_save_dir=None):
#         super().__init__()
#         self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
#         self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
#         self.neighbor_sampler = neighbor_sampler
#         self.device = device
#         self.grad_split_percentile = grad_split_percentile
#         self.grad_threshold = grad_threshold
#         self.mask_save_dir = mask_save_dir or "./saved_masks"

#         self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
#         self.projection_layer = nn.Linear(self.edge_raw_features.shape[1] + time_feat_dim, 64)

#         # self.mlp_mixers = nn.ModuleList([
#         #     MLPMixer(num_tokens=num_tokens, num_channels=64,
#         #              token_dim_expansion_factor=token_dim_expansion_factor,
#         #              channel_dim_expansion_factor=channel_dim_expansion_factor,
#         #              dropout=dropout)
#         #     for _ in range(num_layers)
#         # ])
#         # 1. 为正常样本创建 MLP-Mixer
#         self.normal_mixers = nn.ModuleList([
#             MLPMixer(num_tokens=num_tokens, num_channels=64,
#                      token_dim_expansion_factor=token_dim_expansion_factor,
#                      channel_dim_expansion_factor=channel_dim_expansion_factor,
#                      dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # 2. 为异常样本创建 MLP-Mixer
#         self.abnormal_mixers = nn.ModuleList([
#             MLPMixer(num_tokens=num_tokens, num_channels=64,
#                      token_dim_expansion_factor=token_dim_expansion_factor,
#                      channel_dim_expansion_factor=channel_dim_expansion_factor,
#                      dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.output_layer = nn.Linear(64 + self.node_raw_features.shape[1], self.node_raw_features.shape[1])

#         self.grad_mask = None
#         self.grad_mask_node_ids = None
#         self.grad_mask_dict = None

#     def set_gradients(self, node_embeddings, labels, classifier, node_ids=None, seed=0):
#         classifier.train()
#         for p in classifier.parameters():
#             p.requires_grad_(True)

#         # 不 detach，只设置 requires_grad
#         node_embeddings.requires_grad_(True)

#         logits = classifier(node_embeddings).squeeze()
#         loss_fn = nn.BCEWithLogitsLoss(reduction='none')
#         losses = loss_fn(logits, labels)

#         grads = []
#         for i in range(len(losses)):
#             try:
#                 grad_i = torch.autograd.grad(losses[i], node_embeddings, retain_graph=True)[0][i]
#                 grads.append(grad_i.detach().cpu().numpy())
#             except Exception as e:
#                 grads.append(np.zeros_like(node_embeddings[0].detach().cpu().numpy()))

#         grad_norms = np.linalg.norm(grads, axis=1)
#         # threshold = np.percentile(grad_norms, self.grad_split_percentile)
#         threshold = self.grad_threshold

#         self.grad_mask_node_ids = node_ids if node_ids is not None else np.arange(len(grads))
#         self.grad_mask = grad_norms >= threshold
#         self.grad_mask_dict = dict(zip(self.grad_mask_node_ids.tolist(), self.grad_mask.tolist()))

#         if self.mask_save_dir:
#             os.makedirs(self.mask_save_dir, exist_ok=True)
#             save_path = os.path.join(self.mask_save_dir, f"grad_mask_seed{seed}.pkl")
#             with open(save_path, 'wb') as f:
#                 pickle.dump({"node_ids": self.grad_mask_node_ids, "mask": self.grad_mask}, f)

#     def compute_node_temporal_embeddings(self, node_ids, node_interact_times, num_neighbors, time_gap):
#         neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
#             node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

#         edge_feats = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
#         time_feats = self.time_encoder(torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
#         time_feats[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

#         combined_features = torch.cat([edge_feats, time_feats], dim=-1)
#         combined_features = self.projection_layer(combined_features)

#         # if self.grad_mask_dict is not None:
#         #     grad_mask_tensor = torch.tensor(
#         #         [self.grad_mask_dict.get(int(nid), False) for nid in node_ids],
#         #         dtype=torch.bool,
#         #         device=self.device
#         #     )
#         #     combined_features[grad_mask_tensor] = 0.0  # Dropout-style masking
#         # for mixer in self.mlp_mixers:
#         #     combined_features = mixer(combined_features)

#         if self.grad_mask_dict is not None and len(self.grad_mask_dict) > 0:
#             # print("Using grad_mask_dict to partition samples.")
#             # 1. 创建划分掩码
#             is_abnormal_mask = torch.tensor(
#                 [self.grad_mask_dict.get(int(nid), False) for nid in node_ids],
#                 dtype=torch.bool,
#                 device=self.device
#             )
#             is_normal_mask = ~is_abnormal_mask

#             # 2. 初始化输出张量
#             processed_features = torch.zeros_like(combined_features)

#             # 3. 正常专家处理正常样本
#             if is_normal_mask.any():
#                 # 提取正常样本的特征
#                 normal_features = combined_features[is_normal_mask]
#                 # 逐层通过 normal_mixers
#                 for mixer in self.normal_mixers:
#                     normal_features = mixer(normal_features)
#                 # 将处理后的特征放回输出张量的对应位置
#                 processed_features[is_normal_mask] = normal_features

#             # 4. 异常专家处理异常样本
#             if is_abnormal_mask.any():
#                 # 提取异常样本的特征
#                 abnormal_features = combined_features[is_abnormal_mask]
#                 # 逐层通过 abnormal_mixers
#                 for mixer in self.abnormal_mixers:
#                     abnormal_features = mixer(abnormal_features)
#                 # 将处理后的特征放回输出张量的对应位置
#                 processed_features[is_abnormal_mask] = abnormal_features
            
#             # combined_features 现在是融合了两个专家输出的结果
#             combined_features = processed_features

#         else:
#             # 如果没有掩码（例如在评估或预热时），默认使用正常专家网络
#             # 这是一个设计选择，也可以选择使用一个独立的、更大的网络
#             # print("Error: No grad_mask_dict found during training. Defaulting to normal expert network.")
#             for mixer in self.normal_mixers:
#                 combined_features = mixer(combined_features)



#         agg = torch.mean(combined_features, dim=1)

#         gap_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
#             node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=time_gap)
#         node_feats = self.node_raw_features[torch.from_numpy(gap_ids)]
#         valid_mask = torch.from_numpy((gap_ids > 0).astype(np.float32))
#         valid_mask[valid_mask == 0] = -1e10
#         scores = torch.softmax(valid_mask, dim=1).to(self.device)
#         agg_node = torch.mean(node_feats * scores.unsqueeze(-1), dim=1)

#         final_output = self.output_layer(torch.cat([agg, agg_node + self.node_raw_features[torch.from_numpy(node_ids)]], dim=1))
#         return final_output

#     def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors=20, time_gap=2000):
#         src_embeddings = self.compute_node_temporal_embeddings(src_node_ids, node_interact_times, num_neighbors, time_gap)
#         dst_embeddings = self.compute_node_temporal_embeddings(dst_node_ids, node_interact_times, num_neighbors, time_gap)
#         return src_embeddings, dst_embeddings

#     def set_neighbor_sampler(self, neighbor_sampler):
#         self.neighbor_sampler = neighbor_sampler
#         if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
#             assert self.neighbor_sampler.seed is not None
#             self.neighbor_sampler.reset_random_state()





# # decouple-new
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from models.modules import TimeEncoder
# from models.GraphMixer import MLPMixer
# import pickle

# class PartitionedGraphMixer(nn.Module):
#     def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler,
#                  time_feat_dim, num_tokens, num_layers=2,
#                  token_dim_expansion_factor=0.5, channel_dim_expansion_factor=4.0,
#                  dropout=0.1, device='cpu', grad_split_percentile=99, grad_threshold=1.0,
#                  mask_save_dir=None):
#         super().__init__()
#         self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
#         self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
#         self.neighbor_sampler = neighbor_sampler
#         self.device = device
#         # self.grad_split_percentile = grad_split_percentile
#         # self.grad_threshold = grad_threshold
#         self.mask_save_dir = mask_save_dir or "./saved_masks"

#         self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
#         self.projection_layer = nn.Linear(self.edge_raw_features.shape[1] + time_feat_dim, 64)

#         # 1. 为正常样本创建 MLP-Mixer
#         self.normal_mixers = nn.ModuleList([
#             MLPMixer(num_tokens=num_tokens, num_channels=64,
#                      token_dim_expansion_factor=token_dim_expansion_factor,
#                      channel_dim_expansion_factor=channel_dim_expansion_factor,
#                      dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         # 2. 为异常样本创建 MLP-Mixer
#         self.abnormal_mixers = nn.ModuleList([
#             MLPMixer(num_tokens=num_tokens, num_channels=64,
#                      token_dim_expansion_factor=token_dim_expansion_factor,
#                      channel_dim_expansion_factor=channel_dim_expansion_factor,
#                      dropout=dropout)
#             for _ in range(num_layers)
#         ])

#         self.output_layer = nn.Linear(64 + self.node_raw_features.shape[1], self.node_raw_features.shape[1])

#         self.grad_mask = None
#         self.grad_mask_node_ids = None
#         self.grad_mask_dict = None

#     # def set_gradients(self, node_embeddings, labels, classifier, node_ids=None, seed=0):
#     #     classifier.train()
#     #     for p in classifier.parameters():
#     #         p.requires_grad_(True)

#     #     # 不 detach，只设置 requires_grad
#     #     node_embeddings.requires_grad_(True)

#     #     logits = classifier(node_embeddings).squeeze()
#     #     loss_fn = nn.BCEWithLogitsLoss(reduction='none')
#     #     losses = loss_fn(logits, labels)

#     #     grads = []
#     #     for i in range(len(losses)):
#     #         try:
#     #             grad_i = torch.autograd.grad(losses[i], node_embeddings, retain_graph=True)[0][i]
#     #             grads.append(grad_i.detach().cpu().numpy())
#     #         except Exception as e:
#     #             grads.append(np.zeros_like(node_embeddings[0].detach().cpu().numpy()))

#     #     grad_norms = np.linalg.norm(grads, axis=1)
#     #     # threshold = np.percentile(grad_norms, self.grad_split_percentile)
#     #     threshold = self.grad_threshold

#     #     self.grad_mask_node_ids = node_ids if node_ids is not None else np.arange(len(grads))
#     #     self.grad_mask = grad_norms >= threshold
#     #     self.grad_mask_dict = dict(zip(self.grad_mask_node_ids.tolist(), self.grad_mask.tolist()))

#     #     if self.mask_save_dir:
#     #         os.makedirs(self.mask_save_dir, exist_ok=True)
#     #         save_path = os.path.join(self.mask_save_dir, f"grad_mask_seed{seed}.pkl")
#     #         with open(save_path, 'wb') as f:
#     #             pickle.dump({"node_ids": self.grad_mask_node_ids, "mask": self.grad_mask}, f)

#     def compute_node_temporal_embeddings(self, node_ids, node_interact_times, num_neighbors, time_gap):
#         neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
#             node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

#         edge_feats = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
#         time_feats = self.time_encoder(torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
#         time_feats[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

#         combined_features = torch.cat([edge_feats, time_feats], dim=-1)
#         combined_features = self.projection_layer(combined_features)

#         if self.grad_mask_dict is not None and len(self.grad_mask_dict) > 0:
#             # print("Using grad_mask_dict to partition samples.")
#             # 1. 创建划分掩码
#             is_abnormal_mask = torch.tensor(
#                 [self.grad_mask_dict.get(int(nid), False) for nid in node_ids],
#                 dtype=torch.bool,
#                 device=self.device
#             )
#             is_normal_mask = ~is_abnormal_mask

#             # 2. 初始化输出张量
#             processed_features = torch.zeros_like(combined_features)

#             # 3. 正常专家处理正常样本
#             if is_normal_mask.any():
#                 # 提取正常样本的特征
#                 normal_features = combined_features[is_normal_mask]
#                 # 逐层通过 normal_mixers
#                 for mixer in self.normal_mixers:
#                     normal_features = mixer(normal_features)
#                 # 将处理后的特征放回输出张量的对应位置
#                 processed_features[is_normal_mask] = normal_features

#             # 4. 异常专家处理异常样本
#             if is_abnormal_mask.any():
#                 # 提取异常样本的特征
#                 abnormal_features = combined_features[is_abnormal_mask]
#                 # 逐层通过 abnormal_mixers
#                 for mixer in self.abnormal_mixers:
#                     abnormal_features = mixer(abnormal_features)
#                 # 将处理后的特征放回输出张量的对应位置
#                 processed_features[is_abnormal_mask] = abnormal_features
            
#             # combined_features 现在是融合了两个专家输出的结果
#             combined_features = processed_features

#         else:
#             # 如果没有掩码（例如在评估或预热时），默认使用正常专家网络
#             # 这是一个设计选择，也可以选择使用一个独立的、更大的网络
#             # print("Error: No grad_mask_dict found during training. Defaulting to normal expert network.")
#             for mixer in self.normal_mixers:
#                 combined_features = mixer(combined_features)



#         agg = torch.mean(combined_features, dim=1)

#         gap_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
#             node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=time_gap)
#         node_feats = self.node_raw_features[torch.from_numpy(gap_ids)]
#         valid_mask = torch.from_numpy((gap_ids > 0).astype(np.float32))
#         valid_mask[valid_mask == 0] = -1e10
#         scores = torch.softmax(valid_mask, dim=1).to(self.device)
#         agg_node = torch.mean(node_feats * scores.unsqueeze(-1), dim=1)

#         final_output = self.output_layer(torch.cat([agg, agg_node + self.node_raw_features[torch.from_numpy(node_ids)]], dim=1))
#         return final_output

#     def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors=20, time_gap=2000):
#         src_embeddings = self.compute_node_temporal_embeddings(src_node_ids, node_interact_times, num_neighbors, time_gap)
#         dst_embeddings = self.compute_node_temporal_embeddings(dst_node_ids, node_interact_times, num_neighbors, time_gap)
#         return src_embeddings, dst_embeddings

#     def set_neighbor_sampler(self, neighbor_sampler):
#         self.neighbor_sampler = neighbor_sampler
#         if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
#             assert self.neighbor_sampler.seed is not None
#             self.neighbor_sampler.reset_random_state()
