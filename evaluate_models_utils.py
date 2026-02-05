import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics, get_edge_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data



def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000,
                                       H_v_cpu=None, H_e_cpu=None, H_e_pad_cpu=None, adj_e_cpu=None, adj_v_cpu=None, T_cpu=None, mask_cpu=None, device='cpu'):
    """
    Evaluate models on the node classification task.
    For GeneralDyG, additional sequence data must be provided.
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'TPNet', 'PartitionedGraphMixer']:
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name == 'GeneralDyG':
                batch_node_ids = np.unique(np.concatenate([batch_src_node_ids, batch_dst_node_ids]))
                batch_node_ids = np.array(batch_node_ids)
                valid_node_ids = batch_node_ids[batch_node_ids < len(H_v_cpu)]
                
                H_v_batch = [H_v_cpu[i].to(device) for i in valid_node_ids]
                H_e_batch = [H_e_cpu[i].to(device) for i in valid_node_ids]
                H_e_pad_batch = torch.stack([H_e_pad_cpu[i].to(device) for i in valid_node_ids])
                adj_e_batch = [adj_e_cpu[i].to(device) for i in valid_node_ids]
                adj_v_batch = [adj_v_cpu[i].to(device) for i in valid_node_ids]
                T_batch = [T_cpu[i].to(device) for i in valid_node_ids]
                mask_batch = torch.stack([mask_cpu[i].to(device) for i in valid_node_ids])


                model[0].set_input_sequence(H_v_batch, H_e_batch, H_e_pad_batch, adj_e_batch, adj_v_batch, T_batch, mask_batch)
                _ = model[0](H_v_batch, H_e_batch, H_e_pad_batch, adj_e_batch, adj_v_batch, T_batch, mask_batch)

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer', 'PartitionedGraphMixer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'TPNet']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif model_name == 'GeneralDyG':
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()
            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics, evaluate_y_trues, evaluate_y_predicts



def evaluate_model_edge_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the edge classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'TPNet', 'PartitionedGraphMixer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer', 'PartitionedGraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'TPNet']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            
            logits = model[1](x_1=batch_src_node_embeddings, x_2=batch_dst_node_embeddings, rel_embs=model[0].edge_raw_features).squeeze(-1)
            labels = torch.from_numpy(batch_labels).float().to(logits.device)
            loss = loss_func(logits, labels)
            probs = torch.sigmoid(logits)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(probs)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)


        evaluate_metrics = get_edge_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics, evaluate_y_trues, evaluate_y_predicts


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.DataLoader import Data
from utils.utils import NeighborSampler
from utils.metrics import get_node_classification_metrics

def _get_call_kwargs_for_model(model_name_str: str, num_neighbors: int, time_gap: int) -> dict:
    call_kwargs = {}
    if 'DyGFormer' not in model_name_str and 'TPNet' not in model_name_str:
        call_kwargs['num_neighbors'] = num_neighbors
    if 'DyGFormer' not in model_name_str and 'TPNet' not in model_name_str:
        call_kwargs['time_gap'] = time_gap
        
    return call_kwargs


def evaluate_unsupervised_model_node_classification(model: nn.Module, neighbor_sampler: NeighborSampler, 
                                                    evaluate_idx_data_loader: DataLoader,
                                                    evaluate_data: Data, num_neighbors: int = 20, 
                                                    time_gap: int = 2000):
    if hasattr(model, 'set_neighbor_sampler'):
        model.set_neighbor_sampler(neighbor_sampler)

    model.eval()
    model_name_str = model.__class__.__name__

    with torch.no_grad():
        all_labels, all_anomaly_scores = [], []
        
        for batch_idx, evaluate_data_indices in enumerate(tqdm(evaluate_idx_data_loader, ncols=120, desc="Evaluation")):
            batch_src, batch_dst, batch_ts, batch_eid, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            call_kwargs = _get_call_kwargs_for_model(model_name_str, num_neighbors, time_gap)
            
            anomaly_scores = model.compute_anomaly_score(
                src_node_ids=batch_src, dst_node_ids=batch_dst,
                node_interact_times=batch_ts, edge_ids=batch_eid, **call_kwargs
            )

            all_labels.append(torch.from_numpy(batch_labels).float())
            all_anomaly_scores.append(anomaly_scores.cpu())

        all_labels = torch.cat(all_labels, dim=0)
        all_anomaly_scores = torch.cat(all_anomaly_scores, dim=0)
        
        evaluate_metrics = get_node_classification_metrics(predicts=all_anomaly_scores, labels=all_labels)
        return evaluate_metrics, all_anomaly_scores, all_labels


def extract_embeddings_and_scores(model: nn.Module, neighbor_sampler: NeighborSampler, data_loader: DataLoader, 
                                  data: Data, num_neighbors: int = 20, time_gap: int = 2000):
    model.eval()
    if hasattr(model, 'set_neighbor_sampler'):
        model.set_neighbor_sampler(neighbor_sampler)

    model_name_str = model.__class__.__name__
    all_embeddings, all_predicts, all_labels = [], [], []

    with torch.no_grad():
        for data_indices in tqdm(data_loader, desc="Extracting Data"):
            batch_src, batch_dst, batch_ts, batch_eid, batch_labels = \
                data.src_node_ids[data_indices], data.dst_node_ids[data_indices], \
                data.node_interact_times[data_indices], data.edge_ids[data_indices], \
                data.labels[data_indices]
            
            call_kwargs = _get_call_kwargs_for_model(model_name_str, num_neighbors, time_gap)
            
            if hasattr(model, '_compute_components'):
                components = model._compute_components(
                    src_node_ids=batch_src, dst_node_ids=batch_dst, 
                    node_interact_times=batch_ts, **call_kwargs
                )
                gen_src_m = components[2]
            else:
                raise AttributeError(f"Model {model_name_str} must implement '_compute_components' to extract embeddings.")

            predicts = model.compute_anomaly_score(
                src_node_ids=batch_src, dst_node_ids=batch_dst,
                node_interact_times=batch_ts, edge_ids=batch_eid, **call_kwargs
            )

            all_embeddings.append(gen_src_m)
            all_predicts.append(predicts)
            all_labels.append(batch_labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_predicts = torch.cat(all_predicts, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_predicts, all_labels, all_embeddings
