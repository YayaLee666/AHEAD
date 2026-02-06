import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import time
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F 
import pandas as pd
import networkx as nx

# --- Model Imports ---
# Make sure these model files exist in your `models/` directory
from models.UnsupervisedGraphMixer import UnsupervisedGraphMixer
from models.AHEADGraphMixer import AHEADGraphMixer
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts

# --- Utility Imports ---
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args

# --- Evaluation Function Import ---
from evaluate_models_utils import evaluate_unsupervised_model_node_classification, extract_embeddings_and_scores
from sklearn.metrics import f1_score


def _tensor_max(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        return float(x.detach().max().item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return float(np.max(x))
    return None

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # Get command-line arguments
    args = get_node_classification_args()
    
    # --- Add custom arguments for new models if they are not in the config file ---
    if not hasattr(args, 'anomaly_threshold'):
        args.anomaly_threshold = 0.5 # Default threshold for AHEADGraphMixer
    if not hasattr(args, 'test_interval_epochs'):
        args.test_interval_epochs = 5 # Default interval for periodic testing
    if not hasattr(args, 'pseudo_grad_threshold'):
        args.pseudo_grad_threshold = 1.0 # Default value
    if not hasattr(args, 'save_data_epoch_interval'):
        args.save_data_epoch_interval = 2


    # --- Data Loading ---
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    assert len(train_data.src_node_ids) == len(train_data.labels), \
        f"Length mismatch in train_data! src_node_ids has {len(train_data.src_node_ids)} samples, " \
        f"but labels has {len(train_data.labels)} samples."

    assert len(val_data.src_node_ids) == len(val_data.labels), "Length mismatch in val_data!"
    assert len(test_data.src_node_ids) == len(test_data.labels), "Length mismatch in test_data!"

    num_nodes = node_raw_features.shape[0]

    # --- Neighbor Sampler & Data Loader Initialization ---
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1)
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # --- Main Loop for Multiple Runs ---
    test_metric_all_runs = []
    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f'unsupervised_{args.model_name}_seed{args.seed}'

        # --- Logger Setup ---
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_dir = f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{str(time.time())}.log"))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts for UNSUPERVISED training. **********")
        logger.info(f'Configuration is {args}')
        logger.info(f"dataset_name: {args.dataset_name}")

        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            num_anomalies = np.sum(split_data.labels == 1)
            total_samples = len(split_data.labels)
            anomaly_ratio = num_anomalies / total_samples if total_samples > 0 else 0.0

            logger.info(f"[INFO] Number of anomalies (label=1) in {split_name} set: {num_anomalies}")
            logger.info(f"[INFO] Total samples in {split_name} set: {total_samples}")
            logger.info(f"[INFO] Anomaly ratio in {split_name} set: {anomaly_ratio:.4f}")
            
        # --- Model Creation ---
        logger.info(f"Creating UNSUPERVISED model: {args.model_name}")
        MEMORY_DIM, MESSAGE_DIM = 172, 128

        if args.model_name == 'UnsupervisedGraphMixer':
            model = UnsupervisedGraphMixer(
                node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                num_tokens=args.num_neighbors, memory_dim=MEMORY_DIM, message_dim=MESSAGE_DIM,
                device=args.device, num_layers=args.num_layers, dropout=args.dropout,
                loss_type=args.loss_type,
                temperature=args.temperature # For infonce loss
            )
        elif args.model_name == 'AHEADGraphMixer':
            # Instantiate the model with only the parameters it actually needs
            model = AHEADGraphMixer(
                node_raw_features=node_raw_features, 
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler, 
                time_feat_dim=args.time_feat_dim,
                num_tokens=args.num_neighbors, 
                memory_dim=MEMORY_DIM, 
                message_dim=MESSAGE_DIM,
                device=args.device, 
                num_layers=args.num_layers, 
                dropout=args.dropout,
                loss_type = args.loss_type,
                temperature=args.temperature,
                time_gap=args.time_gap,
                num_hard_negatives=args.num_hard_negatives,
                lcc_loss_weight=args.lcc_loss_weight,
                dataset_name=args.dataset_name,
                run_idx=run,
            )

        else:
            raise ValueError(f"Unsupervised model '{args.model_name}' is not supported in this script.")

        logger.info(f'Model -> {model}')
        logger.info(f'#parameters: {get_parameter_sizes(model) * 4 / 1024 / 1024:.2f} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = convert_to_gpu(model, device=args.device)

        # --- Early Stopping Setup ---
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # --- Initialize states ONLY ONCE before the entire training process ---
        logger.info("Initializing model states for the run...")
        if hasattr(model, 'memory'):
            model.memory.reset_memory()

        # --- Variables to track the best test performance ---
        best_test_metrics = None
        best_test_epoch = 0
        best_test_predicts, best_test_labels = None, None

        epoch_loss_gaps = []
        epoch_val_aurocs = []

        # --- Training Loop ---
        for epoch in range(args.num_epochs):

            # ============================================================
            # EPOCH Stage 1: Indicator / Gradient Collection
            # ============================================================
            logger.info(f"Epoch {epoch + 1}, Stage 1: Collecting indicators...")
            model.eval()
            model.reset_states()
            model.set_neighbor_sampler(train_neighbor_sampler)

                # --- [CRITICAL] Always start Stage 1 from a CLEAN memory state ---
                # reset model.memory (the real source of time rollback)
            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            logger.info("Warming up memory before indicator collection...")
            for train_data_indices in tqdm(train_idx_data_loader, ncols=120, desc="Memory Warm-up"):
                    # IMPORTANT: if your loader returns torch.Tensor indices
                idx = train_data_indices.numpy().copy()
                batch_src = train_data.src_node_ids[idx]
                batch_dst = train_data.dst_node_ids[idx]
                batch_ts  = train_data.node_interact_times[idx]
                batch_eid = train_data.edge_ids[idx]

                with torch.no_grad():
                    call_kwargs = {}
                    call_kwargs["num_neighbors"] = args.num_neighbors
                    call_kwargs["time_gap"] = args.time_gap
                    model.compute_anomaly_score(batch_src, batch_dst, batch_ts, batch_eid, **call_kwargs)


            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            epoch_normal_grad_norms, epoch_abnormal_grad_norms = [], []
            indicator_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, desc=f"Epoch {epoch + 1} [Indicator Collection]")

            for batch_idx, train_data_indices in enumerate(indicator_loader_tqdm):
                idx = train_data_indices.numpy().copy()

                batch_src = train_data.src_node_ids[idx]
                batch_dst = train_data.dst_node_ids[idx]
                batch_ts  = train_data.node_interact_times[idx]
                batch_eid = train_data.edge_ids[idx]
                batch_labels = train_data.labels[idx]

                call_kwargs = {}
                call_kwargs['num_neighbors'] = args.num_neighbors
                call_kwargs['time_gap'] = args.time_gap
                grad_norms, residuals = model.compute_and_cache_indicators(
                    batch_src, batch_dst, batch_ts, batch_eid,
                    **call_kwargs
                )

                is_abnormal_mask = (batch_labels == 1)
                if np.any(~is_abnormal_mask):
                    epoch_normal_grad_norms.extend(grad_norms[~is_abnormal_mask])
                if np.any(is_abnormal_mask):
                     epoch_abnormal_grad_norms.extend(grad_norms[is_abnormal_mask])

            model.indicator_cache.update_global_cache()

            # ============================================================
            # EPOCH Stage 2: Training (must ALSO start from CLEAN state)
            # ============================================================
                
            logger.info(f"Epoch {epoch + 1}, Stage 2: Training model...")
            model.train()
            model.reset_states()
            model.set_neighbor_sampler(train_neighbor_sampler)

            # --- [CRITICAL] Reset memory before training ---
            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            save_logs_this_epoch = (epoch + 1) % 10 == 0
            train_total_loss = 0.0

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            torch.autograd.set_detect_anomaly(True)

            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, desc=f"Epoch {epoch + 1} [Training]")
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):

                idx = train_data_indices.numpy().copy()

                batch_src_node_ids = train_data.src_node_ids[idx]
                batch_dst_node_ids = train_data.dst_node_ids[idx]
                batch_node_interact_times = train_data.node_interact_times[idx]
                batch_edge_ids = train_data.edge_ids[idx]

                optimizer.zero_grad()

                loss_kwargs = dict(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    epoch=epoch + 1,
                    save_logs_this_epoch=save_logs_this_epoch,
                )


                loss_kwargs["num_neighbors"] = args.num_neighbors
                loss_kwargs["time_gap"] = args.time_gap

                out = model.compute_unsupervised_loss(**loss_kwargs)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    loss, mem_payload = out
                else:
                    loss, mem_payload = out, None

                loss.backward()
                optimizer.step()


                if mem_payload is not None:
                    with torch.no_grad():
                        src_nodes, dst_nodes, times, upd_src_m, upd_dst_m = mem_payload
                        model.memory.update_state(src_nodes, upd_src_m.detach(), times)
                        model.memory.update_state(dst_nodes, upd_dst_m.detach(), times)

                train_total_loss += loss.item()
                train_idx_data_loader_tqdm.set_description(f"Epoch: {epoch + 1} [Training], train loss: {loss.item():.4f}")

            train_avg_loss = train_total_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0


            # --- Validation Phase ---
            # state_after_train = copy.deepcopy(model.state_dict())
            state_after_train = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            backup = None 
            logger.info(f"--- Performing validation for epoch {epoch + 1} ---")
            eval_kwargs = {}
            val_metrics, val_predicts, val_labels = evaluate_unsupervised_model_node_classification(model, full_neighbor_sampler, val_idx_data_loader, val_data, **eval_kwargs)

            model.load_state_dict(state_after_train)

            logger.info(f'Epoch: {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_avg_loss:.4f}')
            for metric_name, metric_val in val_metrics.items():
                logger.info(f'Validate {metric_name}: {metric_val:.4f}')


            if (epoch + 1) % args.test_interval_epochs == 0:
                logger.info(f"--- Performing periodic test at epoch {epoch + 1} ---")
                model.load_state_dict(state_after_train, strict=True)
                model = model.to(args.device)
                test_eval_kwargs = {}
                test_metrics, test_predicts, test_labels = evaluate_unsupervised_model_node_classification(
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    **test_eval_kwargs
                )
                model.load_state_dict(state_after_train, strict=True)
                model = model.to(args.device)

                logger.info(f"Saving detailed prediction results for epoch {epoch + 1} ...")

                pred_scores = test_predicts.detach().cpu().numpy().flatten()
                true_labels = test_labels.detach().cpu().numpy().astype(int)

                try:
                    src_ids = test_data.src_node_ids[:len(pred_scores)]
                except Exception:
                    src_ids = np.arange(len(pred_scores))

                df_pred = pd.DataFrame({
                    "src_id": src_ids,
                    "score": pred_scores,
                    "true_label": true_labels
                })

                logger.info(f"--- Test performance at epoch {epoch + 1} ---")
                for metric_name, metric_val in test_metrics.items():
                    logger.info(f'Test {metric_name}: {metric_val:.4f}')

                if best_test_metrics is None or test_metrics['AUROC'] > best_test_metrics['AUROC']:
                    best_test_metrics = test_metrics
                    best_test_epoch = epoch + 1
                    best_test_predicts = test_predicts.cpu()
                    best_test_labels = test_labels.cpu()
                    logger.info(f"*** New best test performance found at epoch {best_test_epoch}! AUROC: {best_test_metrics['AUROC']:.4f} ***")
                    early_stopping.save_checkpoint(model, is_best_test=True)


            # --- Early Stopping Check ---
            val_metric_indicator = [('AUROC', val_metrics['AUROC'], True)]
            early_stop = early_stopping.step(val_metric_indicator, model)

        logger.info("--- Training finished for this run ---")
        best_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pth")

        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path} for final testing and analysis...")
            model.load_state_dict(torch.load(best_model_path))
        else:
            logger.warning(f"Could not find best model checkpoint at {best_model_path}. "
                           "Final testing and analysis will use the model from the last epoch.")

        if best_test_metrics is not None:
            logger.info(f"The best test performance was at epoch {best_test_epoch}.")
            logger.info("--- Best Test Performance Reported ---")
            for metric_name, metric_val in best_test_metrics.items():
                logger.info(f'Best Test {metric_name}: {metric_val:.4f}')
            test_metric_all_runs.append(best_test_metrics)
        else:
            logger.warning("No periodic testing was performed. Reporting performance of the best validation model on the test set.")
            early_stopping.load_checkpoint(model)
            if hasattr(model, 'memory'):
                logger.info("Rebuilding memory state before final test...")
                model.memory.reset_memory()
                _ = evaluate_unsupervised_model_node_classification(
                    model=model, neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=train_idx_data_loader, evaluate_data=train_data,
                    num_neighbors=args.num_neighbors, time_gap=args.time_gap)
                _ = evaluate_unsupervised_model_node_classification(
                    model=model, neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader, evaluate_data=val_data,
                    num_neighbors=args.num_neighbors, time_gap=args.time_gap)
            
            final_test_metrics, final_test_predicts, final_test_labels = evaluate_unsupervised_model_node_classification(
                model=model, neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader, evaluate_data=test_data,
                num_neighbors=args.num_neighbors, time_gap=args.time_gap)
            
            logger.info("--- Final Test Performance (from best validation model) ---")
            for metric_name, metric_val in final_test_metrics.items():
                logger.info(f'Final Test {metric_name}: {metric_val:.4f}')
            test_metric_all_runs.append(final_test_metrics)
            best_test_predicts = final_test_predicts.cpu()
            best_test_labels = final_test_labels.cpu()


        run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {run_time:.2f} seconds.')
        
        # --- Saving results and scores for the best test performance ---
        if best_test_metrics is not None:
            result_json = {metric_name: f'{metric_val:.4f}' for metric_name, metric_val in best_test_metrics.items()}
            result_json = json.dumps(result_json, indent=4)
            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            with open(os.path.join(save_result_folder, f"{args.save_model_name}.json"), 'w') as f:
                f.write(result_json)
            
            save_score_dir = f"./saved_scores/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_score_dir, exist_ok=True)
            if best_test_predicts is not None and best_test_labels is not None:
                np.save(os.path.join(save_score_dir, f"test_predicts_{args.save_model_name}.npy"), best_test_predicts.numpy())
                np.save(os.path.join(save_score_dir, f"test_labels_{args.save_model_name}.npy"), best_test_labels.numpy())
                logger.info(f"Scores from the best test epoch saved for run {run + 1}.")
            else:
                 logger.warning(f"No test scores to save for run {run + 1}.")
        
    # --- Final average metrics calculation across all runs ---
    if args.num_runs > 1 and test_metric_all_runs:
        logger.info(f'===== Summary of Best Test Metrics over {args.num_runs} runs =====')
        if test_metric_all_runs[0]:
            for metric_name in test_metric_all_runs[0].keys():
                metric_values = [run_metric[metric_name] for run_metric in test_metric_all_runs]
                avg_metric = np.mean(metric_values)
                std_metric = np.std(metric_values)
                logger.info(f'Average Best Test {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}')

    sys.exit()

