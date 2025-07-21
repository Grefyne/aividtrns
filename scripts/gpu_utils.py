#!/usr/bin/env python3
"""
GPU utilities for multi-GPU processing and device management.
"""

import os
import torch
import torch.multiprocessing as mp
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    available_gpus = []
    
    for i in range(gpu_count):
        try:
            # Test if GPU is accessible
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            available_gpus.append(i)
        except Exception as e:
            logger.warning(f"GPU {i} not accessible: {e}")
    
    return available_gpus


def get_gpu_memory_info(device_id: int) -> Tuple[int, int]:
    """Get GPU memory info (total, free) in MB."""
    if not torch.cuda.is_available():
        return 0, 0
    
    try:
        torch.cuda.set_device(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated(device_id) / 1024**2
        free_memory = total_memory - allocated_memory
        return int(total_memory), int(free_memory)
    except Exception as e:
        logger.warning(f"Could not get memory info for GPU {device_id}: {e}")
        return 0, 0


def select_best_gpu(min_free_mb: int = 2000) -> Optional[int]:
    """Select the best GPU for single-GPU use: prefer GPU 1 if it has enough free memory, else GPU 0 if it does, else the GPU with the most free memory."""
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.warning("No GPUs available")
        return None
    # Prefer GPU 1 if it has enough free memory
    if 1 in available_gpus:
        total1, free1 = get_gpu_memory_info(1)
        if free1 >= min_free_mb:
            logger.info(f"Preferring GPU 1 for single-GPU use ({free1}MB free)")
            return 1
    # Otherwise, try GPU 0 if it has enough free memory
    if 0 in available_gpus:
        total0, free0 = get_gpu_memory_info(0)
        if free0 >= min_free_mb:
            logger.info(f"Using GPU 0 for single-GPU use ({free0}MB free)")
            return 0
    # Otherwise, pick the GPU with the most free memory
    best_gpu = None
    max_free_memory = 0
    logger.info("Scanning GPUs for best memory availability:")
    for gpu_id in available_gpus:
        total, free = get_gpu_memory_info(gpu_id)
        utilization = ((total - free) / total * 100) if total > 0 else 0
        logger.info(f"  GPU {gpu_id}: {free:,}MB free / {total:,}MB total ({utilization:.1f}% used)")
        if free > max_free_memory:
            max_free_memory = free
            best_gpu = gpu_id
    if best_gpu is not None:
        logger.info(f"Selected GPU {best_gpu} with {max_free_memory:,}MB free memory")
    return best_gpu


def setup_multi_gpu_processing(num_gpus: Optional[int] = None) -> List[int]:
    """Setup multi-GPU processing and return list of available GPUs."""
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        logger.warning("No GPUs available, using CPU")
        return []
    
    if num_gpus is None:
        num_gpus = len(available_gpus)
    else:
        num_gpus = min(num_gpus, len(available_gpus))
    
    selected_gpus = available_gpus[:num_gpus]
    
    logger.info(f"Using {len(selected_gpus)} GPUs: {selected_gpus}")
    
    # Log memory info for each GPU
    for gpu_id in selected_gpus:
        total, free = get_gpu_memory_info(gpu_id)
        logger.info(f"GPU {gpu_id}: {free}MB free / {total}MB total")
    
    return selected_gpus


def get_device(gpu_id: Optional[int] = None, min_free_mb: int = 2000) -> torch.device:
    """Get torch device for specified GPU or best available GPU (prefer GPU 1 if it has enough resources, else GPU 0, else best available)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id is None:
        # Prefer GPU 1 if it has enough free memory, else GPU 0, else best available
        best_gpu = select_best_gpu(min_free_mb=min_free_mb)
        if best_gpu is not None:
            return torch.device(f"cuda:{best_gpu}")
        else:
            return torch.device("cuda:0")
    return torch.device(f"cuda:{gpu_id}")


def set_device(gpu_id: int) -> None:
    """Set the current CUDA device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        logger.info(f"Set CUDA device to GPU {gpu_id}")


def clear_gpu_cache(gpu_id: Optional[int] = None) -> None:
    """Clear GPU cache for specified device or all devices."""
    if not torch.cuda.is_available():
        return
    
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        logger.info(f"Cleared cache for GPU {gpu_id}")
    else:
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Cleared cache for all GPUs")


def parallel_process_with_gpus(func, data_list: List, num_gpus: Optional[int] = None, 
                             chunk_size: Optional[int] = None) -> List:
    """
    Process data in parallel across multiple GPUs.
    
    Args:
        func: Function to apply to each data item
        data_list: List of data items to process
        num_gpus: Number of GPUs to use (default: all available)
        chunk_size: Size of chunks to process per GPU (default: auto)
    
    Returns:
        List of results from processing
    """
    available_gpus = setup_multi_gpu_processing(num_gpus)
    
    if not available_gpus:
        # Fallback to CPU processing
        logger.info("No GPUs available, processing on CPU")
        return [func(item) for item in data_list]
    
    if chunk_size is None:
        chunk_size = max(1, len(data_list) // len(available_gpus))
    
    # Split data into chunks
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    # Process chunks in parallel
    results = []
    with mp.Pool(processes=len(available_gpus)) as pool:
        # Map chunks to GPUs
        chunk_gpu_pairs = [(chunk, available_gpus[i % len(available_gpus)]) 
                          for i, chunk in enumerate(chunks)]
        
        # Process chunks
        chunk_results = pool.starmap(_process_chunk, 
                                   [(func, chunk, gpu_id) for chunk, gpu_id in chunk_gpu_pairs])
        
        # Combine results
        for chunk_result in chunk_results:
            results.extend(chunk_result)
    
    return results


def _process_chunk(func, chunk: List, gpu_id: int) -> List:
    """Process a chunk of data on a specific GPU."""
    try:
        set_device(gpu_id)
        return [func(item) for item in chunk]
    except Exception as e:
        logger.error(f"Error processing chunk on GPU {gpu_id}: {e}")
        return []
    finally:
        clear_gpu_cache(gpu_id)


def get_optimal_batch_size(model_size_mb: float, gpu_id: int = 0) -> int:
    """Calculate optimal batch size based on GPU memory and model size."""
    if not torch.cuda.is_available():
        return 1
    
    try:
        set_device(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
        available_memory = total_memory * 0.8  # Use 80% of GPU memory
        
        # Estimate batch size (rough calculation)
        memory_per_sample = model_size_mb * 2  # Rough estimate
        optimal_batch_size = max(1, int(available_memory / memory_per_sample))
        
        logger.info(f"GPU {gpu_id}: {available_memory:.0f}MB available, "
                   f"optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    except Exception as e:
        logger.warning(f"Could not calculate optimal batch size: {e}")
        return 1


def monitor_gpu_usage(gpu_ids: Optional[List[int]] = None) -> dict:
    """Monitor GPU usage and return statistics."""
    if not torch.cuda.is_available():
        return {"error": "No GPUs available"}
    
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    stats = {}
    for gpu_id in gpu_ids:
        try:
            set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**2
            cached_memory = torch.cuda.memory_reserved(gpu_id) / 1024**2
            
            stats[f"gpu_{gpu_id}"] = {
                "total_memory_mb": total_memory,
                "allocated_memory_mb": allocated_memory,
                "cached_memory_mb": cached_memory,
                "utilization_percent": (allocated_memory / total_memory) * 100
            }
        except Exception as e:
            stats[f"gpu_{gpu_id}"] = {"error": str(e)}
    
    return stats 