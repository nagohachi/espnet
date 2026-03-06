#!/usr/bin/env python3
"""Compare 1GPU batch=1 vs 2GPU Gloo batch=16 each."""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12398'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_model():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    return get_peft_model(model, peft_config)

def benchmark_1gpu_batch1():
    """Benchmark 1 GPU with batch_size=1 (current config)."""
    print("=== 1 GPU, batch_size=1 (current config) ===", flush=True)

    model = create_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 1
    seq_len = 512

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Benchmark - process 32 samples total
    num_samples = 32
    num_iters = num_samples // batch_size  # 32 iterations

    start = time.time()
    for _ in range(num_iters):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"1 GPU batch=1: {num_samples} samples in {elapsed:.2f}s", flush=True)
    print(f"1 GPU batch=1: {elapsed/num_samples*1000:.1f} ms/sample", flush=True)
    return elapsed / num_samples

def benchmark_2gpu_batch16(rank, world_size, result_queue):
    """Benchmark 2 GPU Gloo with batch_size=16 each."""
    setup(rank, world_size)

    if rank == 0:
        print("=== 2 GPU Gloo, batch_size=16 each (optimized) ===", flush=True)

    model = create_model().to(rank)
    model_ddp = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=1e-4)

    batch_size = 16  # per GPU
    seq_len = 512

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        labels = input_ids.clone()
        output = model_ddp(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()

    # Benchmark - process 32 samples total (16 per GPU × 1 iteration)
    # To match, we do 1 iteration (each GPU processes 16 samples = 32 total)
    num_iters = 1
    total_samples = batch_size * world_size * num_iters  # 32

    start = time.time()
    for _ in range(num_iters):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        labels = input_ids.clone()
        output = model_ddp(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()
    elapsed = time.time() - start

    if rank == 0:
        print(f"2 GPU batch=16: {total_samples} samples in {elapsed:.2f}s", flush=True)
        print(f"2 GPU batch=16: {elapsed/total_samples*1000:.1f} ms/sample", flush=True)
        result_queue.put(elapsed / total_samples)

    cleanup()

def main():
    # 1 GPU benchmark
    time_1gpu = benchmark_1gpu_batch1()

    print("", flush=True)

    # 2 GPU benchmark
    result_queue = mp.Queue()
    mp.spawn(
        benchmark_2gpu_batch16,
        args=(2, result_queue),
        nprocs=2,
        join=True
    )
    time_2gpu = result_queue.get()

    print("", flush=True)
    print("=" * 50, flush=True)
    print("=== SUMMARY ===", flush=True)
    print(f"1 GPU batch=1:  {time_1gpu*1000:.1f} ms/sample", flush=True)
    print(f"2 GPU batch=16: {time_2gpu*1000:.1f} ms/sample", flush=True)
    print(f"Speedup: {time_1gpu/time_2gpu:.1f}x", flush=True)
    print("=" * 50, flush=True)

if __name__ == "__main__":
    main()
