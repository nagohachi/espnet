#!/usr/bin/env python3
"""Simple NCCL benchmark - Llama without PEFT first."""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def benchmark(rank, world_size, batch_size):
    setup(rank, world_size)

    if rank == 0:
        print(f"Loading Llama 3.2 3B...", flush=True)

    # Load model WITHOUT PEFT for simplicity
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float16  # Use fp16 to save memory
    )

    # Freeze most parameters, only train lm_head
    for param in model.parameters():
        param.requires_grad = False
    model.lm_head.weight.requires_grad = True

    model = model.to(rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_ddp.parameters()), lr=1e-4)

    seq_len = 512

    if rank == 0:
        print(f"Warmup with batch={batch_size}...", flush=True)

    # Warmup
    for _ in range(2):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        output = model_ddp(input_ids=input_ids, labels=input_ids.clone())
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()

    if rank == 0:
        print(f"Benchmark...", flush=True)

    num_iters = 5
    total_samples = batch_size * world_size * num_iters

    start = time.time()
    for _ in range(num_iters):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        output = model_ddp(input_ids=input_ids, labels=input_ids.clone())
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()
    elapsed = time.time() - start

    if rank == 0:
        ms_per_sample = elapsed / total_samples * 1000
        print(f"=== 2 GPU NCCL Results (batch={batch_size} each) ===", flush=True)
        print(f"Total: {total_samples} samples in {elapsed:.2f}s", flush=True)
        print(f"Per sample: {ms_per_sample:.1f} ms/sample", flush=True)

    cleanup()

if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mp.spawn(benchmark, args=(2, batch_size), nprocs=2, join=True)
