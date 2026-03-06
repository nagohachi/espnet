#!/usr/bin/env python3
"""Quick NCCL benchmark on sacs12."""

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
    os.environ['MASTER_PORT'] = '12388'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def benchmark(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        print("Loading model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config).to(rank)
    model_ddp = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=1e-4)

    batch_size = 12
    seq_len = 512

    if rank == 0:
        print("Warmup...", flush=True)

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        output = model_ddp(input_ids=input_ids, labels=input_ids.clone())
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()

    if rank == 0:
        print("Benchmark...", flush=True)

    # Benchmark - more iterations
    num_iters = 10
    total_samples = batch_size * world_size * num_iters

    torch.cuda.synchronize(rank)
    dist.barrier()
    start = time.time()

    for i in range(num_iters):
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
        ms_per_iter = elapsed / num_iters * 1000
        print(f"=== Results ===", flush=True)
        print(f"Total: {total_samples} samples in {elapsed:.2f}s", flush=True)
        print(f"Per iter: {ms_per_iter:.1f} ms (batch={batch_size}x2={batch_size*2})", flush=True)
        print(f"Per sample: {ms_per_sample:.1f} ms/sample", flush=True)

    cleanup()

if __name__ == "__main__":
    mp.spawn(benchmark, args=(2,), nprocs=2, join=True)
