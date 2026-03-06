#!/usr/bin/env python3
"""Benchmark DDP Gloo vs single GPU for LLM training."""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def setup(rank, world_size, backend="gloo"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12399'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def benchmark_single_gpu():
    """Benchmark single GPU training."""
    print("=== Single GPU Benchmark ===", flush=True)

    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (4, 128)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        labels = input_ids.clone()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    num_iters = 20
    start = time.time()
    for i in range(num_iters):
        input_ids = torch.randint(0, 1000, (4, 128)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        labels = input_ids.clone()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Single GPU: {num_iters} iterations in {elapsed:.2f}s", flush=True)
    print(f"Single GPU: {elapsed/num_iters*1000:.1f} ms/iter", flush=True)
    return elapsed / num_iters

def benchmark_ddp_gloo(rank, world_size):
    """Benchmark 2 GPU DDP Gloo training."""
    setup(rank, world_size, "gloo")

    if rank == 0:
        print("=== 2 GPU DDP Gloo Benchmark ===", flush=True)

    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model = model.to(rank)

    model_ddp = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (4, 128)).to(rank)
        attention_mask = torch.ones_like(input_ids).to(rank)
        labels = input_ids.clone()
        output = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()

    # Benchmark
    num_iters = 20
    start = time.time()
    for i in range(num_iters):
        input_ids = torch.randint(0, 1000, (4, 128)).to(rank)
        attention_mask = torch.ones_like(input_ids).to(rank)
        labels = input_ids.clone()
        output = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(rank)
    dist.barrier()
    elapsed = time.time() - start

    if rank == 0:
        print(f"2 GPU Gloo: {num_iters} iterations in {elapsed:.2f}s", flush=True)
        print(f"2 GPU Gloo: {elapsed/num_iters*1000:.1f} ms/iter", flush=True)
        # Effective throughput is 2x because each GPU processes different data
        print(f"2 GPU Gloo effective: {elapsed/num_iters/2*1000:.1f} ms/iter (per sample)", flush=True)

    cleanup()
    return elapsed / num_iters

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Single GPU benchmark
        single_time = benchmark_single_gpu()
        print(f"\n=== Summary ===")
        print(f"Single GPU: {single_time*1000:.1f} ms/iter")
    else:
        # 2 GPU DDP benchmark
        world_size = 2
        mp.spawn(benchmark_ddp_gloo, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
