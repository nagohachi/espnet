#!/usr/bin/env python3
"""Benchmark 2GPU NCCL on sacs12 (A100)."""

import os
import sys
import time
import gc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def setup(rank, world_size, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12396'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
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

def test_batch_size_single():
    """Test max batch size on single A100."""
    print("=== Testing max batch size on A100 ===", flush=True)
    seq_len = 512

    for batch_size in [1, 2, 4, 8, 12, 14, 16]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = create_model().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
            labels = input_ids.clone()
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Batch {batch_size}: SUCCESS ({mem:.1f} GB)", flush=True)

            del model, optimizer, input_ids, labels, output
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size}: OOM", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            return batch_size - 2  # Return safe max

    return 16

def benchmark_2gpu_nccl(rank, world_size, batch_size):
    """Benchmark 2 GPU NCCL."""
    setup(rank, world_size, "nccl")

    if rank == 0:
        print(f"=== 2 GPU NCCL, batch_size={batch_size} each ===", flush=True)

    model = create_model().to(rank)
    model_ddp = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=1e-4)

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

    # Benchmark
    num_iters = 5
    total_samples = batch_size * world_size * num_iters

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
        ms_per_sample = elapsed / total_samples * 1000
        print(f"2 GPU NCCL batch={batch_size}: {total_samples} samples in {elapsed:.2f}s", flush=True)
        print(f"2 GPU NCCL batch={batch_size}: {ms_per_sample:.1f} ms/sample", flush=True)

    cleanup()

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_nccl_sacs12.py [test_batch|benchmark BATCH_SIZE]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "test_batch":
        test_batch_size_single()
    elif mode == "benchmark":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        mp.spawn(benchmark_2gpu_nccl, args=(2, batch_size), nprocs=2, join=True)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
