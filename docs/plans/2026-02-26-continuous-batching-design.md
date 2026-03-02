# Continuous Batching Demo 设计文档

## 目标

实现 continuous batching（iteration-level scheduling），复现 Orca 论文的核心调度机制。
复用已有的 `TransformerLM`、`KVCacheManager`、`paged_attention_decode`，新增 Scheduler 和 Engine 层。
目的是理解 vLLM serving 架构中调度器与 KV cache 管理的协作方式。

## 参数

- Prefill/Decode 分离执行（不做 chunked prefill）
- 不做 preemption（内存不足时拒绝新请求，等待释放）
- FCFS 调度策略
- 所有层共享一个 `BlockAllocator`，每层独立 `KVCache`
- 支持多请求动态到达（模拟不同 step 到达的请求）
- 验证方式：逐 step 打印 batch 组成，对比 static batching 的吞吐差异

## 架构概览

```
Scheduler (三队列 FCFS 调度)
    ↕ add_request / schedule
Engine (主循环，协调 prefill/decode)
    ↕ 拆层执行，每层: write_kv → attention (prefill: 标准 / decode: paged)
KVCacheManager (多层，共享 BlockAllocator)
    ↕ 统一写入路径 write_kv（≈ vLLM reshape_and_cache）
```

## Static Batching vs Continuous Batching

```
Static:      等最长序列完成才释放 slot，短序列 pad 空转
Continuous:  每个 iteration 后立即踢掉已完成序列，填入新请求
```

核心区别：调度粒度从 request 级降到 iteration 级（Orca, OSDI 2022）。

## vLLM 中的对应关系

| 组件 | vLLM 实现 | 本 demo |
|---|---|---|
| Sequence / SequenceGroup | Python dataclass，含 status 状态机 | `Sequence` dataclass（不实现 SequenceGroup） |
| Scheduler（三队列） | `waiting` / `running` / `swapped` + `SchedulingBudget` | `waiting` / `running` / `finished`（无 swapped） |
| Preemption | Swap to CPU / Recompute | 不实现，内存不足时拒绝接纳 |
| Chunked prefill | 长 prompt 分 chunk，与 decode 混合 batch | 不实现，prefill/decode 分离执行 |
| Block manager | 共享 allocator，统一 block table | 共享 `BlockAllocator`，每层独立 `KVCache` |
| Block 预分配 | Scheduler 调 `allocate_slots` → Worker 构建 `block_tables` / `slot_mapping` → Forward 只写入 | `allocate_slots` → `build_block_tables` → layer 循环内 `write_kv` 只写入 |
| Prefill attention | `reshape_and_cache` + FlashAttention | `write_kv` + 标准 attention（直接用刚算出的 K/V） |
| Decode attention | `reshape_and_cache` + PagedAttention | `write_kv` + `paged_attention_decode`（从 cache 读） |
| Worker / Executor | 多 GPU 分布式执行 | 单进程，直接调用模型 |

## 组件设计

### 1. Sequence

```python
class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

@dataclass
class Sequence:
    seq_id: int
    status: SequenceStatus
    prompt_tokens: list[int]
    output_tokens: list[int]       # 逐步追加
    max_new_tokens: int
```

不实现 `SequenceGroup`（parallel sampling 场景），一个 request = 一个 Sequence。

### 2. Scheduler

**数据结构：**
- `waiting: deque[Sequence]` — FIFO 等待队列
- `running: list[Sequence]` — 正在生成的序列
- `finished: list[Sequence]` — 已完成的序列
- `max_concurrent: int` — 最大并发序列数
- `block_size: int` — 用于估算新请求所需 blocks
- `free_block_num_fn: Callable[[], int]` — 查询当前可用 block 数

**接口：**
- `add_request(seq: Sequence)` — 加入 waiting 队列
- `schedule() -> SchedulerOutput` — 每个 iteration 调用一次
- `has_unfinished() -> bool` — waiting 或 running 非空

**SchedulerOutput：**
```python
@dataclass
class SchedulerOutput:
    prefill_seq: list[Sequence]    # 本轮需要 prefill 的序列
    decode_seq: list[Sequence]     # 本轮需要 decode 的序列
    freed_seq_ids: list[int]       # 本轮完成的序列 ID
```

**schedule() 算法：**
```
1. 从 running 中移除 status == FINISHED 的序列 → freed_seq_ids
2. 从 waiting 中取请求填入空位：
   - 条件：len(running) < max_concurrent AND num_free_blocks >= ceil(prompt_len / block_size)
   - 不满足则停止接纳（不做 preemption）
3. 已在 running 且已完成 prefill（len(output_tokens) > 0）的序列 → decode_seq
```

> **未实现：Preemption**
> vLLM 在内存不足时 preempt 最低优先级的 running 序列（Swap to CPU 或 Recompute）。
> 本 demo 选择等待释放，高负载下 waiting 队列会积压。

> **未实现：Priority scheduling**
> vLLM 支持基于优先级的调度。本 demo 仅 FCFS。

> **未实现：Token budget**
> vLLM 用 `max_num_batched_tokens` 控制每轮处理的总 token 数，平衡 prefill/decode。
> 本 demo 无此限制。

### 3. KVCacheManager（扩展为多层）

不新建类，直接扩展现有 `KVCacheManager`，新增 `n_layers` 参数（默认 1）。
block 管理逻辑（register、free、build_block_tables、fork）与层数无关。

**已完成的改动：**
- `__init__` 新增 `n_layers=1`，`self.kv_cache` → `self.kv_caches: list[KVCache]`（已实现）

**已删除 `append_token` / `append_token_all_layers`，统一为 `write_kv` + `advance_tokens`。**

原有的 `append_token` 系列将"写入"和"推进位置计数器"绑定在一起，按 token 遍历所有层。
拆层执行时（for layer → for token），同一个 token 的不同层会多次触发计数器推进，导致位置错误。
因此将写入和位置管理解耦：`write_kv` 纯写入，`advance_tokens` 显式推进计数器。

**已有方法不变：**
- `register_sequence`、`build_block_tables`、`free_sequence`、`fork_sequence` — 不变

**改动方法：**
- `allocate_slots(seq_id: int, new_num_tokens: int)`
  - 为即将写入的新 token 预分配 block，不写入任何数据，不修改 `seq_to_num_tokens`
  - 对齐 vLLM 的 `KVCacheManager.allocate_slots`：分配阶段和写入阶段完全分离
  - prefill 和 decode 都必须先调 `allocate_slots`，再调 `write_kv`
  - 实现：遍历 `seq_to_num_tokens + i`（i = 0..num_new_tokens-1），`slot == 0` 时分配新 block，`slot > 0` 时检查 CoW
- `write_kv(seq_id: int, layer_idx: int, k: Tensor, v: Tensor)`
  - 纯写入接口，≈ vLLM 的 `reshape_and_cache`，零分配逻辑
  - k/v shape: `(num_tokens, num_heads, head_dim)` 或 `(num_heads, head_dim)`（单 token 时自动 unsqueeze）
  - 内部用 `seq_to_num_tokens + t` 算出 `(block_idx, slot)`，直接从 `seq_to_block` 查找 block ID 并写入
  - **不修改 `seq_to_num_tokens`**，block 必须已由 `allocate_slots` 分配
  - prefill 和 decode 调用同一个方法，区别只是 num_tokens 不同
- `advance_tokens(seq_id: int, num_tokens: int)`
  - 推进 `seq_to_num_tokens`，在所有层的 `write_kv` 完成后由调用方显式调用
  - prefill 后调 `advance_tokens(seq_id, prompt_len)`，decode 每步调 `advance_tokens(seq_id, 1)`

**已删除 `_get_or_allocate_block`。** 原方法将"定位写入位置"和"分配 block"耦合，导致：
1. `write_kv` 多 token 写入时 `seq_to_num_tokens` 不变，每个 token 都算出相同的 slot（prefill bug）
2. 与 `allocate_slots` 预分配冲突，`slot == 0` 时重复分配
拆分为 `allocate_slots`（纯分配）+ `write_kv` 内部纯位置计算，彻底消除耦合。

> **设计说明：写入与位置管理解耦**
> `write_kv` 用当前 `seq_to_num_tokens` 计算写入位置，但不修改它。
> 这样无论按什么顺序遍历层，同一个 token 在所有层都写到相同的 (block, slot)。
> 所有层写完后，调用方调 `advance_tokens` 推进计数器，为下一轮写入准备正确的位置。
> paged attention demo 也统一使用 `write_kv` + `advance_tokens`，不再保留旧接口。

> **设计说明：三阶段分离（对齐 vLLM）**
> 分配、写入、位置推进完全分离为三个独立操作：
> - `allocate_slots`：为新 token 预分配 block + 处理 CoW（不写数据，不动计数器）
> - `write_kv`：纯写入，用 `seq_to_num_tokens + t` 定位 `(block_idx, slot)`，从 `seq_to_block` 查 block ID
> - `advance_tokens`：推进计数器
>
> Prefill 和 decode 都遵循同一时序：`allocate_slots` → `build_block_tables`（decode 需要）→ layer 循环内 `write_kv` → `advance_tokens`。
> 这完全对齐 vLLM 的 `allocate_slots` → `slot_mapping` → `reshape_and_cache` 流程。

### 4. ContinuousBatchingEngine

**初始化参数：**
```python
class ContinuousBatchingEngine:
    model: TransformerLM
    kv_manager: KVCacheManager
    scheduler: Scheduler
    eos_token_id: int
```

**主循环：**
```
def run(requests: list[tuple[int, Sequence]]) -> list[Sequence]:
    # requests: (arrival_step, sequence) 列表
    step = 0
    while scheduler.has_unfinished() or 还有未到达的请求:
        # 注入本 step 到达的请求
        for (arrival, seq) in requests where arrival == step:
            scheduler.add_request(seq)

        output = scheduler.schedule()

        # 释放已完成序列的 KV cache
        for seq_id in output.freed_seq_ids:
            kv_manager.free_sequence(seq_id)

        # Prefill（逐个序列，模型侧封装）
        for seq in output.prefill_seq:
            _prefill(seq)

        # Decode（batch 化，模型侧封装）
        if output.decode_seq:
            _decode_batch(output.decode_seq)

        step += 1
    return scheduler.finished
```

**模型侧封装（在现有 forward 上分支改造）：**

不新增同名方法逐层传递，而是在现有 `forward` 上通过可选参数分支。
真正需要分支的只有 `MultiheadSelfAttention`（attention 计算不同），`TransformerBlock` 只透传参数。
`TransformerLM` 新增方法构造上下文、调用现有 block.forward。

职责划分：
- `MultiheadSelfAttention.forward`：共享 projection + reshape + RoPE，按 `paged_ctx` 分支 attention 计算
- `TransformerBlock.forward`：不感知 paged cache，只透传 `paged_ctx` 给 self.attn
- `TransformerLM`：新增 `prefill_with_paged_cache` / `decode_with_paged_cache`，构造上下文并调用 block.forward

> **与 vLLM 的对齐：**
> vLLM 每层执行: `reshape_and_cache(k, v, cache, slot_mapping)` → attention kernel。
> 本 demo 每层执行: `kv_manager.write_kv(seq_id, layer_idx, k, v)` → attention。
> 写入和计算分离，写入路径 prefill/decode 统一，计算路径不同（标准 vs paged）。

**PagedCacheContext：**
```python
@dataclass
class PagedCacheContext:
    kv_manager: KVCacheManager
    layer_idx: int
    mode: Literal["prefill", "decode"]
    # prefill
    prefill_seq_id: int | None = None
    # decode
    decode_seq_ids: list[int] | None = None
    block_tables: Tensor | None = None
    seq_lens: torch.Tensor | None = None
```

**MultiheadSelfAttention.forward 改造（三条分支）：**

> **Layout 转换：** attention 计算用 `(batch, num_heads, seq_len, head_dim)`（heads-major，PyTorch 标准），
> KV cache 存储用 `(num_tokens, num_heads, head_dim)`（tokens-major，vLLM `reshape_and_cache` 标准）。
> 两种都是各自领域的通用约定，transpose 不可避免。转换在 attention 内部完成（`write_kv` 调用前 rearrange），
> Block 和 LM 不感知 cache 的 layout。

```
def forward(self, x, token_positions, *, use_cache=False, paged_ctx=None):
    q, k, v = project_and_reshape(x)       # 共享，输出 (batch, num_heads, seq_len, head_dim)
    q, k = self.rope(q, k, token_positions) # 共享

    if paged_ctx is not None:
        ctx = paged_ctx
        kv_manager, layer_idx = ctx.kv_manager, ctx.layer_idx

        if ctx.mode == "prefill":
            # 标准 causal attention（直接用刚算出的 k, v，不从 cache 读）
            attn_out = scaled_dot_product_attention(q, k, v, is_causal=True)
            # layout 转换: (1, num_heads, seq_len, head_dim) → (seq_len, num_heads, head_dim)
            k_cache = rearrange(k[0], "heads seq d_k -> seq heads d_k")
            v_cache = rearrange(v[0], "heads seq d_k -> seq heads d_k")
            kv_manager.write_kv(ctx.prefill_seq_id, layer_idx, k_cache, v_cache)

        else:  # decode
            # layout 转换: 逐序列 (num_heads, 1, head_dim) → (num_heads, head_dim)
            for i, sid in enumerate(ctx.decode_seq_ids):
                kv_manager.write_kv(sid, layer_idx, k[i, :, 0, :], v[i, :, 0, :])
            # Paged attention（从 cache 读取所有历史 K/V，包括刚写入的新 token）
            # block_tables 和 seq_lens 由 decode_with_paged_cache 预先构建，
            # 因为 allocate_slots 已在 layer 循环前预分配了 block，block_tables 全程有效
            attn_out = paged_attention_decode(
                q, kv_manager.kv_caches[layer_idx].key_cache,
                kv_manager.kv_caches[layer_idx].value_cache,
                ctx.block_tables, ctx.seq_lens,
            )

    elif use_cache:
        # 现有朴素 concat KV cache（单序列，无调度）
        ...（不变）

    else:
        # 标准路径（训练 / 无 cache 推理）
        ...（不变）

    return self.output_proj(attn_out)
```

**TransformerBlock.forward — 仅透传 paged_ctx：**
```
def forward(self, x, token_positions=None, *, use_cache=False, paged_ctx=None):
    # pre-norm（现有逻辑不变，只多传一个参数）
    x = x + self.attn(self.ln1(x), token_positions, use_cache=use_cache, paged_ctx=paged_ctx)
    x = x + self.ffn(self.ln2(x))
    return x
```

**TransformerLM 新增方法：**

`prefill_with_paged_cache(prompt_tokens, kv_manager, seq_id)`:
```
1. x = self.token_embeddings(prompt_tokens)  # (1, prompt_len, d_model)
2. positions = torch.arange(prompt_len)
3. kv_manager.allocate_slots(seq_id, prompt_len)  # 预分配所有 prompt token 的 block
4. for layer_idx, block in enumerate(self.layers):
       ctx = PagedCacheContext(kv_manager, layer_idx, mode="prefill", prefill_seq_id=seq_id)
       x = block(x, positions, paged_ctx=ctx)
5. kv_manager.advance_tokens(seq_id, prompt_len)
6. logits = self.lm_head(self.ln_final(x))  # (1, prompt_len, vocab)
7. return logits
```

`decode_with_paged_cache(input_ids, kv_manager, seq_ids, token_positions)`:
```
1. x = self.token_embeddings(input_ids)  # (batch, 1, d_model)
2. # 预分配 block（对齐 vLLM：Scheduler 阶段分配，Forward 阶段只写入）
   for sid in seq_ids:
       kv_manager.allocate_slots(sid, new_num_tokens=1)
3. # 构建 block_tables（预分配后已包含新 token 的 block，layer 循环内不会变）
   block_tables, seq_lens = kv_manager.build_block_tables(seq_ids)
   # seq_lens 需要 +1：advance_tokens 尚未调用，seq_to_num_tokens 还是旧值
   seq_lens = seq_lens + 1
4. for layer_idx, block in enumerate(self.layers):
       ctx = PagedCacheContext(kv_manager, layer_idx, mode="decode",
                               decode_seq_ids=seq_ids, block_tables=block_tables, seq_lens=seq_lens)
       x = block(x, token_positions, paged_ctx=ctx)
5. for sid in seq_ids:
       kv_manager.advance_tokens(sid, 1)
6. logits = self.lm_head(self.ln_final(x))  # (batch, 1, vocab)
7. return logits
```

**Engine 的 _prefill / _decode_batch 简化为：**
```
def _prefill(self, seq):
    kv_manager.register_sequence(seq.seq_id)
    logits = self.model.prefill_with_paged_cache(
        seq.prompt_tokens, self.kv_manager, seq.seq_id
    )
    next_token = sample(logits)
    seq.output_tokens.append(next_token)
    check_finished(seq)

def _decode_batch(self, seqs):
    input_ids, positions, seq_ids = prepare_decode_inputs(seqs)
    logits = self.model.decode_with_paged_cache(
        input_ids, self.kv_manager, seq_ids, positions
    )
    for i, seq in enumerate(seqs):
        next_token = sample(logits[i])
        seq.output_tokens.append(next_token)
        check_finished(seq)
```

> **未实现：Chunked Prefill**
> 本 demo prefill/decode 分离执行。vLLM V1 将长 prompt 切 chunk 与 decode 混合 batch，
> 好处：decode 延迟稳定（不被长 prefill 阻塞），compute-bound prefill 与 memory-bound decode 互补。

> **未实现：Selective Batching（Orca）**
> Orca 论文中，非 attention 操作将所有序列 token 拼接成大 tensor batch 计算，
> attention 操作因 KV 长度不同而逐序列执行。本 demo 的 decode batch 已通过
> `paged_attention_decode` 实现了 attention 的 batch 化。

## 设计决策

### 两条 KV cache 路径：朴素 vs Paged

`MultiheadSelfAttention.forward` 中 `use_cache` 和 `paged_ctx` 是同一个问题（避免重复计算 K/V）的两种解法，互斥，不会同时启用。

```
朴素 KV cache (use_cache)              → 避免重复计算 K/V
  + Paged allocation                   → 避免连续内存浪费，按需分块
    + 多序列隔离 (seq_id)              → 同时服务多个请求
      + Scheduler (continuous batching) → iteration 级调度，动态进出
```

| | `use_cache`（朴素） | `paged_ctx`（本 demo 新增） |
|---|---|---|
| 解决的问题 | 避免重复计算 | 避免重复计算 + 内存管理 + 多序列 + 调度 |
| 存储方式 | 每层自持 tensor，`torch.cat` 拼接 | 外部 `KVCacheManager`，按 block 分配 |
| 内存增长 | 连续内存，每步 realloc | 按需分配固定大小 block |
| 多序列 | 不支持，cache 绑定在模型实例 | 按 `seq_id` 隔离，支持并发 |
| 释放 | `clear_kv_cache()` 全清 | `free_sequence(seq_id)` 按序列释放 |
| 调度 | 无，单序列串行 | 配合 Scheduler 做 continuous batching |
| 适用场景 | 单序列推理、教学演示 | serving、多请求并发 |

保留 `use_cache` 分支：它是最简单的 KV cache 实现，用于单序列推理和已有测试。
paged 分支在其基础上叠加了内存管理、多序列隔离和调度能力，是 serving 场景的完整方案。

### Prefill/Decode 统一写入路径

vLLM 中 prefill 和 decode 的 KV cache 写入使用同一个 CUDA kernel（`reshape_and_cache`），区别只是 token 数不同。
本 demo 对齐这一设计：`write_kv` 同时服务 prefill（多 token）和 decode（单 token），写入逻辑完全一致。

写入和计算分离：
```
每层: write_kv(k, v → paged cache)  →  attention(读取方式不同)
       ↑ 统一                           ↑ prefill: 标准 attention（用刚算出的 k, v）
                                         ↑ decode:  paged attention（从 cache 读）
```

Prefill 时标准 attention 直接用刚投影出的 k, v（连续内存，更高效），不从 paged cache 回读。
写入 paged cache 是 side effect，为后续 decode 步骤准备数据。

### Block 分配与 block_tables 构建的时序问题

**问题发现：** 原始设计中 `write_kv` 内部调用 `_get_or_allocate_block`，将"分配 block"和"写入数据"耦合在一起。
这导致了一个时序问题：`build_block_tables` 必须在 `write_kv` 之后调用，否则可能拿到过时的 block_tables。

具体场景（decode 写入 1 个新 token）：
```
seq 0 当前有 8 个 token，block_size = 4
→ 已占满 block [3, 7]（每个 block 4 个 token）
→ 新 token 需要分配第 3 个 block

如果在 write_kv 之前构建 block_tables：
  block_tables = [[3, 7]]          ← 只有 2 个 block
  write_kv → _get_or_allocate_block → 分配 block 12
  paged_attention_decode(block_tables=[[3, 7]])  ← 缺了 block 12，读不到新 token！

如果在 write_kv 之后构建 block_tables：
  write_kv → 分配 block 12
  block_tables = [[3, 7, 12]]      ← 正确，包含新 block
  paged_attention_decode(block_tables=[[3, 7, 12]])  ← 能读到所有 token
```

多层场景下问题更微妙：block 分配只在 layer 0 的 `write_kv` 时发生（因为所有层共享 block table，
同一个 token 在所有层写到相同的 (block, slot)）。如果在 layer 循环开头构建 block_tables，
layer 0 拿到的是过时的，layer 1+ 拿到的是正确的（因为 layer 0 已经分配过了）。

**vLLM 的解法：分配和写入完全分离。**

查阅 vLLM V1 源码（`vllm/v1/core/kv_cache_manager.py`），其时序为：
```
Scheduler.schedule()
  └─ kv_cache_manager.allocate_slots(request, num_new_tokens)
     └─ 为新 token 预分配 block（不写入任何 KV 数据）

Worker.execute_model()
  ├─ block_table.append_row(new_block_ids)   ← 更新 block table
  ├─ 构建 slot_mapping                       ← 此时 block table 已完整
  └─ Model Forward（逐层）:
       ├─ reshape_and_cache(k, v, slot_mapping)  ← 按预算好的位置写入
       └─ attention(q, block_tables)              ← 按完整的 block_tables 读取
```

block 分配发生在 Scheduler 阶段，Forward 阶段的 `reshape_and_cache` 只负责写入数据到已分配的 block。
因此 `block_tables` 在 Forward 开始前就已经是完整的，不存在时序问题。

**本 demo 的对齐方案：** `allocate_slots` 统一处理 prefill 和 decode，`write_kv` 变为纯写入。

Prefill 时序：
```
allocate_slots(seq_id, prompt_len)  ← 预分配所有 prompt token 的 block
for layer in layers:
    write_kv(...)                   ← 纯写入，用 num_tokens + t 定位 (block_idx, slot)
advance_tokens(seq_id, prompt_len)
```

Decode 时序：
```
allocate_slots(seq_id, 1)           ← 预分配 1 个 token 的 block
build_block_tables(seq_ids)         ← 此时 block_tables 已包含新 block
for layer in layers:
    write_kv(...)                   ← 纯写入
    paged_attention_decode(block_tables)  ← block_tables 全程有效
advance_tokens(seq_id, 1)
```

Prefill 不需要 `build_block_tables`，因为 prefill attention 直接用刚算出的 k, v，不从 cache 读。

**`write_kv` vs vLLM `reshape_and_cache`：现已对齐为纯写入**

vLLM 的 `reshape_and_cache` 是一个纯写入的 CUDA kernel：
```python
# vLLM: reshape_and_cache 签名
reshape_and_cache(
    key, value,        # 刚算出的 K/V
    key_cache,         # 物理 cache pool
    value_cache,       # 物理 cache pool
    slot_mapping,      # 每个 token 写到哪个 slot（扁平整数索引，由外部预先算好）
)
# 实现：cache[slot_mapping[i]] = k[i]，纯写入，零分配逻辑
```

本 demo 的 `write_kv` 现在也是纯写入：
```python
# 本 demo: write_kv 内部
base = seq_to_num_tokens[seq_id]
for t, (token_key, token_value) in enumerate(zip(layer_key, layer_value)):
    pos = base + t
    block_id = seq_to_block[seq_id][pos // block_size]  # 纯查找，不分配
    slot = pos % block_size
    cache.write(block_id, slot, token_key, token_value)
```

区别仅在于 vLLM 用扁平 `slot_mapping` 索引，本 demo 用 `(block_idx, slot)` 二级索引。
分配链条完全一致：
```
vLLM:     allocate_slots (Scheduler) → slot_mapping (Worker) → reshape_and_cache (CUDA)
本 demo:  allocate_slots             → write_kv 内部算 (block_idx, slot) → cache.write
```

### 拆层逻辑放模型侧还是 Engine 侧？

本 demo 选择封装到模型侧，Engine 只管调度 + sample。
不新增同名方法逐层传递，而是在现有 `forward` 上通过 `paged_ctx` 参数分支改造：
- `MultiheadSelfAttention.forward`：共享 projection + reshape + RoPE，按 `paged_ctx` 分支 attention 计算（与现有 `use_cache` 分支并列）
- `TransformerBlock.forward`：不感知 paged cache，只透传 `paged_ctx` 给 self.attn
- `TransformerLM`：新增 `prefill_with_paged_cache` / `decode_with_paged_cache`，构造 `PagedCacheContext` 并调用现有 block.forward

**划分标准：Model 定义 what to compute，Engine 定义 how to compute。**

| 放模型里 | 放 Engine/Runner 里 |
|---|---|
| 改了就不是同一个模型 | 改了模型还是同一个模型 |
| 权重、计算图、架构选择 | 并行策略、kernel 选择、调度策略 |
| 跟硬件/部署无关 | 跟硬件/部署强相关 |

**模型内部分层。**
LM 管上层架构（embedding → 层迭代 → final norm → lm_head），Block 管 norm + residual + FFN，Attention 管投影、RoPE、write_kv、attention 变体。
每层的 paged cache 逻辑（write_kv + attention 选择）是 attention 内部细节，不应泄漏到 Block 或 LM。
paged 分支只支持 pre-norm（当前主流），不处理 post/none 分支。

**vLLM 选的是 Engine 侧拆层（Worker/ModelRunner）。** 原因是生产系统需要在层间插入模型本身不该关心的逻辑：
- Tensor parallelism 的跨 GPU 通信（取决于几张卡，同一个模型单卡不需要）
- 不同 attention backend 的切换（FlashAttention / FlashInfer，数学等价，选择取决于硬件）
- Speculative decoding 的 draft/verify 交替（额外的推理策略，跟模型定义无关）
- 动态 memory profiling（纯运行时监控）

这些优化的共同特点：同一个模型，不同部署环境下选择不同。如果塞进模型，每加一种部署方式就多一堆 if-else，且分支之间交叉组合会指数膨胀。

**本 demo 选模型侧封装的理由：** 只有一种 how（单卡、paged attention、无 speculative）。当 how 只有一种时，放模型里更简洁，Engine 保持干净。

> **注意：** `MultiheadSelfAttention.forward` 的 paged 分支与标准分支存在部分逻辑重复（causal mask 构造等）。
> 这是有意为之——demo 目标是理解 serving 机制，不是消除重复。生产系统（vLLM）通过在 Engine 侧统一拆层来避免这个问题。

## 文件结构

```
cs336_basics/
    continuous_batching.py       # Sequence, Scheduler, Engine（KVCacheManager 在 paged_attention.py 中扩展）
    paged_attention.py           # 已有，复用

scripts/
    demo_continuous_batching.py  # Demo 脚本：模拟多请求到达，逐 step 打印 batch 组成
```

## 验证方案

1. 正确性对比：单个请求走 continuous batching，输出应与直接 `model.generate` 逐 token 完全一致（相同 seed / temperature）
2. 构造 4-6 个请求，不同 step 到达，不同 prompt 长度
3. 逐 step 打印：`Step N: [prefill: SeqX] [decode: SeqY, SeqZ]`，验证序列动态进出
4. 验证已完成序列的 KV cache blocks 被正确释放（`num_free_blocks` 恢复）
5. 对比 static batching：相同请求集，测量总 step 数差异

## 未实现功能汇总

| 功能 | 本 demo | vLLM 生产实现 |
|---|---|---|
| Prefill/Decode 混合 | 分离执行 | Chunked prefill，混合 batch |
| Preemption | 不做，等待释放 | Swap / Recompute |
| 优先级调度 | FCFS | Priority-based |
| Token budget | 无 | `max_num_batched_tokens` |
| Parallel sampling | 不支持 | SequenceGroup + CoW |
| Prefix caching | 不支持 | 共享 system prompt blocks |
| Speculative decoding | 不支持 | Draft model + verification |
| Tensor parallelism | 单 GPU | 多 GPU 分片 |
| CUDA kernel | PyTorch（`write_kv` ≈ `reshape_and_cache`） | `reshape_and_cache` CUDA kernel |
| 动态请求到达 | 预定义列表 | 异步 HTTP server |
