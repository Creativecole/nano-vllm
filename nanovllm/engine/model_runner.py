import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.attention import HybridAttentionMetadataBuilder
from nanovllm.config import Config
from nanovllm.engine.cache_coordinator import HybridCacheCoordinator
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import ScheduledRequest
from nanovllm.engine.layer_state import (
    DeltaNetStateSpec,
    delta_state_bytes_per_sequence,
    paged_kv_bytes_per_block,
)
from nanovllm.models.registry import get_model_class
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.profiler import profile_range


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_text_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.reset_execution_stats()

        model_class = get_model_class(config.hf_config, hf_config)
        if not getattr(model_class, "supports_cuda_graph", True):
            self.enforce_eager = True
            config.enforce_eager = True
        config.enable_prefix_cache = config.enable_prefix_cache and bool(
            getattr(model_class, "supports_prefix_cache", True)
        )

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        torch.set_default_device("cuda")
        hf_config.nanovllm_deltanet_backend = config.deltanet_backend
        hf_config.nanovllm_deltanet_chunk_size = config.deltanet_chunk_size
        self.model = model_class(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        get_specs = getattr(self.model, "get_layer_state_specs", None)
        self.layer_state_specs = get_specs() if get_specs is not None else []
        self.is_hybrid = any(
            isinstance(spec, DeltaNetStateSpec) for spec in self.layer_state_specs
        )
        self.attention_metadata_builder = (
            HybridAttentionMetadataBuilder() if self.is_hybrid else None
        )
        self.hybrid_cache_coordinator = None
        self.hybrid_state_manager = None
        if self.is_hybrid:
            if self.world_size != 1:
                raise NotImplementedError("Qwen3.5 hybrid serving currently supports TP=1")
            self.allocate_hybrid_cache()
        else:
            self.warmup_model()
            self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "kv_cache"):
            del self.kv_cache
        self.hybrid_cache_coordinator = None
        self.hybrid_state_manager = None
        reset_context()
        torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_text_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * config.dtype.itemsize
        )
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_hybrid_cache(self):
        config = self.config
        # Hybrid allocation has no warmup peak to reserve. Release cached blocks
        # from a previous runtime in the same process and budget from live usage.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        available = min(
            free,
            int(total * config.gpu_memory_utilization - used),
        )
        state_bytes = delta_state_bytes_per_sequence(self.layer_state_specs)
        block_bytes = paged_kv_bytes_per_block(
            self.layer_state_specs, self.block_size
        )
        if state_bytes <= 0 or block_bytes <= 0:
            raise RuntimeError(
                "Hybrid serving requires both DeltaNet state and full-attention KV layers"
            )

        if config.hybrid_state_capacity:
            state_capacity = min(config.max_num_seqs, config.hybrid_state_capacity)
        else:
            state_budget = int(available * config.hybrid_state_memory_fraction)
            state_capacity = min(
                config.max_num_seqs,
                max(1, state_budget // state_bytes),
            )
        reserved_state_bytes = state_capacity * state_bytes
        num_blocks = (available - reserved_state_bytes) // block_bytes
        if num_blocks <= 0:
            raise RuntimeError(
                "Insufficient GPU memory after reserving DeltaNet request states: "
                f"available={available}, state_capacity={state_capacity}, "
                f"state_bytes_per_sequence={state_bytes}, kv_block_bytes={block_bytes}"
            )

        coordinator = HybridCacheCoordinator(
            self.layer_state_specs,
            state_capacity=state_capacity,
            num_kv_blocks=num_blocks,
            block_size=self.block_size,
            device=torch.device("cuda", self.rank),
            compact_delta_slots=config.resident_deltanet_state,
        )
        coordinator.bind_model(self.model)
        self.hybrid_cache_coordinator = coordinator
        # Compatibility alias for existing diagnostics and serving benchmarks.
        self.hybrid_state_manager = coordinator.delta_states
        config.max_num_seqs = min(config.max_num_seqs, state_capacity)
        config.num_kvcache_blocks = int(num_blocks)
        cache_stats = coordinator.get_stats()
        print(
            "Hybrid cache: "
            f"state_slots={state_capacity}, state_bytes_per_sequence={state_bytes}, "
            f"kv_blocks={num_blocks}, "
            f"full_attention_layers={cache_stats['full_attention_layers']}, "
            f"deltanet_layers={cache_stats['deltanet_layers']}",
            flush=True,
        )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefill_seq_lens = tuple(seq.num_scheduled_tokens for seq in seqs)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            prefill_seq_lens,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_mixed(
        self,
        requests: list[ScheduledRequest],
    ) -> tuple[torch.Tensor, torch.Tensor, list[Sequence], list[int]]:
        input_ids = []
        positions = []
        slot_mapping = []
        sample_indices = []
        sample_seqs = []

        decode_requests = [item for item in requests if not item.is_prefill]
        prefill_requests = [item for item in requests if item.is_prefill]
        if requests != decode_requests + prefill_requests:
            raise ValueError(
                "Unified execution requires decode requests before prefill requests"
            )

        decode_context_lens = []
        for item in decode_requests:
            seq = item.sequence
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            decode_context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size
                + seq.last_block_num_tokens
                - 1
            )
            sample_indices.append(len(input_ids) - 1)
            sample_seqs.append(seq)

        prefill_cu_seqlens_q = [0]
        prefill_cu_seqlens_k = [0]
        prefill_max_seqlen_q = 0
        prefill_max_seqlen_k = 0
        prefill_has_prefix = False
        prefill_seq_lens = []
        for item in prefill_requests:
            seq = item.sequence
            start = seq.num_cached_tokens
            end = start + item.num_scheduled_tokens
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            prefill_seq_lens.append(item.num_scheduled_tokens)
            prefill_cu_seqlens_q.append(
                prefill_cu_seqlens_q[-1] + item.num_scheduled_tokens
            )
            prefill_cu_seqlens_k.append(prefill_cu_seqlens_k[-1] + end)
            prefill_max_seqlen_q = max(
                prefill_max_seqlen_q, item.num_scheduled_tokens
            )
            prefill_max_seqlen_k = max(prefill_max_seqlen_k, end)
            prefill_has_prefix = prefill_has_prefix or end > item.num_scheduled_tokens

            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for block_idx in range(start_block, end_block):
                slot_start = seq.block_table[block_idx] * self.block_size
                if block_idx == start_block:
                    slot_start += start % self.block_size
                if block_idx != end_block - 1:
                    slot_end = (
                        seq.block_table[block_idx] * self.block_size
                        + self.block_size
                    )
                else:
                    slot_end = (
                        seq.block_table[block_idx] * self.block_size
                        + end
                        - block_idx * self.block_size
                    )
                slot_mapping.extend(range(slot_start, slot_end))
            if item.sample:
                sample_indices.append(len(input_ids) - 1)
                sample_seqs.append(seq)

        input_ids_tensor = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions_tensor = torch.tensor(
            positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        sample_indices_tensor = torch.tensor(
            sample_indices, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)

        decode_seqs = [item.sequence for item in decode_requests]
        decode_context_lens_tensor = (
            torch.tensor(
                decode_context_lens,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            if decode_seqs
            else None
        )
        decode_block_tables = (
            self.prepare_block_tables(decode_seqs) if decode_seqs else None
        )

        prefill_seqs = [item.sequence for item in prefill_requests]
        prefill_cu_seqlens_q_tensor = (
            torch.tensor(
                prefill_cu_seqlens_q,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            if prefill_seqs
            else None
        )
        prefill_cu_seqlens_k_tensor = (
            torch.tensor(
                prefill_cu_seqlens_k,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            if prefill_seqs
            else None
        )
        prefill_block_tables = (
            self.prepare_block_tables(prefill_seqs)
            if prefill_seqs and prefill_has_prefix
            else None
        )
        set_context(
            False,
            slot_mapping=slot_mapping_tensor,
            is_mixed=True,
            num_decode_requests=len(decode_requests),
            decode_context_lens=decode_context_lens_tensor,
            decode_block_tables=decode_block_tables,
            prefill_cu_seqlens_q=prefill_cu_seqlens_q_tensor,
            prefill_cu_seqlens_k=prefill_cu_seqlens_k_tensor,
            prefill_max_seqlen_q=prefill_max_seqlen_q,
            prefill_max_seqlen_k=prefill_max_seqlen_k,
            prefill_block_tables=prefill_block_tables,
            prefill_seq_lens=tuple(prefill_seq_lens),
            sequence_query_lens=tuple(
                item.num_scheduled_tokens for item in requests
            ),
            sample_indices=sample_indices_tensor,
        )
        return input_ids_tensor, positions_tensor, sample_seqs, sample_indices

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        layer_states=None,
        compute_logits: bool = True,
        attention_metadata=None,
    ):
        if layer_states is not None:
            hidden_states = self.model(
                input_ids,
                positions,
                layer_states=layer_states,
                attention_metadata=attention_metadata,
            )
            return (
                self.model.compute_logits(hidden_states)
                if compute_logits
                else hidden_states
            )
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)
            return (
                self.model.compute_logits(hidden_states)
                if compute_logits
                else hidden_states
            )
        else:
            if not compute_logits:
                raise ValueError("decode always requires logits")
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _build_hybrid_attention_metadata(
        self,
        *,
        request_ids: list[int],
        query_lens: list[int],
        is_prefilling: list[bool],
        positions: torch.Tensor,
    ):
        if not self.is_hybrid:
            return None
        return self.attention_metadata_builder.build(
            context=get_context(),
            request_ids=request_ids,
            query_lens=query_lens,
            is_prefilling=is_prefilling,
            positions=positions,
        )

    def _prepare_hybrid_state_batch(
        self,
        seqs: list[Sequence],
        *,
        allow_reorder: bool,
    ) -> tuple[list[Sequence], dict | None, bool]:
        coordinator = self.hybrid_cache_coordinator
        if coordinator is None:
            return seqs, None, False

        requested_ids = [seq.seq_id for seq in seqs]
        execution_ids, layer_states, is_resident = (
            coordinator.prepare_request_batch(
                requested_ids,
                allow_reorder=allow_reorder,
            )
        )
        if is_resident:
            self.execution_stats["state_resident_view_calls"] += 1
        else:
            self.execution_stats["state_gather_calls"] += 1
        if execution_ids == requested_ids:
            return seqs, layer_states, is_resident
        seq_by_id = {seq.seq_id: seq for seq in seqs}
        ordered_seqs = [seq_by_id[seq_id] for seq_id in execution_ids]
        return ordered_seqs, layer_states, is_resident

    @staticmethod
    def _restore_request_order(
        values,
        execution_seqs: list[Sequence],
        requested_seqs: list[Sequence],
    ):
        if values is None or execution_seqs == requested_seqs:
            return values
        row_by_id = {
            seq.seq_id: row for row, seq in enumerate(execution_seqs)
        }
        rows = [row_by_id[seq.seq_id] for seq in requested_seqs]
        if isinstance(values, torch.Tensor):
            return values[rows]
        return [values[row] for row in rows]

    def _finish_hybrid_state_batch(
        self,
        layer_states: dict | None,
        is_resident: bool,
    ) -> None:
        if layer_states is None:
            return
        committed = self.hybrid_cache_coordinator.finish_request_batch(
            layer_states,
            is_resident=is_resident,
        )
        if not committed:
            self.execution_stats["state_commit_skipped_calls"] += 1
            return
        self.execution_stats["state_commit_calls"] += 1

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        return_logits: bool = False,
        skip_sampling: bool = False,
    ):
        self.execution_stats["model_runner_calls"] += 1
        self.execution_stats[
            "prefill_model_runner_calls"
            if is_prefill
            else "decode_model_runner_calls"
        ] += 1
        if skip_sampling and (not is_prefill or return_logits):
            raise ValueError(
                "skip_sampling is only valid for intermediate prefill chunks"
            )
        requested_seqs = seqs
        seqs, layer_states, resident_states = (
            self._prepare_hybrid_state_batch(
                requested_seqs,
                allow_reorder=True,
            )
        )
        with profile_range("qwen35_metadata_prepare"):
            input_ids, positions = (
                self.prepare_prefill(seqs)
                if is_prefill
                else self.prepare_decode(seqs)
            )
            attention_metadata = self._build_hybrid_attention_metadata(
                request_ids=[seq.seq_id for seq in seqs],
                query_lens=[
                    seq.num_scheduled_tokens if is_prefill else 1
                    for seq in seqs
                ],
                is_prefilling=[is_prefill] * len(seqs),
                positions=positions,
            )
        temperatures = (
            self.prepare_sample(seqs)
            if self.rank == 0 and not skip_sampling
            else None
        )
        phase = "prefill" if is_prefill else "decode"
        with profile_range(f"qwen35_{phase}_model"):
            model_output = self.run_model(
                input_ids,
                positions,
                is_prefill,
                layer_states,
                compute_logits=not skip_sampling,
                attention_metadata=attention_metadata,
            )
        self._finish_hybrid_state_batch(layer_states, resident_states)
        if skip_sampling:
            token_ids = [0] * len(seqs) if self.rank == 0 else None
            logits_cpu = None
        else:
            logits = model_output
            token_ids = (
                self.sampler(logits, temperatures).tolist()
                if self.rank == 0
                else None
            )
            logits_cpu = (
                logits.float().cpu()
                if return_logits and self.rank == 0
                else None
            )
        token_ids = self._restore_request_order(
            token_ids,
            seqs,
            requested_seqs,
        )
        logits_cpu = self._restore_request_order(
            logits_cpu,
            seqs,
            requested_seqs,
        )
        reset_context()
        if return_logits:
            return token_ids, logits_cpu
        return token_ids

    def run_mixed(
        self,
        requests: list[ScheduledRequest],
        return_logits: bool = False,
    ):
        self.execution_stats["model_runner_calls"] += 1
        self.execution_stats["unified_model_runner_calls"] += 1
        if not requests:
            raise ValueError("run_mixed requires at least one scheduled request")
        if not self.is_hybrid:
            raise RuntimeError(
                "Unified mixed execution currently supports hybrid models only"
            )
        with profile_range("qwen35_metadata_prepare_mixed"):
            input_ids, positions, sample_seqs, _ = self.prepare_mixed(requests)
            attention_metadata = self._build_hybrid_attention_metadata(
                request_ids=[item.request_id for item in requests],
                query_lens=[
                    item.num_scheduled_tokens for item in requests
                ],
                is_prefilling=[item.is_prefill for item in requests],
                positions=positions,
            )

        temperatures = (
            self.prepare_sample(sample_seqs)
            if self.rank == 0 and sample_seqs
            else None
        )
        seqs = [item.sequence for item in requests]
        _, layer_states, resident_states = self._prepare_hybrid_state_batch(
            seqs,
            allow_reorder=False,
        )

        with profile_range("qwen35_unified_model"):
            model_output = self.run_model(
                input_ids,
                positions,
                True,
                layer_states,
                compute_logits=bool(sample_seqs),
                attention_metadata=attention_metadata,
            )
        self._finish_hybrid_state_batch(layer_states, resident_states)

        sampled_tokens = {}
        logits_cpu = None
        if sample_seqs:
            token_ids = (
                self.sampler(model_output, temperatures).tolist()
                if self.rank == 0
                else None
            )
            if self.rank == 0:
                sampled_tokens = {
                    seq.seq_id: token_id
                    for seq, token_id in zip(sample_seqs, token_ids)
                }
                if return_logits:
                    logits_cpu = model_output.float().cpu()
        reset_context()
        if return_logits:
            return sampled_tokens, logits_cpu
        return sampled_tokens

    def reset_execution_stats(self) -> None:
        self.execution_stats = {
            "model_runner_calls": 0,
            "prefill_model_runner_calls": 0,
            "decode_model_runner_calls": 0,
            "unified_model_runner_calls": 0,
            "state_resident_view_calls": 0,
            "state_gather_calls": 0,
            "state_commit_calls": 0,
            "state_commit_skipped_calls": 0,
        }
        if getattr(self, "hybrid_state_manager", None) is not None:
            self.hybrid_state_manager.max_allocated_count = (
                self.hybrid_state_manager.allocated_count
            )
            self.hybrid_state_manager.reset_runtime_stats()

    def get_execution_stats(self) -> dict[str, int]:
        return dict(self.execution_stats)

    def release_states(self, seq_ids: list[int]):
        if self.hybrid_cache_coordinator is not None:
            self.hybrid_cache_coordinator.free_requests(seq_ids)

    def set_deltanet_diagnostics(
        self, enabled: bool, reset: bool = True
    ) -> None:
        configure = getattr(self.model, "set_deltanet_diagnostics", None)
        if configure is None or self.hybrid_state_manager is None:
            raise RuntimeError("DeltaNet diagnostics require a hybrid Qwen3.5 model")
        configure(enabled, reset=reset)
        self.hybrid_state_manager.set_diagnostics(enabled, reset=reset)

    def reset_deltanet_diagnostics(self) -> None:
        reset = getattr(self.model, "reset_deltanet_diagnostics", None)
        if reset is None or self.hybrid_state_manager is None:
            raise RuntimeError("DeltaNet diagnostics require a hybrid Qwen3.5 model")
        reset()
        self.hybrid_state_manager.reset_diagnostics()

    def get_deltanet_diagnostics(self) -> dict[str, object]:
        collect = getattr(self.model, "get_deltanet_diagnostics", None)
        if collect is None or self.hybrid_state_manager is None:
            raise RuntimeError("DeltaNet diagnostics require a hybrid Qwen3.5 model")
        return {
            "layers": collect(),
            "state_manager": self.hybrid_state_manager.get_diagnostics(),
        }

    def get_hybrid_state_stats(self):
        if self.hybrid_state_manager is None:
            return None
        delta_bytes_per_sequence = delta_state_bytes_per_sequence(
            self.layer_state_specs
        )
        kv_bytes_per_block = paged_kv_bytes_per_block(
            self.layer_state_specs, self.block_size
        )
        return {
            "capacity": self.hybrid_state_manager.capacity,
            "allocated": self.hybrid_state_manager.allocated_count,
            "free": self.hybrid_state_manager.free_count,
            "max_allocated": self.hybrid_state_manager.max_allocated_count,
            "utilization": (
                self.hybrid_state_manager.allocated_count
                / self.hybrid_state_manager.capacity
            ),
            "max_utilization": (
                self.hybrid_state_manager.max_allocated_count
                / self.hybrid_state_manager.capacity
            ),
            "delta_bytes_per_sequence": delta_bytes_per_sequence,
            "delta_pool_bytes": self.hybrid_state_manager.capacity
            * delta_bytes_per_sequence,
            "kv_bytes_per_block": kv_bytes_per_block,
            "num_kv_blocks": self.config.num_kvcache_blocks,
            "kv_cache_bytes": self.config.num_kvcache_blocks * kv_bytes_per_block,
            "state_execution": self.hybrid_state_manager.get_runtime_stats(),
        }

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
