import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def _finish_step(self, seqs, token_ids, is_prefill):
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)
        return [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]

    def step(self):
        if self._scheduler_policy() == "unified":
            return self._step_unified()
        if self._scheduler_policy() == "interleave":
            return self._step_interleaved()
        seqs, is_prefill = self.scheduler.schedule()
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        outputs = self._finish_step(seqs, token_ids, is_prefill)
        return outputs, num_tokens

    def _run_scheduled(self, seqs, is_prefill, return_logits=False):
        if not seqs:
            return [], 0, {}
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else len(seqs)
        )
        skip_sampling = (
            is_prefill
            and not return_logits
            and all(
                seq.num_cached_tokens + seq.num_scheduled_tokens
                < seq.num_tokens
                for seq in seqs
            )
        )
        result = self.model_runner.call(
            "run", seqs, is_prefill, return_logits, skip_sampling
        )
        if return_logits:
            token_ids, logits = result
            step_logits = {
                seq.seq_id: logits[row] for row, seq in enumerate(seqs)
            }
        else:
            token_ids = result
            step_logits = {}
        outputs = self._finish_step(seqs, token_ids, is_prefill)
        return outputs, num_tokens, step_logits

    def _step_interleaved(self, return_logits=False):
        outputs = []
        step_logits = {}
        token_budget = self.config.max_num_batched_tokens

        decode_seqs = self.scheduler.schedule_decode(token_budget)
        decode_outputs, decode_tokens, decode_logits = self._run_scheduled(
            decode_seqs, False, return_logits
        )
        outputs.extend(decode_outputs)
        step_logits.update(decode_logits)
        token_budget -= decode_tokens

        prefill_budget = min(
            token_budget,
            self.config.max_prefill_chunk_tokens,
        )
        prefill_seqs = self.scheduler.schedule_prefill(
            prefill_budget,
            chunk_tokens=self.config.max_prefill_chunk_tokens,
        )
        prefill_outputs, prefill_tokens, prefill_logits = (
            self._run_scheduled(
                prefill_seqs, True, return_logits
            )
        )
        outputs.extend(prefill_outputs)
        step_logits.update(prefill_logits)

        if not decode_seqs and not prefill_seqs:
            raise RuntimeError("Interleaved scheduler produced no work")
        num_tokens = -decode_tokens if decode_tokens else prefill_tokens
        if return_logits:
            return outputs, num_tokens, step_logits
        return outputs, num_tokens

    def _step_unified(self, return_logits=False):
        scheduler_output = self.scheduler.schedule_unified()
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)

        result = self.model_runner.call(
            "run_mixed",
            scheduler_output.requests,
            return_logits,
        )
        if return_logits:
            sampled_token_ids, logits = result
            sample_ids = [
                item.request_id
                for item in scheduler_output.requests
                if item.sample
            ]
            step_logits = {
                seq_id: logits[row]
                for row, seq_id in enumerate(sample_ids)
            }
        else:
            sampled_token_ids = result
            step_logits = {}

        self.scheduler.postprocess_unified(
            scheduler_output,
            sampled_token_ids,
        )
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)
        outputs = [
            (item.sequence.seq_id, item.sequence.completion_token_ids)
            for item in scheduler_output.requests
            if item.sequence.is_finished
        ]
        num_tokens = (
            -scheduler_output.num_decode_requests
            if scheduler_output.num_decode_requests
            else scheduler_output.total_num_scheduled_tokens
        )
        if return_logits:
            return outputs, num_tokens, step_logits
        return outputs, num_tokens

    def _scheduler_policy(self):
        config = getattr(self, "config", None)
        return getattr(config, "scheduler_policy", "prefill_first")

    def step_with_logits(self):
        """Run the normal serving step and return per-request logits for validation."""
        if self._scheduler_policy() == "unified":
            return self._step_unified(return_logits=True)
        if self._scheduler_policy() == "interleave":
            return self._step_interleaved(return_logits=True)
        seqs, is_prefill = self.scheduler.schedule()
        released_seq_ids = self.scheduler.drain_released_seq_ids()
        if released_seq_ids:
            self.model_runner.call("release_states", released_seq_ids)
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        token_ids, logits = self.model_runner.call(
            "run", seqs, is_prefill, True
        )
        step_logits = {
            seq.seq_id: logits[row] for row, seq in enumerate(seqs)
        }
        outputs = self._finish_step(seqs, token_ids, is_prefill)
        return outputs, num_tokens, step_logits

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
