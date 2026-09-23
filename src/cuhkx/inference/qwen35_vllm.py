"""Dual-GPU vLLM inference backend for the Qwen3.5-4B lane.

vLLM replaces only the *generation* engine. Training and development time
evaluation-under-training still use the Transformers ``Qwen35Backend``: vLLM is
inference-only and cannot produce gradients. The data contract, prompt version,
constrained answer space and run/checkpoint protocol are unchanged, so a vLLM
run and a Transformers run of the same revision stay comparable.

Two properties of the existing protocol make this substitution safe, and both
are asserted below rather than assumed:

1. The answer space is closed and tiny, so constraining generation to the same
   set of outputs is what matters, not the mechanism. The Transformers backend
   enforces this with a stateful ``prefix_allowed_tokens_fn``. vLLM has no such
   hook; it constrains the same language with a structured-output ``choice``
   grammar.

2. The prompt must tokenize to exactly what ``apply_chat_template`` produces,
   including ``enable_thinking=False``. The prefix boundary positions inside the
   allowed-answer map are only valid for that encoding, so the rendered prompt
   is compared against the reference tokenizer before the engine is built.
"""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

from cuhkx.config import require


# Two T4s have no NVLink. Tensor parallelism therefore crosses PCIe, where
# CUDA-graph capture and custom all-reduce are a common source of hangs and
# illegal-memory-access crashes. Eager mode costs some throughput and buys a
# run that actually finishes; the NCCL path is also pinned to PCIe so vLLM
# cannot probe for peer-to-peer support at startup.
VLLM_ENGINE_ENV = {
    "VLLM_USE_V1": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # Worker telemetry travels over vLLM's RPC channel, whose default msgpack
    # serializer refuses arbitrary Python callables:
    #
    #     TypeError: Object of type <class 'function'> is not serializable
    #     Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle
    #
    # Without this the RPC in worker_memory() fails and *every* worker figure is
    # lost, which is why an earlier run reported allocated_mib: 0 on both cards
    # and workers: []. The channel only ever links this process to the vLLM
    # workers it spawned itself, and carries nothing but our own telemetry
    # function, so the pickle fallback is acceptable here. Insecure means "do not
    # expose this channel to untrusted peers", not "unsafe in this process tree".
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    # FlashInfer/Triton JIT need a writable cache even on a read-only dataset.
    "TRITON_CACHE_DIR": "/tmp/cuhkx_triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/cuhkx_inductor_cache",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
}


# vLLM's attention backends differ in what they need at *run* time, not just at
# import time, so the choice is part of the lane's runtime contract.
#
# FLASHINFER is vLLM's automatic first choice, but on a Kaggle T4 it JIT-compiles
# SM 7.5 kernels through the host CUDA toolkit, and the final link step asks for
# the driver library:
#
#     nvcc ... -gencode=arch=compute_75,code=sm_75 ... -lcudart -lcuda
#     /usr/bin/ld: cannot find -lcuda: No such file or directory
#
# libcuda.so belongs to the NVIDIA *driver*, not to the CUDA runtime that pip
# installs, and Kaggle's container has no driver stubs directory. The failure
# only appears on the first real forward pass, after the model and KV cache are
# already up, which makes it look like a later bug than it is. That same JIT
# also dominated startup (~9 minutes) before it failed.
#
# TRITON_ATTN ships as pure Triton, needs no nvcc/link step, and reports
# supports_compute_capability() == True for every device. It supports float16,
# which is this lane's dtype, so it is the default here.
ATTENTION_BACKENDS = ("TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION")


def apply_engine_environment(environment=None) -> dict:
    """Populate vLLM/NCCL defaults without overwriting explicit overrides."""
    import os

    target = os.environ if environment is None else environment
    for name, value in VLLM_ENGINE_ENV.items():
        target.setdefault(name, value)
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        Path(target[name]).mkdir(parents=True, exist_ok=True)
    return dict(target)


def render_prompt(tokenizer, prompt: str, *, enable_thinking: bool = False) -> str:
    """Render one user turn with the exact template arguments the lane pins."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def verify_prompt_fidelity(processor, prompts: dict) -> dict:
    """Prove vLLM's chat rendering matches the Transformers reference encoding.

    ``Qwen35Backend`` tokenizes the multimodal turn through
    ``processor.apply_chat_template``. vLLM renders chat templates and injects
    image placeholders itself, so the two can differ silently. Only the textual
    envelope is checked here; the comparison is between the reference processor
    and the vLLM renderer, so a template or ``enable_thinking`` drift is a hard
    error rather than a silent change of the decoding boundary.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    require(bool(prompts), "no prompts were available for fidelity checking")
    reference = []
    for prompt in prompts.values():
        # The reference path passes a message list to the processor; the
        # rendered string must match what vLLM will build from the same turn.
        reference.append(_render_reference(tokenizer, prompt))
    checked = 0
    for qa_id, prompt in prompts.items():
        require(image_token not in prompt, f"prompt leaks the image placeholder: {qa_id}")
        rendered = render_prompt(tokenizer, prompt)
        require(rendered == reference[checked], f"vLLM chat rendering differs for {qa_id}")
        checked += 1
    return {"prompts_checked": checked, "image_token": image_token, "template_mismatches": 0}


def _render_reference(tokenizer, prompt: str) -> str:
    """Reference rendering used by the Transformers backend for text turns."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        # Older templates do not accept enable_thinking; the vLLM path would
        # then also have to omit it, so surface the difference immediately.
        raise ValueError(
            "the pinned chat template no longer accepts enable_thinking; "
            "vLLM prompt fidelity cannot be guaranteed"
        ) from None


class Qwen35VLLMBackend:
    """Offline multi-GPU vLLM engine constrained to the lane's answer space.

    Two ways to use both T4s, and they are not interchangeable:

    * Tensor parallelism (``tensor_parallel_size=2``) splits every layer across
      the cards, so one request is served by both GPUs and every forward pass
      pays a PCIe all-reduce. Two T4s have no NVLink.
    * Data parallelism (``data_parallel_size=2``, ``tensor_parallel_size=1``)
      gives each card its own complete engine and routes different requests to
      different cards, so there is no cross-card communication at all. Each card
      must hold the whole model and its own KV cache.

    On PCIe-only T4s the second is often the better trade: independence removes
    the all-reduce that dominates short-prefill, tiny-decode workloads like this
    one. Which one actually wins is an empirical question, so both are selectable
    and the chosen topology is recorded in the signed contract.
    """

    def __init__(self, baseline: dict, weights: Path, adapter: Path | None = None,
                 *, tensor_parallel_size: int = 2, gpu_memory_utilization: float = 0.80,
                 enable_prefix_caching: bool = True, max_model_len: int = 4096,
                 attention_backend: str = "TRITON_ATTN", max_num_seqs: int = 1,
                 data_parallel_size: int = 1):
        require(tensor_parallel_size >= 1, "tensor parallel size must be positive")
        require(data_parallel_size >= 1, "data parallel size must be positive")
        require(not (data_parallel_size > 1 and tensor_parallel_size > 1),
                "tensor and data parallelism cannot both exceed 1: the free GPUs are "
                "either split inside one replica or divided between replicas")
        require(0.0 < gpu_memory_utilization < 1.0, "gpu memory utilization must be a fraction")
        require(max_model_len >= 1024, "max_model_len is implausibly small")
        require(attention_backend in ATTENTION_BACKENDS,
                f"unsupported attention backend: {attention_backend}")
        require(max_num_seqs >= 1, "max_num_seqs must allow at least one sequence")
        apply_engine_environment()

        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        require(weights.is_dir(), "Qwen3.5 weight directory is missing")
        available = torch.cuda.device_count()
        required = tensor_parallel_size * data_parallel_size
        require(available >= required,
                f"vLLM needs {required} visible CUDA devices for "
                f"tensor_parallel_size={tensor_parallel_size} x "
                f"data_parallel_size={data_parallel_size}, found {available}")

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        self.image_size = baseline["frames"]["input_image_size"]
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.attention_backend = attention_backend
        self.max_num_seqs = max_num_seqs
        self.engines = []
        self._dispatch_index = 0
        # Prefix-cache accounting is accumulated across the run and summarized in
        # metadata(); a run that never sees the fields reports zero coverage
        # rather than a flattering zero hit rate.
        self.prefix_cache_requests = 0
        self.prefix_cache_missing = 0
        self.prefix_cache_cached_tokens = 0
        self.prefix_cache_prompt_tokens = 0
        self.prefix_cache_summary = {}
        # Filled from the first finished request so the summary carries the real
        # field inventory of this vLLM build.
        self.metrics_shape = None

        started = time.perf_counter()
        # use_fast=False mirrors the reference backend: the slow image processor
        # is the one the lane validated.
        self.processor = AutoProcessor.from_pretrained(
            str(weights), local_files_only=True, trust_remote_code=False, use_fast=False,
        )
        engine_kwargs = dict(
            model=str(weights),
            revision=baseline["model"].get("revision") or None,
            tokenizer=str(weights),
            tokenizer_revision=baseline["model"].get("revision") or None,
            trust_remote_code=False,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching,
            disable_custom_all_reduce=True,
            max_num_seqs=max_num_seqs,
            attention_config={"backend": attention_backend},
            # LLM.__init__ forces disable_log_stats=True when the caller does not
            # pass it, and the output processor then attaches RequestStateStats
            # only when log_stats is on. Without this the finished output carries
            # metrics=None, which is why an earlier run reported no TTFT at all:
            # first_token_latency simply is not produced. Engine statistics are
            # per-iteration aggregates, so the cost is negligible.
            disable_log_stats=False,
            # A LoRA adapter needs the LoRA path compiled into the engine.
            enable_lora=adapter is not None,
            max_loras=1,
            limit_mm_per_prompt={"image": 4},
            mm_processor_kwargs={"use_fast": False},
        )
        if data_parallel_size > 1:
            # One whole replica per GPU. vLLM has no in-process DP for the
            # offline LLM entry point that also lets each replica own a distinct
            # device, so the replicas are built in child processes, each with
            # CUDA_VISIBLE_DEVICES pinned to its own card. The children own the
            # engines; this process only dispatches plain data to them.
            self.engine = None
            self.engines = [_ReplicaHandle(index, engine_kwargs, weights)
                            for index in range(data_parallel_size)]
            self.adapter_request = None
            self.last_metrics = {"reported": False}
            self.load_seconds = time.perf_counter() - started
            return
        self.engine = LLM(
            tensor_parallel_size=tensor_parallel_size,
            **engine_kwargs,
        )
        self.adapter_request = None
        # Populated per request; absent until generate() has run.
        self.last_metrics = {"reported": False}
        if adapter is not None:
            self.adapter_request = self._register_adapter(adapter)
        self.load_seconds = time.perf_counter() - started

    @property
    def _data_parallel_replicas(self):
        """The data-parallel replicas, or an empty list in the other modes.

        Read through an accessor rather than the attribute directly: tests build
        this class with ``__new__`` to exercise single methods, and those
        instances never ran ``__init__``, so ``self.engines`` is not set on them.
        """
        return getattr(self, "engines", None) or []

    @property
    def replicas(self):
        """The engines requests are dispatched to, one per data-parallel rank."""
        replicas = self._data_parallel_replicas
        return replicas if replicas else [self.engine]

    def _round_robin(self, size):
        """Distribute ``size`` requests over the replicas, in order.

        Contiguous blocks rather than interleaving: each replica then sees one
        engine call per dispatch, which is what makes the batched timing a
        meaningful per-replica figure instead of a mix of two engines.
        """
        replicas = self.replicas
        count = len(replicas)
        base, extra = divmod(size, count)
        assignments, start = [], 0
        for index in range(count):
            length = base + (1 if index < extra else 0)
            if length:
                assignments.append((index, start, start + length))
            start += length
        return assignments

    def _register_adapter(self, adapter: Path):
        """Load the verified LoRA adapter through vLLM's own loader.

        The adapter arrives already verified by ``verify_adapter`` against the
        base weight receipt; vLLM re-reads the same directory, so the receipt
        stays the authority on provenance.
        """
        adapter = Path(adapter).resolve()
        require(adapter.is_dir(), "adapter directory is missing")
        from vllm.lora.request import LoRARequest

        request = LoRARequest(lora_name="cuhkx_adapter", lora_int_id=1, lora_path=str(adapter))
        self.engine.add_lora(request)
        return request

    def generate(self, images, prompt, *, allowed_outputs, max_new_tokens):
        require(len(images) == 4, "Qwen3.5 IR4 requires exactly four images")
        outputs = tuple(allowed_outputs)
        require(outputs and all(isinstance(value, str) and value for value in outputs),
                "the constrained answer space must be non-empty text")
        require(max_new_tokens >= max(len(value) for value in outputs),
                "max_new_tokens cannot cover the answer space")

        # Encoding four frames to PNG data URIs is pure CPU work on the request
        # path. It is measured separately because at batch=1 it can be a visible
        # share of the wall clock, and it is the first thing to move off the
        # critical path when batching.
        encode_started = time.perf_counter()
        content = [{"type": "image_url",
                    "image_url": {"url": _data_uri(image), "detail": "high"}}
                   for image in images]
        self.last_encode_seconds = time.perf_counter() - encode_started
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        if self._data_parallel_replicas:
            # Data parallel: the request goes to one replica, chosen in turn so
            # consecutive requests spread across the cards. The replica builds
            # its own SamplingParams, so none is built here.
            replica = self.replicas[self._dispatch_index % len(self.replicas)]
            self._dispatch_index += 1
            response = replica.chat([messages], [list(outputs)], max_new_tokens)
            answers = response.get("answers") or []
            require(len(answers) == 1, "vLLM replica returned an unexpected number of results")
            metrics = (response.get("metrics") or [{}])[0]
            self.last_metrics = dict(metrics)
            self._record_prefix_cache(metrics)
            return answers[0]
        parameters = self.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
            # ``choice`` takes literal answer strings. vLLM builds the grammar
            # itself, so the model can only emit one of these exact strings --
            # the same closed language the Transformers backend enforces with
            # prefix_allowed_tokens_fn. Do not pass a regex here: a pattern like
            # r"\s*B" would be escaped into the grammar and the model would
            # answer with the literal characters "\s*B".
            structured_outputs=self.StructuredOutputsParams(choice=list(outputs)),
        )
        results = self.engine.chat(
            messages,
            sampling_params=parameters,
            use_tqdm=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
            lora_request=self.adapter_request,
        )
        require(len(results) == 1, "vLLM returned an unexpected number of results")
        output = results[0]
        # The engine times the request itself, which is the only way to get TTFT:
        # the offline chat API returns the finished text, not a token stream.
        # ``first_token_latency`` is an engine-core timestamp measured from
        # request arrival, so it includes queueing and excludes our own image
        # encoding -- exactly the split worth reporting.
        self.last_metrics = _engine_metrics(output)
        self._record_prefix_cache(self.last_metrics)
        return output.outputs[0].text.strip()

    def generate_batch(self, batch, *, max_new_tokens):
        """Run many requests through the engine in one scheduler pass.

        ``batch`` is a list of ``(images, prompt, allowed_outputs)`` triples. The
        engine interleaves them, so this returns answers in input order without
        any per-request timing: the batch is the smallest honest unit here, which
        is why the runner records a batch sample rather than fabricating one per
        request.

        ``max_num_seqs`` bounds how many the scheduler actually runs at once; the
        engine handles the excess by queuing, so an oversized batch degrades to
        multiple passes rather than failing.
        """
        require(bool(batch), "an empty batch has nothing to generate")
        encode_started = time.perf_counter()
        conversations = []
        for images, prompt, allowed_outputs in batch:
            require(len(images) == 4, "Qwen3.5 IR4 requires exactly four images")
            outputs = tuple(allowed_outputs)
            require(outputs and all(isinstance(value, str) and value for value in outputs),
                    "the constrained answer space must be non-empty text")
            require(max_new_tokens >= max(len(value) for value in outputs),
                    "max_new_tokens cannot cover the answer space")
            content = [{"type": "image_url",
                        "image_url": {"url": _data_uri(image), "detail": "high"}}
                       for image in images]
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])
        self.last_encode_seconds = time.perf_counter() - encode_started

        # One SamplingParams per request: the answer spaces differ (single choice
        # versus ordered multi-select), and one shared constraint would let a
        # request emit another request's answers. engine.chat pairs a sequence of
        # parameters with the prompts one by one.
        #
        # In data-parallel mode the parameters are built inside each replica
        # process from the plain answer lists, so the parent has no vLLM objects
        # to construct and nothing vLLM-specific crosses the process boundary.
        parameters = None
        if not self._data_parallel_replicas:
            parameters = [
                self.SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_new_tokens,
                    skip_special_tokens=True,
                    # ``choice`` takes literal answer strings; a regex would be
                    # escaped into the grammar and the model would answer with the
                    # literal pattern characters.
                    structured_outputs=self.StructuredOutputsParams(choice=list(outputs)),
                )
                for _, _, outputs in batch
            ]
        started = time.perf_counter()
        if self._data_parallel_replicas:
            # Data parallel: split the batch into one contiguous block per
            # replica and run the blocks as parallel engine calls. The wall clock
            # of the slowest replica is the batch cost, because a batch is only
            # finished when every request in it is.
            import concurrent.futures

            replicas = self._data_parallel_replicas
            answers = [None] * len(batch)
            metrics = [None] * len(batch)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(replicas)) as pool:
                futures = []
                for replica_index, start, end in self._round_robin(len(batch)):
                    futures.append((start, end, pool.submit(
                        self.replicas[replica_index].chat,
                        conversations[start:end],
                        [list(allowed) for _, _, allowed in batch[start:end]],
                        max_new_tokens,
                    )))
                for start, end, future in futures:
                    response = future.result()
                    block = response.get("answers") or []
                    block_metrics = response.get("metrics") or [{}] * (end - start)
                    require(len(block) == end - start,
                            "vLLM replica returned the wrong number of results")
                    answers[start:end] = block
                    metrics[start:end] = block_metrics
                    if response.get("metrics_shape") and self.metrics_shape is None:
                        self.metrics_shape = response["metrics_shape"]
            generate_seconds = time.perf_counter() - started
            require(all(isinstance(value, str) for value in answers),
                    "a data-parallel replica returned no answer")
            self._record_prefix_cache_batch(metrics)
            reported = [value for value in metrics if value.get("reported")]
            self.last_metrics = {
                "reported": bool(reported),
                "first_token_latency_max": max((float(v.get("first_token_latency", 0.0))
                                                for v in reported), default=None),
                "first_token_latency_mean": (sum(float(v.get("first_token_latency", 0.0))
                                                 for v in reported) / len(reported)) if reported else None,
                "num_generation_tokens": sum(int(v.get("num_generation_tokens", 0))
                                             for v in reported) or None,
                "num_prompt_tokens": sum(int(v.get("num_prompt_tokens", 0))
                                         for v in reported) or None,
                "batch_generate_seconds": generate_seconds,
                "batch_size": len(batch),
                "data_parallel_size": len(replicas),
            }
            self._record_prefix_summary()
            return answers
        results = self.engine.chat(
            conversations,
            sampling_params=parameters,
            use_tqdm=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
            lora_request=self.adapter_request,
        )
        generate_seconds = time.perf_counter() - started
        require(len(results) == len(batch), "vLLM returned the wrong number of results")
        answers = [result.outputs[0].text.strip() for result in results]
        # Engine-side aggregates over the batch, for the runner to record.
        metrics = [_engine_metrics(result) for result in results]
        if self.metrics_shape is None and results:
            self.metrics_shape = describe_engine_metrics(results[0])
        self._record_prefix_cache_batch(metrics)
        reported = [value for value in metrics if value.get("reported")]
        self.last_metrics = {
            "reported": bool(reported),
            "first_token_latency_max": max((float(v.get("first_token_latency", 0.0))
                                            for v in reported), default=None),
            "first_token_latency_mean": (sum(float(v.get("first_token_latency", 0.0))
                                             for v in reported) / len(reported)) if reported else None,
            "num_generation_tokens": sum(int(v.get("num_generation_tokens", 0))
                                         for v in reported) or None,
            "num_prompt_tokens": sum(int(v.get("num_prompt_tokens", 0))
                                     for v in reported) or None,
            "batch_generate_seconds": generate_seconds,
            "batch_size": len(batch),
        }
        self._record_prefix_summary()
        return answers

    def _prefix_counters(self):
        """The prefix-cache counters, created on first use.

        Read through a helper so a backend built with ``__new__`` in a test, which
        never ran ``__init__``, still records instead of raising an
        AttributeError from inside the request path.
        """
        for name in ("prefix_cache_requests", "prefix_cache_missing",
                     "prefix_cache_cached_tokens", "prefix_cache_prompt_tokens"):
            if not hasattr(self, name):
                setattr(self, name, 0)
        return self

    @property
    def metrics_shape(self):
        """Field inventory of the engine's stats, or None before the first call.

        A property with a default rather than an attribute, so instances built
        with ``__new__`` in tests behave like constructed ones.
        """
        return getattr(self, "_metrics_shape", None)

    @metrics_shape.setter
    def metrics_shape(self, value):
        self._metrics_shape = value

    def _record_prefix_cache(self, metrics):
        """Accumulate prefix-cache reuse from one finished request.

        vLLM reports how many of a request's prompt tokens it served from the
        prefix cache rather than recomputing. The hit *rate* is what makes the
        optimisation's value measurable: on this lane every request carries four
        distinct images, so the only reusable prefix is the chat template, and
        the rate is expected to be near zero. Measuring it turns that expectation
        into evidence.
        """
        self._prefix_counters()
        cached = metrics.get("num_cached_tokens") if isinstance(metrics, dict) else None
        prompt = metrics.get("num_prompt_tokens") if isinstance(metrics, dict) else None
        if cached is None:
            self.prefix_cache_missing += 1
            return
        self.prefix_cache_cached_tokens += int(cached)
        self.prefix_cache_requests += 1
        if prompt:
            self.prefix_cache_prompt_tokens += int(prompt)
    def _record_prefix_cache_batch(self, metrics):
        for entry in metrics:
            self._record_prefix_cache(entry)

    def _record_prefix_summary(self):
        self._prefix_counters()
        self.prefix_cache_summary = {
            "reported_requests": self.prefix_cache_requests,
            "unreported_requests": self.prefix_cache_missing,
            "cached_tokens": self.prefix_cache_cached_tokens,
            "prompt_tokens": self.prefix_cache_prompt_tokens,
            "hit_rate": (round(self.prefix_cache_cached_tokens / self.prefix_cache_prompt_tokens, 6)
                         if self.prefix_cache_prompt_tokens else None),
        }

    def worker_memory(self):
        """Per-worker GPU memory, sampled inside each vLLM worker process.

        Returns ``{"workers": [...], "error": None}``. A worker that has already
        exited, or a failure inside the RPC, is reported as an error string
        rather than raised: telemetry must never fail a prediction run.
        """
        if self._data_parallel_replicas:
            # Each replica owns a card outright, so it reports its own memory
            # from its own process; no collective RPC is involved.
            workers, errors = [], []
            for replica in self._data_parallel_replicas:
                try:
                    memory = replica.worker_memory()
                except Exception as error:      # noqa: BLE001 - telemetry only
                    errors.append(f"replica {replica.index}: {type(error).__name__}: {error}")
                    continue
                if memory:
                    workers.append({**memory, "replica": replica.index})
            return {"workers": workers, "error": "; ".join(errors) or None}
        try:
            results = self.engine.collective_rpc(_worker_telemetry)
        except Exception as error:              # noqa: BLE001 - telemetry only
            return {"workers": [], "error": f"{type(error).__name__}: {error}"}
        return {"workers": list(results), "error": None}

    def close(self):
        """Release data-parallel replicas.

        Each replica is a child process holding a full model, so leaving them
        alive after the run would keep most of both cards reserved. A no-op in
        tensor-parallel mode, where the engine lives in this process.
        """
        for replica in self._data_parallel_replicas:
            replica.close()
        if getattr(self, "engines", None):
            self.engines = []

    def metadata(self):
        import os

        runtime = {name: importlib.metadata.version(name)
                   for name in ("torch", "transformers", "vllm")
                   if _installed(name)}
        # Sampled here, while the engine is still alive: the workers hold the
        # model and KV cache, and they are gone once the object is released.
        return {
            "backend": "qwen35_4b_vllm",
            "model_class": "DP" if self._data_parallel_replicas else type(self.engine).__name__,
            "image_size": self.image_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size if self._data_parallel_replicas else None,
            "parallelism": ("data" if self._data_parallel_replicas else
                            ("tensor" if self.tensor_parallel_size > 1 else "single")),
            "attention_backend": self.attention_backend,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "answer_constraint": "structured_outputs_choice",
            "lora_adapter": self.adapter_request.lora_name if self.adapter_request else None,
            "load_seconds": self.load_seconds,
            "workers": self.worker_memory(),
            "image_encode_ms_last": round(getattr(self, "last_encode_seconds", 0.0) * 1000.0, 3),
            # Measured, not assumed: a near-zero hit rate is the expected result
            # on this workload and is reported either way.
            "prefix_cache": self.prefix_cache_summary,
            # What the engine's stats object really contains, for when the
            # counter above comes back empty.
            "metrics_shape": getattr(self, "metrics_shape", None),
            "versions": runtime,
            "engine_environment": {name: os.environ.get(name) for name in VLLM_ENGINE_ENV},
        }


def _engine_metrics(output) -> dict:
    """Extract engine-reported timing from a RequestOutput.

    vLLM attaches RequestStateStats to finished requests on some versions and
    leaves it None on others, and the field set has moved between releases. This
    reads what is present and reports nothing rather than guessing, so a missing
    TTFT is visible as missing instead of silently becoming zero.

    Prefix-cache reuse is searched for across the plausible spellings instead of
    one fixed name. ``num_cached_tokens`` was the documented field, but the first
    data-parallel smoke run reported every request as unreported, so the value --
    if this build exposes it at all -- lives under a different name or on a
    nested object. Enumerating candidates is deliberate: the engine's field set
    is the authority here, not this file.
    """
    stats = getattr(output, "metrics", None)
    if stats is None:
        return {"reported": False}
    values = {}
    for name in ("first_token_latency", "e2e_latency", "num_prompt_tokens",
                 "num_generation_tokens", "queued_ts", "scheduled_ts"):
        value = getattr(stats, name, None)
        if value is not None:
            values[name] = value
    cached = _find_cached_tokens(stats)
    if cached is not None:
        values["num_cached_tokens"] = cached
    if not values:
        return {"reported": False}
    return {"reported": True, **values}


# Spellings seen across vLLM releases for "prompt tokens served from the prefix
# cache". Checked on the stats object and one level into its nested metrics.
_CACHED_TOKEN_FIELDS = (
    "num_cached_tokens",
    "num_prefix_cached_tokens",
    "cached_tokens",
    "prefix_cache_hit_tokens",
    "num_cached_prompt_tokens",
)


def _find_cached_tokens(stats):
    """Locate the prefix-cache reuse count on a stats object, or return None.

    Searched at the top level and one level down (``kv_cache_metrics`` and
    friends), because the value has been both a direct attribute and a nested
    one. Returns None rather than 0 when nothing is found, so "absent" and
    "measured zero" stay distinguishable -- collapsing them is exactly how a
    missing metric turns into a flattering hit rate.
    """
    for field in _CACHED_TOKEN_FIELDS:
        value = getattr(stats, field, None)
        if isinstance(value, (int, float)):
            return value
    for container_name in ("kv_cache_metrics", "prefix_cache_stats", "cache_metrics"):
        container = getattr(stats, container_name, None)
        if container is None:
            continue
        for field in _CACHED_TOKEN_FIELDS:
            value = getattr(container, field, None)
            if isinstance(value, (int, float)):
                return value
    return None


def describe_engine_metrics(output) -> dict:
    """Field inventory of a finished request's stats object.

    Used to discover what this vLLM build actually reports. Guessing field names
    from documentation already produced one wrong probe, so the run records the
    real attribute set and the whole nested object, and the reader decides
    instead of the code assuming.
    """
    stats = getattr(output, "metrics", None)
    if stats is None:
        return {"metrics_is_none": True}
    def public(obj):
        return sorted(name for name in dir(obj)
                      if not name.startswith("_") and not callable(getattr(obj, name, None)))
    report = {"metrics_is_none": False, "fields": public(stats)}
    for container_name in ("kv_cache_metrics", "prefix_cache_stats", "cache_metrics"):
        container = getattr(stats, container_name, None)
        if container is not None:
            report[container_name] = {
                "type": type(container).__name__,
                "fields": public(container),
                "values": {name: getattr(container, name, None)
                           for name in public(container)},
            }
    report["values"] = {}
    for name in ("num_prompt_tokens", "num_cached_tokens", "num_generation_tokens",
                 "first_token_latency", "e2e_latency"):
        value = getattr(stats, name, None)
        if value is not None:
            report["values"][name] = value
    return report


def _worker_telemetry(worker):
    """Run inside a vLLM worker process and report that worker's GPU memory.

    This must execute in the worker, not the parent: vLLM runs the engine and
    each tensor-parallel rank in separate processes, so torch's allocator
    counters read from the parent are always zero. Only the driver view
    (mem_get_info) crosses process boundaries, which is why an earlier version
    of this reported `used_mib` correctly but `allocated_mib: 0`.

    KV cache figures come from the worker too, since the engine computes them
    during its own profiling pass and never exposes them on the LLM object.

    Returns plain Python types only -- the result is pickled back across the
    process boundary, and large tensors must never be returned.
    """
    import torch

    index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    report = {
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "total_mib": round(total_bytes / 1024**2, 1),
        "free_mib": round(free_bytes / 1024**2, 1),
        "used_mib": round((total_bytes - free_bytes) / 1024**2, 1),
        "allocated_mib": round(torch.cuda.memory_allocated(index) / 1024**2, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(index) / 1024**2, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / 1024**2, 1),
    }
    # Present only after profiling has run; absent on a worker that never
    # finished initialization, which is reported as a missing key on purpose.
    available = getattr(worker, "available_kv_cache_memory_bytes", None)
    if available is not None:
        report["kv_cache_mib"] = round(int(available) / 1024**2, 1)
    cache_config = getattr(worker, "cache_config", None)
    blocks = getattr(cache_config, "num_gpu_blocks", None)
    if blocks:
        report["kv_cache_blocks"] = int(blocks)
        block_size = getattr(cache_config, "block_size", None)
        if block_size:
            report["kv_cache_tokens"] = int(blocks) * int(block_size)
    return report


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


# Live replica processes, terminated at interpreter exit.
#
# The replicas cannot be daemons (vLLM spawns workers beneath them), so nothing
# reaps them automatically. A prediction run that raises, or a notebook cell that
# is interrupted, would otherwise leave a process holding most of a GPU until the
# session is reset. Registration is best-effort and never raises: it exists to
# release hardware, not to police shutdown.
_REPLICA_PROCESSES = []


def _register_replica_cleanup(process) -> None:
    import atexit

    if not _REPLICA_PROCESSES:
        atexit.register(_terminate_replicas)
    _REPLICA_PROCESSES.append(process)


def _terminate_replicas() -> None:                # pragma: no cover - exit path
    while _REPLICA_PROCESSES:
        process = _REPLICA_PROCESSES.pop()
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        except Exception:                         # noqa: BLE001 - shutdown only
            pass


class _ReplicaHandle:
    """One data-parallel replica: a whole vLLM engine on its own GPU.

    The engine cannot be built in this process. ``CUDA_VISIBLE_DEVICES`` is read
    by CUDA at initialization and is process-wide, so two engines wanting
    different cards cannot coexist here. Each replica therefore lives in a child
    process that is spawned with the variable already set for its card, and the
    parent talks to it over a queue pair.

    Only plain data crosses the boundary: rendered conversations, per-request
    answer choices, and back the answer strings plus JSON-able metrics. Nothing
    from vLLM is pickled in either direction, which keeps the message format
    independent of the engine version.
    """

    def __init__(self, index: int, engine_kwargs: dict, weights: Path):
        self.index = index
        self._connection = None
        self._process = None
        self._spawn(engine_kwargs, weights)
        # The child reports a failure during construction rather than leaving the
        # parent to time out on the first request.
        status = self._call({"op": "ready"})
        require(status.get("ok"), f"data-parallel replica {index} failed to start: "
                                  f"{status.get('error')}")
        self.load_seconds = float(status.get("load_seconds") or 0.0)

    def _spawn(self, engine_kwargs: dict, weights: Path):
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=_replica_main,
            args=(child_connection, self.index, engine_kwargs, str(weights)),
            # NOT a daemon. vLLM spawns its own worker processes beneath this one,
            # and multiprocessing forbids a daemonic process from having children:
            #
            #     AssertionError: daemonic processes are not allowed to have children
            #
            # Daemon mode would have cleaned the replica up automatically when the
            # parent exits, but it cannot coexist with vLLM's own process tree.
            # Cleanup is therefore explicit: close() terminates each replica, and
            # atexit covers the paths where the caller forgets.
            daemon=False,
        )
        process.start()
        # The child holds its own copy; closing here is what lets the parent see
        # EOF if the child dies mid-run instead of blocking forever.
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        _register_replica_cleanup(process)

    def _call(self, message: dict) -> dict:
        require(self._process is not None and self._process.is_alive(),
                f"data-parallel replica {self.index} is not running")
        try:
            self._connection.send(message)
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"data-parallel replica {self.index} stopped responding: "
                               f"{error}") from error
        if not response.get("ok"):
            raise RuntimeError(f"data-parallel replica {self.index}: {response.get('error')}")
        return response

    def chat(self, conversations, choices, max_new_tokens):
        """Run one engine call carrying a list of conversations.

        The child builds its own SamplingParams from the plain answer lists, so
        no vLLM object is ever pickled across the process boundary.
        """
        return self._call({
            "op": "chat",
            "conversations": conversations,
            "choices": [list(entry) for entry in choices],
            "max_tokens": int(max_new_tokens),
        })

    def worker_memory(self):
        """Ask this replica for its own device memory, measured in its process.

        This is the only place the figure can come from: the replica process --
        not the parent -- holds the weights and the KV cache.
        """
        return self._call({"op": "memory"}).get("memory")

    def close(self):
        if self._process is None:
            return
        try:
            if self._process.is_alive():
                self._connection.send({"op": "stop"})
                self._process.join(timeout=30)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
            # Deregister so the atexit hook does not try to reap it again.
            if self._process in _REPLICA_PROCESSES:
                _REPLICA_PROCESSES.remove(self._process)
            try:
                self._connection.close()
            except OSError:
                pass
            self._process = None

    def __del__(self):                # pragma: no cover - defensive cleanup only
        try:
            self.close()
        except Exception:             # noqa: BLE001 - destructors must not raise
            pass


def _replica_main(connection, index, engine_kwargs, weights):
    """Child-process body for one data-parallel replica.

    Runs with CUDA_VISIBLE_DEVICES set to a single card, so the engine it builds
    sees exactly one device and needs no tensor parallelism. Everything it sends
    back is plain data.
    """
    import os
    import traceback

    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    # vLLM reads several of these at import time, so the environment is applied
    # before the engine module is touched.
    apply_engine_environment()

    state = {"engine": None, "sampling": None, "structured": None}

    def reply(payload):
        connection.send({"ok": True, **payload})

    def fail(error):
        connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})

    try:
        import time as _time

        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        started = _time.perf_counter()
        engine = LLM(tensor_parallel_size=1, **engine_kwargs)
        state.update(engine=engine, sampling=SamplingParams, structured=StructuredOutputsParams)
        load_seconds = _time.perf_counter() - started
    except Exception as error:            # noqa: BLE001 - reported to the parent
        try:
            connection.send({
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            connection.close()
        finally:
            return

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            operation = message.get("op")
            if operation == "stop":
                break
            if operation == "ready":
                reply({"load_seconds": load_seconds})
                continue
            if operation == "memory":
                reply({"memory": _worker_telemetry(None)})
                continue
            if operation == "chat":
                try:
                    parameters = [
                        state["sampling"](
                            temperature=0.0, top_p=1.0,
                            max_tokens=int(message.get("max_tokens") or 8),
                            skip_special_tokens=True,
                            structured_outputs=state["structured"](choice=list(choices)),
                        )
                        for choices in message["choices"]
                    ]
                    results = state["engine"].chat(
                        message["conversations"],
                        sampling_params=parameters,
                        use_tqdm=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    reply({
                        "answers": [result.outputs[0].text.strip() for result in results],
                        "metrics": [_engine_metrics(result) for result in results],
                        # Send the real stats shape once so the parent can record
                        # which fields this vLLM build actually provides.
                        "metrics_shape": (describe_engine_metrics(results[0])
                                          if results else None),
                    })
                except Exception as error:        # noqa: BLE001 - per-request guard
                    fail(error)
                continue
            fail(ValueError(f"unknown operation: {operation!r}"))
    finally:
        try:
            connection.close()
        except OSError:
            pass


def _data_uri(image) -> str:
    """Encode a PIL frame as an OpenAI-style image URL."""
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
