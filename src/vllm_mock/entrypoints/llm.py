from collections.abc import Sequence
from collections.abc import Sequence as ABCSequence
from typing import Any, Callable, Optional, Union

from tqdm.auto import tqdm
from vllm import PromptType, RequestOutput
from vllm.config import HfOverrides, ModelDType, PoolerConfig, TaskOption, TokenizerMode
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest, LLMGuidedOptions
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.sampling_params import SamplingParams

from vllm_mock.outputs import get_request_output


class LLM:
    def __init__(
        self,
        model: str,
        *,
        task: TaskOption = "auto",
        tokenizer: Optional[str] = None,
        tokenizer_mode: TokenizerMode = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: Optional[QuantizationMethods] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.task = task
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.skip_tokenizer_init = skip_tokenizer_init
        self.trust_remote_code = trust_remote_code
        self.allowed_local_media_path = allowed_local_media_path
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.quantization = quantization
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.cpu_offload_gb = cpu_offload_gb
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.disable_async_output_proc = disable_async_output_proc
        self.hf_token = hf_token
        self.hf_overrides = hf_overrides
        self.mm_processor_kwargs = mm_processor_kwargs
        self.override_pooler_config = override_pooler_config
        self.compilation_config = compilation_config

    def generate(  # noqa: C901
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]], Optional[Union[str, list[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams, Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions, GuidedDecodingRequest]] = None,
        priority: Optional[list[int]] = None,
    ) -> list[RequestOutput]:
        # Check if prompts is a sequence
        if prompt_token_ids is not None:
            raise ValueError(
                "The 'prompt_token_ids' argument is deprecated and will be removed in a future version. Please use 'prompts' instead."
            )

        if prompts is not None:
            if not isinstance(prompts, ABCSequence) or isinstance(prompts, (str, bytes)):
                prompts = [prompts]  # Convert to a list
            input_prompt_length = len(prompts)
        else:
            raise ValueError(
                "The 'prompts' argument cannot be None. Please provide a valid prompt or a sequence of prompts."
            )

        n_list = [1] * input_prompt_length
        logprobs = [None] * input_prompt_length
        prompt_logprobs = [None] * input_prompt_length
        if sampling_params is not None:
            if isinstance(sampling_params, SamplingParams):
                # not a sequence
                n_list = [sampling_params.n] * input_prompt_length
                logprobs = [sampling_params.logprobs] * input_prompt_length
                prompt_logprobs = [sampling_params.prompt_logprobs] * input_prompt_length
            else:
                if input_prompt_length != len(sampling_params):
                    raise ValueError("The length of 'sampling_params' must match the number of prompts provided.")
                # a sequence
                n_list = [sp.n for sp in sampling_params]
                logprobs = [sp.logprobs for sp in sampling_params]
                prompt_logprobs = [sp.prompt_logprobs for sp in sampling_params]

        if lora_request is None and guided_options_request is None:
            # The normal situation
            request_outputs = []
            for i, (n, logprob, prompt_logprob) in enumerate(zip(n_list, logprobs, prompt_logprobs)):
                request_outputs.append(get_request_output(str(i), n, logprob, prompt_logprob))
            return request_outputs
        elif lora_request is not None and guided_options_request is None:
            raise NotImplementedError  # TODO: Implement LoRARequest handling
        elif lora_request is None and guided_options_request is not None:
            raise NotImplementedError  # TODO: Implement GuidedDecodingRequest handling
        else:
            raise NotImplementedError  # TODO: Implement both LoRARequest and GuidedDecodingRequest handling
