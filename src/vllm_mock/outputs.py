"""
The outputs for vLLM mock.
"""

from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.outputs import CompletionOutput, RequestOutput


def get_request_output(
    request_id: str, output_cnt: int, logprobs: int | None = None, prompt_logprobs: int | None = None
) -> RequestOutput:
    mock_prompt_token_ids = [1, 2, 3, 4]
    mock_completion_token_ids = [5, 6, 7, 8]

    # Make prompt_logprobs
    prompt_logprobs_instance = None
    if prompt_logprobs is not None:
        prompt_logprobs_instance = make_prompt_logprobs(prompt_logprobs, mock_prompt_token_ids)

    logprobs_instance = None
    if logprobs is not None:
        logprobs_instance = make_logprobs(logprobs, mock_completion_token_ids)

    return RequestOutput(
        request_id=request_id,
        prompt="This is a mock prompt.",
        prompt_token_ids=mock_prompt_token_ids,
        prompt_logprobs=prompt_logprobs_instance,
        outputs=[
            CompletionOutput(
                index=i,
                text=f"This is mock completion {i}.",
                token_ids=mock_completion_token_ids,
                cumulative_logprob=None,
                logprobs=logprobs_instance,
            )
            for i in range(output_cnt)
        ],
        finished=True,
    )


def make_prompt_logprobs(prompt_logprobs_cnt: int, prompt_token_ids: list[int]) -> PromptLogprobs:
    return [
        {
            token + k: Logprob(
                logprob=-1.0,
                rank=k,
                decoded_token=f"token_{k + token}",
            )
            for k in range(prompt_logprobs_cnt)
        }
        for token in prompt_token_ids
    ]


def make_logprobs(logprobs_cnt: int, generated_token_ids: list[int]) -> SampleLogprobs:
    return [
        {
            k + token: Logprob(
                logprob=-1.0,
                rank=k,
                decoded_token=f"token_{k + token}",
            )
            for k in range(logprobs_cnt)
        }
        for token in generated_token_ids
    ]
