"""
The outputs for vLLM mock.
"""

from vllm.outputs import CompletionOutput, RequestOutput

default_completion_output = CompletionOutput(
    index=0,
    text="This is a mock completion.",
    token_ids=[5, 6, 7, 8],
    cumulative_logprob=None,
    logprobs=None,
)

default_reqeust_output = RequestOutput(
    request_id="",
    prompt="This is a mock prompt.",
    prompt_token_ids=[1, 2, 3, 4],
    prompt_logprobs=None,
    outputs=[default_completion_output],
    finished=True,
)


def get_default_request_output(request_id: str, output_cnt: int) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="This is a mock prompt.",
        prompt_token_ids=[1, 2, 3, 4],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=i,
                text=f"This is mock completion {i}.",
                token_ids=[5 + i, 6 + i, 7 + i, 8 + i],
                cumulative_logprob=None,
                logprobs=None,
            )
            for i in range(output_cnt)
        ],
        finished=True,
    )
