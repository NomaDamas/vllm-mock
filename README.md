# vllm-mock

Provide a mock instance to test vLLM without CUDA or any GPUs.

- **Github repository**: <https://github.com/vkehfdl1/vllm-mock/>

# Features

- `vllm.LLM.generate` mock
- `vllm.LLM.chat` mock

# Usage

It is highly recommended to use the mock instance with `pytest-mock`.

```python
from vllm_mock import LLM
from vllm import SamplingParams

def test_vllm(mocker):
    mock_class = mocker.patch("vllm.LLM")
    mock_class.return_value = LLM(model="mock-model")

    llm = mock_class()
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, logprobs=1)
    response = llm.generate("Hello, world!", sampling_params=sampling_params)
    assert isinstance(response[0].outputs[0].text, str)

    chat_response = llm.chat([
		{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ], sampling_params=sampling_params)
    assert isinstance(chat_response[0].outputs[0].text, str)
```

# Installation

```bash
pip install vllm-mock pytest-mock
```

or in a UV environment

```bash
uv add --dev vllm-mock pytest-mock
```

## To-do List

- [ ] Mock vLLM API server
- [ ] Mock Reasoning model features
- [ ] Mock quantization features
- [ ] Mock LoRA features
- [ ] vLM models mock
- [ ] `vllm.LLM.beam_search` mock
- [ ] `vllm.LLM.embed` mock
- [ ] `vllm.LLM.classify` mock
- [ ] `vllm.LLM.encode` mock
- [ ] `vllm.LLM.reward` mock

## For Contributors

### 1. Setup Environment

First, clone a repository

```bash
git clone https://github.com/NomaDamas/vllm-mock.git
cd vllm-mock
```

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 2. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

You can create any issue or PR to support this project. Thank you!


## Builder of this repository

- [Jeffrey](https://github.com/vkehfdl1) is a creator of this repo. Made this because he desperately needed it for his research.
- [NomaDamas](https://github.com/NomaDamas) is an AI open-source Hacker House in Seoul, Korea.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
