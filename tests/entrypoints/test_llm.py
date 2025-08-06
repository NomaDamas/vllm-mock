import pytest
from vllm.sampling_params import SamplingParams

from vllm_mock import LLM


@pytest.fixture
def llm_instance():
    """Create a mock LLM instance for testing."""
    return LLM(model="mock-model")


class TestLLMGenerate:
    """Test cases for LLM.generate() method."""

    def test_single_prompt_no_logprobs(self, llm_instance):
        """Test single prompt without logprobs or prompt_logprobs."""
        prompt = "Hello, world!"
        outputs = llm_instance.generate(prompts=prompt)

        assert len(outputs) == 1
        assert outputs[0].prompt == "This is a mock prompt."
        assert outputs[0].prompt_logprobs is None
        assert len(outputs[0].outputs) == 1
        assert outputs[0].outputs[0].logprobs is None

    def test_single_prompt_with_logprobs_only(self, llm_instance):
        """Test single prompt with logprobs but no prompt_logprobs."""
        prompt = "Hello, world!"
        sampling_params = SamplingParams(logprobs=3)
        outputs = llm_instance.generate(prompts=prompt, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert outputs[0].prompt_logprobs is None
        assert outputs[0].outputs[0].logprobs is not None
        assert len(outputs[0].outputs[0].logprobs) == 4  # mock_completion_token_ids length
        assert len(list(outputs[0].outputs[0].logprobs[0].keys())) == 3

    def test_single_prompt_with_prompt_logprobs_only(self, llm_instance):
        """Test single prompt with prompt_logprobs but no logprobs."""
        prompt = "Hello, world!"
        sampling_params = SamplingParams(prompt_logprobs=2)
        outputs = llm_instance.generate(prompts=prompt, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert outputs[0].prompt_logprobs is not None
        assert len(outputs[0].prompt_logprobs) == 4  # mock_prompt_token_ids length
        assert outputs[0].outputs[0].logprobs is None
        assert len(list(outputs[0].prompt_logprobs[0].keys())) == 2

    def test_single_prompt_with_both_logprobs(self, llm_instance):
        """Test single prompt with both logprobs and prompt_logprobs."""
        prompt = "Hello, world!"
        sampling_params = SamplingParams(logprobs=3, prompt_logprobs=2)
        outputs = llm_instance.generate(prompts=prompt, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert outputs[0].prompt_logprobs is not None
        assert len(outputs[0].prompt_logprobs) == 4  # mock_prompt_token_ids length
        assert len(list(outputs[0].prompt_logprobs[0].keys())) == 2
        assert outputs[0].outputs[0].logprobs is not None
        assert len(outputs[0].outputs[0].logprobs) == 4  # mock_completion_token_ids length
        assert len(list(outputs[0].outputs[0].logprobs[0].keys())) == 3

    def test_multiple_prompts_no_logprobs(self, llm_instance):
        """Test multiple prompts without logprobs or prompt_logprobs."""
        prompts = ["Hello, world!", "How are you?", "Goodbye!"]
        outputs = llm_instance.generate(prompts=prompts)

        assert len(outputs) == 3
        for i, output in enumerate(outputs):
            assert output.request_id == str(i)
            assert output.prompt_logprobs is None
            assert len(output.outputs) == 1
            assert output.outputs[0].logprobs is None

    def test_multiple_prompts_with_logprobs_only(self, llm_instance):
        """Test multiple prompts with logprobs but no prompt_logprobs."""
        prompts = ["Hello, world!", "How are you?"]
        sampling_params = SamplingParams(logprobs=5)
        outputs = llm_instance.generate(prompts=prompts, sampling_params=sampling_params)

        assert len(outputs) == 2
        for output in outputs:
            assert output.prompt_logprobs is None
            assert output.outputs[0].logprobs is not None
            assert len(output.outputs[0].logprobs) == 4
            assert len(list(output.outputs[0].logprobs[0].keys())) == 5

    def test_multiple_prompts_with_prompt_logprobs_only(self, llm_instance):
        """Test multiple prompts with prompt_logprobs but no logprobs."""
        prompts = ["Hello, world!", "How are you?"]
        sampling_params = SamplingParams(prompt_logprobs=4)
        outputs = llm_instance.generate(prompts=prompts, sampling_params=sampling_params)

        assert len(outputs) == 2
        for output in outputs:
            assert output.prompt_logprobs is not None
            assert len(output.prompt_logprobs) == 4
            assert output.outputs[0].logprobs is None
            assert len(list(output.prompt_logprobs[0].keys())) == 4

    def test_multiple_prompts_with_both_logprobs(self, llm_instance):
        """Test multiple prompts with both logprobs and prompt_logprobs."""
        prompts = ["Hello, world!", "How are you?", "What's up?"]
        sampling_params = SamplingParams(logprobs=2, prompt_logprobs=3)
        outputs = llm_instance.generate(prompts=prompts, sampling_params=sampling_params)

        assert len(outputs) == 3
        for output in outputs:
            assert output.prompt_logprobs is not None
            assert len(output.prompt_logprobs) == 4
            assert output.outputs[0].logprobs is not None
            assert len(output.outputs[0].logprobs) == 4
            assert len(list(output.prompt_logprobs[0].keys())) == 3
            assert len(list(output.outputs[0].logprobs[0].keys())) == 2

    def test_multiple_prompts_with_sequence_sampling_params(self, llm_instance):
        """Test multiple prompts with different sampling parameters for each."""
        prompts = ["Hello, world!", "How are you?", "Goodbye!"]
        sampling_params = [
            SamplingParams(logprobs=2),
            SamplingParams(prompt_logprobs=3),
            SamplingParams(logprobs=1, prompt_logprobs=2),
        ]
        outputs = llm_instance.generate(prompts=prompts, sampling_params=sampling_params)

        assert len(outputs) == 3

        # First prompt: only logprobs
        assert outputs[0].prompt_logprobs is None
        assert outputs[0].outputs[0].logprobs is not None

        # Second prompt: only prompt_logprobs
        assert outputs[1].prompt_logprobs is not None
        assert outputs[1].outputs[0].logprobs is None

        # Third prompt: both logprobs
        assert outputs[2].prompt_logprobs is not None
        assert outputs[2].outputs[0].logprobs is not None

    def test_multiple_outputs_per_prompt(self, llm_instance):
        """Test generating multiple outputs per prompt using sampling_params.n."""
        prompt = "Hello, world!"
        sampling_params = SamplingParams(n=3, logprobs=2)
        outputs = llm_instance.generate(prompts=prompt, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 3
        for completion in outputs[0].outputs:
            assert completion.logprobs is not None

    def test_deprecated_prompt_token_ids_error(self, llm_instance):
        """Test that using deprecated prompt_token_ids raises ValueError."""
        with pytest.raises(ValueError, match="deprecated"):
            llm_instance.generate(prompt_token_ids=[1, 2, 3, 4])

    def test_none_prompts_error(self, llm_instance):
        """Test that None prompts raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            llm_instance.generate(prompts=None)

    def test_mismatched_sampling_params_length_error(self, llm_instance):
        """Test error when sampling_params length doesn't match prompts length."""
        prompts = ["Hello", "World"]
        sampling_params = [SamplingParams()]  # Only one param for two prompts

        with pytest.raises(ValueError, match="must match the number of prompts"):
            llm_instance.generate(prompts=prompts, sampling_params=sampling_params)
