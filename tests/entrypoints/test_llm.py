import pytest
from vllm.sampling_params import SamplingParams

from vllm_mock import LLM


@pytest.fixture
def llm_instance():
    """Create a mock LLM instance for testing."""
    return LLM(model="mock-model")


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def multiple_chat_conversations():
    """Multiple chat conversations for testing."""
    return [
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
        [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm good!"}],
    ]


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
        for _i, output in enumerate(outputs):
            assert output.request_id == str(_i)
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


class TestLLMChat:
    """Test cases for LLM.chat() method."""

    def test_single_conversation_basic(self, llm_instance, sample_chat_messages):
        """Test basic single conversation chat."""
        outputs = llm_instance.chat(messages=sample_chat_messages)

        assert len(outputs) == 1
        assert outputs[0].prompt == "This is a mock prompt."
        assert len(outputs[0].outputs) == 1

    def test_single_conversation_with_sampling_params(self, llm_instance, sample_chat_messages):
        """Test single conversation with sampling parameters."""
        sampling_params = SamplingParams(logprobs=3, n=2)
        outputs = llm_instance.chat(messages=sample_chat_messages, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 2
        for completion in outputs[0].outputs:
            assert completion.logprobs is not None

    def test_multiple_conversations_basic(self, llm_instance, multiple_chat_conversations):
        """Test basic multiple conversations chat."""
        outputs = llm_instance.chat(messages=multiple_chat_conversations)

        assert len(outputs) == 2
        for _i, output in enumerate(outputs):
            assert output.request_id is not None
            assert len(output.outputs) == 1

    def test_multiple_conversations_with_sampling_params_list(self, llm_instance, multiple_chat_conversations):
        """Test multiple conversations with different sampling parameters."""
        sampling_params = [SamplingParams(logprobs=2), SamplingParams(prompt_logprobs=3)]
        outputs = llm_instance.chat(messages=multiple_chat_conversations, sampling_params=sampling_params)

        assert len(outputs) == 2
        assert outputs[0].outputs[0].logprobs is not None
        assert outputs[0].prompt_logprobs is None
        assert outputs[1].outputs[0].logprobs is None
        assert outputs[1].prompt_logprobs is not None

    def test_chat_with_optional_parameters(self, llm_instance, sample_chat_messages):
        """Test chat with various optional parameters."""
        chat_template_kwargs = {"enable_thinking": True}
        outputs = llm_instance.chat(
            messages=sample_chat_messages,
            chat_template="custom_template",
            chat_template_content_format="openai",
            add_generation_prompt=True,
            continue_final_message=False,
            tools=[{"type": "function", "name": "test_tool"}],
            chat_template_kwargs=chat_template_kwargs,
            mm_processor_kwargs={"key": "value"},
        )

        assert len(outputs) == 1

    def test_chat_invalid_messages_not_sequence(self, llm_instance):
        """Test error when messages is not a proper sequence."""
        with pytest.raises(TypeError, match="must be a sequence"):
            llm_instance.chat(messages="invalid string message")

    def test_chat_invalid_sampling_params_length(self, llm_instance, multiple_chat_conversations):
        """Test error when sampling_params length doesn't match conversations."""
        sampling_params = [SamplingParams()]  # Only one param for two conversations

        with pytest.raises(ValueError, match="must match the number of message sequences"):
            llm_instance.chat(messages=multiple_chat_conversations, sampling_params=sampling_params)

    def test_chat_invalid_content_format(self, llm_instance, sample_chat_messages):
        """Test error with invalid chat_template_content_format."""
        with pytest.raises(ValueError, match="Invalid 'chat_template_content_format'"):
            llm_instance.chat(messages=sample_chat_messages, chat_template_content_format="invalid_format")

    def test_chat_conflicting_generation_params(self, llm_instance, sample_chat_messages):
        """Test error when add_generation_prompt and continue_final_message are both True."""
        with pytest.raises(ValueError, match="cannot be True when 'add_generation_prompt' is True"):
            llm_instance.chat(messages=sample_chat_messages, add_generation_prompt=True, continue_final_message=True)

    def test_chat_missing_role_key(self, llm_instance):
        """Test error when message is missing 'role' key."""
        invalid_messages = [
            {"content": "Hello"},  # Missing 'role'
            {"role": "user", "content": "How are you?"},
        ]

        with pytest.raises(ValueError, match="missing the 'role' key"):
            llm_instance.chat(messages=invalid_messages)

    def test_chat_missing_content_key(self, llm_instance):
        """Test error when message is missing 'content' key."""
        invalid_messages = [
            {"role": "user"},  # Missing 'content'
            {"role": "assistant", "content": "I'm good!"},
        ]

        with pytest.raises(ValueError, match="missing the 'content' key"):
            llm_instance.chat(messages=invalid_messages)

    def test_chat_missing_enable_thinking_warning(self, llm_instance, sample_chat_messages):
        """Test UserWarning when enable_thinking is not in chat_template_kwargs."""
        chat_template_kwargs = {"other_key": "value"}  # Missing 'enable_thinking'

        with pytest.warns(UserWarning, match="If you are using a reasoning model"):
            llm_instance.chat(messages=sample_chat_messages, chat_template_kwargs=chat_template_kwargs)

    def test_chat_with_use_tqdm_false(self, llm_instance, sample_chat_messages):
        """Test chat with use_tqdm set to False."""
        outputs = llm_instance.chat(messages=sample_chat_messages, use_tqdm=False)

        assert len(outputs) == 1

    def test_chat_with_lora_request(self, llm_instance, sample_chat_messages):
        """Test chat with LoRA request (should work with current implementation)."""
        # Note: This will use the generate method which currently raises NotImplementedError for LoRA
        # But the chat method should handle the basic flow
        try:
            outputs = llm_instance.chat(
                messages=sample_chat_messages,
                lora_request=None,  # Keeping it None for now since LoRA is not implemented
            )
            assert len(outputs) == 1
        except NotImplementedError:
            pytest.skip("LoRA request handling not yet implemented")

    def test_chat_complex_conversation_flow(self, llm_instance):
        """Test a more complex conversation with multiple turns."""
        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, I need help with Python."},
            {"role": "assistant", "content": "I'd be happy to help you with Python! What specific topic?"},
            {"role": "user", "content": "How do I create a list?"},
            {"role": "assistant", "content": "You can create a list using square brackets: my_list = []"},
            {"role": "user", "content": "Thanks! How about adding items?"},
        ]

        sampling_params = SamplingParams(logprobs=2, prompt_logprobs=1)
        outputs = llm_instance.chat(messages=complex_messages, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert outputs[0].outputs[0].logprobs is not None
        assert outputs[0].prompt_logprobs is not None
