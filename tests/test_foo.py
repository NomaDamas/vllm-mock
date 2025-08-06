from vllm_mock.foo import foo


def test_foo():
    assert foo("foo") == "foo"
