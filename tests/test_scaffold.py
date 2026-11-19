import llmprobe


def test_package_version_is_defined() -> None:
    assert llmprobe.__version__ == "0.1.0"
