import llmbench


def test_package_version_is_defined() -> None:
    assert llmbench.__version__ == "0.1.0"
