import photometrypipeline


def test_version():
    """Check to see that we can get the package version"""
    assert photometrypipeline.__version__ is not None
