import logging

from observability.logging import configure_logging, get_logger


def test_configure_logging() -> None:
    configure_logging()

    logger = get_logger("test")

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.NOTSET