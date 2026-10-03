"""Custom exceptions raised by EasyPyML."""


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` has been called.

    It inherits from both ``ValueError`` and ``AttributeError`` (like
    scikit-learn's), so existing ``except ValueError`` or
    ``except AttributeError`` blocks keep catching it.
    """
