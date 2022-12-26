class InvalidSampleError(ValueError):
    """Exception raised when a sample is not valid"""


class OutOfBoundsError(ValueError):
    """Exception raised when a value is out of bounds"""


class BadSampleRow(ValueError):
    """Exception raised when a sample row is not valid"""
