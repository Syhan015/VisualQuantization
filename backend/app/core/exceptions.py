class VQException(Exception):
    """Base exception for VisualQuantization"""

    pass


class ModelNotFoundError(VQException):
    """Raised when model file is not found"""

    pass


class DiffResultNotFoundError(VQException):
    """Raised when diff result is not found"""

    pass


class ONNXParseError(VQException):
    """Raised when ONNX model parsing fails"""

    pass
