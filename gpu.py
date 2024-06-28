import torch
import numpy as np
import cupy as cp


class ArrayConverter(object):
    """A utility class for converting arrays between NumPy, PyTorch, and CuPy formats.
    
    This class provides methods to convert arrays to and from NumPy, PyTorch, and CuPy,
    allowing for easy manipulation of array data across these popular libraries. It also
    retains the original array type for convenient reversion to its initial state.

    Parameters
    ----------
    array : numpy.ndarray, torch.Tensor, or cupy.ndarray
        The array to be converted between different formats. This can be an array from
        NumPy, PyTorch, or CuPy.

    Attributes
    ----------
    array : numpy.ndarray, torch.Tensor, or cupy.ndarray
        The current state of the array after any conversions.
    original_type : str
        The type of the array ('numpy', 'torch', or 'cupy') when the instance was created.

    Methods
    -------
    to_numpy()
        Converts the current array to a NumPy array.
    to_torch(device="cuda")
        Converts the current array to a PyTorch tensor on the specified device.
    to_cupy()
        Converts the current array to a CuPy array.
    to_original()
        Converts the current array back to its original format.
    get_type(array)
        Static method to determine the type of the given array.
    get_conversion_fn(array, to_type)
        Static method to get the appropriate conversion function based on the array type and target type.
    numpy_to_cupy(numpy_array)
        Static method to convert a NumPy array to a CuPy array.
    numpy_to_torch(numpy_array, device="cuda")
        Static method to convert a NumPy array to a PyTorch tensor.
    cupy_to_numpy(cupy_array)
        Static method to convert a CuPy array to a NumPy array.
    cupy_to_torch(cupy_array, device="cuda")
        Static method to convert a CuPy array to a PyTorch tensor.
    torch_to_numpy(torch_tensor)
        Static method to convert a PyTorch tensor to a NumPy array.
    torch_to_cupy(torch_tensor)
        Static method to convert a PyTorch tensor to a CuPy array.
    torch_to_torch(torch_tensor, device="cuda")
        Static method to change the device of a PyTorch tensor.
    """
    def __init__(self, array):
        self.array = array
        self.original_type = ArrayConverter.get_type(array)

    @property
    def type(self):
        return self.get_type(self.array)
    
    @property
    def dtype(self):
        return self.array.dtype

    def to_numpy(self):
        self.array = self.get_conversion_fn(self.array, "numpy")(self.array)
        return self.array

    def to_torch(self, device="cuda"):
        self.array = self.get_conversion_fn(self.array, "torch")(self.array, device=device)
        return self.array

    def to_cupy(self):
        self.array = self.get_conversion_fn(self.array, "cupy")(self.array)
        return self.array

    def to_original(self):
        self.array = self.get_conversion_fn(self.array, self.original_type)(self.array)
        return self.array
    
    @staticmethod
    def get_type(array):
        if isinstance(array, np.ndarray):
            return "numpy"
        
        elif torch.is_tensor(array):
            return "torch"

        elif isinstance(array, cp.ndarray):
            return "cupy"

        else:
            raise NotImplementedError(f"Unsupported type {type(array)}!")
            
    @staticmethod
    def get_conversion_fn(array, to_type):
        from_type = ArrayConverter.get_type(array)
        if from_type != "torch" and from_type == to_type:
            return lambda x: x
        
        conversion_fn_name = f"{from_type}_to_{to_type}"
        if hasattr(ArrayConverter, conversion_fn_name):
            return getattr(ArrayConverter, conversion_fn_name)

        else:
            raise NotImplementedError(f"The conversion {conversion_fn_name} has not been implemented yet!")
    
    @staticmethod
    def numpy_to_cupy(numpy_array):
        return cp.asarray(numpy_array)

    @staticmethod
    def numpy_to_torch(numpy_array, device="cuda"):
        assert device in ["cpu", "cuda"], "Please choose device from 'cpu' or 'cuda'!"
        return torch.from_numpy(numpy_array).to(device)

    @staticmethod
    def cupy_to_numpy(cupy_array):
        return cp.asnumpy(cupy_array)
    
    @staticmethod
    def cupy_to_torch(cupy_array, device="cuda"):
        return torch.as_tensor(cupy_array).to(device)

    @staticmethod
    def torch_to_numpy(torch_tensor):
        return torch_tensor.cpu().numpy()
    
    @staticmethod
    def torch_to_cupy(torch_tensor):
        return cp.asarray(torch_tensor.to("cuda").clone())

    @staticmethod
    def torch_to_torch(torch_tensor, device="cuda"):
        return torch_tensor.to(device)
