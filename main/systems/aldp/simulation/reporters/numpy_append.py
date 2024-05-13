# Source: https://github.com/davidjtoomer/openmm-numpy-reporters

# Adapted from https://github.com/xor2k/npy-append-array/blob/master/npy_append_array/npy_append_array.py
# Improvements: compatibility with fortran_order=True and magic != (2, 0)

import os
import io
import struct
import numpy as np


class NumpyAppendFile:
    def __init__(self, filename: str) -> None:
        """
        Initialize the object, and associate it with a file if the filename already exists.

        Parameters
        ----------
        filename : str
            The filename of the NumPy file.
        """
        self.filename = filename
        if os.path.isfile(filename):
            self.initialize()

    def __del__(self) -> None:
        """
        Close the file.
        """
        if self.is_initialized:
            self.fp.close()

    def __enter__(self):
        """
        Return the object when the context is entered.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the file when the context is exited.
        """
        self.__del__()

    def initialize(self, array: np.ndarray = None) -> None:
        """
        Read in the header of a NumPy file.
        If the file does not exist, create a new file with the given array.

        Parameters
        ----------
        array : np.ndarray
            The array to write if the file doesn't exist yet.
        """
        self.fp = open(self.filename, "wb" if array is not None else "rb+")

        if array is not None:
            # file does not exist yet. create the file with the given array
            # default: Fortran order = False, magic = None (oldest version that
            # works)
            self.shape = array.shape
            self.fortran_order = False
            self.dtype = array.dtype
            self.magic = None

            header = self.create_header()
            self.fp.write(header)
            self.header_length = self.fp.tell()

            array.tofile(self.fp)
        else:
            # file exists already. read the header
            self.magic = np.lib.format.read_magic(self.fp)
            header = np.lib.format._read_array_header(self.fp, version=self.magic)

            self.shape, self.fortran_order, self.dtype = header
            self.header_length = self.fp.tell()

            new_header = self.create_header()
            if new_header.nbytes != self.header_length:
                raise ValueError(
                    f"Cannot expand header size in target file: {self.filename}"
                )

            self.fp.seek(0, os.SEEK_END)

    def append(self, array: np.ndarray) -> None:
        """
        Append the given array to the end of the file.

        Parameters
        ----------
        array : np.ndarray
            The array to append.
        """
        if not isinstance(array, np.ndarray):
            if not isinstance(array, list):
                raise TypeError(
                    f"Input has wrong type: expected type np.ndarray, got {type(array)}"
                )
            array = np.array(array)

        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            raise ValueError("Array must be contiguous in memory (C or Fortran order)")

        if not hasattr(self, "fp"):
            self.initialize(array=array)
            return

        # check if the array is compatible with the file
        if array.dtype != self.dtype:
            raise TypeError(
                f"Array has wrong dtype: expected {self.dtype}, got {array.dtype}"
            )

        if len(array.shape) != len(self.shape):
            raise ValueError(
                f"Array has wrong number of dimensions: expected {len(self.shape)} != {len(array.shape)}"
            )

        if self.shape[1:] != array.shape[1:]:
            raise ValueError(
                f"Arrays must match outside the zeroth dimension: expected {self.shape[1:]}, got {array.shape[1:]}"
            )

        # append the array to the file
        self.shape = list(self.shape)
        self.shape[0] += array.shape[0]
        self.shape = tuple(self.shape)
        array.tofile(self.fp)
        self.write_header()

    def create_header(self, padding: int = 64) -> memoryview:
        """
        Create the header for the NumPy file.

        Parameters
        ----------
        padding : int
            The padding at the end of the header to use. Must be a multiple of 64 bytes.
        """
        header = {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": self.fortran_order,
            "shape": self.shape,
        }
        buffer = io.BytesIO()
        np.lib.format._write_array_header(buffer, header, version=self.magic)

        # determine the format version if automatically generated
        if not self.magic:
            buffer.seek(0, os.SEEK_SET)
            self.magic = np.lib.format.read_magic(buffer)
            buffer.seek(0, os.SEEK_END)

        # https://github.com/numpy/numpy/blob/c9b5893039759f08ad949a43f7ad8dbabb322b62/numpy/lib/format.py#L183
        header_info = {
            (1, 0): (10, "<H", "latin1"),
            (2, 0): (12, "<I", "latin1"),
            (3, 0): (12, "<I", "utf8"),
        }
        # prefix = magic prefix + magic + header size, 10-12 bytes
        (prefix_length, ctype, encoding) = header_info[self.magic]

        buffer.getbuffer()[8:prefix_length] = struct.pack(
            ctype, int(buffer.getbuffer().nbytes - prefix_length + padding)
        )
        # replace current newline character with empty space
        buffer.getbuffer()[-1] = ord(" ".encode(encoding))
        buffer.write(b" " * 64)
        # insert padded newline character
        buffer.getbuffer()[-1] = ord("\n".encode(encoding))

        return buffer.getbuffer()

    def write_header(self) -> None:
        """
        Write the header to the file.
        """
        self.fp.seek(0, os.SEEK_SET)
        new_header = self.create_header()

        assert len(new_header) == self.header_length

        self.fp.write(new_header)
        self.fp.seek(0, os.SEEK_END)

    @property
    def is_initialized(self) -> bool:
        """
        Returns True if the file has been initialized.
        """
        return hasattr(self, "fp")


if __name__ == "__main__":
    # create a random array
    array = np.random.rand(1000, 10)

    with NumpyAppendFile("test.npy") as file:
        file.append(array)
        file.append(array)

    with NumpyAppendFile("test.npy") as file:
        file.append(array)
        file.append(array)

    test = np.load("test.npy")
