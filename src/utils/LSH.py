from typing import Callable, Optional
import numpy as np
import numpy.typing as npt


type hash_func_array = list[Callable[[npt.NDArray], float]]


class LSH:
    type f64_arrya = npt.NDArray[np.float64]

    def __init__(
        self,
        input_size: int,
        array_len: int,
        bin_size: int | float,
        n_hash: int,
    ):
        self.input_size = input_size
        self.bin_size = bin_size
        self.n_hash = n_hash
        self.array_len = array_len
        self.data = np.zeros((array_len, n_hash))  # indices x hash outputs

        self.hash_functions: hash_func_array = [
            self._gen_h_func() for _ in range(n_hash)
        ]

    def _gen_h_func(self) -> Callable[[npt.NDArray], int]:

        a = np.random.randn(self.input_size)
        b = np.random.random() * self.bin_size

        def h_func(array: npt.NDArray) -> int:
            h = np.floor((a @ array + b) / self.bin_size)
            return int(h)

        return h_func

    def __getitem__(self, index: int) -> npt.NDArray:
        return self.data[index, :]

    def __setitem__(self, index: int, array: npt.NDArray) -> None:
        h_array = np.array([f(array) for f in self.hash_functions])
        self.data[index, :] = h_array

    def compare(self, index1: int, index2: int) -> float:
        return sum(self[index1] == self[index2]) / self.n_hash


def _test_lsh_basic_behavior():
    np.random.seed(42)  # make tests reproducible

    dim = 10
    n_hash = 20
    bin_size = 1.0

    lsh = LSH(
        input_size=dim,
        array_len=dim,
        bin_size=bin_size,
        n_hash=n_hash,
    )

    # identical vectors
    x = np.ones(dim)
    lsh[0] = x
    lsh[1] = x.copy()

    assert lsh.compare(0, 1), "Identical vectors should collide"

    # very similar vectors (small noise)
    y = x + 0.01 * np.random.randn(dim)
    lsh[2] = y

    assert lsh.compare(0, 2), "Very similar vectors should collide"

    # clearly different vectors
    z = np.random.randn(dim) * 10
    lsh[3] = z

    assert not lsh.compare(0, 3), "Dissimilar vectors should not collide"

    # symmetry check
    assert lsh.compare(1, 0) == lsh.compare(0, 1), "Comparison should be symmetric"

    # self-comparison
    assert lsh.compare(0, 0) > 0.99, "Vector should always match itself"

    print("All LSH tests passed ✔️")


if __name__ == "__main__":
    _test_lsh_basic_behavior()
