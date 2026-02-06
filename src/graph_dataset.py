import numpy as np
import numpy.typing as npt


class MyGraph:
    """loads a tensor representation of a dynamic graph"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.size: tuple[int]
        self.data: npt.NDArray = np.load(path)
        self.ts = self.data.shape[2]
        self.n_nodes = self.data.shape[0]

    def __getitem__(self, idx) -> npt.NDArray:
        return self.data[idx]

    def __array__(self) -> npt.NDArray:
        return self.data

    def __repr__(self) -> str:
        return f"MyGraph(nodes={self.n_nodes}, timesteps={self.ts}"

    def __getattr__(self, name):
        return getattr(self.data, name)
