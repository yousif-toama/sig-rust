"""Type stubs for the native Rust extension module."""

from typing import Any

from numpy.typing import NDArray
import numpy as np

class PreparedData:
    @property
    def d(self) -> int: ...
    @property
    def m(self) -> int: ...

class RotInv2DPreparedData:
    @property
    def m(self) -> int: ...

def sig(path: NDArray[np.float64], m: int, format: int = 0) -> Any: ...
def siglength(d: int, m: int) -> int: ...
def sigcombine(
    sig1: NDArray[np.float64],
    sig2: NDArray[np.float64],
    d: int,
    m: int,
) -> Any: ...
def prepare(d: int, m: int, method: str = "auto") -> PreparedData: ...
def basis(s: PreparedData) -> list[str]: ...
def logsig(path: NDArray[np.float64], s: PreparedData) -> Any: ...
def logsig_expanded(
    path: NDArray[np.float64], s: PreparedData
) -> Any: ...
def logsiglength(d: int, m: int) -> int: ...
def sigbackprop(
    deriv: NDArray[np.float64],
    path: NDArray[np.float64],
    m: int,
) -> Any: ...
def sigjacobian(path: NDArray[np.float64], m: int) -> Any: ...
def logsigbackprop(
    deriv: NDArray[np.float64],
    path: NDArray[np.float64],
    s: PreparedData,
) -> Any: ...
def sigjoin(
    sig_flat: NDArray[np.float64],
    segment: NDArray[np.float64],
    d: int,
    m: int,
    fixedLast: float = ...,
) -> Any: ...
def sigjoinbackprop(
    deriv: NDArray[np.float64],
    sig_flat: NDArray[np.float64],
    segment: NDArray[np.float64],
    d: int,
    m: int,
    fixedLast: float = ...,
) -> Any: ...
def sigscale(
    sig_flat: NDArray[np.float64],
    scales: NDArray[np.float64],
    d: int,
    m: int,
) -> Any: ...
def sigscalebackprop(
    deriv: NDArray[np.float64],
    sig_flat: NDArray[np.float64],
    scales: NDArray[np.float64],
    d: int,
    m: int,
) -> Any: ...
def rotinv2dprepare(
    m: int, inv_type: str = "a"
) -> RotInv2DPreparedData: ...
def rotinv2d(
    path: NDArray[np.float64], s: RotInv2DPreparedData
) -> Any: ...
def rotinv2dlength(s: RotInv2DPreparedData) -> int: ...
def rotinv2dcoeffs(s: RotInv2DPreparedData) -> Any: ...
def version() -> str: ...
