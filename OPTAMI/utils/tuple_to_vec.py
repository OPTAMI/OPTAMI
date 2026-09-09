"""Helpers for treating a list/tuple of parameter tensors as one flat vector,
and for reductions over such lists with a *single* device-to-host synchronisation.

The naming rule: a function whose name ends in ``_item`` returns a Python number
and therefore blocks on the device; everything else returns a tensor and does
not.  Prefer the tensor-valued form and read the result once, at the point where
Python actually has to branch on it.
"""
import warnings
from typing import List, Sequence, Set, Tuple

import torch

Tensors = Sequence[torch.Tensor]


# ---------------------------------------------------------------------------
# zero-copy flattening
# ---------------------------------------------------------------------------
class NonContiguousCopyWarning(UserWarning):
    """Flattening a tensor to 1-D required a copy instead of a view.

    It is a distinct class so it can be handled with the standard machinery::

        warnings.filterwarnings("error", category=ttv.NonContiguousCopyWarning)   # raise
        warnings.filterwarnings("ignore", category=ttv.NonContiguousCopyWarning)  # silence
    """


#: ``(shape, stride)`` keys already reported, so a copy inside an optimizer loop
#: warns once instead of once per iteration.  ``clear()`` it to warn again.
_reported_copies: Set[Tuple[tuple, tuple]] = set()


def _is_alias(out: torch.Tensor, src: torch.Tensor) -> bool:
    """Whether ``out`` shares ``src``'s memory.  Assumes it does when it cannot tell."""
    try:
        return out.data_ptr() == src.data_ptr()
    except (RuntimeError, NotImplementedError):
        # A BatchedTensor under ``torch.func.vmap``, a FakeTensor or a meta tensor
        # has no address to compare; stay quiet rather than warn spuriously.
        return True


def flat_view(t: torch.Tensor, warn: bool = True) -> torch.Tensor:
    """Flatten ``t`` to 1-D, preferring a zero-copy view.

    A contiguous tensor is flattened with ``view(-1)``, which never allocates.
    Otherwise ``reshape(-1)`` is used: it still returns a view whenever the
    strides allow one, and copies only when they do not.  Only a *real* copy
    (checked by comparing storage addresses, not by ``is_contiguous`` alone)
    emits a :class:`NonContiguousCopyWarning`, once per distinct shape/stride, so
    an unnoticed non-contiguous parameter or gradient becomes visible instead of
    silently enlarging the working set every iteration.

    The copy is a *transient*: at every call site in OPTAMI the flat result is
    consumed immediately by ``torch.cat`` / ``torch.dot`` and freed, so it lifts
    the peak by one tensor rather than leaking.  Pass ``warn=False`` where that
    is known and expected.
    """
    if t.is_contiguous():
        return t.view(-1)

    out = t.reshape(-1)
    if warn and not _is_alias(out, t):
        key = (tuple(t.shape), tuple(t.stride()))
        if key not in _reported_copies:
            _reported_copies.add(key)
            warnings.warn(
                f"flattening a non-contiguous tensor of shape {tuple(t.shape)} "
                f"(stride {tuple(t.stride())}) required a copy of "
                f"{t.numel() * t.element_size()} bytes; make it contiguous once "
                f"instead of paying for the copy on every iteration.",
                NonContiguousCopyWarning, stacklevel=2,
            )
    return out


# ---------------------------------------------------------------------------
# flatten / unflatten
# ---------------------------------------------------------------------------
def tuple_to_vector(tensors: Tensors, warn: bool = True) -> torch.Tensor:
    """Concatenate a sequence of tensors into one flat 1-D tensor."""
    return torch.cat([flat_view(t, warn) for t in tensors])


def tuple_numel(tensors: Tensors) -> List[int]:
    """Number of elements of every tensor in the sequence."""
    return [t.numel() for t in tensors]


def rollup_vector(flat_vector: torch.Tensor, like: Tensors) -> List[torch.Tensor]:
    """Split a flat vector into views shaped like the tensors in ``like``.

    The returned tensors are *views* of ``flat_vector`` (no copy).
    """
    if not flat_vector.is_contiguous():
        raise ValueError(
            "rollup_vector needs a contiguous flat_vector: the results must alias it, "
            "so they cannot be produced by a copy. Pass flat_vector.contiguous() and "
            "copy the result back yourself if the aliasing is not wanted."
        )
    chunks = torch.split(flat_vector, tuple_numel(like))
    return [v.view_as(t) for v, t in zip(chunks, like)]


# ---------------------------------------------------------------------------
# reductions (one synchronisation each)
# ---------------------------------------------------------------------------
def sum_of_squares(tensors: Tensors) -> torch.Tensor:
    """Return ``sum_i ||t_i||_2^2`` as a 0-d tensor (no host sync)."""
    tensors = list(tensors)
    if len(tensors) == 1:
        return tensors[0].square().sum()
    norms = torch._foreach_norm(tensors)  # one fused kernel on CUDA
    return torch.stack(norms).square().sum()


def norm_square_item(tensors: Tensors) -> float:
    """``sum_i ||t_i||^2`` as a Python float.

    The ``_item`` suffix is the warning: this calls ``.item()``,
    which is a device->host synchronisation.  On CUDA the CPU
    blocks until the whole queue has drained, so call it only when a Python
    number is genuinely needed (a branch, or a scalar ``alpha=`` argument).  Use
    :func:`sum_of_squares` when the value only scales further tensor maths, and
    :func:`items` to read several such scalars in one sync instead of several.
    """
    return sum_of_squares(tensors).item()


def norm_item(tensors: Tensors) -> float:
    """Euclidean norm of the concatenation of ``tensors`` as a Python float."""
    return sum_of_squares(tensors).sqrt().item()


def dot_tensor(a: Tensors, b: Tensors) -> torch.Tensor:
    """``sum_i <a_i, b_i>`` as a 0-d tensor (no host sync).

    ``torch.dot`` on the flattened views is a single BLAS call per tensor and
    does not allocate the elementwise product that ``(a*b).sum()`` would.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        # zip() would silently drop the tail and return a partial sum.
        raise ValueError(f"dot_tensor needs two sequences of the same length, "
                         f"got {len(a)} and {len(b)}")
    if len(a) == 1:
        return torch.dot(flat_view(a[0]), flat_view(b[0]))
    return torch.stack([torch.dot(flat_view(x), flat_view(y)) for x, y in zip(a, b)]).sum()


def dot_item(a: Tensors, b: Tensors) -> float:
    """``sum_i <a_i, b_i>`` as a Python float.

    See :func:`norm_square_item` on the ``_item`` suffix and the cost it marks.
    """
    return dot_tensor(a, b).item()


def items(*scalars: torch.Tensor) -> List[float]:
    """Read several 0-d tensors as Python floats with **one** synchronisation.

    Two independent reductions cost two stalls when read separately::

        rr   = norm_square_item(r)      # sync
        pAp  = dot_item(p, Ap)          # sync

    and one when they are stacked and read together::

        rr, pAp = items(sum_of_squares(r), dot_tensor(p, Ap))

    Only worth it for values that do not depend on one another; a value needed to
    compute the next one has to be read first anyway.
    """
    return torch.stack(list(scalars)).tolist()


# ---------------------------------------------------------------------------
# deprecated aliases (PEP 562): the old names still work, with a warning naming
# the sync the new ones advertise.
# ---------------------------------------------------------------------------
_RENAMED = {
    "tuple_norm_square": "norm_square_item",
    "norm_of_tensors": "norm_item",
    "dot_of_tensors": "dot_item",
}


def __getattr__(name: str):
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(
            f"tuple_to_vec.{name} was renamed to {new}: the '_item' suffix marks the "
            f"device->host synchronisation it performs. The old name still works but "
            f"will be removed in a future release.",
            DeprecationWarning, stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# multi-tensor updates (must be called under ``torch.no_grad()`` when the
# targets are leaf parameters)
# ---------------------------------------------------------------------------
def add_tensors_(tensors: Tensors, others: Tensors, alpha: float = 1.0) -> None:
    """In-place ``t_i += alpha * o_i`` for all i (one fused kernel on CUDA)."""
    torch._foreach_add_(list(tensors), list(others), alpha=alpha)


def copy_tensors_(dst: Tensors, src: Tensors) -> None:
    """In-place ``dst_i.copy_(src_i)`` for all i."""
    torch._foreach_copy_(list(dst), list(src))


def clone_tensors(tensors: Tensors) -> List[torch.Tensor]:
    """Detached clones of all tensors (``detach()`` first: no autograd bookkeeping)."""
    return [t.detach().clone() for t in tensors]
