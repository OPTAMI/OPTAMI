r"""Derivative oracles (gradient, Hessian-vector products, full Hessian, third
derivative along a direction) for closure-based optimizers.

Gradient modes
==============
``gradient(closure, params, mode=...)`` obtains the gradient in one of three ways, because
different methods need different things from the same closure:

``"autograd"``
    ``torch.autograd.grad`` on the loss returned by the closure.  The only mode that can
    return a *differentiable* gradient (``create_graph=True``), so every second- and
    higher-order method here uses it.  Nothing is written to ``p.grad``; the closure need
    not call ``zero_grad()`` and must not call ``backward()``.

``"backward"``
    ``loss.backward()`` and read ``p.grad``.  Equivalent numerically, but it runs the
    gradient through the ``.grad`` machinery: autograd hooks, DDP bucketing and all-reduce,
    AMP unscaling, gradient clipping and anything else the user attached to the parameters.
    That is what a stochastic first-order method wants and what a Hessian-vector product
    cannot use, so it is opt-in per method (``UniversalSGD.GRAD_MODE``).

``"closure"``
    the closure computes the gradient itself and leaves it in ``p.grad``; OPTAMI only reads
    it.  Selected automatically for a closure marked with :func:`provides_grad`, whatever
    the method asked for.  This is the escape hatch for gradients an optimizer cannot form
    from the outside: variance reduction against a stored snapshot, a hand-written backward,
    a gradient accumulated over micro-batches or gathered from other workers.

In the last two modes ``p.grad`` is set to ``None`` before the closure runs, so a stale
gradient can never leak into a step.

Some quantities live in the data, not in the optimizer.  A closure may accept ``full_grad``
to be asked for the gradient over the whole dataset (:func:`full_gradient`), which is how a
variance-reduced method gets its snapshot gradient::

    @derivatives.provides_grad
    def closure(full_grad=False):
        xb, yb = (X, Y) if full_grad else next(batches)
        loss = criterion(model(xb), yb)
        loss.backward()
        return loss

Closure contract: unless it is marked with :func:`provides_grad`, ``closure()``
re-evaluates the model and returns the loss **without** calling ``backward()``.  Whether
``p.grad`` is written is then the method's business (``mode``), and no mode depends on the
closure calling ``zero_grad()``.
"""
import functools
import inspect
import warnings
from typing import Callable, List, Sequence, Tuple, Union

import torch

from . import tuple_to_vec

Tensor = torch.Tensor
Tensors = Sequence[Tensor]


def _as_list(x) -> List[Tensor]:
    """Accept a single tensor or a sequence of tensors, always return a list."""
    if isinstance(x, Tensor):
        return [x]
    return list(x)


# ---------------------------------------------------------------------------
# first order
# ---------------------------------------------------------------------------
#: Ways of obtaining a gradient; see "Gradient modes" in the module docstring.
GRAD_MODES = ("autograd", "backward", "closure")

#: Attribute that :func:`provides_grad` sets on a closure.  It survives
#: ``torch.enable_grad()(closure)``, which copies ``__dict__`` through ``functools.wraps``.
PROVIDES_GRAD_ATTR = "optami_provides_grad"

#: Keyword a closure may accept to be asked for a gradient over the whole dataset.
FULL_GRAD_KWARG = "full_grad"


class ClosureContractError(RuntimeError):
    """The closure cannot deliver what the optimizer asked of it."""


class FullGradientUnavailable(ClosureContractError):
    """A full-dataset gradient was requested from a closure that cannot produce one."""


def provides_grad(closure):
    r"""Mark ``closure`` as computing its own gradients into ``p.grad``.

    Use it when the optimizer must not be the one deciding how the gradient is formed:
    variance reduction over a stored snapshot, a hand-written or fused backward, a
    manually accumulated gradient over micro-batches, a gradient that arrives from
    another process.  The marked closure must

    * return the loss, and
    * leave the gradient of that loss in ``p.grad`` for every parameter it touches.

    It may accept ``full_grad`` (see :data:`FULL_GRAD_KWARG` and :func:`full_gradient`)::

        @derivatives.provides_grad
        def closure(full_grad=False):
            xb, yb = (X, Y) if full_grad else next(batches)
            loss = criterion(model(xb), yb)
            loss.backward()          # OPTAMI never calls backward on such a closure
            return loss

    ``p.grad`` is set to ``None`` before the closure runs, so it does not have to call
    ``zero_grad()``; anything it accumulates *inside* one call is kept.

    Only first-order methods can use such a closure: second- and higher-order methods
    need to differentiate the gradient again, which ``backward()`` has already discarded.
    They raise :class:`ClosureContractError` instead of failing later inside autograd.
    """
    try:
        setattr(closure, PROVIDES_GRAD_ATTR, True)
        return closure
    except AttributeError:  # bound methods and other objects with no writable __dict__
        @functools.wraps(closure)
        def marked(*args, **kwargs):
            return closure(*args, **kwargs)
        setattr(marked, PROVIDES_GRAD_ATTR, True)
        return marked


def closure_provides_grad(closure) -> bool:
    """Whether ``closure`` was marked with :func:`provides_grad`."""
    return bool(getattr(closure, PROVIDES_GRAD_ATTR, False))


def closure_accepts(closure, name: str = FULL_GRAD_KWARG) -> bool:
    """Whether ``closure`` can be called with the keyword argument ``name``.

    ``inspect.signature`` follows ``__wrapped__``, so a closure keeps its signature
    through ``torch.enable_grad()`` and any ``functools.wraps``-based decorator.
    """
    try:
        parameters = inspect.signature(closure).parameters
    except (TypeError, ValueError):  # a builtin or otherwise unintrospectable callable
        return False
    for p in parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == name and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                         inspect.Parameter.KEYWORD_ONLY):
            return True
    return False


def accepts_full_grad(closure) -> bool:
    """Whether ``closure`` accepts ``full_grad=True``, i.e. can be asked for a full gradient."""
    return closure_accepts(closure, FULL_GRAD_KWARG)


def require_autograd_closure(closure, who: str = "this method") -> None:
    """Raise if ``closure`` computes its own gradients, for methods that need derivatives of it.

    Called at the top of ``step`` by every method that differentiates the gradient
    (second and higher order), so the user gets one clear message instead of an autograd
    error about a graph that was already freed.
    """
    if closure_provides_grad(closure):
        raise ClosureContractError(
            f"{who} needs second (or higher) derivatives, so it has to build the gradient "
            f"itself with torch.autograd.grad(..., create_graph=True).  The closure it was "
            f"given is marked with @provides_grad, i.e. it calls backward() and leaves the "
            f"gradient in p.grad, which destroys the graph the method needs.  Pass a closure "
            f"that only re-evaluates the model and returns the loss.")


def _resolve_grad_mode(mode: str, closure, create_graph: bool) -> str:
    """Validate ``mode`` against the closure and ``create_graph``; return one of :data:`GRAD_MODES`."""
    if mode not in GRAD_MODES and mode != "auto":
        raise ValueError(f"mode must be 'auto' or one of {GRAD_MODES}, got {mode!r}")
    if closure_provides_grad(closure):
        if create_graph:
            require_autograd_closure(closure)
        if mode == "autograd":
            raise ClosureContractError(
                "mode='autograd' was requested for a closure marked with @provides_grad: it "
                "calls backward() itself, so autograd.grad would differentiate a freed graph.")
        return "closure"  # the closure's own gradient wins over the optimizer's preference
    if mode == "closure":
        raise ClosureContractError(
            "mode='closure' expects a closure that fills p.grad itself; decorate it with "
            "@OPTAMI.utils.derivatives.provides_grad to say so.")
    if create_graph and mode == "backward":
        raise ValueError("create_graph=True needs mode='autograd' (backward() on a graph that "
                         "is kept alive is not how the higher-order methods here work)")
    return "autograd" if mode == "auto" else mode


def _grads_from_param_grad(params: List[Tensor], mode: str) -> Tuple[Tensor, ...]:
    """Read ``p.grad`` for every parameter, materialising zeros for the ones autograd never reached."""
    grads = []
    missing = 0
    for p in params:
        if p.grad is None:
            missing += 1
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad)
    if missing == len(params) and params:
        raise ClosureContractError(
            "no gradient reached p.grad" + (
                ": the closure is marked with @provides_grad but never called backward() "
                "(or called it on a different set of tensors)." if mode == "closure" else
                ": the loss returned by the closure does not depend on the parameters."))
    return tuple(grads)


def gradient(closure: Callable[..., Tensor], params: Tensors, create_graph: bool = False,
             mode: str = "auto", **closure_kwargs) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    r"""Evaluate ``closure`` and return ``(loss, grads)``, one gradient tensor per parameter.

    Arguments:
        closure: re-evaluates the model and returns the loss.  In the ``"closure"`` mode it
            also computes the gradient itself; see :func:`provides_grad`.
        params: the parameters to differentiate with respect to.
        create_graph: keep the graph of the gradient so it can be differentiated again
            (Hessian-vector products, third derivatives).  ``"autograd"`` mode only.
        mode: how the gradient is obtained -- ``"autograd"``, ``"backward"``, ``"closure"``,
            or ``"auto"`` (the default: ``"autograd"``, unless the closure provides its own
            gradient, which always wins).  See "Gradient modes" in the module docstring.
        closure_kwargs: forwarded to the closure, e.g. ``full_grad=True``
            (:func:`full_gradient` is the readable spelling of that one).

    In the ``"backward"`` and ``"closure"`` modes the returned tensors *are* the ``p.grad``
    buffers.  They are freshly allocated -- ``p.grad`` is set to ``None`` before the closure
    runs, so nothing accumulates from a previous step and the tensors returned by an earlier
    call stay valid -- but ``optimizer.zero_grad(set_to_none=False)`` in between would zero
    them in place.  Parameters the backward pass does not reach get a zero tensor (in
    ``"autograd"`` mode they raise, which is torch's default and a useful signal).
    """
    params = list(params)
    mode = _resolve_grad_mode(mode, closure, create_graph)

    if mode == "autograd":
        loss = closure(**closure_kwargs)
        return loss, torch.autograd.grad(loss, params, create_graph=create_graph)

    for p in params:
        p.grad = None
    loss = closure(**closure_kwargs)
    if mode == "backward":
        # inputs=params keeps the accumulation inside this optimizer's parameters; hooks
        # (DDP all-reduce, AMP scaling, clipping) still fire, which is the point of this mode.
        torch.autograd.backward(loss, inputs=params)
    return loss, _grads_from_param_grad(params, mode)


def full_gradient(closure: Callable[..., Tensor], params: Tensors, mode: str = "auto",
                  strict: bool = True, **closure_kwargs) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    r"""Evaluate ``closure`` over the whole dataset and return ``(loss, grads)``.

    An optimizer cannot compute a full gradient out of a stochastic closure -- it never sees
    the data -- so the closure has to offer it, by accepting ``full_grad`` (see
    :func:`provides_grad`).  Variance-reduced methods need this for their snapshot gradient.

    With ``strict=False`` a closure that does not accept ``full_grad`` is called as-is, which
    is right only when it is deterministic (full batch).  Otherwise
    :class:`FullGradientUnavailable` is raised.  Use :func:`accepts_full_grad` to check once
    in ``__init__`` rather than failing in the middle of a run.
    """
    if accepts_full_grad(closure):
        closure_kwargs[FULL_GRAD_KWARG] = True
        return gradient(closure, params, mode=mode, **closure_kwargs)
    if strict:
        raise FullGradientUnavailable(
            "this method needs the gradient over the whole dataset, which only the closure can "
            "compute.  Give it a 'full_grad' argument:\n\n"
            "    def closure(full_grad=False):\n"
            "        xb, yb = (X, Y) if full_grad else next(batches)\n"
            "        return criterion(model(xb), yb)\n\n"
            "(add @OPTAMI.utils.derivatives.provides_grad and call loss.backward() inside if "
            "you also want to compute the gradient yourself).  Pass strict=False if the "
            "closure is deterministic and its gradient is already the full one.")
    return gradient(closure, params, mode=mode, **closure_kwargs)


def value(closure: Callable[..., Tensor], **closure_kwargs) -> Tensor:
    r"""The loss only, detached, for line searches and acceptance tests.

    Runs under ``torch.no_grad()`` -- except for a closure that computes its own gradient,
    which needs grad mode to call ``backward()``.  That gradient is discarded, and it does
    overwrite ``p.grad``: read the gradient before the value if you need both.
    """
    if closure_provides_grad(closure):
        with torch.enable_grad():
            return closure(**closure_kwargs).detach()
    with torch.no_grad():
        return closure(**closure_kwargs)


# ---------------------------------------------------------------------------
# second order
# ---------------------------------------------------------------------------
def hvp_from_grad(grads_tuple: Tensors, list_params: Tensors, vec_tuple,
                  retain_graph: bool = True, create_graph: bool = False) -> Tuple[Tensor, ...]:
    """Hessian-vector product ``H v`` from a gradient computed with ``create_graph=True``.

    ``grad_outputs=v`` makes autograd compute the vector-Jacobian product
    ``v^T (d grads / d params) = (H v)^T`` directly; no scalar ``<grads, v>`` is
    ever formed.  ``retain_graph=True`` (default) keeps the forward + gradient
    graph alive so the same ``grads_tuple`` can be reused for many products
    (this is what makes CG cheap).  Pass ``create_graph=True`` to differentiate
    the product once more (third derivatives).
    """
    return torch.autograd.grad(grads_tuple, list(list_params), grad_outputs=_as_list(vec_tuple),
                               retain_graph=retain_graph, create_graph=create_graph)


def hess_vec_prod(closure: Callable[[], Tensor], list_params: Tensors, vec_tuple):
    """Evaluate the closure and return ``(H v, grads)`` (grads keep their graph)."""
    _, grads = gradient(closure, list_params, create_graph=True)
    return hvp_from_grad(grads, list_params, vec_tuple), grads


def flat_hvp(closure: Callable[[], Tensor], list_params: Tensors, vector: Tensor):
    """Evaluate the closure and return the flat ``(H v, grad)`` for a flat ``vector``."""
    list_params = list(list_params)
    _, grads = gradient(closure, list_params, create_graph=True)
    v = tuple_to_vec.rollup_vector(vector, list_params)
    hvp = hvp_from_grad(grads, list_params, v, retain_graph=False)
    return tuple_to_vec.tuple_to_vector(hvp), tuple_to_vec.tuple_to_vector(grads)


#: At or below this many parameters, ``chunk_size="auto"`` batches every row of the
#: Hessian in one pass.  All of the datasets used in the OPTAMI paper sit below it
#: (a9a n=123, w8a n=300, madelon n=500), so their results are bit-identical to the
#: unchunked path.
AUTO_CHUNK_THRESHOLD = 512

#: Rows per chunk that ``chunk_size="auto"`` uses above that threshold.  See
#: ``auto_chunk_size`` for the measurements behind the value.
AUTO_CHUNK_ROWS = 512


def auto_chunk_size(n: int) -> Union[int, None]:
    r"""Rows per vmap chunk for a Hessian of size ``n``, or ``None`` for "all at once".

    Batching every row at once is fastest when the graph has no large per-sample
    activations, and much slower when it does.  Measured (torch 2.14, 4 CPU threads):

    ==========================================  =========  =========  =========
    problem                                     no chunk   chunk 512  chunk 128
    ==========================================  =========  =========  =========
    dense logistic regression, n = 1,000          17 ms      18 ms      24 ms
    dense logistic regression, n = 3,000         139 ms     147 ms     194 ms
    LeNet-5 on MNIST, N = 3,108                13,141 ms   7,756 ms   6,318 ms
    ==========================================  =========  =========  =========

    On a dense problem chunking costs a few percent; on a convolutional net it is
    worth a factor of two, because batching all ``n`` cotangents through a
    convolution makes its input-gradient huge and that shape dispatches to a naive
    ``slow_conv_transpose2d`` kernel instead of the tuned one (and inflates peak
    memory: 7.8 GB versus a few hundred MB in the measurement above).

    Since the cost of guessing wrong is a few percent and the benefit is a factor of
    two, the policy chunks whenever ``n`` is big enough for the question to arise,
    and never below ``AUTO_CHUNK_THRESHOLD``, which keeps the library's own
    benchmark problems on exactly the code path they used before.
    """
    return None if n <= AUTO_CHUNK_THRESHOLD else AUTO_CHUNK_ROWS


def _resolve_chunk_size(chunk_size, n: int) -> Union[int, None]:
    """Turn the ``chunk_size`` argument (``"auto"`` / ``None`` / a positive int) into a value for vmap."""
    if isinstance(chunk_size, str):
        if chunk_size != "auto":
            raise ValueError(f"chunk_size must be 'auto', None or a positive int, got {chunk_size!r}")
        return auto_chunk_size(n)
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be 'auto', None or a positive int, got {chunk_size!r}")
    return chunk_size


def flat_hessian(flat_grads: Tensor, params: Tensors, chunk_size="auto") -> Tensor:
    r"""Dense Hessian (n x n) from a flat gradient that carries a graph.

    torc.func is a JAX-like composable function transforms for PyTorch.
    Essentially, it transforms the computational block into a function f(x): x-> R
    This allows faster differentiation and high-order derivatives.
    ``torch.func.vmap`` maps the vector-Jacobian product over the rows of the
    identity matrix, so every autograd node receives a batch of ``n``
    cotangents at once and the ``mv`` backward becomes a single GEMM.
    However, in some rare cases it can fail, then we perform a direct for-loop computation
    with torch.autograd.grad. If you interested, we write we don't use func.hessian in the comments

    On a9a vmap measured 12x faster than the loop  (17 ms vs 204 ms, float64 CPU);
    on GPU the gain is larger because the loop paid ``n`` kernel-launch latencies.

    Arguments:
        flat_grads: output of ``tuple_to_vector(autograd.grad(..., create_graph=True))``.
        params: the parameters the gradient was taken with respect to.
        chunk_size: rows of the Hessian per vmap chunk.  ``"auto"`` (default) applies
            ``auto_chunk_size``; ``None`` batches all rows in one pass; a positive int
            is used as given.  The batched backward materialises intermediates of size
            ``chunk_size x (size of every activation)``, so lower it when memory is tight.

    Ops without a vmap batching rule fall back automatically to a per-row loop
    inside ``vmap`` (torch prints a one-time performance warning; the result is
    still exact).  If ``vmap`` itself raises, we fall back to the explicit loop.
    """
    params = list(params)
    n = flat_grads.numel()
    chunk_size = _resolve_chunk_size(chunk_size, n)
    eye = torch.eye(n, dtype=flat_grads.dtype, device=flat_grads.device)

    def vjp_row(e: Tensor) -> Tensor:
        rows = torch.autograd.grad(flat_grads, params, grad_outputs=e, retain_graph=True)
        return torch.cat([tuple_to_vec.flat_view(r) for r in rows])

    try:
        return torch.func.vmap(vjp_row, chunk_size=chunk_size)(eye)
    except RuntimeError as err:  # e.g. an op that cannot be vmapped at all
        warnings.warn(f"flat_hessian: vmap failed ({err}); falling back to a row-by-row loop.")
        return torch.stack([vjp_row(eye[i]) for i in range(n)])

    # Why we do not use torch.func.hessian (= jacfwd(jacrev))?
    #
    # 1. It cannot be driven from a closure. A torch.optim optimizer receives
    #    model.parameters() and a closure, never the model, but every torch.func
    #    transform needs a pure function whose argument is the parameter vector.
    #    Both ways to bridge that fail: assigning the traced vector to p.data
    #    segfaults the interpreter, and p.copy_(traced) raises "in-place operation
    #    during a grad transform". The only working route, functional_call, needs
    #    the nn.Module that this library deliberately never accepts.
    #
    # 2. Its operator coverage is incomplete. Forward-over-reverse needs forward-mode
    #    AD, and torch 2.14 has no forward-AD rule for soft_margin_loss, the loss in
    #    our own test suite. It raises NotImplementedError only after the forward
    #    pass is already paid (~740 ms on a9a), so "try it and fall back" would cost
    #    that on every step.
    #
    # 3. It is not faster. Same work, torch 2.14, 4 CPU threads:
    #      a9a, n = 123:        flat_hessian    19 ms  vs  torch.func.hessian    96 ms
    #      LeNet-5, N = 6,080:  flat_hessian  17.9 s   vs  torch.func.hessian  36.8 s
    #    (the LeNet numbers need chunk_size; see auto_chunk_size)
    #
    # 4. Our methods build one gradient graph with create_graph=True and reuse it for
    #    many Hessian-vector products. torch.func rebuilds the forward pass on every
    #    call, which is also why it loses on the Hessian-vector-product path.


# ---------------------------------------------------------------------------
# third order
# ---------------------------------------------------------------------------
def third_derivative_from_grad(grads: Tensors, params: Tensors, vector, flat: bool = False,
                               retain_graph: bool = False):
    r"""``(D^3 f(x)[v, v], H v)`` from a gradient computed with ``create_graph=True``.

    ``H v`` is obtained with ``create_graph=True`` so that it can be
    differentiated once more in the direction ``v``:
    ``D^3 f[v, v] = d/dx <H(x) v, v>``.  The returned ``H v`` is detached.
    """
    params = list(params)
    v = _as_list(vector)
    hvp = torch.autograd.grad(grads, params, grad_outputs=v, create_graph=True)
    hvp_det = [h.detach() for h in hvp]
    # The next block is introduce to work with the parts where the Hessian is constant in a parameter
    # (a quadratic objective, or a parameter entering only through a quadratic penalty).
    # That block of ``H v`` carries no graph, and autograd rejects the whole outputs list with "does not require grad".
    # Such a block contributes zero to every block of the third derivative, so it is dropped here;
    # ``allow_unused``/``materialize_grads`` pairs with it to cover unused *inputs*.
    live = [(h, vec) for h, vec in zip(hvp, v) if h.requires_grad]
    if not live:
        third = tuple(torch.zeros_like(h) for h in hvp_det)
    else:
        third = torch.autograd.grad([h for h, _ in live], params,
                                    grad_outputs=[vec for _, vec in live],
                                    retain_graph=retain_graph,
                                    allow_unused=True, materialize_grads=True)
    # ``retain_graph=True`` lets the caller keep reusing ``grads`` afterwards
    # (BasicTensorMethod does this to avoid re-evaluating the closure in every inner iteration)
    if flat:
        return tuple_to_vec.tuple_to_vector(third), tuple_to_vec.tuple_to_vector(hvp_det)
    return third, hvp_det


def third_derivative_vec(closure: Callable[[], Tensor], params: Tensors, vector, flat: bool = False):
    r"""Evaluate the closure and return ``(D^3 f(x)[v, v], H v)``."""
    params = list(params)
    _, grads = gradient(closure, params, create_graph=True)
    return third_derivative_from_grad(grads, params, vector, flat=flat, retain_graph=False)
