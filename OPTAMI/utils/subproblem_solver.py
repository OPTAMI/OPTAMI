import torch
from OPTAMI.utils import tuple_to_vec, line_search, derivatives

EPS = 1e-8


# Solve cubic
def cubic_exact(params, grad_approx, hessian_approx, L, delta=0., precision=EPS, testing=False):
    """Implements a Solver of Cubic Newton Subproblem with additional quadratic regularizer
    by SVD with torch.linalg.eigh and line-search:
             Returns solution h = $h^T\nabla f(x) + 0.5  h^T \nabla^2 f(x) h + \frac{L}{6} \|h\|^3 + \frac{\delta}{2}\|h\|^2 $.
                Contributors:
                    Dmitry Kamzolov
                Arguments:
                    params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
                    grad (list of tensors): gradient with create_graph=True for Newton Step.
                    lambd (float): estimated value of Hessian regularizer. (default: 0.)
                    testing (bool): if True, it may compute some additional tests. (default: False)
                """
    c = grad_approx.detach().clone().to(torch.double)
    A = hessian_approx.detach().clone().to(torch.double)

    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if testing and (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    if A.dim() == 1:
        T = A.clone()
        ct = c.clone()
    else:
        T, U = torch.linalg.eigh(A)
        ct = U.t().mv(c)

    T.add_(delta)

    def inv(T, L, tau):
        return (T + L / 2 * tau).reciprocal()

    def dual(tau):
        return L / 6 * tau ** 3 + inv(T, L, tau).mul(ct.square()).sum()

    tau_best = line_search.ray_line_search(
        dual,
        left_point=0.,
        middle_point=2.,
        delta=precision)
    invert = inv(T, L, tau_best)

    if A.dim() == 1:
        x = - invert.mul(ct).type_as(grad_approx)
    else:
        x = - U.mv(invert.mul(ct)).type_as(grad_approx)

    if testing and (c + (L / 2 * x.norm() + delta) * x + A.mv(x)).abs().max().item() >= 0.01:
        raise ValueError('obtained `x` is not optimal')

    return tuple_to_vec.rollup_vector(x, list(params))


def quadratic_exact_solve(params, grad, reg: float = 0., hessian_damping: float = 1., testing: bool = False):
    """Implements a Solver of Regularized Newton Subproblem by torch.linalg.solve:
         Returns h = $(\nabla^2 f(x) * c + \lambda I)^{-1}\nabla f(x)$,
         where we denote $A = \nabla^2 f(x), g = \nabla f(x)$, and $c$ is hessian_damping.
            Contributors:
                Dmitry Kamzolov
                Dmitry Vilensky-Pasechnyuk
            Arguments:
                params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
                grad (tuple of tensors): gradient with create_graph=True for Newton Step.
                reg (float): estimated value of Hessian regularizer. (default: 0.)
                hessian_damping (float): damping coefficient for the Hessian. (default: 1.)
                testing (bool): if True, it computes some additional tests. (default: False)
            """
    grad_flat = tuple_to_vec.tuple_to_vector(grad)
    hessian_flat = derivatives.flat_hessian(grad_flat, list(params))
    c = grad_flat.detach()  # .to(torch.double)
    A = hessian_flat.detach()  # .to(torch.double)

    if c.dim() != 1:
        raise ValueError(f"`c` must be a vector, but c = {c}")

    if A.dim() > 2:
        raise ValueError(f"`A` must be a matrix, but A = {A}")

    if c.size()[0] != A.size()[0]:
        raise ValueError("`c` and `A` mush have the same 1st dimension")

    if testing and (A.t() - A).max() > 0.1:
        raise ValueError("`A` is not symmetric")

    h = torch.linalg.solve(A.mul(hessian_damping) + torch.diag(torch.ones_like(c)).mul_(reg), c)

    return tuple_to_vec.rollup_vector(h, list(params))


def CG_subsolver(params, grad, reg: float = 0., hessian_damping: float = 1., max_iter: int = None,
                 precision: float = EPS,
                 rel_precision: float = EPS, iter_counter: bool = False, testing: bool = False):
    """Implements a Solver of Regularized Newton Subproblem by CG:
     Returns h = $(\nabla^2 f(x) + \lambda I)^{-1}\nabla f(x)$,
     where we denote $A = \nabla^2 f(x), g = \nabla f(x)$.
        Conjugate gradient is Algorithm 5.2 (CG) from "Numerical optimization" J. Nocedal, S. Wright
        Contributors:
            Dmitry Kamzolov
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            grad (tuple of tensors): gradient with create_graph=True for Newton Step.
            reg (float): estimated value of Hessian regularizer. (default: 0.)
            hessian_damping (float): damping coefficient for the Hessian. (default: 1.)
            max_iter (int): maximal number of CG iterations. (default: None)
            precision (float): the precision in terms of $\|r_k\| = \|Ax-g\|\leq \eps. (default: EPS)
            rel_precision (float): the relative precision with respect to gradient norm in terms of
            $\|r_k\| = \|Ax-g\|\leq eps \|g\|$. (default: EPS)
            iter_counter (bool): if True, it returns number of CG steps. (default: False)
            testing (bool): if True, it computes some additional tests. (default: False)
        """

    # Calculating maximal number of iterations. It is equal to the number of params.
    if max_iter is None:
        max_iter = 0
        for p in params:
            total_p = 1
            for t in p.size():
                total_p *= t
            max_iter += total_p

    x_k = []
    r_k = []
    h_k = []
    grad_norm_square = 0.
    with torch.no_grad():
        for g in grad:
            c = g.detach()
            grad_norm_square += c.square().sum().item()
            r_k.append(-c)
            x_k.append(torch.zeros_like(c))
            h_k.append(c)

    iter = 0
    flag = True
    while iter < max_iter and flag:
        Ap = derivatives.hvp_from_grad(grad, params, vec_tuple=h_k)

        # Computation of $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$
        r_k_square = 0.
        alpha_down = 0.
        with torch.no_grad():
            for h, r, a, g in zip(h_k, r_k, Ap, grad):
                a.mul_(hessian_damping).add_(h, alpha=reg)
                r_k_square += r.mul(r).sum()
                alpha_down += a.mul(h).sum()

        alpha_k = r_k_square / alpha_down
        with torch.no_grad():
            for x, r, h, a, g in zip(x_k, r_k, h_k, Ap, grad):
                x.add_(h, alpha=alpha_k)
                r.add_(a, alpha=alpha_k)

        r_k1_square = 0.
        with torch.no_grad():
            for r in r_k:
                r_k1_square += r.mul(r).sum()
        beta_k = r_k1_square / r_k_square

        with torch.no_grad():
            for r, h in zip(r_k, h_k):
                h.mul_(beta_k).sub_(r)
        iter += 1
        r_k1 = r_k1_square ** (1 / 2)
        if r_k1 < precision or r_k1_square < rel_precision * grad_norm_square ** (1 / 2):
            flag = False

    if iter_counter:
        return x_k, iter
    else:
        return x_k

