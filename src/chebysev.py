import numpy as np
import cvxpy as cp

from cvxpy import Maximize, Minimize, Problem, Variable


def chebysev_center(d, U, epsilon=1e-3):
    """Chebysev center

    Parameters:
        d: dimension
        U: set U
        epsilon: epsilon of set U

    Returns:
        radius: radius of the ball
        u: Chebysev center
    """

    # Variables initialization
    radius = cp.Variable(1, pos=True)
    u = cp.Variable(d)
    l_U = len(U)
    constraints = []

    # Linear constraint
    for i in range(d):
        indices_i = np.zeros(d)
        indices_i[i] = 1
        constraints += [indices_i @ u + radius <= 1]
        constraints += [-indices_i @ u + radius <= 0]

    if l_U > 0:
        for i, (v_i, v_j) in enumerate(U):
            constraints += [2 * u.T @ (v_i - v_j) + 2 * radius * cp.norm(v_i - v_j) + v_j.T @ v_j - v_i.T @ v_i <= epsilon]

    # Bounded constraint
    constraints += [u <= 1] + [0 <= u]

    # Objective and solve
    objective = cp.Maximize(radius)
    p = cp.Problem(objective, constraints)
    result = p.solve()
    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return radius.value, u.value


def chebysev_center_analytic(d, U):
    """Chebysev center

    Parameters:
        d: dimension
        U: set U
        epsilon: epsilon of set U

    Returns:
        radius: radius of the ball
        u: Chebysev center
    """

    # Variables initialization
    u = cp.Variable(d)
    l_U = len(U)
    constraints = [u <= 1] + [0 <= u]

    # Objective and solve
    objective = 0
    for v_i, v_j in U:
        objective += -cp.log(-2 * u.T @ (v_i - v_j) - cp.sum_squares(v_j) + cp.sum_squares(v_i))
        
    objective = cp.Minimize(objective)
    p = cp.Problem(objective, constraints)
    result = p.solve()
    print(result)
    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return u.value


if __name__ == '__main__':
    # variables
    d = 64
    U = [np.random.rand(2, d) for i in range(100)]
    import time 
    t1 = time.time()
    for i in range(1000):
        radius, u_opt = chebysev_center(d, U, 1e-5)
    
    u_opt_ = chebysev_center_analytic(d, U)
