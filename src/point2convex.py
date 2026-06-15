import numpy as np
import cvxpy as cp

from cvxpy import Maximize, Minimize, Problem, Variable


np.random.seed(42)

def item2set(d, U, vi, epsilon=1e-3):
    # Variables initialization
    v = cp.Variable(d)
    l_U = len(U)
    constraints = []

    if l_U > 0:
        for i, (v_i, v_j) in enumerate(U):
            constraints += [2 * v.T @ (v_i - v_j) + v_j.T @ v_j - v_i.T @ v_i <= epsilon]

    # Bounded constraint
    constraints += [v <= 1] + [0 <= v]

    # Objective and solve
    objective = cp.Minimize(cp.sum_squares(v-vi))
    p = cp.Problem(objective, constraints)
    result = p.solve()

    # Results
    if p.status not in ["infeasible", "unbounded"]:
        return result


if __name__ == '__main__':
    # variables
    d = 5
    U = [np.random.rand(2, d) for i in range(10)]
    v = np.random.randn(d)
    result = item2set(d, U, v, 0.1)
    print(result)


# normalize ve 0-1