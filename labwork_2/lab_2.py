import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg as ConjugateGradients

EPSILON = 1e-4


def fill_matrix(size, alpha=0.01, beta=10):
    main_diag = np.eye(size, k=0)
    upper_diag = np.eye(size, k=1)
    lower_diag = np.eye(size, k=-1)
    
    A = (main_diag * alpha + main_diag * 2) - upper_diag - lower_diag
    A[0, 0] = 2 + beta
    
    return A


def mat_vec_mult(matrix, vector):
    return (matrix @ vector).flatten()

def _solve_iterative_process(A, f, x_0, B, F, tol=EPSILON):
    x_k = x_0
    res_0_norm = np.linalg.norm(mat_vec_mult(A, x_0) - f, ord=2)
    res_k_norm = res_0_norm
    error = []

    while res_k_norm / res_0_norm >= tol:
        error.append(res_k_norm / res_0_norm)
        curr_solution = x_k
        x_k = mat_vec_mult(B, x_k) + F
        res_k_norm = np.linalg.norm(mat_vec_mult(A, curr_solution) - f, ord=2)

    iterations = len(error)
    return curr_solution, error, iterations


class SolveLinearEquation():
    def __init__(self, A, f, method: str = 'Jacobi', 
                 starting_point = None, tau: int = None):
        """
        Jacobi, FPI, CG
        """
        self.A = A
        self.f = f

        if starting_point == None: 
            self.starting_point = np.zeros(shape=(A.shape[0], 1))
        else:
            self.starting_point = starting_point

        if method == "Jacobi":
            self.solution, self.error_arr, self.iters = self.JacobiMethod()
        elif method == "FPI":
            self.solution, self.error_arr, self.iters = self.FixedPointIter(tau)
        elif method == "CG":
            self.solution, self.error_arr, self.iters = self.ConjugateGradient()
        else:
            raise ValueError("This method does not exist!", 
                             "Please check available methods names:",
                             "'FPI', 'Jacobi', 'CG'.")

    def FixedPointIter(self, tau, tol=EPSILON):
        if tau == None: 
            eig_values, _ = np.linalg.eig(self.A)
            tau = 2 / (eig_values.min() + eig_values.max())
        
        B = np.eye(self.A.shape[0]) - tau * self.A
        F = tau * self.f
        return _solve_iterative_process(self.A, self.f, self.starting_point, B, F, tol=tol)
    
    
    def JacobiMethod(self, tol=EPSILON):
        L = np.tril(self.A, -1)
        D = np.diag(np.diag(self.A))
        U = np.triu(self.A, 1)

        D_invert = np.linalg.inv(D)
        B = np.dot(-D_invert, L + U)
        F = np.dot(D_invert, self.f)

        return _solve_iterative_process(self.A, self.f, self.starting_point, B, F, tol=tol)

    def ConjugateGradient(self, tol=EPSILON):
        i = 0
        res_0_norm = np.linalg.norm(self.f - mat_vec_mult(self.A, self.starting_point))
        res_k_norm = res_0_norm
        error = []

        while res_k_norm / res_0_norm >= tol:
            error.append(res_k_norm / res_0_norm)
            i += 1
            x_k, _ = ConjugateGradients(self.A, self.f, self.starting_point, maxiter=i)
            res_k_norm = np.linalg.norm(self.f - mat_vec_mult(self.A, x_k), ord=2)

        iterations = len(error)
        return x_k, error, iterations



def main():
    system_size = 1000
    A = fill_matrix(system_size)

    f = np.zeros((system_size,))
    f[495:505] = 1
    
    Jacobi_solve = SolveLinearEquation(A, f, 'Jacobi')
    FPI_solve = SolveLinearEquation(A, f, 'FPI')
    CG_solve = SolveLinearEquation(A, f, 'CG')

    #First task: r_k / r_0(iter)

    plt.semilogy(np.arange(FPI_solve.iters), FPI_solve.error_arr, 'blue',
                 np.arange(Jacobi_solve.iters), Jacobi_solve.error_arr, 'green', 
                 np.arange(CG_solve.iters), CG_solve.error_arr, 'red')

    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend(['Simple iterations, no precond', 'Simple iterations diag. precond', 'Conjugate gradients'])
    plt.grid()
    plt.show()

    #Second task: Solution(component)


    plt.plot(np.arange(system_size), FPI_solve.solution, 'blue', marker="o")
    plt.plot(np.arange(system_size), Jacobi_solve.solution, 'green', marker="s") 
    plt.plot(np.arange(system_size), CG_solve.solution, 'red', marker=",")

    plt.xlabel('Component index')
    plt.ylabel('Solution x')
    plt.legend(['Simple iterations, no precond', 'Simple iterations diag. precond', 'Conjugate gradients'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
