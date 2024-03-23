import numpy as np
from scipy.optimize import fsolve
INTERVAL_STEPS_NUMBER = 1000

def ChemicalReaction(y):
    k_1 = 0.04
    k_2 = 3 * 1e7
    k_3 = 1e4
    return np.array([-k_1 * y[0] + k_3 * y[1] * y[2],
            k_1 * y[0] - k_3 * y[1] * y[2] - k_2 * y[1]**2, 
            k_2 * y[1] ** 2
            ])

class ImplicitRungeKutta():
    def __init__(self, diff_eq_func, butcher_table, interval=[0, 5000], 
                  y_init=[1, 0, 0], tau=0.1, 
                  tol=1e-3, save_y_hist=True):
        self.butcher_table = butcher_table
        self.tol = tol
        self.tau = tau
        self.A, self.b, self.c = self.get_butcher_vals()
        self.function = diff_eq_func
        self.iterations = np.arange(*interval, tau)
        self.y_init = np.array(y_init).reshape(-1, 1)
        self.y_hist = self.y_init if save_y_hist else None

    def solve(self):
        self._ks_shape = (self.y_init.size, self.b.size)

        new_y = self.y_init
        for _ in self.iterations:
            Ks = fsolve(self._find_k, np.zeros(self._ks_shape), args=new_y, xtol=0.1)
            Ks = Ks.reshape(self._ks_shape)
            new_y = new_y + self.tau * (Ks @ self.b)
            if self.y_hist is not None: 
                self.y_hist = np.hstack((self.y_hist, new_y))
        return new_y
    
    def _find_k(self, Ks, y):
        Ks = Ks.reshape(self._ks_shape)
        func_argument = y + self.tau * (Ks @ self.A.T)
        rows, columns = func_argument.shape
        new_Ks = np.ravel([self.function(func_argument[:, column]) - Ks[:, column] for column in range(columns)])
        return new_Ks
    
    def get_butcher_vals(self, butcher_table=None):
        if butcher_table is None:
            butcher_table = self.butcher_table
        c = butcher_table[:-1, 0].reshape(-1, 1)
        b = butcher_table[-1, 1:].reshape(-1, 1)
        A_matrix = butcher_table[:-1, 1:]

        for i in range(0, len(c)):
            if np.abs(np.sum(A_matrix[i, :]) - c[i]) > self.tol:
                raise ValueError('Butcher table doesnt meet conditions!')
        return A_matrix, b, c
    


if __name__ == '__main__':
    butcher_table = np.array([[1/2, 1/2, 0], [3/2, -1/2, 2], [0, -1/2, 3/2]])

    a = ImplicitRungeKutta(ChemicalReaction, butcher_table, save_y_hist=True)
    a.solve()
    concentrations = a.y_hist

    import matplotlib.pyplot as plt
    iteration = [i for i in range(concentrations.shape[1])]
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(iteration, concentrations[0, :],  color='red', label='1st concentration')
    axs[0, 0].plot(iteration, concentrations[1, :],  color='green', label='2nd concentration')
    axs[0, 0].plot(iteration, concentrations[2, :],  color='blue', label='3rd concentration')
    axs[0, 0].title.set_text('All concentrations')

    axs[0, 1].plot(iteration, concentrations[0, :])
    axs[0, 1].title.set_text('1st concentration')
    axs[1, 0].plot(iteration, concentrations[1, :])
    axs[1, 0].title.set_text('2nd concentration')
    axs[1, 1].plot(iteration, concentrations[2, :])
    axs[1, 1].title.set_text('3rd concentration')

    plt.xlabel('iteration')
    plt.ylabel('concentration value')
    plt.legend()

    plt.show()
        