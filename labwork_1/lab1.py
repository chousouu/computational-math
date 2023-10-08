import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from typing import Callable
import time 


def _diff_matrix(matrix_size: tuple, axis: np.ndarray, step: float, border_value: float) -> tuple:
    row_num, col_num = matrix_size
    
    if col_num == 0 and axis[0] == 1:
        raise ValueError("Call diff_y() instead!")
    if row_num == 0 and axis[0] == 0:
        raise ValueError("Call diff_x() instead!")
    
    row_num = np.max([row_num, 1])
    
    max_eq_num = row_num * col_num 
    result_matrix = np.zeros(shape=(max_eq_num, max_eq_num))
    free_term_matrix = np.zeros(max_eq_num)

    for row in range(row_num):
        for column in range(col_num):
            curr_pos = np.array([row, column])
            max_pos = np.array([row_num, col_num])

            eq_num = _equation_ind(curr_pos, axis, max_pos)
            var_num = _closest_elements_ind(curr_pos, axis, max_pos)

            if column == 0:
                free_term_matrix[eq_num] -= border_value
            else: 
                result_matrix[eq_num, var_num[0]] = 1

            result_matrix[eq_num, var_num[1]] = -2

            if column == (col_num - 1):
                free_term_matrix[eq_num] -= border_value
            else:
                result_matrix[eq_num, var_num[2]] = 1
    
    result_matrix /= step**2
    free_term_matrix /= step**2

    return result_matrix, free_term_matrix


def _prep_func(func: Callable[[float, float], float], x: np.ndarray, y:np.ndarray):
    func_arr = []
    for x_i in x:
        for y_i in y:
            func_arr.append(func(x_i, y_i))
    return np.array(func_arr)


def _equation_ind(curr_pos: np.ndarray, axis: np.ndarray, max_pos: np.ndarray):
    #curr_pos / pos_max -- [row, col]
    return np.sum(axis * (curr_pos * max_pos[::-1] + curr_pos[::-1]))


def _closest_elements_ind(curr_pos: np.ndarray, axis: np.ndarray, max_pos: np.ndarray):
    if axis[0] == 1: #meaning vertical 
        curr_pos = curr_pos[::-1] # now [col, row]
    max = np.sum(max_pos * axis) 

    return [np.sum((curr_pos + (elem_pos * axis)) * [1, max]) for elem_pos in [-1, 0, 1]]


class LaplaceEquation():
    def __init__(self, function, grid_x_step, grid_y_step, 
                 x_bounds=(0, 1), y_bounds=(0, 1), border_value=0, c=0.1):

        x_start, x_end = x_bounds
        y_start, y_end = y_bounds
        x_num, y_num = int((x_end - x_start) / grid_x_step) + 1, int((y_end - y_start) / grid_y_step) + 1
        x = np.linspace(x_start, x_end, num=x_num)
        y = np.linspace(y_start, y_end, num=y_num)

        self._x, self._y = x, y
        self.function_values = _prep_func(function, x[1:-1], y[1:-1]).astype('float64')
        self.free_term = self.function_values
        self.matrix_size = (x_num - 2, y_num - 2)
        self.x_step = grid_x_step
        self.y_step = grid_y_step
        self.border_value = border_value
        self.c = c


    def diff_x(self):
        axis = np.array([1, 0])

        matrix, free_term = _diff_matrix(self.matrix_size, axis, self.x_step, self.border_value)
        self.free_term += free_term

        return matrix
    
    def diff_y(self):
        axis = np.array([0, 1])

        matrix, free_term = _diff_matrix(self.matrix_size, axis, self.y_step, self.border_value)
        self.free_term += free_term

        return matrix

    def get_free_term(self):
        return self.free_term
    
    def get_equation_matrix(self):
        diag_u_matrix = np.eye(self.matrix_size[0] * self.matrix_size[1])
        A = -self.diff_x() - self.diff_y() + self.c * diag_u_matrix
        return A

    def solve_equation(self):
        A = self.get_equation_matrix()
        return lu_solve(lu_factor(A), self.free_term)

    def plot_solution(self):
        nx, ny = np.meshgrid(self._x[1:-1], self._y[1:-1])
        solution = self.solve_equation()
        solution = np.reshape(solution, self.matrix_size)

        plt.pcolormesh(nx, ny, solution)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def main():     
    print("-------------------------")
    print("First task")
    
    task_1a = LaplaceEquation(lambda x, y: x * y, 1/4, 1/4, x_bounds=(0, 1), y_bounds=(0, 1))
    matrix_1a = task_1a.diff_x() # matrix of inner nodes (here 6,7,8,11,12,13,16,17,18)

    plt.spy(matrix_1a, marker='.', color='black')
    plt.title("Task 1 visualization.")
    plt.show()

    print("-------------------------")
    print("Second task")

    print("Plotting solution")
    task_2a = LaplaceEquation(lambda x, y: 1, 1/120, 1/120) #above 1/120 -> breaks.
    task_2a.plot_solution()

    print("Time(Greed size) plot")
    
    grid_step_arr = []
    time_arr = []
    for step_size in range(10, 101, 10):
        eq = LaplaceEquation(lambda x, y: 1, 1/step_size, 1/step_size)
        time_start = time.time()
        eq.solve_equation()
        time_end = time.time()
        grid_step_arr.append(step_size)
        time_arr.append(time_end - time_start)
    
    plt.plot(grid_step_arr, time_arr)
    plt.xlabel("Grid step size")
    plt.ylabel("Time")
    plt.show()


if __name__ == "__main__":
    main()
