import numpy as np 
import matplotlib.pyplot as plt


CONST_c = 0.1

def _diff_matrix(matrix: np.ndarray, step: float, border_value: float) -> tuple:
    """
    NOTE! returns +1 * d^2/dx^2
    """
    print(f"{matrix.shape=}")
    row_num, col_num = np.shape(matrix[1:-1, 1:-1])
    
    if col_num == 0:
        raise ValueError("Call diff_y() instead!")
    
    row_num = np.max([row_num, 1])
    
    print(f"{row_num, col_num=}")

    max_eq_num = row_num * col_num 
    result_matrix = np.zeros(shape=(max_eq_num, max_eq_num))
    free_term_matrix = np.zeros(max_eq_num)

    for row in range(row_num):
        for column in range(col_num):
            eq_num = row * col_num + column
            var_num = row * col_num + column 
            if column == 0:
                free_term_matrix[eq_num] -= border_value
            else: 
                result_matrix[eq_num, var_num - 1] = 1

            result_matrix[eq_num, var_num] = -2

            if column == (col_num - 1):
                free_term_matrix[eq_num] -= border_value
            else:
                result_matrix[eq_num, var_num + 1] = 1
    
    result_matrix /= step**2
    free_term_matrix /= step**2

    return result_matrix, free_term_matrix

def _diff_vert_matrix(matrix: np.ndarray, step: float, border_value: float) -> tuple:
    """
    NOTE! returns +1 * d^2/dx^2
    """
    print(f"{matrix.shape=}")
    row_num, col_num = np.shape(matrix[1:-1, 1:-1])
    
    if col_num == 0:
        raise ValueError("Call diff_y() instead!")
    
    row_num = np.max([row_num, 1])
    
    print(f"{row_num, col_num=}")

    max_eq_num = row_num * col_num 
    result_matrix = np.zeros(shape=(max_eq_num, max_eq_num))
    free_term_matrix = np.zeros(max_eq_num)

    for row in range(row_num):
        print('in row')
        for column in range(col_num):
            eq_num = row * col_num + column
            var_num = row * col_num + column 
            print(f"{row, row_num=}")
            if row == 0:
                free_term_matrix[eq_num] -= border_value
            else:
                result_matrix[eq_num, col_num * (row - 1) + column] = 1

            result_matrix[eq_num, var_num] = -2

            if row == (row_num - 1):
                free_term_matrix[eq_num] -= border_value
            else:
                result_matrix[eq_num, col_num * (row + 1) + column] = 1
    
    result_matrix /= step**2
    free_term_matrix /= step**2

    return result_matrix, free_term_matrix

class LaplaceEquation():
    def __init__(self, grid_x_step, grid_y_step, 
                 x_bounds=(0, 1), y_bounds=(0, 1), border_value=0):
        
        x_start, x_end = x_bounds
        y_start, y_end = y_bounds
        x = np.linspace(x_start, x_end, num=int((x_end - x_start) / grid_x_step) + 1)
        y = np.linspace(y_start, y_end, num=int((y_end - y_start) / grid_y_step) + 1)

        self.x_step = grid_x_step
        self.y_step = grid_y_step
        self.x_points, self.y_points = np.meshgrid(x, y) # removing borders because we know border_value
        self.border_value = border_value

    def diff_x(self):
        return _diff_matrix(self.x_points, self.x_step, self.border_value)
    
    def diff_y(self):
        #TODO: fix ПОРЯДОК u_{i}
        diff_y_matrix, free_term_matrix = _diff_matrix(self.y_points, self.y_step, self.border_value)
        # return _diff_matrix_2(self.y_points, self.y_step, self.border_value)
        return _diff_vert_matrix(self.y_points, self.y_step, self.border_value)
        return (diff_y_matrix, free_term_matrix)


def main():  
    # a = LaplaceEquation(1/2, 0.25, y_bounds=(0,0))
    a = LaplaceEquation(0.25, 0.25, x_bounds=(0,0))
    a = LaplaceEquation(0.25, 0.25, x_bounds=(0.25, 1), y_bounds=(0.25, 1))

    # print(-a.diff_x()[0])
    print(-a.diff_y()[0])

    # print(a.diff_x()[1])


if __name__ == "__main__":
    main()
