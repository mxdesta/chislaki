"""
Задача 1.3: Метод простых итераций и метод Зейделя
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import MATRIX_1_3, VECTOR_1_3
from utils.logger import IterationLogger, print_final_summary


def L1_norm(X):
    """Вычисление L1 нормы
    
    Для вектора: ||x||₁ = Σ|x_i|
    Для матрицы: ||A||₁ = max по j (Σ|a_ij|) — максимум суммы модулей по столбцам
    """
    if X.ndim == 1:  # Вектор
        norm = 0
        for i in range(len(X)):
            norm += abs(X[i])
        return norm
    else:  # Матрица
        n = X.shape[1]  # Количество столбцов
        max_col_sum = 0
        for j in range(n):  # Идем по столбцам
            col_sum = 0
            for i in range(n):  # Суммируем по строкам в столбце j
                col_sum += abs(X[i][j])
            if col_sum > max_col_sum:
                max_col_sum = col_sum
        return max_col_sum


def solve_iterative(A, b, eps, logger=None):
    """Метод простых итераций"""
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')

    # Преобразование к виду x = alpha * x + beta
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
        beta[i] = b[i] / A[i][i]

    if logger:
        logger.log_matrix("Матрица alpha", alpha)
        logger.log_matrix("Вектор beta", beta)
        logger.log_iteration(0, {
            "Норма alpha": L1_norm(alpha),
            "Условие сходимости": f"||alpha|| = {L1_norm(alpha)} < 1: {L1_norm(alpha) < 1}"
        })

    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    
    while not converge:
        prev_x = np.copy(cur_x)
        cur_x = alpha @ prev_x + beta
        iterations += 1
        
        if L1_norm(alpha) < 1:
            error_estimate = L1_norm(alpha) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x)
            converge = error_estimate <= eps
        else:
            converge = L1_norm(cur_x - prev_x) <= eps
        
        if logger and (iterations <= 5 or iterations % 5 == 0):
            logger.log_iteration(iterations, {
                "Итерация": iterations,
                "x": cur_x.copy(),
                "Погрешность ||x_new - x_old||": L1_norm(cur_x - prev_x),
                "Оценка погрешности": error_estimate if L1_norm(alpha) < 1 else "N/A",
                "Критерий останова": f"{error_estimate if L1_norm(alpha) < 1 else L1_norm(cur_x - prev_x)} <= {eps}",
                "Сходимость": converge
            })
            
    return cur_x, iterations


def seidel_multiplication(alpha, x, beta):
    """Шаг итерации метода Зейделя"""
    res = np.copy(x)
    c = np.copy(alpha)
    for i in range(alpha.shape[0]):
        res[i] = beta[i]
        for j in range(alpha.shape[1]):
            res[i] += alpha[i][j] * res[j]
            if j < i:
                c[i][j] = 0
    return res, c


def solve_seidel(A, b, eps, logger=None):
    """Метод Зейделя"""
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')
    
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
        beta[i] = b[i] / A[i][i]

    if logger:
        logger.log_matrix("Матрица alpha", alpha)
        logger.log_matrix("Вектор beta", beta)

    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    
    while not converge:
        prev_x = np.copy(cur_x)
        cur_x, c = seidel_multiplication(alpha, prev_x, beta)
        iterations += 1
        
        if L1_norm(alpha) < 1:
            error_estimate = L1_norm(c) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x)
            converge = error_estimate <= eps
        else:
            converge = L1_norm(prev_x - cur_x) <= eps
        
        if logger and (iterations <= 5 or iterations % 5 == 0):
            logger.log_iteration(iterations, {
                "Итерация": iterations,
                "x": cur_x.copy(),
                "Погрешность ||x_new - x_old||": L1_norm(cur_x - prev_x),
                "Матрица c": c,
                "Оценка погрешности": error_estimate if L1_norm(alpha) < 1 else "N/A",
                "Сходимость": converge
            })
            
    return cur_x, iterations


logger_si = IterationLogger("Простые_итерации")
logger_gs = IterationLogger("Метод_Зейделя")

A = MATRIX_1_3.astype(float)
b = VECTOR_1_3.astype(float)
eps = 0.000000001

logger_si.log_matrix("Матрица A", A)
logger_si.log_matrix("Вектор b", b)
logger_gs.log_matrix("Матрица A", A)
logger_gs.log_matrix("Вектор b", b)

print('Метод простых итераций')
x_iter, i_iter = solve_iterative(A, b, eps, logger_si)
print('Решение:', x_iter)
print('Кол-во итераций:', i_iter)
print()

print('Метод Зейделя')
x_seidel, i_seidel = solve_seidel(A, b, eps, logger_gs)
print('Решение:', x_seidel)
print('Кол-во итераций:', i_seidel)

x_exact = np.linalg.solve(A, b)

final_results_si = {
    "Решение": x_iter,
    "Количество итераций": i_iter,
    "Погрешность ||Ax - b||": np.linalg.norm(A @ x_iter - b),
    "Погрешность от точного решения": np.linalg.norm(x_iter - x_exact),
    "Метод": "Простые итерации"
}

final_results_gs = {
    "Решение": x_seidel,
    "Количество итераций": i_seidel,
    "Погрешность ||Ax - b||": np.linalg.norm(A @ x_seidel - b),
    "Погрешность от точного решения": np.linalg.norm(x_seidel - x_exact),
    "Ускорение относительно простых итераций": i_iter / i_seidel if i_seidel > 0 else "N/A"
}

logger_si.log_final_result(final_results_si)
logger_gs.log_final_result(final_results_gs)

print_final_summary("Метод простых итераций", final_results_si)
print_final_summary("Метод Зейделя", final_results_gs)

print(f"\nЛоги сохранены в:")
print(f"Простые итерации: {logger_si.get_log_file_path()}")
print(f"Метод Зейделя: {logger_gs.get_log_file_path()}")
