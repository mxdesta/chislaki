"""
Задача 1.2: Метод прогонки для трехдиагональных матриц
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import TRIDIAG_A, TRIDIAG_B, TRIDIAG_C, TRIDIAG_D
from utils.logger import IterationLogger, print_final_summary


def tridiagonal_solve(A, b, logger=None):
    """Метод прогонки"""
    n = len(A)
    
    if logger:
        logger.log_matrix("Трехдиагональная матрица A", A)
        logger.log_matrix("Вектор правых частей b", b)
    
    # Forward (прямой ход)
    P = [0 for _ in range(n)]
    Q = [0 for _ in range(n)]
    
    P[0] = A[0][1] / -A[0][0]
    Q[0] = b[0] / A[0][0]
    
    if logger:
        logger.log_iteration(1, {
            "Прямой ход - начало": "Вычисляем P[0] и Q[0]",
            "P[0]": P[0],
            "Q[0]": Q[0],
            "Формула P[0]": f"{A[0][1]} / (-{A[0][0]}) = {P[0]}",
            "Формула Q[0]": f"{b[0]} / {A[0][0]} = {Q[0]}"
        })
    
    for i in range(1, n-1):
        denominator = -A[i][i] - A[i][i-1] * P[i-1]
        P[i] = A[i][i+1] / denominator
        Q[i] = (A[i][i-1] * Q[i-1] - b[i]) / denominator
        
        if logger:
            logger.log_iteration(i + 1, {
                "Прямой ход - шаг": i + 1,
                f"P[{i}]": P[i],
                f"Q[{i}]": Q[i],
                "Знаменатель": denominator,
                f"Формула P[{i}]": f"{A[i][i+1]} / {denominator} = {P[i]}",
                f"Формула Q[{i}]": f"({A[i][i-1]} * {Q[i-1]} - {b[i]}) / {denominator} = {Q[i]}"
            })
    
    P[n-1] = 0
    Q[n-1] = (A[n-1][n-2] * Q[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * P[n-2])
    
    if logger:
        logger.log_iteration(n, {
            "Прямой ход - финал": f"Вычисляем Q[{n-1}]",
            f"P[{n-1}]": P[n-1],
            f"Q[{n-1}]": Q[n-1],
            "Массив P": P,
            "Массив Q": Q
        })

    # Backward (обратный ход)
    x = [0 for _ in range(n)]
    x[n-1] = Q[n-1]
    
    if logger:
        logger.log_iteration(n + 1, {
            "Обратный ход - начало": f"x[{n-1}] = Q[{n-1}] = {Q[n-1]}",
            f"x[{n-1}]": x[n-1]
        })
    
    for i in range(n-1, 0, -1):
        x[i-1] = P[i-1] * x[i] + Q[i-1]
        
        if logger:
            logger.log_iteration(n + 1 + (n-i), {
                "Обратный ход - шаг": n - i + 1,
                f"x[{i-1}]": x[i-1],
                f"Формула x[{i-1}]": f"{P[i-1]} * {x[i]} + {Q[i-1]} = {x[i-1]}"
            })
    
    return x


# Инициализация логгера
logger = IterationLogger("Метод_прогонки")

# Создаем полную матрицу из диагоналей
n = 5
A = [[0 for _ in range(n)] for _ in range(n)]
for i in range(n):
    A[i][i] = TRIDIAG_B[i]
    if i > 0:
        A[i][i-1] = TRIDIAG_A[i]
    if i < n - 1:
        A[i][i+1] = TRIDIAG_C[i]

b = TRIDIAG_D.tolist()

# Решение методом прогонки
x = tridiagonal_solve(A, b, logger)

# Проверка результатов
import numpy as np
A_np = np.array(A)
b_np = np.array(b)
x_np = np.array(x)

# Финальные результаты
final_results = {
    "Решение x": x,
    "Проверка ||Ax - b||": float(np.linalg.norm(A_np @ x_np - b_np)),
    "Размерность системы": n,
    "Метод": "Прогонка"
}

logger.log_final_result(final_results)
print_final_summary("Метод прогонки", final_results)

print(f"\nПодробный лог сохранен в: {logger.get_log_file_path()}")
