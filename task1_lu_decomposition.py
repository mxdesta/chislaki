"""
Задача 1.1: LU-разложение с выбором главного элемента
Решение СЛАУ, вычисление определителя и обратной матрицы
"""
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import MATRIX_1_1, VECTOR_1_1
from utils.logger import IterationLogger, print_final_summary


def LU_decompose(A, logger=None):
    """LU-разложение с выбором главного элемента (как в методичке)
    
    Алгоритм:
    1. Применяем метод Гаусса к матрице A, получаем верхнюю треугольную U
    2. Множители μ(k)ij запоминаем в нижнюю треугольную матрицу L
    3. В результате: A = L × U (с учетом перестановок P)
    """
    n = len(A)
    
    # A(k) - матрица на k-м шаге (изначально копия A)
    A_k = [row[:] for row in A]
    
    # L - нижняя треугольная с единицами на диагонали
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    
    # P - вектор перестановок
    P = list(range(n))
    swaps = 0
    
    if logger:
        logger.log_matrix("Исходная матрица A(0)", A_k)
        logger.log_iteration(0, {
            "Этап": "Начало метода Гаусса",
            "Цель": "Привести A к верхней треугольной матрице U",
            "Попутно": "Запоминать множители в матрицу L"
        })
    
    # Прямой ход метода Гаусса (n-1 шагов)
    for k in range(n - 1):
        if logger:
            logger.log_iteration(k + 0.05, {
                f"=== ШАГ {k + 1} МЕТОДА ГАУССА ===": f"Обнуляем столбец {k} под диагональю",
                "Текущая матрица A(k)": [row[:] for row in A_k]
            })
        
        # ШАГ 1: Выбор главного элемента в столбце k
        max_row = k
        max_val = abs(A_k[k][k])
        
        for i in range(k + 1, n):
            if abs(A_k[i][k]) > max_val:
                max_val = abs(A_k[i][k])
                max_row = i
        
        # ШАГ 2: Перестановка строк (если нужно)
        if max_row != k:
            A_k[k], A_k[max_row] = A_k[max_row], A_k[k]
            P[k], P[max_row] = P[max_row], P[k]
            
            # Меняем уже вычисленные элементы L
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            
            swaps += 1
            
            if logger:
                logger.log_iteration(k + 0.1, {
                    "Выбор главного элемента": f"a({k},{k}) = {max_val:.4f}",
                    "Перестановка": f"Строка {k} ↔ Строка {max_row}",
                    "Матрица после перестановки": [row[:] for row in A_k]
                })
        
        # ШАГ 3: Проверка на вырожденность
        if abs(A_k[k][k]) < 1e-10:
            raise ValueError(f"Матрица вырождена на шаге {k}")
        
        # ШАГ 4: Вычисление множителей μ(k)ij и обнуление
        multipliers = {}
        
        for i in range(k + 1, n):
            # Вычисляем множитель μ(k)ij = a(k)ik / a(k)kk
            mu = A_k[i][k] / A_k[k][k]
            L[i][k] = mu  # Запоминаем в L
            multipliers[f"μ({k+1}){i+1}{k+1}"] = mu
            
            # Вычитаем: строка_i = строка_i - μ × строка_k
            for j in range(k, n):
                A_k[i][j] = A_k[i][j] - mu * A_k[k][j]
        
        if logger:
            logger.log_iteration(k + 1, {
                f"Множители на шаге {k + 1}": multipliers,
                "Формула": f"строка_i = строка_i - μ(k)_ik × строка_{k}",
                f"Матрица A({k + 1}) после обнуления": [row[:] for row in A_k],
                "Текущая матрица L": [row[:] for row in L]
            })
    
    # A_k теперь содержит верхнюю треугольную матрицу U
    U = A_k
    
    if logger:
        logger.log_iteration(n, {
            "=== ПРЯМОЙ ХОД ЗАВЕРШЕН ===": "Получена верхняя треугольная матрица U",
            "Финальная матрица U": U,
            "Финальная матрица L": L,
            "Проверка": "L × U должно дать исходную A (с учетом перестановок)"
        })
    
    return L, U, P, swaps


def solve_system(L, U, b, P, logger=None):
    """Решение системы LUx = Pb через прямой и обратный ход"""
    n = len(L)
    
    # Применяем перестановки к вектору b
    Pb = [b[P[i]] for i in range(n)]
    
    if logger:
        logger.log_iteration(0, {
            "Этап": "Применение перестановок к вектору b",
            "Исходный b": b,
            "Вектор перестановок P": P,
            "Pb после перестановок": Pb
        })
    
    # Прямой ход: Ly = Pb
    y = [0 for _ in range(n)]
    
    if logger:
        logger.log_iteration(0.5, {
            "Этап": "Прямой ход (Ly = Pb)",
            "Матрица L": L,
            "Вектор Pb": Pb
        })

    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (Pb[i] - s) / L[i][i]
        
        if logger:
            logger.log_iteration(i + 1, {
                "Прямой ход - шаг": i + 1,
                f"y[{i}]": y[i],
                "Сумма s": s,
                "Формула": f"({Pb[i]} - {s}) / {L[i][i]} = {y[i]}"
            })

    if logger:
        logger.log_iteration(n + 1, {
            "Этап": "Обратный ход (Ux = y)",
            "Матрица U": U,
            "Вектор y": y
        })

    # Обратный ход: Ux = y
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
        
        if logger:
            logger.log_iteration(n + 2 + (n-1-i), {
                "Обратный ход - шаг": n - i,
                f"x[{i}]": x[i],
                "Сумма s": s,
                "Формула": f"({y[i]} - {s}) / {U[i][i]} = {x[i]}"
            })
    
    return x


def determinant(A):
    """Вычисление определителя через LU-разложение с учетом перестановок"""
    _, U, P, swaps = LU_decompose(A)
    
    # det(A) = (-1)^swaps × произведение диагональных элементов U
    det = (-1) ** swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det


def inverse_matrix(A):
    """Вычисление обратной матрицы"""
    n = len(A)
    E = [[float(i == j) for i in range(n)] for j in range(n)]
    L, U, P, swaps = LU_decompose(A)
    A_inv = []
    for e in E:
        inv_row = solve_system(L, U, e, P)
        A_inv.append(inv_row)
    return transpose(A_inv)


def transpose(X):
    """Транспонирование матрицы"""
    m = len(X)
    n = len(X[0])
    transpose_matrix = [[X[j][i] for j in range(n)] for i in range(m)]
    return transpose_matrix


# Инициализация логгера
logger = IterationLogger("LU_разложение")

# Преобразуем numpy матрицы в списки
A = MATRIX_1_1.tolist()
b = VECTOR_1_1.tolist()

logger.log_matrix("Исходная матрица A", A)
logger.log_matrix("Вектор правых частей b", b)

print("LU разложение с выбором главного элемента")
L, U, P, swaps = LU_decompose(A, logger)
logger.log_matrix("Финальная матрица L", L)
logger.log_matrix("Финальная матрица U", U)
logger.log_iteration(999, {
    "Вектор перестановок P": P,
    "Количество перестановок": swaps,
    "Знак определителя": f"(-1)^{swaps} = {(-1)**swaps}"
})

print("Решение системы")
x = solve_system(L, U, b, P, logger)
logger.log_matrix("Решение x", x)

print("Детерминант")
det = determinant(A)

print("Обратная матрица")
A_inv = inverse_matrix(A)
logger.log_matrix("Обратная матрица", A_inv)

# Проверка результатов
import numpy as np
A_np = np.array(A)
b_np = np.array(b)
x_np = np.array(x)

# Финальные результаты
final_results = {
    "Решение СЛАУ": x,
    "Определитель": det,
    "Количество перестановок строк": swaps,
    "Вектор перестановок": P,
    "Погрешность решения ||Ax - b||": float(np.linalg.norm(A_np @ x_np - b_np)),
    "Обратная матрица (первая строка)": A_inv[0] if A_inv else "Ошибка вычисления",
    "Метод": "LU-разложение с выбором главного элемента"
}

logger.log_final_result(final_results)
print_final_summary("LU-разложение", final_results)

print(f"\nПодробный лог сохранен в: {logger.get_log_file_path()}")
