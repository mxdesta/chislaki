"""
Задача 1.4: Метод вращений (метод Якоби) для нахождения собственных значений
"""
import numpy as np
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import MATRIX_1_4
from utils.logger import IterationLogger, print_final_summary

EPS = 0.000000000000000001


def find_max_upper_element(X):
    """Находит позицию максимального по модулю элемента в верхнем треугольнике матрицы"""
    n = X.shape[0]
    i_max, j_max = 0, 1
    max_elem = abs(X[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(X[i][j]) > max_elem:
                max_elem = abs(X[i][j])
                i_max = i
                j_max = j

    return i_max, j_max


def matrix_norm(X):
    """Норма матрицы"""
    norm = 0
    for i in range(len(X[0])):
        for j in range(i + 1, len(X[0])):
            norm += X[i][j] * X[i][j]
    return np.sqrt(norm)


def rotation_method(A, logger=None):
    """Вычисляет СЗ и СВ с помощью метода вращений"""
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_vectors = np.eye(n)
    iterations = 0

    if logger:
        logger.log_matrix("Исходная матрица A", A)
        logger.log_iteration(0, {
            "Начальная норма внедиагональных элементов": matrix_norm(A_i),
            "Критерий останова": f"норма < {EPS}"
        })

    while matrix_norm(A_i) > EPS:
        i_max, j_max = find_max_upper_element(A_i)
        
        if A_i[i_max][i_max] - A_i[j_max][j_max] == 0:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A_i[i_max][j_max] / (A_i[i_max][i_max] - A_i[j_max][j_max]))

        U = np.eye(n)
        U[i_max][j_max] = -np.sin(phi)
        U[j_max][i_max] = np.sin(phi)
        U[i_max][i_max] = np.cos(phi)
        U[j_max][j_max] = np.cos(phi)

        A_i = U.T @ A_i @ U
        eigen_vectors = eigen_vectors @ U
        iterations += 1

        if logger and (iterations <= 10 or iterations % 5 == 0):
            logger.log_iteration(iterations, {
                "Итерация": iterations,
                "Максимальный элемент": f"A[{i_max}][{j_max}] = {A_i[i_max][j_max] if iterations == 1 else 'обнулен'}",
                "Позиция максимального элемента": f"({i_max}, {j_max})",
                "Угол поворота phi": phi,
                "cos(phi)": np.cos(phi),
                "sin(phi)": np.sin(phi),
                "Норма внедиагональных элементов": matrix_norm(A_i),
                "Текущие диагональные элементы": np.diag(A_i),
                "Матрица A_i": A_i.copy() if iterations <= 5 else "Слишком большая для лога"
            })

    eigen_values = np.array([A_i[i][i] for i in range(n)])
    
    if logger:
        logger.log_iteration(iterations + 1, {
            "Статус": "СХОДИМОСТЬ ДОСТИГНУТА",
            "Финальная норма": matrix_norm(A_i),
            "Собственные значения": eigen_values,
            "Собственные векторы": eigen_vectors
        })
    
    return eigen_values, eigen_vectors, iterations


logger = IterationLogger("Метод_вращений")

A = MATRIX_1_4.astype(float)

print('Метод вращений')
values, vectors, iters = rotation_method(A, logger)

print('Собственные значения:', values)
print('Собственные векторы:')
print(vectors)
print('Итерации:', iters)

errors = []
for i in range(len(values)):
    v = vectors[:, i]
    Av = A @ v
    lambda_v = values[i] * v
    error = np.linalg.norm(Av - lambda_v)
    errors.append(error)

eigenvalues_np, eigenvectors_np = np.linalg.eigh(A)

final_results = {
    "Собственные значения": values,
    "Собственные векторы (первый столбец)": vectors[:, 0],
    "Количество итераций": iters,
    "Максимальная погрешность проверки": max(errors),
    "Собственные значения (numpy)": eigenvalues_np,
    "Разница с numpy": np.linalg.norm(np.sort(values) - np.sort(eigenvalues_np)),
    "Метод": "Вращения"
}

logger.log_final_result(final_results)
print_final_summary("Метод вращений", final_results)

print(f"\nПодробный лог сохранен в: {logger.get_log_file_path()}")

n = len(A)
Q = np.array([vectors[i][0] for i in range(n)])
print("\nДополнительная проверка первого собственного вектора:")
print("Q:", Q)
print("A @ Q:", A @ Q)
print("Q * λ_1:", Q * values[0])
