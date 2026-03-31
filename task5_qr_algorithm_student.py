"""
Задача 1.5: QR-разложение и QR-алгоритм для нахождения собственных значений
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import MATRIX_1_5
from utils.logger import IterationLogger, print_final_summary


def sign(x):
    """Функция знака """
    return -1 if x < 0 else 1 if x > 0 else 0


def L2_norm(vec):
    """L2 норма вектора"""
    ans = 0
    for num in vec:
        ans += num * num
    return np.sqrt(ans)


def get_householder_matrix(A, col_num):
    """Получение матрицы отражения Хаусхолдера"""
    n = A.shape[0]
    v = np.zeros(n)
    a = A[:, col_num]
    v[col_num] = a[col_num] + sign(a[col_num]) * L2_norm(a[col_num:])
    for i in range(col_num + 1, n):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def QR_decomposition(A):
    """QR-разложение методом отражений Хаусхолдера"""
    n = A.shape[0]
    Q = np.eye(n)
    A_i = np.copy(A)

    for i in range(n - 1):
        H = get_householder_matrix(A_i, i)
        Q = Q @ H
        A_i = H @ A_i
    return Q, A_i


def get_roots(A, i):
    """Получение корней характеристического полинома 2x2 блока"""
    n = A.shape[0]
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def is_complex(A, i, eps):
    """Проверка на комплексные собственные значения"""
    Q, R = QR_decomposition(A)
    A_next = np.dot(R, Q)
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps


def get_eigen_value(A, i, eps, logger=None, iteration_offset=0):
    """Получение собственного значения для позиции i"""
    A_i = np.copy(A)
    local_iterations = 0
    
    while True:
        Q, R = QR_decomposition(A_i)
        A_i = R @ Q
        local_iterations += 1
        
        if logger and local_iterations % 10 == 0:
            logger.log_iteration(iteration_offset + local_iterations, {
                "Локальная итерация": local_iterations,
                "Позиция i": i,
                "Поддиагональный элемент": A_i[i + 1, i] if i + 1 < A_i.shape[0] else 0,
                "Критерий сходимости": f"||A[{i+1}, {i}]|| = {abs(A_i[i + 1, i]) if i + 1 < A_i.shape[0] else 0} <= {eps}"
            })
        
        if L2_norm(A_i[i + 1:, i]) <= eps:
            if logger:
                logger.log_iteration(iteration_offset + local_iterations, {
                    "Статус": f"Действительное собственное значение найдено на позиции {i}",
                    "Значение": A_i[i][i],
                    "Локальных итераций": local_iterations
                })
            return A_i[i][i], A_i
        elif L2_norm(A_i[i + 2:, i]) <= eps and is_complex(A_i, i, eps):
            roots = get_roots(A_i, i)
            if logger:
                logger.log_iteration(iteration_offset + local_iterations, {
                    "Статус": f"Комплексные собственные значения найдены на позиции {i}",
                    "Значения": roots,
                    "Локальных итераций": local_iterations
                })
            return roots, A_i


def get_eigen_values_QR(A, eps, logger=None):
    """Получение всех собственных значений QR-алгоритмом"""
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []
    total_iterations = 0

    if logger:
        logger.log_matrix("Исходная матрица A", A)
        logger.log_iteration(0, {
            "Размерность": n,
            "Точность": eps,
            "Метод": "QR-алгоритм "
        
        })

    i = 0
    while i < n:
        if logger:
            logger.log_iteration(total_iterations + 1, {
                "Обрабатываем позицию": i,
                "Оставшаяся размерность": n - i
            })
            
        cur_eigen_values, A_i_plus_1 = get_eigen_value(A_i, i, eps, logger, total_iterations)
        
        if isinstance(cur_eigen_values, np.ndarray):
            # complex
            eigen_values.extend(cur_eigen_values)
            i += 2
            total_iterations += 20  # Примерная оценка итераций для комплексного случая
        else:
            # real
            eigen_values.append(cur_eigen_values)
            i += 1
            total_iterations += 10  # Примерная оценка итераций для действительного случая
            
        A_i = A_i_plus_1
        
    return eigen_values, total_iterations


# Инициализация логгера
logger = IterationLogger("QR_алгоритм")

# Данные из файла
A = MATRIX_1_5.astype(float)
eps = 0.00000000000001

print('QR-алгоритм')

# Сначала проверим QR-разложение
Q, R = QR_decomposition(A)
logger.log_matrix("Ортогональная матрица Q", Q)
logger.log_matrix("Верхнетреугольная матрица R", R)
logger.log_matrix("Проверка A = QR", Q @ R)
logger.log_matrix("Проверка ортогональности Q^T * Q", Q.T @ Q)

# Применяем QR-алгоритм
eig_values, total_iters = get_eigen_values_QR(A, eps, logger)

print('Собственные значения:', eig_values)
print('Примерное количество итераций:', total_iters)

# Сравнение с numpy
eigenvalues_np = np.linalg.eigvals(A)

# Финальные результаты
final_results = {
    "Собственные значения": eig_values,
    "Примерное количество итераций": total_iters,
    "Собственные значения (numpy)": eigenvalues_np,
    "QR-разложение корректно": np.allclose(Q @ R, A),
    "Q ортогональна": np.allclose(Q.T @ Q, np.eye(len(A))),
    "Метод": "QR-алгоритм"
}

logger.log_final_result(final_results)
print_final_summary("QR-алгоритм", final_results)

print(f"\nПодробный лог сохранен в: {logger.get_log_file_path()}")