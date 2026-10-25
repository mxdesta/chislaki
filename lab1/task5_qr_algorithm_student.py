"""
Задача 1.5: QR-разложение и QR-алгоритм для нахождения собственных значений
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.matrices import MATRIX_1_5, TEST_MATRIX_5x5
from utils.logger import IterationLogger, print_final_summary


def sign(x):
    """Функция знака числа
    
    Возвращает: -1 если x < 0, 1 если x > 0, 0 если x = 0
    """
    return -1 if x < 0 else 1 if x > 0 else 0


def L2_norm(vec):
    """L2 норма вектора (евклидова норма)
    
    ||v|| = sqrt(v1² + v2² + ... + vn²)
    """
    ans = 0
    for num in vec:
        ans += num * num  # Суммируем квадраты элементов
    return np.sqrt(ans)  # Возвращаем корень из суммы


def get_householder_matrix(A, col_num):
    """Получение матрицы отражения Хаусхолдера для обнуления столбца
    
    Матрица Хаусхолдера: H = Е - 2vv^T / (v^T v)
    где v - специальный вектор отражения
    """
    n = A.shape[0]  # Размерность матрицы
    v = np.zeros(n)  # Вектор отражения
    a = A[:, col_num]  # Берем col_num-й столбец матрицы
    
    # Вычисляем первый элемент вектора v
    # v[col_num] = a[col_num] + sign(a[col_num]) × ||a[col_num:]||
    v[col_num] = a[col_num] + sign(a[col_num]) * L2_norm(a[col_num:])
    
    # Остальные элементы v копируем из столбца a
    for i in range(col_num + 1, n):
        v[i] = a[i]
    
    # Преобразуем v в вектор-столбец (n×1)
    v = v[:, np.newaxis]
    
    # Вычисляем матрицу Хаусхолдера: H = Е - 2vv^T / (v^T v)
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def QR_decomposition(A):
    """QR-разложение методом отражений Хаусхолдера
    
    Разлагаем A = QR, где:
    Q - ортогональная матрица (Q^T Q = I)
    R - верхнетреугольная матрица
    """
    n = A.shape[0]  # Размерность матрицы
    Q = np.eye(n)  # Изначально Q = единичная матрица
    A_i = np.copy(A)  # Рабочая копия матрицы

    # Применяем n-1 отражений Хаусхолдера (для каждого столбца)
    for i in range(n - 1):
        H = get_householder_matrix(A_i, i)  # Получаем матрицу отражения
        Q = Q @ H  # Накапливаем произведение отражений
        A_i = H @ A_i  # Применяем отражение к матрице
    
    # A_i теперь верхнетреугольная (это R)
    return Q, A_i


def get_roots(A, i):
    """Получение корней характеристического полинома 2×2 блока
    
    Для блока [[a11, a12], [a21, a22]] решаем:
    det(A - λI) = λ² - (a11+a22)λ + (a11×a22 - a12×a21) = 0
    
    Решаем квадратное уравнение: a·λ² + b·λ + c = 0
    где a = 1, b = -(a11 + a22), c = a11·a22 - a12·a21
    
    Корни находим по формуле: λ = [-b ± √(b² - 4ac)] / (2a)
    """
    n = A.shape[0]
    # Извлекаем элементы 2×2 блока начиная с позиции (i, i)
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    
    # Коэффициенты квадратного уравнения: λ² + b·λ + c = 0
    b = -a11 - a22 
    c = a11 * a22 - a12 * a21  # свободный член
    
    # Вычисляем дискриминант: D = b² - 4·a·c, где a = 1
    D = b * b - 4 * c
    
    if D >= 0:
        # Действительные корни
        sqrt_D = np.sqrt(D)
        lambda1 = (-b + sqrt_D) / 2
        lambda2 = (-b - sqrt_D) / 2
    else:
        # Комплексные корни
        sqrt_D = np.sqrt(complex(D))  # комплексное извлечение корня
        lambda1 = (-b + sqrt_D) / 2
        lambda2 = (-b - sqrt_D) / 2
    
    return np.array([lambda1, lambda2])


def is_complex(A, i, eps):
    """Проверка на комплексные собственные значения
    
    Если 2×2 блок сходится к одним и тем же комплексным корням,
    значит у матрицы есть пара комплексно-сопряженных собственных значений
    """
    # Делаем одну итерацию QR-алгоритма
    Q, R = QR_decomposition(A)
    A_next = R @ Q
    # Вычисляем корни до и после итерации
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    # Если корни почти не изменились, значит сошлись
    return abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps


def get_eigen_value(A, i, eps, logger=None, iteration_offset=0):
    """Получение собственного значения для позиции i
    
    Применяем QR-итерации пока элементы под диагональю не станут малыми
    """
    A_i = np.copy(A)  # Рабочая копия матрицы
    local_iterations = 0  # Счетчик локальных итераций
    
    while True:
        # Одна итерация QR-алгоритма: A_i = R × Q, где A_i = Q × R
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
        
        # Проверяем, сошелся ли один элемент (действительное собственное значение)
        if L2_norm(A_i[i + 1:, i]) <= eps:
            if logger:
                logger.log_iteration(iteration_offset + local_iterations, {
                    "Статус": f"Действительное собственное значение найдено на позиции {i}",
                    "Значение": A_i[i][i],
                    "Локальных итераций": local_iterations
                })
            return A_i[i][i], A_i
        # Проверяем, сошелся ли 2×2 блок (комплексные собственные значения)
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
    """Получение всех собственных значений QR-алгоритмом
    
    Идея: последовательно находим собственные значения, начиная с первого
    После нахождения каждого значения работаем с оставшейся подматрицей
    """
    n = A.shape[0]  # Размерность матрицы
    A_i = np.copy(A)  # Рабочая копия
    eigen_values = []  # Список собственных значений
    total_iterations = 0  # Общий счетчик итераций

    if logger:
        logger.log_matrix("Исходная матрица A", A)
        logger.log_iteration(0, {
            "Размерность": n,
            "Точность": eps,
            "Метод": "QR-алгоритм "
        
        })

    i = 0  # Текущая позиция
    while i < n:
        if logger:
            logger.log_iteration(total_iterations + 1, {
                "Обрабатываем позицию": i,
                "Оставшаяся размерность": n - i
            })
            
        # Находим собственное значение на позиции i
        cur_eigen_values, A_i_plus_1 = get_eigen_value(A_i, i, eps, logger, total_iterations)
        
        if isinstance(cur_eigen_values, np.ndarray):
            # Нашли пару комплексных собственных значений
            eigen_values.extend(cur_eigen_values)
            i += 2  # Пропускаем 2 позиции
            total_iterations += 20  # Примерная оценка итераций
        else:
            # Нашли действительное собственное значение
            eigen_values.append(cur_eigen_values)
            i += 1  # Переходим к следующей позиции
            total_iterations += 10  # Примерная оценка итераций
            
        A_i = A_i_plus_1  # Обновляем матрицу
        
    return eigen_values, total_iterations


# Инициализация логгера
logger = IterationLogger("QR_алгоритм")

# Данные из файла
A = TEST_MATRIX_5x5
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