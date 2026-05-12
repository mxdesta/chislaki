"""
Лабораторная работа №3. Задание 3.2
Кубический сплайн

Вариант 14: X* = 1.5
Таблица:
i  | 0   | 1      | 2      | 3      | 4
xi | 0.0 | 0.9    | 1.8    | 2.7    | 3.6
yi | 0.0 | 0.72235| 1.5609 | 2.8459 | 7.7275
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# Добавляем путь для импорта из chislaki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем функции LU разложения из task1
from task1_lu_decomposition import LU_decompose, solve_system


def solve_tridiagonal_with_lu(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Решение трехдиагональной системы с использованием LU разложения
    
    Система имеет вид:
    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    
    Args:
        a: нижняя диагональ (коэффициенты при x[i-1])
        b: главная диагональ (коэффициенты при x[i])
        c: верхняя диагональ (коэффициенты при x[i+1])
        d: правая часть системы
    
    Returns:
        Решение системы x
    """
    n = len(d)
    
    # Создаем полную матрицу из трех диагоналей
    A_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Заполняем главную диагональ
    for i in range(n):
        A_matrix[i][i] = b[i]
    
    # Заполняем верхнюю диагональ (кроме последнего элемента)
    for i in range(n - 1):
        A_matrix[i][i + 1] = c[i]
    
    # Заполняем нижнюю диагональ (кроме первого элемента)
    for i in range(1, n):
        A_matrix[i][i - 1] = a[i]
    
    # Преобразуем вектор d в список
    d_list = d.tolist() if isinstance(d, np.ndarray) else list(d)
    
    # Используем LU разложение для решения системы
    L, U, P, swaps = LU_decompose(A_matrix)
    x = solve_system(L, U, d_list, P)
    
    return np.array(x)


def build_cubic_spline(x_nodes: np.ndarray, y_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x_nodes) - 1  # количество интервалов (n интервалов для n+1 узлов)
    
    # Вычисляем шаги между узлами: h_i = x_i - x_{i-1}
    h = np.diff(x_nodes)  # h[0] = x_1 - x_0, ..., h[n-1] = x_n - x_{n-1}
    
    
    if n == 1:
        # Только один интервал - вырожденный случай
        # c_1 = c_2 = 0 (оба граничных условия)
        c_full = np.zeros(2) 
    else:
        
        system_size = n - 1  
        
        A = np.zeros(system_size)  # нижняя диагональ
        B = np.zeros(system_size)  # главная диагональ
        C = np.zeros(system_size)  # верхняя диагональ
        D = np.zeros(system_size)  # правая часть
        
        # Уравнения системы (3.13):
        # h_{i-1}*c_{i-1} + 2(h_{i-1} + h_i)*c_i + h_i*c_{i+1} = 3[(f_i - f_{i-1})/h_i - (f_{i-1} - f_{i-2})/h_{i-1}]
        # для i = 2, 3, ..., n-1
        
        B[0] = 2 * (h[0] + h[1])
        if system_size > 1:
            C[0] = h[1]
        D[0] = 3 * ((y_nodes[2] - y_nodes[1]) / h[1] - (y_nodes[1] - y_nodes[0]) / h[0])
        
        # Средние уравнения (для c_3, ..., c_{n-1})
        for i in range(1, system_size - 1):
            # Уравнение для c_{i+1} (индекс i+1 в массиве c, т.к. c_1=0)
            A[i] = h[i]  # коэффициент при c_i
            B[i] = 2 * (h[i] + h[i + 1])  # коэффициент при c_{i+1}
            C[i] = h[i + 1]  # коэффициент при c_{i+2}
            D[i] = 3 * ((y_nodes[i + 2] - y_nodes[i + 1]) / h[i + 1] - 
                        (y_nodes[i + 1] - y_nodes[i]) / h[i])
        
        # Последнее уравнение (для c_n):
        # h_{n-1}*c_{n-1} + 2(h_{n-1} + h_n)*c_n = 3[(f_n - f_{n-1})/h_n - (f_{n-1} - f_{n-2})/h_{n-1}]
        if system_size > 1:
            A[system_size - 1] = h[n - 2]
            B[system_size - 1] = 2 * (h[n - 2] + h[n - 1])
            # C[system_size - 1] = 0 (т.к. c_{n+1} = 0 из граничного условия)
            D[system_size - 1] = 3 * ((y_nodes[n] - y_nodes[n - 1]) / h[n - 1] - 
                                      (y_nodes[n - 1] - y_nodes[n - 2]) / h[n - 2])
        
        # Решаем трехдиагональную систему
        c_inner = solve_tridiagonal_with_lu(A, B, C, D)
        
        c_full = np.zeros(n + 1)
        c_full[1:n] = c_inner  # c_inner содержит c_2, ..., c_n

    # Возвращаем коэффициенты для n интервалов
    
    a = np.zeros(n)  # a_i = f_{i-1}
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        # Интервал i в Python = интервал i+1 в методичке: [x_i, x_{i+1}]
        a[i] = y_nodes[i]
        c[i] = c_full[i]  # c[i] = c_{i+1} в методичке
        
        if i < n - 1:
            # b_i = (f_i - f_{i-1})/h_i - (1/3)*h_i*(c_{i+1} + 2*c_i)
            # d_i = (c_{i+1} - c_i) / (3*h_i)
            b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (c_full[i + 1] + 2 * c_full[i]) / 3
            d[i] = (c_full[i + 1] - c_full[i]) / (3 * h[i])
        else:

            # b_n = (f_n - f_{n-1})/h_n - (2/3)*h_n*c_n
            # d_n = -c_n/(3*h_n)
            # где c_n = c_full[n-1] (c_n для последнего интервала)
            c_n = c_full[n - 1]  # c_n для последнего интервала
            b[i] = (y_nodes[n] - y_nodes[n - 1]) / h[n - 1] - 2 * h[n - 1] * c_n / 3
            d[i] = -c_n / (3 * h[n - 1])
    
    return a, b, c, d


def evaluate_spline(x: float, x_nodes: np.ndarray, a: np.ndarray, b: np.ndarray, 
                   c: np.ndarray, d: np.ndarray) -> float:
    """
    Вычисление значения сплайна в точке x
    
    Сплайн на интервале [x_{i-1}, x_i] имеет вид:
    S(x) = a_i + b_i*(x - x_{i-1}) + c_i*(x - x_{i-1})^2 + d_i*(x - x_{i-1})^3
    """
    n = len(a)  # количество интервалов
    
    # Обработка граничных случаев
    if x <= x_nodes[0]:
        i = 0
    elif x >= x_nodes[-1]:
        i = n - 1  # последний интервал
    else:
        i = np.searchsorted(x_nodes, x, side='right') - 1
        # i указывает на левый конец интервала, значит интервал [x_i, x_{i+1}]
        # но коэффициенты a[i], b[i], c[i], d[i] соответствуют интервалу [x_i, x_{i+1}]
    
    # Вычисляем значение сплайна
    dx = x - x_nodes[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3


def print_spline_coefficients(x_nodes: np.ndarray, a: np.ndarray, b: np.ndarray, 
                              c: np.ndarray, d: np.ndarray):
    """Вывод таблицы коэффициентов сплайна"""
    print("\nТаблица коэффициентов кубического сплайна:")
    print("-" * 90)
    print(f"{'i':<5} {'[x_{i-1}, x_i]':<20} {'a_i':<15} {'b_i':<15} {'c_i':<15} {'d_i':<15}")
    print("-" * 90)
    
    for i in range(len(a)):
        interval = f"[{x_nodes[i]:.1f}, {x_nodes[i+1]:.1f}]"
        print(f"{i+1:<5} {interval:<20} {a[i]:<15.8f} {b[i]:<15.8f} {c[i]:<15.8f} {d[i]:<15.8f}")
    
    print("-" * 90)


def solve_task():
    """Решение задания 3.2"""
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. ЗАДАНИЕ 3.2")
    print("КУБИЧЕСКИЙ СПЛАЙН")
    print("Вариант 14: X* = 1.5")
    print("=" * 80)
    
    # Исходные данные
    x_nodes = np.array([0.0, 0.9, 1.8, 2.7, 3.6])
    y_nodes = np.array([0.0, 0.72235, 1.5609, 2.8459, 7.7275])
    x_star = 1.5
    
    print("\nИсходные данные:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
    print("-" * 30)
    for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
        print(f"{i:<5} {x:<10.1f} {y:<15.5f}")
    
    # Построение сплайна
    a, b, c, d = build_cubic_spline(x_nodes, y_nodes)
    
    # Вывод коэффициентов
    print_spline_coefficients(x_nodes, a, b, c, d)
    
    # Вычисление значения в точке X*
    y_star = evaluate_spline(x_star, x_nodes, a, b, c, d)
    
    print(f"\nВычисление значения функции в точке X* = {x_star}:")
    print(f"Точка X* = {x_star} принадлежит интервалу [{x_nodes[1]:.1f}, {x_nodes[2]:.1f}]")
    print(f"\nНа этом интервале сплайн имеет вид:")
    print(f"S(x) = {a[1]:.8f} + {b[1]:.8f}(x - {x_nodes[1]}) + {c[1]:.8f}(x - {x_nodes[1]})² + {d[1]:.8f}(x - {x_nodes[1]})³")
    print(f"\nS({x_star}) = {y_star:.8f}")
    
    # Построение графика
    x_plot = np.linspace(x_nodes[0], x_nodes[-1], 500)
    y_plot = [evaluate_spline(x, x_nodes, a, b, c, d) for x in x_plot]
    
    plt.figure(figsize=(12, 7))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Кубический сплайн S(x)')
    plt.plot(x_nodes, y_nodes, 'ro', markersize=10, label='Узлы интерполяции')
    plt.plot(x_star, y_star, 'g^', markersize=15, label=f'X* = {x_star}, S(X*) = {y_star:.5f}')
    
    # Добавляем вертикальные линии для границ интервалов
    for x in x_nodes[1:-1]:
        plt.axvline(x, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Кубический сплайн-интерполяция (Вариант 14)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task3_2_spline.png', dpi=150)
    print(f"\nГрафик сохранен: task3_2_spline.png")
    
    # Проверка непрерывности в узлах
    print("\n" + "=" * 80)
    print("ПРОВЕРКА НЕПРЕРЫВНОСТИ СПЛАЙНА В УЗЛАХ")
    print("=" * 80)
    eps = 1e-10
    for i in range(1, len(x_nodes) - 1):
        left = evaluate_spline(x_nodes[i] - eps, x_nodes, a, b, c, d)
        right = evaluate_spline(x_nodes[i] + eps, x_nodes, a, b, c, d)
        exact = y_nodes[i]
        print(f"x = {x_nodes[i]:.1f}: S(x-ε) = {left:.8f}, S(x+ε) = {right:.8f}, f(x) = {exact:.8f}")
        print(f"  Разрыв: {abs(right - left):.2e}")
    
    print("=" * 80)
    
    return y_star


if __name__ == "__main__":
    result = solve_task()
    plt.show()
