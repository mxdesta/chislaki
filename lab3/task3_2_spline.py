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


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Решение трехдиагональной системы методом прогонки
    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    """
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    # Прямой ход
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
    
    d_prime[n - 1] = (d[n - 1] - a[n - 1] * d_prime[n - 2]) / (b[n - 1] - a[n - 1] * c_prime[n - 2])
    
    # Обратный ход
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x


def build_cubic_spline(x_nodes: np.ndarray, y_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Построение кубического сплайна с нулевой кривизной на концах
    
    S(x) = a_i + b_i(x - x_{i-1}) + c_i(x - x_{i-1})^2 + d_i(x - x_{i-1})^3
    для x ∈ [x_{i-1}, x_i], i = 1,...,n
    
    Возвращает: (a, b, c, d) - коэффициенты сплайнов
    """
    n = len(x_nodes)
    h = np.diff(x_nodes)  # h_i = x_i - x_{i-1}
    
    # Построение системы для нахождения c_i (i = 2,...,n)
    # Система имеет размерность (n-1) x (n-1)
    # n - это количество узлов, количество интервалов = n-1
    num_intervals = n - 1
    system_size = num_intervals - 1  # размер системы для c_2, ..., c_n
    
    A = np.zeros(system_size)  # нижняя диагональ
    B = np.zeros(system_size)  # главная диагональ
    C = np.zeros(system_size)  # верхняя диагональ
    D = np.zeros(system_size)  # правая часть
    
    # Первое уравнение: 2(h_1 + h_2)c_2 + h_2*c_3 = 3[(f_2-f_1)/h_2 - (f_1-f_0)/h_1]
    B[0] = 2 * (h[0] + h[1])
    if system_size > 1:
        C[0] = h[1]
    D[0] = 3 * ((y_nodes[2] - y_nodes[1]) / h[1] - (y_nodes[1] - y_nodes[0]) / h[0])
    
    # Средние уравнения
    for i in range(1, system_size - 1):
        A[i] = h[i]
        B[i] = 2 * (h[i] + h[i + 1])
        C[i] = h[i + 1]
        D[i] = 3 * ((y_nodes[i + 2] - y_nodes[i + 1]) / h[i + 1] - 
                    (y_nodes[i + 1] - y_nodes[i]) / h[i])
    
    # Последнее уравнение
    if system_size > 1:
        A[system_size - 1] = h[system_size - 1]
        B[system_size - 1] = 2 * (h[system_size - 1] + h[system_size])
        D[system_size - 1] = 3 * ((y_nodes[system_size + 1] - y_nodes[system_size]) / h[system_size] - 
                        (y_nodes[system_size] - y_nodes[system_size - 1]) / h[system_size - 1])
    
    # Решаем систему для c_2, ..., c_n
    c_inner = solve_tridiagonal(A, B, C, D)
    
    # Формируем полный массив c (c_1 = 0, c_2,...,c_n)
    c = np.zeros(n)
    c[1:1+len(c_inner)] = c_inner
    
    # Вычисляем остальные коэффициенты
    a = y_nodes[:-1]  # a_i = f_{i-1}, i = 1,...,n
    
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    
    for i in range(n - 2):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    
    # Последний интервал
    b[n - 2] = (y_nodes[n - 1] - y_nodes[n - 2]) / h[n - 2] - 2 * h[n - 2] * c[n - 1] / 3
    d[n - 2] = -c[n - 1] / (3 * h[n - 2])
    
    return a, b, c[:-1], d


def evaluate_spline(x: float, x_nodes: np.ndarray, a: np.ndarray, b: np.ndarray, 
                   c: np.ndarray, d: np.ndarray) -> float:
    """
    Вычисление значения сплайна в точке x
    """
    # Находим интервал, которому принадлежит x
    i = np.searchsorted(x_nodes[1:], x)
    
    if i >= len(a):
        i = len(a) - 1
    
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
    plt.savefig('chislaki/lab3/task3_2_spline.png', dpi=150)
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
