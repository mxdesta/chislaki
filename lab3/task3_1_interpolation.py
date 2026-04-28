"""
Лабораторная работа №3. Задание 3.1
Интерполяция функций многочленами Лагранжа и Ньютона

Вариант 14:
y = tg(x) + x
а) Xi = 0, π/8, 2π/8, 3π/8; X* = 3π/16
б) Xi = 0, π/8, π/3, 3π/8; X* = 3π/16
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def lagrange_polynomial(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    """
    Вычисление значения интерполяционного многочлена Лагранжа
    
    L_n(x) = Σ f_i * ω_{n+1}(x) / ((x - x_i) * ω'_{n+1}(x_i))
    """
    n = len(x_nodes)
    result = 0.0
    
    for i in range(n):
        # Вычисляем базисный многочлен l_i(x)
        l_i = 1.0
        for j in range(n):
            if i != j:
                l_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        
        result += y_nodes[i] * l_i
    
    return result


def divided_differences(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """
    Построение таблицы разделенных разностей для многочлена Ньютона
    """
    n = len(x_nodes)
    # Создаем таблицу разделенных разностей
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_nodes[i + j] - x_nodes[i])
    
    return table


def newton_polynomial(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    """
    Вычисление значения интерполяционного многочлена Ньютона
    
    P_n(x) = f(x_0) + (x-x_0)f(x_1,x_0) + (x-x_0)(x-x_1)f(x_0,x_1,x_2) + ...
    """
    table = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    result = table[0, 0]
    product = 1.0
    
    for i in range(1, n):
        product *= (x - x_nodes[i - 1])
        result += table[0, i] * product
    
    return result


def print_divided_differences_table(x_nodes: np.ndarray, y_nodes: np.ndarray):
    """Вывод таблицы разделенных разностей"""
    table = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    
    print("\nТаблица разделенных разностей:")
    print("-" * 80)
    header = "i | x_i      | f(x_i)   |"
    for j in range(1, n):
        header += f" f[x_i,...,x_{{i+{j}}}] |"
    print(header)
    print("-" * 80)
    
    for i in range(n):
        row = f"{i} | {x_nodes[i]:8.5f} | {y_nodes[i]:8.5f} |"
        for j in range(1, n - i):
            row += f" {table[i, j]:17.10f} |"
        print(row)
    print("-" * 80)


def solve_variant_a():
    """Решение варианта а) Xi = 0, π/8, 2π/8, 3π/8"""
    print("=" * 80)
    print("ВАРИАНТ А: Xi = 0, π/8, 2π/8, 3π/8; X* = 3π/16")
    print("=" * 80)
    
    # Узлы интерполяции
    x_nodes = np.array([0, np.pi/8, 2*np.pi/8, 3*np.pi/8])
    
    # Функция y = tg(x) + x
    y_nodes = np.tan(x_nodes) + x_nodes
    
    # Точка для вычисления
    x_star = 3 * np.pi / 16
    
    # Точное значение
    y_exact = np.tan(x_star) + x_star
    
    print(f"\nУзлы интерполяции:")
    print(f"{'i':<5} {'x_i':<12} {'y_i = tg(x_i) + x_i':<20}")
    print("-" * 40)
    for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
        print(f"{i:<5} {x:<12.8f} {y:<20.10f}")
    
    # Многочлен Лагранжа
    y_lagrange = lagrange_polynomial(x_nodes, y_nodes, x_star)
    error_lagrange = abs(y_exact - y_lagrange)
    
    print(f"\n--- Многочлен Лагранжа ---")
    print(f"L_3({x_star:.8f}) = {y_lagrange:.10f}")
    print(f"Точное значение: y({x_star:.8f}) = {y_exact:.10f}")
    print(f"Абсолютная погрешность: {error_lagrange:.10e}")
    
    # Многочлен Ньютона
    print_divided_differences_table(x_nodes, y_nodes)
    
    y_newton = newton_polynomial(x_nodes, y_nodes, x_star)
    error_newton = abs(y_exact - y_newton)
    
    print(f"\n--- Многочлен Ньютона ---")
    print(f"P_3({x_star:.8f}) = {y_newton:.10f}")
    print(f"Точное значение: y({x_star:.8f}) = {y_exact:.10f}")
    print(f"Абсолютная погрешность: {error_newton:.10e}")
    
    # Построение графика
    x_plot = np.linspace(0, 3*np.pi/8, 200)
    y_exact_plot = np.tan(x_plot) + x_plot
    y_lagrange_plot = [lagrange_polynomial(x_nodes, y_nodes, x) for x in x_plot]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_exact_plot, 'b-', label='Точная функция y = tg(x) + x', linewidth=2)
    plt.plot(x_plot, y_lagrange_plot, 'r--', label='Многочлен Лагранжа L₃(x)', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'go', markersize=10, label='Узлы интерполяции')
    plt.plot(x_star, y_exact, 'bs', markersize=12, label=f'X* = 3π/16 (точное)')
    plt.plot(x_star, y_lagrange, 'r^', markersize=12, label=f'X* = 3π/16 (интерполяция)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция функции y = tg(x) + x (вариант а)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chislaki/lab3/task3_1_variant_a.png', dpi=150)
    print(f"\nГрафик сохранен: task3_1_variant_a.png")
    
    return error_lagrange, error_newton


def solve_variant_b():
    """Решение варианта б) Xi = 0, π/8, π/3, 3π/8"""
    print("\n" + "=" * 80)
    print("ВАРИАНТ Б: Xi = 0, π/8, π/3, 3π/8; X* = 3π/16")
    print("=" * 80)
    
    # Узлы интерполяции
    x_nodes = np.array([0, np.pi/8, np.pi/3, 3*np.pi/8])
    
    # Функция y = tg(x) + x
    y_nodes = np.tan(x_nodes) + x_nodes
    
    # Точка для вычисления
    x_star = 3 * np.pi / 16
    
    # Точное значение
    y_exact = np.tan(x_star) + x_star
    
    print(f"\nУзлы интерполяции:")
    print(f"{'i':<5} {'x_i':<12} {'y_i = tg(x_i) + x_i':<20}")
    print("-" * 40)
    for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
        print(f"{i:<5} {x:<12.8f} {y:<20.10f}")
    
    # Многочлен Лагранжа
    y_lagrange = lagrange_polynomial(x_nodes, y_nodes, x_star)
    error_lagrange = abs(y_exact - y_lagrange)
    
    print(f"\n--- Многочлен Лагранжа ---")
    print(f"L_3({x_star:.8f}) = {y_lagrange:.10f}")
    print(f"Точное значение: y({x_star:.8f}) = {y_exact:.10f}")
    print(f"Абсолютная погрешность: {error_lagrange:.10e}")
    
    # Многочлен Ньютона
    print_divided_differences_table(x_nodes, y_nodes)
    
    y_newton = newton_polynomial(x_nodes, y_nodes, x_star)
    error_newton = abs(y_exact - y_newton)
    
    print(f"\n--- Многочлен Ньютона ---")
    print(f"P_3({x_star:.8f}) = {y_newton:.10f}")
    print(f"Точное значение: y({x_star:.8f}) = {y_exact:.10f}")
    print(f"Абсолютная погрешность: {error_newton:.10e}")
    
    # Построение графика
    x_plot = np.linspace(0, 3*np.pi/8, 200)
    y_exact_plot = np.tan(x_plot) + x_plot
    y_lagrange_plot = [lagrange_polynomial(x_nodes, y_nodes, x) for x in x_plot]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_exact_plot, 'b-', label='Точная функция y = tg(x) + x', linewidth=2)
    plt.plot(x_plot, y_lagrange_plot, 'r--', label='Многочлен Лагранжа L₃(x)', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'go', markersize=10, label='Узлы интерполяции')
    plt.plot(x_star, y_exact, 'bs', markersize=12, label=f'X* = 3π/16 (точное)')
    plt.plot(x_star, y_lagrange, 'r^', markersize=12, label=f'X* = 3π/16 (интерполяция)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция функции y = tg(x) + x (вариант б)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chislaki/lab3/task3_1_variant_b.png', dpi=150)
    print(f"\nГрафик сохранен: task3_1_variant_b.png")
    
    return error_lagrange, error_newton


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. ЗАДАНИЕ 3.1")
    print("ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНАМИ ЛАГРАНЖА И НЬЮТОНА")
    print("Вариант 14: y = tg(x) + x")
    print("=" * 80)
    
    # Решение варианта а
    error_lag_a, error_newt_a = solve_variant_a()
    
    # Решение варианта б
    error_lag_b, error_newt_b = solve_variant_b()
    
    # Итоговая сводка
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА ПОГРЕШНОСТЕЙ")
    print("=" * 80)
    print(f"Вариант А:")
    print(f"  Погрешность Лагранжа:  {error_lag_a:.10e}")
    print(f"  Погрешность Ньютона:   {error_newt_a:.10e}")
    print(f"\nВариант Б:")
    print(f"  Погрешность Лагранжа:  {error_lag_b:.10e}")
    print(f"  Погрешность Ньютона:   {error_newt_b:.10e}")
    print("=" * 80)
    
    plt.show()
