"""
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
    
    Формула: L_n(x) = Σ f_i * l_i(x)
    где l_i(x) = Π (x - x_j) / (x_i - x_j), j≠i
    
    Идея: Каждый базисный многочлен l_i(x) равен 1 в узле x_i и 0 в остальных узлах.
    Поэтому сумма f_i * l_i(x) проходит через все заданные точки.
    
    Args:
        x_nodes: массив узлов интерполяции [x_0, x_1, ..., x_n]
        y_nodes: массив значений функции в узлах [f_0, f_1, ..., f_n]
        x: точка, в которой вычисляем значение многочлена
    
    Returns:
        Значение интерполяционного многочлена в точке x
    """
    n = len(x_nodes)  # количество узлов
    result = 0.0      # итоговое значение многочлена
    
    # Проходим по всем узлам интерполяции
    for i in range(n):
        # Вычисляем базисный многочлен l_i(x)
        # l_i(x_i) = 1, l_i(x_j) = 0 при j≠i
        l_i = 1.0
        
        # Перемножаем все множители (x - x_j) / (x_i - x_j) для j≠i
        for j in range(n):
            if i != j:  # пропускаем случай i=j
                l_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        
        # Добавляем вклад i-го узла: f_i * l_i(x)
        result += y_nodes[i] * l_i
    
    return result


def divided_differences(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """
    Построение таблицы разделенных разностей для многочлена Ньютона
    
    Разделенная разность - это обобщение понятия производной для дискретных данных.
    
    Порядок 0: f[x_i] = f_i
    Порядок 1: f[x_i, x_j] = (f_i - f_j) / (x_i - x_j)
    Порядок 2: f[x_i, x_j, x_k] = (f[x_i, x_j] - f[x_j, x_k]) / (x_i - x_k)
    И так далее...
    
    Args:
        x_nodes: массив узлов интерполяции
        y_nodes: массив значений функции в узлах
    
    Returns:
        Таблица разделенных разностей размера n×n
        table[i, j] содержит разделенную разность порядка j, начинающуюся с узла i
    """
    n = len(x_nodes)
    
    # Создаем таблицу разделенных разностей
    # Строки - начальный узел, столбцы - порядок разности
    table = np.zeros((n, n))
    
    # Нулевой столбец - сами значения функции (разности порядка 0)
    table[:, 0] = y_nodes
    
    # Заполняем таблицу по диагоналям (увеличивая порядок разности)
    for j in range(1, n):  # порядок разности
        for i in range(n - j):  # начальный узел
            # Рекуррентная формула для разделенных разностей:
            """
    Простыми словами
    Сначала заполняем самый левый столбец — просто копируем значения функции.
    Потом берём этот столбец и вычисляем следующий столбец: для каждой ячейки берём ДВЕ ячейки из предыдущего столбца (чуть ниже и ту же строку).
    Потом берём новый столбец и вычисляем следующий.
    И так до самого правого верхнего угла таблицы.
    Получается заполнение «диагоналями» или «лесенкой»: сначала первый столбец, потом второй, потом третий... 
    Внутри каждого столбца идём сверху вниз, но сам переход между столбцами — это движение вправо."""
            # - i=0: `table[0][1] = (table[1][0] - table[0][0]) / (x[1] - x[0]) = (0.4709 - 0) / (0.3927 - 0) = 1.1990`
            # f[x_i, ..., x_{i+j}] = (f[x_{i+1}, ..., x_{i+j}] - f[x_i, ..., x_{i+j-1}]) / (x_{i+j} - x_i)
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_nodes[i + j] - x_nodes[i])
    
    return table


def newton_polynomial(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    """
    Вычисление значения интерполяционного многочлена Ньютона
    
    Формула: P_n(x) = f[x_0] + (x-x_0)·f[x_0,x_1] + (x-x_0)(x-x_1)·f[x_0,x_1,x_2] + ...
    
    Коэффициенты f[x_0], f[x_0,x_1], f[x_0,x_1,x_2], ... — это разделённые разности,
    которые хранятся в верхней строке таблицы: table[0][0], table[0][1], table[0][2], ...
    
    Args:
        x_nodes: массив узлов интерполяции
        y_nodes: массив значений функции в узлах
        x: точка, в которой вычисляем значение многочлена
    
    Returns:
        Значение интерполяционного многочлена в точке x
    """
    # Строим таблицу разделенных разностей
    table = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    
    # Первый член: f[x_0] — значение функции в первом узле
    result = table[0, 0]
    
    # Произведение (x - x_0)(x - x_1)...(x - x_{i-1})
    # На каждом шаге умножаем на очередную скобку
    product = 1.0
    
    # Добавляем остальные члены многочлена
    # i = 1 → добавляем член с f[x₀, x₁]
    # i = 2 → добавляем член с f[x₀, x₁, x₂]
    # и так далее
    for i in range(1, n):
        # Умножаем накопленное произведение на очередную скобку (x - x_{i-1})
        product *= (x - x_nodes[i - 1])
        
        # Добавляем очередной член многочлена Ньютона:
        # (x-x₀)(x-x₁)...(x-x_{i-1}) · (разделённая разность i-го порядка из первого узла)
        # table[0, i] — это разделённая разность f[x₀, x₁, ..., x_i] (коэффициент Ньютона)
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
    y_newton_plot = [newton_polynomial(x_nodes, y_nodes, x) for x in x_plot]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_exact_plot, 'b-', label='Точная функция y = tg(x) + x', linewidth=2)
    plt.plot(x_plot, y_lagrange_plot, 'r--', label='Многочлен Лагранжа L₃(x)', linewidth=2)
    plt.plot(x_plot, y_newton_plot, 'm:', label='Многочлен Ньютона P₃(x)', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'go', markersize=10, label='Узлы интерполяции')
    plt.plot(x_star, y_exact, 'bs', markersize=12, label=f'X* = 3π/16 (точное)')
    plt.plot(x_star, y_lagrange, 'r^', markersize=12, label=f'X* = 3π/16 (Лагранж)')
    plt.plot(x_star, y_newton, 'mv', markersize=12, label=f'X* = 3π/16 (Ньютон)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция функции y = tg(x) + x (вариант а)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task3_1_variant_a.png', dpi=150)
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
    y_newton_plot = [newton_polynomial(x_nodes, y_nodes, x) for x in x_plot]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_exact_plot, 'b-', label='Точная функция y = tg(x) + x', linewidth=2)
    plt.plot(x_plot, y_lagrange_plot, 'r--', label='Многочлен Лагранжа L₃(x)', linewidth=2)
    plt.plot(x_plot, y_newton_plot, 'm:', label='Многочлен Ньютона P₃(x)', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'go', markersize=10, label='Узлы интерполяции')
    plt.plot(x_star, y_exact, 'bs', markersize=12, label=f'X* = 3π/16 (точное)')
    plt.plot(x_star, y_lagrange, 'r^', markersize=12, label=f'X* = 3π/16 (Лагранж)')
    plt.plot(x_star, y_newton, 'mv', markersize=12, label=f'X* = 3π/16 (Ньютон)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция функции y = tg(x) + x (вариант б)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task3_1_variant_b.png', dpi=150)
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
