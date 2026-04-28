"""
Лабораторная работа №3. Задание 3.4
Численное дифференцирование

Вариант 14: X* = 3.0
Таблица:
i  | 0   | 1   | 2   | 3   | 4
xi | 1.0 | 2.0 | 3.0 | 4.0 | 5.0
yi | 1.0 | 2.6931 | 4.0986 | 5.3863 | 6.6094
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def first_derivative_forward(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """
    Правосторонняя производная (первый порядок точности)
    y'(x_i) ≈ (y_{i+1} - y_i) / (x_{i+1} - x_i)
    """
    return (y[i + 1] - y[i]) / (x[i + 1] - x[i])


def first_derivative_backward(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """
    Левосторонняя производная (первый порядок точности)
    y'(x_i) ≈ (y_i - y_{i-1}) / (x_i - x_{i-1})
    """
    return (y[i] - y[i - 1]) / (x[i] - x[i - 1])


def first_derivative_central(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """
    Центральная производная (второй порядок точности)
    Использует интерполяционный многочлен второй степени
    
    y'(x) ≈ (y_{i+1} - y_i)/(x_{i+1} - x_i) + 
            [((y_{i+2} - y_{i+1})/(x_{i+2} - x_{i+1}) - (y_{i+1} - y_i)/(x_{i+1} - x_i)) / (x_{i+2} - x_i)] * 
            (2x - x_i - x_{i+1})
    """
    if i == 0:
        # Используем точки i, i+1, i+2
        h1 = x[i + 1] - x[i]
        h2 = x[i + 2] - x[i + 1]
        f1 = (y[i + 1] - y[i]) / h1
        f2 = (y[i + 2] - y[i + 1]) / h2
        return f1 + ((f2 - f1) / (x[i + 2] - x[i])) * (2 * x[i] - x[i] - x[i + 1])
    elif i == len(x) - 1:
        # Используем точки i-2, i-1, i
        h1 = x[i - 1] - x[i - 2]
        h2 = x[i] - x[i - 1]
        f1 = (y[i - 1] - y[i - 2]) / h1
        f2 = (y[i] - y[i - 1]) / h2
        return f2 + ((f2 - f1) / (x[i] - x[i - 2])) * (2 * x[i] - x[i - 1] - x[i])
    else:
        # Используем точки i-1, i, i+1
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        f1 = (y[i] - y[i - 1]) / h1
        f2 = (y[i + 1] - y[i]) / h2
        return f1 + ((f2 - f1) / (x[i + 1] - x[i - 1])) * (2 * x[i] - x[i - 1] - x[i])


def second_derivative(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """
    Вторая производная (второй порядок точности)
    
    y''(x) ≈ 2 * [((y_{i+1} - y_i)/(x_{i+1} - x_i) - (y_i - y_{i-1})/(x_i - x_{i-1})) / (x_{i+1} - x_{i-1})]
    """
    if i == 0:
        # Используем точки 0, 1, 2
        h1 = x[1] - x[0]
        h2 = x[2] - x[1]
        f1 = (y[1] - y[0]) / h1
        f2 = (y[2] - y[1]) / h2
        return 2 * (f2 - f1) / (x[2] - x[0])
    elif i == len(x) - 1:
        # Используем точки n-2, n-1, n
        h1 = x[i - 1] - x[i - 2]
        h2 = x[i] - x[i - 1]
        f1 = (y[i - 1] - y[i - 2]) / h1
        f2 = (y[i] - y[i - 1]) / h2
        return 2 * (f2 - f1) / (x[i] - x[i - 2])
    else:
        # Используем точки i-1, i, i+1
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        f1 = (y[i] - y[i - 1]) / h1
        f2 = (y[i + 1] - y[i]) / h2
        return 2 * (f2 - f1) / (x[i + 1] - x[i - 1])


def solve_task():
    """Решение задания 3.4"""
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. ЗАДАНИЕ 3.4")
    print("ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ")
    print("Вариант 14: X* = 3.0")
    print("=" * 80)
    
    # Исходные данные
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([1.0, 2.6931, 4.0986, 5.3863, 6.6094])
    x_star = 3.0
    
    # Находим индекс точки X*
    i_star = np.where(x_data == x_star)[0][0]
    
    print("\nИсходные данные:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
    print("-" * 30)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        marker = " <-- X*" if x == x_star else ""
        print(f"{i:<5} {x:<10.1f} {y:<15.4f}{marker}")
    
    # ========== ПЕРВАЯ ПРОИЗВОДНАЯ ==========
    print("\n" + "=" * 80)
    print("ВЫЧИСЛЕНИЕ ПЕРВОЙ ПРОИЗВОДНОЙ В ТОЧКЕ X* = 3.0")
    print("=" * 80)
    
    # Правосторонняя производная
    if i_star < len(x_data) - 1:
        y_prime_forward = first_derivative_forward(x_data, y_data, i_star)
        print(f"\n1) Правосторонняя производная (первый порядок точности):")
        print(f"   y'({x_star}) ≈ (y_{i_star+1} - y_{i_star}) / (x_{i_star+1} - x_{i_star})")
        print(f"   y'({x_star}) ≈ ({y_data[i_star+1]:.4f} - {y_data[i_star]:.4f}) / ({x_data[i_star+1]:.1f} - {x_data[i_star]:.1f})")
        print(f"   y'({x_star}) ≈ {y_prime_forward:.8f}")
    
    # Левосторонняя производная
    if i_star > 0:
        y_prime_backward = first_derivative_backward(x_data, y_data, i_star)
        print(f"\n2) Левосторонняя производная (первый порядок точности):")
        print(f"   y'({x_star}) ≈ (y_{i_star} - y_{i_star-1}) / (x_{i_star} - x_{i_star-1})")
        print(f"   y'({x_star}) ≈ ({y_data[i_star]:.4f} - {y_data[i_star-1]:.4f}) / ({x_data[i_star]:.1f} - {x_data[i_star-1]:.1f})")
        print(f"   y'({x_star}) ≈ {y_prime_backward:.8f}")
    
    # Центральная производная (второй порядок точности)
    y_prime_central = first_derivative_central(x_data, y_data, i_star)
    print(f"\n3) Центральная производная (второй порядок точности):")
    print(f"   Использует интерполяционный многочлен 2-й степени")
    print(f"   на точках x_{i_star-1}, x_{i_star}, x_{i_star+1}")
    
    h1 = x_data[i_star] - x_data[i_star - 1]
    h2 = x_data[i_star + 1] - x_data[i_star]
    f1 = (y_data[i_star] - y_data[i_star - 1]) / h1
    f2 = (y_data[i_star + 1] - y_data[i_star]) / h2
    
    print(f"   f₁ = (y_{i_star} - y_{i_star-1}) / h₁ = {f1:.8f}")
    print(f"   f₂ = (y_{i_star+1} - y_{i_star}) / h₂ = {f2:.8f}")
    print(f"   y'({x_star}) ≈ {y_prime_central:.8f}")
    
    # ========== ВТОРАЯ ПРОИЗВОДНАЯ ==========
    print("\n" + "=" * 80)
    print("ВЫЧИСЛЕНИЕ ВТОРОЙ ПРОИЗВОДНОЙ В ТОЧКЕ X* = 3.0")
    print("=" * 80)
    
    y_double_prime = second_derivative(x_data, y_data, i_star)
    
    print(f"\nВторая производная (второй порядок точности):")
    print(f"y''(x) ≈ 2 * [(f₂ - f₁) / (x_{i_star+1} - x_{i_star-1})]")
    print(f"y''({x_star}) ≈ 2 * [({f2:.8f} - {f1:.8f}) / ({x_data[i_star+1]:.1f} - {x_data[i_star-1]:.1f})]")
    print(f"y''({x_star}) ≈ {y_double_prime:.8f}")
    
    # ========== ТАБЛИЦА ПРОИЗВОДНЫХ ВО ВСЕХ ТОЧКАХ ==========
    print("\n" + "=" * 80)
    print("ТАБЛИЦА ПРОИЗВОДНЫХ ВО ВСЕХ УЗЛАХ")
    print("=" * 80)
    
    print(f"\n{'i':<5} {'x_i':<10} {'y_i':<15} {'y_prime(x_i)':<15} {'y_double_prime(x_i)':<15}")
    print("-" * 65)
    
    for i in range(len(x_data)):
        y_prime = first_derivative_central(x_data, y_data, i)
        y_double = second_derivative(x_data, y_data, i)
        marker = " <-- X*" if x_data[i] == x_star else ""
        print(f"{i:<5} {x_data[i]:<10.1f} {y_data[i]:<15.4f} {y_prime:<15.8f} {y_double:<15.8f}{marker}")
    
    # ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
    # Вычисляем производные во всех точках
    y_prime_all = np.array([first_derivative_central(x_data, y_data, i) for i in range(len(x_data))])
    y_double_prime_all = np.array([second_derivative(x_data, y_data, i) for i in range(len(x_data))])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # График функции
    axes[0].plot(x_data, y_data, 'bo-', linewidth=2, markersize=8, label='y(x)')
    axes[0].plot(x_star, y_data[i_star], 'r^', markersize=15, label=f'X* = {x_star}')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Функция y(x)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # График первой производной
    axes[1].plot(x_data, y_prime_all, 'go-', linewidth=2, markersize=8, label="y'(x)")
    axes[1].plot(x_star, y_prime_central, 'r^', markersize=15, label=f"y'({x_star}) = {y_prime_central:.4f}")
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel("y'", fontsize=12)
    axes[1].set_title('Первая производная y\'(x)', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # График второй производной
    axes[2].plot(x_data, y_double_prime_all, 'mo-', linewidth=2, markersize=8, label="y''(x)")
    axes[2].plot(x_star, y_double_prime, 'r^', markersize=15, label=f"y''({x_star}) = {y_double_prime:.4f}")
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel("y''", fontsize=12)
    axes[2].set_title('Вторая производная y\'\'(x)', fontsize=13)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chislaki/lab3/task3_4_differentiation.png', dpi=150)
    print(f"\nГрафик сохранен: task3_4_differentiation.png")
    
    # ========== ИТОГОВАЯ СВОДКА ==========
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 80)
    print(f"Точка: X* = {x_star}")
    print(f"Значение функции: y({x_star}) = {y_data[i_star]:.8f}")
    print(f"Первая производная: y'({x_star}) = {y_prime_central:.8f}")
    print(f"Вторая производная: y''({x_star}) = {y_double_prime:.8f}")
    print("=" * 80)
    
    return y_prime_central, y_double_prime


if __name__ == "__main__":
    y_prime, y_double_prime = solve_task()
    plt.show()
