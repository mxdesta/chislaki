"""
Лабораторная работа №3. Задание 3.3
Метод наименьших квадратов (МНК)

Вариант 14:
i  | 0    | 1   | 2   | 3   | 4   | 5
xi | -0.9 | 0.0 | 0.9 | 1.8 | 2.7 | 3.6
yi | -1.2689 | 0.0 | 1.2689 | 2.6541 | 4.4856 | 9.9138
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def least_squares_polynomial(x_data: np.ndarray, y_data: np.ndarray, degree: int) -> np.ndarray:
    """
    Построение приближающего многочлена методом наименьших квадратов
    
    Решает нормальную систему МНК:
    Σ a_i Σ x_j^{k+i} = Σ y_j x_j^k, k = 0,1,...,n
    
    Возвращает коэффициенты [a_0, a_1, ..., a_n]
    """
    N = len(x_data)
    n = degree
    
    # Формируем матрицу системы и правую часть
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    
    for k in range(n + 1):
        for i in range(n + 1):
            A[k, i] = np.sum(x_data**(k + i))
        b[k] = np.sum(y_data * x_data**k)
    
    # Решаем систему
    coeffs = np.linalg.solve(A, b)
    
    return coeffs


def evaluate_polynomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Вычисление значения многочлена"""
    result = np.zeros_like(x)
    for i, a in enumerate(coeffs):
        result += a * x**i
    return result


def compute_error_sum(x_data: np.ndarray, y_data: np.ndarray, coeffs: np.ndarray) -> float:
    """Вычисление суммы квадратов ошибок"""
    y_approx = evaluate_polynomial(x_data, coeffs)
    return np.sum((y_approx - y_data)**2)


def print_normal_system(x_data: np.ndarray, y_data: np.ndarray, degree: int):
    """Вывод нормальной системы МНК"""
    N = len(x_data)
    n = degree
    
    print(f"\nНормальная система МНК для многочлена степени {n}:")
    print("-" * 80)
    
    # Вычисляем суммы
    sums = {}
    for k in range(2 * n + 1):
        sums[k] = np.sum(x_data**k)
    
    for k in range(n + 1):
        sum_yx = np.sum(y_data * x_data**k)
        sums[f'yx{k}'] = sum_yx
    
    # Выводим систему
    for k in range(n + 1):
        equation = ""
        for i in range(n + 1):
            coeff = sums[k + i]
            if i == 0:
                equation += f"{coeff:.4f}·a_{i}"
            else:
                equation += f" + {coeff:.4f}·a_{i}"
        equation += f" = {sums[f'yx{k}']:.4f}"
        print(equation)
    
    print("-" * 80)


def solve_task():
    """Решение задания 3.3"""
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. ЗАДАНИЕ 3.3")
    print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ")
    print("Вариант 14")
    print("=" * 80)
    
    # Исходные данные
    x_data = np.array([-0.9, 0.0, 0.9, 1.8, 2.7, 3.6])
    y_data = np.array([-1.2689, 0.0, 1.2689, 2.6541, 4.4856, 9.9138])
    
    print("\nИсходные данные:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
    print("-" * 30)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        print(f"{i:<5} {x:<10.1f} {y:<15.4f}")
    
    # ========== МНОГОЧЛЕН 1-ОЙ СТЕПЕНИ ==========
    print("\n" + "=" * 80)
    print("А) ПРИБЛИЖАЮЩИЙ МНОГОЧЛЕН 1-ОЙ СТЕПЕНИ: F₁(x) = a₀ + a₁x")
    print("=" * 80)
    
    print_normal_system(x_data, y_data, 1)
    
    coeffs_1 = least_squares_polynomial(x_data, y_data, 1)
    print(f"\nРешение системы:")
    print(f"a₀ = {coeffs_1[0]:.8f}")
    print(f"a₁ = {coeffs_1[1]:.8f}")
    print(f"\nПриближающий многочлен 1-ой степени:")
    print(f"F₁(x) = {coeffs_1[0]:.8f} + {coeffs_1[1]:.8f}x")
    
    # Вычисляем значения и ошибки
    y_approx_1 = evaluate_polynomial(x_data, coeffs_1)
    error_sum_1 = compute_error_sum(x_data, y_data, coeffs_1)
    
    print(f"\nТаблица значений:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15} {'F₁(x_i)':<15} {'Ошибка':<15}")
    print("-" * 60)
    for i, (x, y, y_app) in enumerate(zip(x_data, y_data, y_approx_1)):
        error = y_app - y
        print(f"{i:<5} {x:<10.1f} {y:<15.4f} {y_app:<15.4f} {error:<15.4f}")
    
    print(f"\nСумма квадратов ошибок: Φ₁ = {error_sum_1:.8f}")
    
    # ========== МНОГОЧЛЕН 2-ОЙ СТЕПЕНИ ==========
    print("\n" + "=" * 80)
    print("Б) ПРИБЛИЖАЮЩИЙ МНОГОЧЛЕН 2-ОЙ СТЕПЕНИ: F₂(x) = a₀ + a₁x + a₂x²")
    print("=" * 80)
    
    print_normal_system(x_data, y_data, 2)
    
    coeffs_2 = least_squares_polynomial(x_data, y_data, 2)
    print(f"\nРешение системы:")
    print(f"a₀ = {coeffs_2[0]:.8f}")
    print(f"a₁ = {coeffs_2[1]:.8f}")
    print(f"a₂ = {coeffs_2[2]:.8f}")
    print(f"\nПриближающий многочлен 2-ой степени:")
    print(f"F₂(x) = {coeffs_2[0]:.8f} + {coeffs_2[1]:.8f}x + {coeffs_2[2]:.8f}x²")
    
    # Вычисляем значения и ошибки
    y_approx_2 = evaluate_polynomial(x_data, coeffs_2)
    error_sum_2 = compute_error_sum(x_data, y_data, coeffs_2)
    
    print(f"\nТаблица значений:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15} {'F₂(x_i)':<15} {'Ошибка':<15}")
    print("-" * 60)
    for i, (x, y, y_app) in enumerate(zip(x_data, y_data, y_approx_2)):
        error = y_app - y
        print(f"{i:<5} {x:<10.1f} {y:<15.4f} {y_app:<15.4f} {error:<15.4f}")
    
    print(f"\nСумма квадратов ошибок: Φ₂ = {error_sum_2:.8f}")
    
    # ========== СРАВНЕНИЕ ==========
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"Сумма квадратов ошибок для F₁(x): Φ₁ = {error_sum_1:.8f}")
    print(f"Сумма квадратов ошибок для F₂(x): Φ₂ = {error_sum_2:.8f}")
    print(f"Улучшение: {(error_sum_1 - error_sum_2) / error_sum_1 * 100:.2f}%")
    
    # ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
    x_plot = np.linspace(x_data[0] - 0.5, x_data[-1] + 0.5, 500)
    y_plot_1 = evaluate_polynomial(x_plot, coeffs_1)
    y_plot_2 = evaluate_polynomial(x_plot, coeffs_2)
    
    plt.figure(figsize=(14, 8))
    
    # График 1: Все вместе
    plt.subplot(1, 2, 1)
    plt.plot(x_data, y_data, 'ko', markersize=10, label='Табличные данные', zorder=5)
    plt.plot(x_plot, y_plot_1, 'b-', linewidth=2, label=f'F₁(x) (Φ₁={error_sum_1:.4f})')
    plt.plot(x_plot, y_plot_2, 'r--', linewidth=2, label=f'F₂(x) (Φ₂={error_sum_2:.4f})')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Приближение методом наименьших квадратов', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # График 2: Ошибки
    plt.subplot(1, 2, 2)
    errors_1 = y_approx_1 - y_data
    errors_2 = y_approx_2 - y_data
    x_pos = np.arange(len(x_data))
    width = 0.35
    
    plt.bar(x_pos - width/2, errors_1, width, label='Ошибки F₁(x)', alpha=0.8)
    plt.bar(x_pos + width/2, errors_2, width, label='Ошибки F₂(x)', alpha=0.8)
    plt.xlabel('Номер точки i', fontsize=12)
    plt.ylabel('Ошибка', fontsize=12)
    plt.title('Ошибки приближения в узлах', fontsize=13)
    plt.xticks(x_pos, [str(i) for i in range(len(x_data))])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('chislaki/lab3/task3_3_least_squares.png', dpi=150)
    print(f"\nГрафик сохранен: task3_3_least_squares.png")
    
    return coeffs_1, coeffs_2, error_sum_1, error_sum_2


if __name__ == "__main__":
    coeffs_1, coeffs_2, error_1, error_2 = solve_task()
    plt.show()
