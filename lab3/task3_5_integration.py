"""
Лабораторная работа №3. Задание 3.5
Численное интегрирование

Вариант 14:
y = 1 / (x⁴ + 16)
X₀ = 0, Xₖ = 2
h₁ = 0.5, h₂ = 0.25
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from scipy import integrate


def function(x: float) -> float:
    """Подынтегральная функция y = 1 / (x⁴ + 16)"""
    return 1.0 / (x**4 + 16)


def rectangle_method(f: Callable, a: float, b: float, h: float) -> float:
    """
    Метод прямоугольников (средних точек)
    F ≈ Σ h_i * f((x_{i-1} + x_i) / 2)
    """
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    result = 0.0
    
    for i in range(n):
        x_mid = (x[i] + x[i + 1]) / 2
        result += h * f(x_mid)
    
    return result


def trapezoid_method(f: Callable, a: float, b: float, h: float) -> float:
    """
    Метод трапеций
    F ≈ (1/2) Σ (f_i + f_{i-1}) * h_i
    """
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    
    result = 0.0
    for i in range(n):
        result += 0.5 * (y[i] + y[i + 1]) * h
    
    return result


def simpson_method(f: Callable, a: float, b: float, h: float) -> float:
    """
    Метод Симпсона (парабол)
    F ≈ (h/3) [f_0 + 4f_1 + 2f_2 + 4f_3 + ... + 2f_{n-2} + 4f_{n-1} + f_n]
    
    Требует четное число интервалов
    """
    n = int((b - a) / h)
    if n % 2 != 0:
        n += 1  # Делаем четным
    
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    
    result = y[0] + y[n]
    
    for i in range(1, n, 2):
        result += 4 * y[i]
    
    for i in range(2, n, 2):
        result += 2 * y[i]
    
    result *= h / 3
    
    return result


def runge_romberg_richardson(F_h: float, F_kh: float, k: int, p: int) -> float:
    """
    Метод Рунге-Ромберга-Ричардсона для уточнения результата
    
    F = F_h + (F_h - F_{kh}) / (k^p - 1)
    
    где:
    - F_h: значение интеграла с шагом h
    - F_kh: значение интеграла с шагом k*h
    - k: коэффициент изменения шага (обычно 2)
    - p: порядок точности метода (2 для прямоугольников и трапеций, 4 для Симпсона)
    """
    return F_h + (F_h - F_kh) / (k**p - 1)


def solve_task():
    """Решение задания 3.5"""
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. ЗАДАНИЕ 3.5")
    print("ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ")
    print("Вариант 14: y = 1 / (x⁴ + 16)")
    print("=" * 80)
    
    # Параметры
    a = 0.0
    b = 2.0
    h1 = 0.5
    h2 = 0.25
    k = 2  # h1 / h2
    
    print(f"\nПараметры:")
    print(f"Интервал интегрирования: [{a}, {b}]")
    print(f"Шаг h₁ = {h1}")
    print(f"Шаг h₂ = {h2}")
    print(f"Коэффициент k = h₁/h₂ = {k}")
    
    # Точное значение (численное интегрирование высокой точности)
    exact_value, _ = integrate.quad(function, a, b)
    print(f"\nТочное значение интеграла (scipy.integrate.quad):")
    print(f"I = {exact_value:.10f}")
    
    # ========== ВЫЧИСЛЕНИЯ С ШАГОМ h1 ==========
    print("\n" + "=" * 80)
    print(f"ВЫЧИСЛЕНИЯ С ШАГОМ h₁ = {h1}")
    print("=" * 80)
    
    n1 = int((b - a) / h1)
    x1 = np.linspace(a, b, n1 + 1)
    y1 = np.array([function(xi) for xi in x1])
    
    print(f"\nТаблица значений (N = {n1} интервалов):")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
    print("-" * 30)
    for i, (x, y) in enumerate(zip(x1, y1)):
        print(f"{i:<5} {x:<10.2f} {y:<15.10f}")
    
    rect_h1 = rectangle_method(function, a, b, h1)
    trap_h1 = trapezoid_method(function, a, b, h1)
    simp_h1 = simpson_method(function, a, b, h1)
    
    print(f"\nРезультаты:")
    print(f"Метод прямоугольников: I_rect(h₁) = {rect_h1:.10f}")
    print(f"Метод трапеций:        I_trap(h₁) = {trap_h1:.10f}")
    print(f"Метод Симпсона:        I_simp(h₁) = {simp_h1:.10f}")
    
    print(f"\nАбсолютные погрешности:")
    print(f"Метод прямоугольников: |I - I_rect(h₁)| = {abs(exact_value - rect_h1):.10e}")
    print(f"Метод трапеций:        |I - I_trap(h₁)| = {abs(exact_value - trap_h1):.10e}")
    print(f"Метод Симпсона:        |I - I_simp(h₁)| = {abs(exact_value - simp_h1):.10e}")
    
    # ========== ВЫЧИСЛЕНИЯ С ШАГОМ h2 ==========
    print("\n" + "=" * 80)
    print(f"ВЫЧИСЛЕНИЯ С ШАГОМ h₂ = {h2}")
    print("=" * 80)
    
    n2 = int((b - a) / h2)
    x2 = np.linspace(a, b, n2 + 1)
    y2 = np.array([function(xi) for xi in x2])
    
    print(f"\nТаблица значений (N = {n2} интервалов):")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
    print("-" * 30)
    for i, (x, y) in enumerate(zip(x2, y2)):
        print(f"{i:<5} {x:<10.2f} {y:<15.10f}")
    
    rect_h2 = rectangle_method(function, a, b, h2)
    trap_h2 = trapezoid_method(function, a, b, h2)
    simp_h2 = simpson_method(function, a, b, h2)
    
    print(f"\nРезультаты:")
    print(f"Метод прямоугольников: I_rect(h₂) = {rect_h2:.10f}")
    print(f"Метод трапеций:        I_trap(h₂) = {trap_h2:.10f}")
    print(f"Метод Симпсона:        I_simp(h₂) = {simp_h2:.10f}")
    
    print(f"\nАбсолютные погрешности:")
    print(f"Метод прямоугольников: |I - I_rect(h₂)| = {abs(exact_value - rect_h2):.10e}")
    print(f"Метод трапеций:        |I - I_trap(h₂)| = {abs(exact_value - trap_h2):.10e}")
    print(f"Метод Симпсона:        |I - I_simp(h₂)| = {abs(exact_value - simp_h2):.10e}")
    
    # ========== УТОЧНЕНИЕ ПО МЕТОДУ РУНГЕ-РОМБЕРГА ==========
    print("\n" + "=" * 80)
    print("УТОЧНЕНИЕ ПО МЕТОДУ РУНГЕ-РОМБЕРГА-РИЧАРДСОНА")
    print("=" * 80)
    
    print(f"\nФормула: I = I_h + (I_h - I_{{kh}}) / (k^p - 1)")
    print(f"где k = {k}, p = 2 (для прямоугольников и трапеций), p = 4 (для Симпсона)")
    
    # Уточнение для метода прямоугольников (p = 2)
    rect_refined = runge_romberg_richardson(rect_h2, rect_h1, k, 2)
    print(f"\nМетод прямоугольников (p = 2):")
    print(f"I_уточн = {rect_h2:.10f} + ({rect_h2:.10f} - {rect_h1:.10f}) / ({k}² - 1)")
    print(f"I_уточн = {rect_refined:.10f}")
    print(f"Абсолютная погрешность: {abs(exact_value - rect_refined):.10e}")
    
    # Уточнение для метода трапеций (p = 2)
    trap_refined = runge_romberg_richardson(trap_h2, trap_h1, k, 2)
    print(f"\nМетод трапеций (p = 2):")
    print(f"I_уточн = {trap_h2:.10f} + ({trap_h2:.10f} - {trap_h1:.10f}) / ({k}² - 1)")
    print(f"I_уточн = {trap_refined:.10f}")
    print(f"Абсолютная погрешность: {abs(exact_value - trap_refined):.10e}")
    
    # Уточнение для метода Симпсона (p = 4)
    simp_refined = runge_romberg_richardson(simp_h2, simp_h1, k, 4)
    print(f"\nМетод Симпсона (p = 4):")
    print(f"I_уточн = {simp_h2:.10f} + ({simp_h2:.10f} - {simp_h1:.10f}) / ({k}⁴ - 1)")
    print(f"I_уточн = {simp_refined:.10f}")
    print(f"Абсолютная погрешность: {abs(exact_value - simp_refined):.10e}")
    
    # ========== ИТОГОВАЯ ТАБЛИЦА ==========
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    print(f"\n{'Метод':<25} {'I(h₁)':<15} {'I(h₂)':<15} {'I_уточн':<15} {'Погрешность':<15}")
    print("-" * 85)
    print(f"{'Точное значение':<25} {'':<15} {'':<15} {exact_value:<15.10f} {0.0:<15.10e}")
    print(f"{'Прямоугольников':<25} {rect_h1:<15.10f} {rect_h2:<15.10f} {rect_refined:<15.10f} {abs(exact_value - rect_refined):<15.10e}")
    print(f"{'Трапеций':<25} {trap_h1:<15.10f} {trap_h2:<15.10f} {trap_refined:<15.10f} {abs(exact_value - trap_refined):<15.10e}")
    print(f"{'Симпсона':<25} {simp_h1:<15.10f} {simp_h2:<15.10f} {simp_refined:<15.10f} {abs(exact_value - simp_refined):<15.10e}")
    print("-" * 85)
    
    # ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
    x_plot = np.linspace(a, b, 500)
    y_plot = np.array([function(x) for x in x_plot])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Функция
    axes[0, 0].plot(x_plot, y_plot, 'b-', linewidth=2, label='y = 1/(x⁴+16)')
    axes[0, 0].fill_between(x_plot, 0, y_plot, alpha=0.3)
    axes[0, 0].plot(x1, y1, 'ro', markersize=8, label=f'Узлы (h={h1})')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('y', fontsize=11)
    axes[0, 0].set_title('Подынтегральная функция', fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Метод прямоугольников
    axes[0, 1].plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.5)
    for i in range(len(x1) - 1):
        x_mid = (x1[i] + x1[i + 1]) / 2
        y_mid = function(x_mid)
        axes[0, 1].add_patch(plt.Rectangle((x1[i], 0), h1, y_mid, 
                                          fill=True, alpha=0.3, edgecolor='red', linewidth=2))
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('y', fontsize=11)
    axes[0, 1].set_title(f'Метод прямоугольников (h={h1})', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Метод трапеций
    axes[1, 0].plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.5)
    for i in range(len(x1) - 1):
        axes[1, 0].fill_between([x1[i], x1[i + 1]], 0, [y1[i], y1[i + 1]], 
                               alpha=0.3, edgecolor='green', linewidth=2)
    axes[1, 0].plot(x1, y1, 'go', markersize=8)
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('y', fontsize=11)
    axes[1, 0].set_title(f'Метод трапеций (h={h1})', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Сравнение погрешностей
    methods = ['Прямоуг.', 'Трапеций', 'Симпсона']
    errors_h1 = [abs(exact_value - rect_h1), abs(exact_value - trap_h1), abs(exact_value - simp_h1)]
    errors_h2 = [abs(exact_value - rect_h2), abs(exact_value - trap_h2), abs(exact_value - simp_h2)]
    errors_refined = [abs(exact_value - rect_refined), abs(exact_value - trap_refined), abs(exact_value - simp_refined)]
    
    x_pos = np.arange(len(methods))
    width = 0.25
    
    axes[1, 1].bar(x_pos - width, errors_h1, width, label=f'h={h1}', alpha=0.8)
    axes[1, 1].bar(x_pos, errors_h2, width, label=f'h={h2}', alpha=0.8)
    axes[1, 1].bar(x_pos + width, errors_refined, width, label='Уточненное', alpha=0.8)
    axes[1, 1].set_xlabel('Метод', fontsize=11)
    axes[1, 1].set_ylabel('Абсолютная погрешность', fontsize=11)
    axes[1, 1].set_title('Сравнение погрешностей', fontsize=12)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('chislaki/lab3/task3_5_integration.png', dpi=150)
    print(f"\nГрафик сохранен: task3_5_integration.png")
    
    return {
        'exact': exact_value,
        'rectangle': {'h1': rect_h1, 'h2': rect_h2, 'refined': rect_refined},
        'trapezoid': {'h1': trap_h1, 'h2': trap_h2, 'refined': trap_refined},
        'simpson': {'h1': simp_h1, 'h2': simp_h2, 'refined': simp_refined}
    }


if __name__ == "__main__":
    results = solve_task()
    plt.show()
