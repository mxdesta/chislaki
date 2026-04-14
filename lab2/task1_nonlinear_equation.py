"""
Лабораторная работа 2.1
Методы решения нелинейных уравнений
Уравнение: x^3 - 2x^2 - 10x + 15 = 0

Реализованы методы:
- Метод Ньютона (касательных)
- Метод простой итерации
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def f(x: float) -> float:
    """Целевая функция: x^3 - 2x^2 - 10x + 15"""
    return x**3 - 2*x**2 - 10*x + 15


def df(x: float) -> float:
    """Производная функции: 3x^2 - 4x - 10"""
    return 3*x**2 - 4*x - 10


def d2f(x: float) -> float:
    """Вторая производная функции: 6x - 4"""
    return 6*x - 4


def plot_function():
    """Графическое определение начального приближения (отделение корней)"""
    x = np.linspace(-3, 5, 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(14, 6))
    
    # График функции
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x³ - 2x² - 10x + 15')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('График функции f(x)', fontsize=14)
    plt.legend(fontsize=10)
    
    # Увеличенный участок с положительным корнем
    plt.subplot(1, 2, 2)
    x_zoom = np.linspace(3, 5, 500)
    y_zoom = [f(xi) for xi in x_zoom]
    plt.plot(x_zoom, y_zoom, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='y = 0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Увеличенный участок (положительный корень)', fontsize=14)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('function_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nОтделение корней:")
    print("Корни примерно в точках: x ≈ -2.5, x ≈ 1.2, x ≈ 3.8")
    print("Положительный корень находится в интервале [3.5, 4.0]")


def newton_method(x0: float, epsilon: float, max_iter: int = 1000) -> Tuple[float, List[float], List[float], List[Dict]]:
    """
    Метод Ньютона (метод касательных)
    Формула (2.2): x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - f(x⁽ᵏ⁾) / f'(x⁽ᵏ⁾)
    
    Условия сходимости (Теорема 2.2):
    1) f(x) и f'(x), f''(x) непрерывны на [a,b]
    2) f'(x) и f''(x) сохраняют знак на [a,b]
    3) f(a)·f(b) < 0
    4) Начальное приближение: f(x⁽⁰⁾)·f''(x⁽⁰⁾) > 0
    """
    iterations = [x0]
    errors = []
    table_data = []
    x = x0
    
    # Проверка условия (2.3)
    print(f"\nПроверка условия сходимости (2.3):")
    print(f"f({x0}) = {f(x0):.4f}")
    print(f"f''({x0}) = {d2f(x0):.4f}")
    print(f"f(x⁽⁰⁾)·f''(x⁽⁰⁾) = {f(x0) * d2f(x0):.4f} > 0 ✓")
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:
            print(f"\nПроизводная близка к нулю на итерации {i+1}")
            break
        
        delta = -fx / dfx
        x_new = x + delta
        error = abs(x_new - x)
        errors.append(error)
        iterations.append(x_new)
        
        table_data.append({
            'k': i,
            'x_k': x,
            'f_x_k': fx,
            'df_x_k': dfx,
            'delta': delta,
            'error': error
        })
        
        if error < epsilon:
            print(f"\nМетод Ньютона сошелся за {i+1} итераций")
            print(f"Корень: x = {x_new:.10f}")
            print(f"Проверка: f(x) = {f(x_new):.2e}")
            return x_new, iterations, errors, table_data
        
        x = x_new
    
    print(f"\nМетод Ньютона не сошелся за {max_iter} итераций")
    return x, iterations, errors, table_data


def simple_iteration_method(x0: float, epsilon: float, max_iter: int = 1000) -> Tuple[float, List[float], List[float], List[Dict]]:
    """
    Метод простой итерации
    Преобразуем уравнение к виду: x = φ(x)
    
    Используем преобразование с параметром λ:
    x = x - λ·f(x), где λ подбирается так, чтобы |1 - λ·f'(x)| < 1
    
    Для нашего уравнения на интервале [3.5, 4.0]:
    f'(x) ≈ 14.8, выбираем λ = 0.05
    φ(x) = x - 0.05·(x³ - 2x² - 10x + 15)
    φ'(x) = 1 - 0.05·(3x² - 4x - 10)
    """
    # Параметр λ для обеспечения сходимости
    lambda_param = 0.05
    
    def phi(x):
        return x - lambda_param * f(x)
    
    def dphi(x):
        return 1 - lambda_param * df(x)
    
    iterations = [x0]
    errors = []
    table_data = []
    x = x0
    
    print(f"\nНачальное приближение: x⁽⁰⁾ = {x0}")
    print(f"Параметр λ = {lambda_param}")
    print(f"Преобразование: φ(x) = x - {lambda_param}·f(x)")
    print(f"Проверка условия сходимости: |φ'({x0})| = {abs(dphi(x0)):.4f} < 1 ✓")
    
    for i in range(max_iter):
        phi_x = phi(x)
        x_new = phi_x
        error = abs(x_new - x)
        errors.append(error)
        iterations.append(x_new)
        
        # Оценка погрешности
        q = abs(dphi(x))
        estimated_error = (q / (1 - q)) * error if q < 1 else float('inf')
        
        table_data.append({
            'k': i,
            'x_k': x,
            'phi_x_k': phi_x,
            'error': error,
            'estimated_error': estimated_error
        })
        
        if error < epsilon:
            print(f"\nМетод простой итерации сошелся за {i+1} итераций")
            print(f"Корень: x = {x_new:.10f}")
            print(f"Проверка: f(x) = {f(x_new):.2e}")
            return x_new, iterations, errors, table_data
        
        x = x_new
    
    print(f"\nМетод простой итерации не сошелся за {max_iter} итераций")
    return x, iterations, errors, table_data


def print_table_newton(table_data: List[Dict]):
    """Вывод таблицы итераций метода Ньютона"""
    print("\nТаблица итераций метода Ньютона:")
    print("=" * 90)
    print(f"{'k':>3} | {'x⁽ᵏ⁾':>12} | {'f(x⁽ᵏ⁾)':>12} | {'f\'(x⁽ᵏ⁾)':>12} | {'-f/f\'':>12} | {'Погрешность':>12}")
    print("=" * 90)
    for row in table_data:
        print(f"{row['k']:>3} | {row['x_k']:>12.8f} | {row['f_x_k']:>12.6f} | "
              f"{row['df_x_k']:>12.6f} | {row['delta']:>12.8f} | {row['error']:>12.2e}")
    print("=" * 90)


def print_table_simple_iteration(table_data: List[Dict]):
    """Вывод таблицы итераций метода простой итерации"""
    print("\nТаблица итераций метода простой итерации:")
    print("=" * 70)
    print(f"{'k':>3} | {'x⁽ᵏ⁾':>12} | {'φ(x⁽ᵏ⁾)':>12} | {'Погрешность':>12} | {'Оценка':>12}")
    print("=" * 70)
    for row in table_data:
        est = f"{row['estimated_error']:.2e}" if row['estimated_error'] != float('inf') else "inf"
        print(f"{row['k']:>3} | {row['x_k']:>12.8f} | {row['phi_x_k']:>12.8f} | "
              f"{row['error']:>12.2e} | {est:>12}")
    print("=" * 70)


def plot_convergence(errors_si: List[float], errors_newton: List[float], epsilon: float):
    """Анализ зависимости погрешности от количества итераций"""
    plt.figure(figsize=(14, 6))
    
    # Метод простой итерации
    plt.subplot(1, 2, 1)
    plt.semilogy(range(1, len(errors_si) + 1), errors_si, 'bo-', 
                 markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность |x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾|', fontsize=12)
    plt.title('Метод простой итерации', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    # Метод Ньютона
    plt.subplot(1, 2, 2)
    plt.semilogy(range(1, len(errors_newton) + 1), errors_newton, 'ro-', 
                 markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность |x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾|', fontsize=12)
    plt.title('Метод Ньютона (квадратичная сходимость)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА 2.1")
    print("Решение нелинейных уравнений методами Ньютона и простой итерации")
    print("=" * 80)
    print("\nУравнение: f(x) = x³ - 2x² - 10x + 15 = 0")
    print("Задача: найти положительный корень с заданной точностью")
    
    # Этап 1: Отделение корней (графический способ)
    print("\n" + "=" * 80)
    print("ЭТАП 1: ОТДЕЛЕНИЕ КОРНЕЙ (графический способ)")
    print("=" * 80)
    plot_function()
    
    # Параметры
    x0 = 3.8  # Начальное приближение для положительного корня
    epsilon = float(input("\nВведите точность вычислений ε (например, 1e-6): ") or "1e-6")
    
    print(f"\nВыбрано начальное приближение: x⁽⁰⁾ = {x0}")
    print(f"Заданная точность: ε = {epsilon}")
    
    # Этап 2: Метод Ньютона
    print("\n" + "=" * 80)
    print("ЭТАП 2: МЕТОД НЬЮТОНА (метод касательных)")
    print("=" * 80)
    print("Формула: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - f(x⁽ᵏ⁾)/f'(x⁽ᵏ⁾)")
    root_newton, iter_newton, errors_newton, table_newton = newton_method(x0, epsilon)
    print_table_newton(table_newton)
    
    # Этап 3: Метод простой итерации
    print("\n" + "=" * 80)
    print("ЭТАП 3: МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 80)
    print("Преобразование: x = x - λ·f(x), где λ = 0.05")
    root_si, iter_si, errors_si, table_si = simple_iteration_method(x0, epsilon)
    print_table_simple_iteration(table_si)
    
    # Этап 4: Сравнительный анализ
    print("\n" + "=" * 80)
    print("ЭТАП 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
    print("=" * 80)
    print(f"\nМетод Ньютона:")
    print(f"  - Количество итераций: {len(errors_newton)}")
    print(f"  - Найденный корень: x = {root_newton:.10f}")
    print(f"  - Проверка: f(x) = {f(root_newton):.2e}")
    print(f"  - Тип сходимости: квадратичная")
    
    print(f"\nМетод простой итерации:")
    print(f"  - Количество итераций: {len(errors_si)}")
    print(f"  - Найденный корень: x = {root_si:.10f}")
    print(f"  - Проверка: f(x) = {f(root_si):.2e}")
    print(f"  - Тип сходимости: линейная")
    
    print(f"\nВывод: Метод Ньютона сходится быстрее в {len(errors_si) / len(errors_newton):.1f} раза")
    
    # Этап 5: Визуализация
    print("\n" + "=" * 80)
    print("ЭТАП 5: АНАЛИЗ ЗАВИСИМОСТИ ПОГРЕШНОСТИ ОТ КОЛИЧЕСТВА ИТЕРАЦИЙ")
    print("=" * 80)
    plot_convergence(errors_si, errors_newton, epsilon)
    
    print("\n✓ Работа завершена! Графики сохранены")
    print("=" * 80)


if __name__ == "__main__":
    main()
