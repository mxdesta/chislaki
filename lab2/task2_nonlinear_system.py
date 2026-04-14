"""
Лабораторная работа 2.2
Методы решения систем нелинейных уравнений

Система уравнений (a = 3):
x₁²/a² + x₂²/(a/2)² - 1 = 0
ax₂ - e^(x₁) - x₁ = 0

Реализованы методы:
- Метод Ньютона
- Метод простой итерации
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# Параметр системы
A = 3


def f1(x1: float, x2: float) -> float:
    """Первое уравнение: x₁²/a² + x₂²/(a/2)² - 1 = 0"""
    return x1**2 / A**2 + x2**2 / (A/2)**2 - 1


def f2(x1: float, x2: float) -> float:
    """Второе уравнение: ax₂ - e^(x₁) - x₁ = 0"""
    return A * x2 - np.exp(x1) - x1


# Частные производные для матрицы Якоби
def df1_dx1(x1: float, x2: float) -> float:
    """∂f₁/∂x₁ = 2x₁/a²"""
    return 2 * x1 / A**2


def df1_dx2(x1: float, x2: float) -> float:
    """∂f₁/∂x₂ = 2x₂/(a/2)² = 8x₂/a²"""
    return 2 * x2 / (A/2)**2


def df2_dx1(x1: float, x2: float) -> float:
    """∂f₂/∂x₁ = -e^(x₁) - 1"""
    return -np.exp(x1) - 1


def df2_dx2(x1: float, x2: float) -> float:
    """∂f₂/∂x₂ = a"""
    return A


def plot_system():
    """Графическое определение начального приближения"""
    x1 = np.linspace(-1, 3, 500)
    x2 = np.linspace(-1, 3, 500)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Вычисляем значения функций
    F1 = X1**2 / A**2 + X2**2 / (A/2)**2 - 1
    F2 = A * X2 - np.exp(X1) - X1
    
    plt.figure(figsize=(12, 6))
    
    # График системы
    plt.subplot(1, 2, 1)
    plt.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=2, label='f₁(x₁,x₂) = 0')
    plt.contour(X1, X2, F2, levels=[0], colors='red', linewidths=2, label='f₂(x₁,x₂) = 0')
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Графическое решение системы', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(['f₁ = 0 (эллипс)', 'f₂ = 0'], fontsize=10)
    plt.axis('equal')
    
    # Увеличенный участок с положительным решением
    plt.subplot(1, 2, 2)
    x1_zoom = np.linspace(0, 2, 500)
    x2_zoom = np.linspace(0.5, 2, 500)
    X1_zoom, X2_zoom = np.meshgrid(x1_zoom, x2_zoom)
    F1_zoom = X1_zoom**2 / A**2 + X2_zoom**2 / (A/2)**2 - 1
    F2_zoom = A * X2_zoom - np.exp(X1_zoom) - X1_zoom
    
    plt.contour(X1_zoom, X2_zoom, F1_zoom, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1_zoom, X2_zoom, F2_zoom, levels=[0], colors='red', linewidths=2)
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Увеличенный участок (положительное решение)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(['f₁ = 0', 'f₂ = 0'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('system_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nОтделение корней:")
    print("Положительное решение находится примерно в области:")
    print("0.5 < x₁ < 1.5, 1.0 < x₂ < 1.5")


def newton_method_system(x1_0: float, x2_0: float, epsilon: float, max_iter: int = 1000) -> Tuple:
    """
    Метод Ньютона для системы уравнений
    Формулы (2.13), (2.14), (2.20)
    
    Для системы двух уравнений:
    x₁⁽ᵏ⁺¹⁾ = x₁⁽ᵏ⁾ - det(A₁⁽ᵏ⁾) / det(J⁽ᵏ⁾)
    x₂⁽ᵏ⁺¹⁾ = x₂⁽ᵏ⁾ - det(A₂⁽ᵏ⁾) / det(J⁽ᵏ⁾)
    
    где J - матрица Якоби, A₁, A₂ - вспомогательные матрицы
    """
    x1, x2 = x1_0, x2_0
    table_data = []
    errors = []
    
    print(f"\nНачальное приближение: x₁⁽⁰⁾ = {x1_0}, x₂⁽⁰⁾ = {x2_0}")
    
    for k in range(max_iter):
        # Вычисляем значения функций
        f1_val = f1(x1, x2)
        f2_val = f2(x1, x2)
        
        # Вычисляем элементы матрицы Якоби
        j11 = df1_dx1(x1, x2)
        j12 = df1_dx2(x1, x2)
        j21 = df2_dx1(x1, x2)
        j22 = df2_dx2(x1, x2)
        
        # Определитель матрицы Якоби
        det_J = j11 * j22 - j12 * j21
        
        if abs(det_J) < 1e-10:
            print(f"\nМатрица Якоби вырождена на итерации {k+1}")
            break
        
        # Определители вспомогательных матриц A₁ и A₂
        det_A1 = f1_val * j22 - f2_val * j12
        det_A2 = j11 * f2_val - j21 * f1_val
        
        # Новое приближение
        x1_new = x1 - det_A1 / det_J
        x2_new = x2 - det_A2 / det_J
        
        # Погрешность (норма бесконечности)
        error = max(abs(x1_new - x1), abs(x2_new - x2))
        errors.append(error)
        
        table_data.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'f1': f1_val,
            'f2': f2_val,
            'j11': j11,
            'j12': j12,
            'j21': j21,
            'j22': j22,
            'det_A1': det_A1,
            'det_A2': det_A2,
            'det_J': det_J,
            'error': error
        })
        
        if error < epsilon:
            print(f"\nМетод Ньютона сошелся за {k+1} итераций")
            print(f"Решение: x₁ = {x1_new:.10f}, x₂ = {x2_new:.10f}")
            print(f"Проверка: f₁ = {f1(x1_new, x2_new):.2e}, f₂ = {f2(x1_new, x2_new):.2e}")
            return x1_new, x2_new, errors, table_data
        
        x1, x2 = x1_new, x2_new
    
    print(f"\nМетод Ньютона не сошелся за {max_iter} итераций")
    return x1, x2, errors, table_data


def simple_iteration_method_system(x1_0: float, x2_0: float, epsilon: float, max_iter: int = 1000) -> Tuple:
    """
    Метод простой итерации для системы уравнений
    Формулы (2.23), (2.24)
    
    Преобразуем систему к виду x = φ(x) используя метод с параметром:
    x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - λ·f(x⁽ᵏ⁾)
    
    Для нашей системы выбираем λ = 0.1 для обеспечения сходимости
    """
    lambda_param = 0.1
    
    def phi1(x1, x2):
        """φ₁(x₁, x₂) = x₁ - λ·f₁(x₁, x₂)"""
        return x1 - lambda_param * f1(x1, x2)
    
    def phi2(x1, x2):
        """φ₂(x₁, x₂) = x₂ - λ·f₂(x₁, x₂)"""
        return x2 - lambda_param * f2(x1, x2)
    
    x1, x2 = x1_0, x2_0
    table_data = []
    errors = []
    
    print(f"\nНачальное приближение: x₁⁽⁰⁾ = {x1_0}, x₂⁽⁰⁾ = {x2_0}")
    print(f"Параметр λ = {lambda_param}")
    print("Преобразование системы:")
    print("x₁ = x₁ - λ·f₁(x₁, x₂)")
    print("x₂ = x₂ - λ·f₂(x₁, x₂)")
    
    for k in range(max_iter):
        # Новое приближение
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)
        
        # Погрешность (норма бесконечности)
        error = max(abs(x1_new - x1), abs(x2_new - x2))
        errors.append(error)
        
        table_data.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'phi1': x1_new,
            'phi2': x2_new,
            'error': error
        })
        
        if error < epsilon:
            print(f"\nМетод простой итерации сошелся за {k+1} итераций")
            print(f"Решение: x₁ = {x1_new:.10f}, x₂ = {x2_new:.10f}")
            print(f"Проверка: f₁ = {f1(x1_new, x2_new):.2e}, f₂ = {f2(x1_new, x2_new):.2e}")
            return x1_new, x2_new, errors, table_data
        
        x1, x2 = x1_new, x2_new
    
    print(f"\nМетод простой итерации не сошелся за {max_iter} итераций")
    return x1, x2, errors, table_data


def print_table_newton_system(table_data: List[Dict]):
    """Вывод таблицы итераций метода Ньютона для системы"""
    print("\nТаблица итераций метода Ньютона:")
    print("=" * 120)
    print(f"{'k':>3} | {'x₁⁽ᵏ⁾':>12} | {'x₂⁽ᵏ⁾':>12} | {'f₁':>12} | {'f₂':>12} | "
          f"{'∂f₁/∂x₁':>10} | {'∂f₁/∂x₂':>10} | {'∂f₂/∂x₁':>10} | {'∂f₂/∂x₂':>10} | {'Погрешность':>12}")
    print("=" * 120)
    for row in table_data:
        print(f"{row['k']:>3} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
              f"{row['f1']:>12.6f} | {row['f2']:>12.6f} | "
              f"{row['j11']:>10.6f} | {row['j12']:>10.6f} | "
              f"{row['j21']:>10.6f} | {row['j22']:>10.6f} | {row['error']:>12.2e}")
    print("=" * 120)


def print_table_simple_iteration_system(table_data: List[Dict]):
    """Вывод таблицы итераций метода простой итерации для системы"""
    print("\nТаблица итераций метода простой итерации:")
    print("=" * 80)
    print(f"{'k':>3} | {'x₁⁽ᵏ⁾':>12} | {'x₂⁽ᵏ⁾':>12} | {'φ₁(x)':>12} | {'φ₂(x)':>12} | {'Погрешность':>12}")
    print("=" * 80)
    for row in table_data:
        print(f"{row['k']:>3} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
              f"{row['phi1']:>12.8f} | {row['phi2']:>12.8f} | {row['error']:>12.2e}")
    print("=" * 80)


def plot_convergence_system(errors_si: List[float], errors_newton: List[float], epsilon: float):
    """Анализ зависимости погрешности от количества итераций для системы"""
    plt.figure(figsize=(14, 6))
    
    # Метод простой итерации
    plt.subplot(1, 2, 1)
    plt.semilogy(range(1, len(errors_si) + 1), errors_si, 'bo-', 
                 markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность ||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾||', fontsize=12)
    plt.title('Метод простой итерации (система)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    # Метод Ньютона
    plt.subplot(1, 2, 2)
    plt.semilogy(range(1, len(errors_newton) + 1), errors_newton, 'ro-', 
                 markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность ||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾||', fontsize=12)
    plt.title('Метод Ньютона (система, квадратичная сходимость)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('system_convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА 2.2")
    print("Решение систем нелинейных уравнений")
    print("=" * 80)
    print(f"\nСистема уравнений (a = {A}):")
    print("f₁(x₁, x₂) = x₁²/a² + x₂²/(a/2)² - 1 = 0")
    print("f₂(x₁, x₂) = ax₂ - e^(x₁) - x₁ = 0")
    print("\nЗадача: найти положительное решение с заданной точностью")
    
    # Этап 1: Отделение корней (графический способ)
    print("\n" + "=" * 80)
    print("ЭТАП 1: ОТДЕЛЕНИЕ КОРНЕЙ (графический способ)")
    print("=" * 80)
    plot_system()
    
    # Параметры
    x1_0 = 1.0
    x2_0 = 1.2
    epsilon = float(input("\nВведите точность вычислений ε (например, 1e-4): ") or "1e-4")
    
    print(f"\nВыбрано начальное приближение: x₁⁽⁰⁾ = {x1_0}, x₂⁽⁰⁾ = {x2_0}")
    print(f"Заданная точность: ε = {epsilon}")
    
    # Этап 2: Метод Ньютона
    print("\n" + "=" * 80)
    print("ЭТАП 2: МЕТОД НЬЮТОНА для системы")
    print("=" * 80)
    x1_n, x2_n, errors_newton, table_newton = newton_method_system(x1_0, x2_0, epsilon)
    print_table_newton_system(table_newton)
    
    # Этап 3: Метод простой итерации
    print("\n" + "=" * 80)
    print("ЭТАП 3: МЕТОД ПРОСТОЙ ИТЕРАЦИИ для системы")
    print("=" * 80)
    x1_si, x2_si, errors_si, table_si = simple_iteration_method_system(x1_0, x2_0, epsilon)
    print_table_simple_iteration_system(table_si)
    
    # Этап 4: Сравнительный анализ
    print("\n" + "=" * 80)
    print("ЭТАП 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
    print("=" * 80)
    print(f"\nМетод Ньютона:")
    print(f"  - Количество итераций: {len(errors_newton)}")
    print(f"  - Найденное решение: x₁ = {x1_n:.10f}, x₂ = {x2_n:.10f}")
    print(f"  - Проверка: f₁ = {f1(x1_n, x2_n):.2e}, f₂ = {f2(x1_n, x2_n):.2e}")
    
    print(f"\nМетод простой итерации:")
    print(f"  - Количество итераций: {len(errors_si)}")
    print(f"  - Найденное решение: x₁ = {x1_si:.10f}, x₂ = {x2_si:.10f}")
    print(f"  - Проверка: f₁ = {f1(x1_si, x2_si):.2e}, f₂ = {f2(x1_si, x2_si):.2e}")
    
    print(f"\nВывод: Метод Ньютона сходится быстрее в {len(errors_si) / len(errors_newton):.1f} раза")
    
    # Этап 5: Визуализация
    print("\n" + "=" * 80)
    print("ЭТАП 5: АНАЛИЗ ЗАВИСИМОСТИ ПОГРЕШНОСТИ ОТ КОЛИЧЕСТВА ИТЕРАЦИЙ")
    print("=" * 80)
    plot_convergence_system(errors_si, errors_newton, epsilon)
    
    print("\n✓ Работа завершена! Графики сохранены")
    print("=" * 80)


if __name__ == "__main__":
    main()
