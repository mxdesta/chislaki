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
    Метод Ньютона для системы уравнения
    
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
        
        # вспомогательные
        det_A1 = f1_val * j22 - f2_val * j12
        det_A2 = j11 * f2_val - j21 * f1_val
        
        x1_new = x1 - det_A1 / det_J
        x2_new = x2 - det_A2 / det_J
        
        # погрешности считаем
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
    Метод простой итерации для системы уравнений (в соответствии с методичкой)
    
    Преобразуем систему к виду x = φ(x) используя метод с параметром:
    x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - λ·f(x⁽ᵏ⁾)
    
    Параметр λ подбирается так, чтобы обеспечить сходимость: ||I - λ·J_f(x)|| < 1
    
    Согласно формуле (2.27) из методички:
    q = max_{x∈G} ||φ'(x)|| = max_{x∈G} { max_i Σ|∂φᵢ/∂xⱼ| }
    """
    
    # Определим область G вокруг начального приближения
    # Расширяем область для надежного нахождения q
    x1_grid = np.linspace(max(0, x1_0 - 1.0), min(2.5, x1_0 + 1.0), 30)
    x2_grid = np.linspace(max(0.5, x2_0 - 0.8), min(2.0, x2_0 + 0.8), 30)
    
    print(f"\nНачальное приближение: x₁⁽⁰⁾ = {x1_0}, x₂⁽⁰⁾ = {x2_0}")
    print("\n" + "=" * 70)
    print("ПОДБОР ПАРАМЕТРА λ И ПРОВЕРКА УСЛОВИЯ СХОДИМОСТИ")
    print("=" * 70)
    
    # Кандидаты параметра λ для перебора
    lambda_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    lambda_param = None
    q_best = float('inf')
    best_candidate_info = None
    
    # Подбираем оптимальный λ
    for lambda_test in lambda_candidates:
        # Вычисляем q = max||φ'(x)|| для текущего λ по формуле (2.27)
        q = 0
        for x1_val in x1_grid:
            for x2_val in x2_grid:
                # ∂φ₁/∂x₁ = 1 - λ·∂f₁/∂x₁
                dphi1_dx1_val = abs(1 - lambda_test * df1_dx1(x1_val, x2_val))
                # ∂φ₁/∂x₂ = -λ·∂f₁/∂x₂
                dphi1_dx2_val = abs(-lambda_test * df1_dx2(x1_val, x2_val))
                # ∂φ₂/∂x₁ = -λ·∂f₂/∂x₁
                dphi2_dx1_val = abs(-lambda_test * df2_dx1(x1_val, x2_val))
                # ∂φ₂/∂x₂ = 1 - λ·∂f₂/∂x₂
                dphi2_dx2_val = abs(1 - lambda_test * df2_dx2(x1_val, x2_val))
                
                # Норма по формуле (2.27): max(сумма по 1-й строке, сумма по 2-й строке)
                norm = max(dphi1_dx1_val + dphi1_dx2_val, 
                          dphi2_dx1_val + dphi2_dx2_val)
                q = max(q, norm)
        
        status = "✓" if q < 1 else "✗"
        print(f"  λ = {lambda_test:.3f} -> q = {q:.6f} {status}")
        
        # Выбираем λ с наименьшим q (обеспечивает максимальную скорость сходимости)
        if q < 1 and q < q_best:
            q_best = q
            lambda_param = lambda_test
            best_candidate_info = (lambda_test, q)
    
    print("=" * 70)
    
    # Если не нашли подходящий λ, используем метод проб и ошибок
    if lambda_param is None:
        print("\n⚠ Внимание: Не найден λ с q < 1 для начальной области.")
        print("   Пробуем другие значения λ...")
        
        # Расширенный поиск
        for lambda_test in [0.02, 0.04, 0.06, 0.08, 0.55, 0.6, 0.65, 0.7]:
            q = 0
            for x1_val in x1_grid:
                for x2_val in x2_grid:
                    dphi1_dx1_val = abs(1 - lambda_test * df1_dx1(x1_val, x2_val))
                    dphi1_dx2_val = abs(-lambda_test * df1_dx2(x1_val, x2_val))
                    dphi2_dx1_val = abs(-lambda_test * df2_dx1(x1_val, x2_val))
                    dphi2_dx2_val = abs(1 - lambda_test * df2_dx2(x1_val, x2_val))
                    norm = max(dphi1_dx1_val + dphi1_dx2_val, 
                              dphi2_dx1_val + dphi2_dx2_val)
                    q = max(q, norm)
            
            if q < 1 and q < q_best:
                q_best = q
                lambda_param = lambda_test
                best_candidate_info = (lambda_test, q)
                print(f"  λ = {lambda_test:.3f} -> q = {q:.6f} ✓")
        
        if lambda_param is None:
            lambda_param = 0.2  # значение по умолчанию
            print(f"\n  Используем λ = {lambda_param} (значение по умолчанию)")
            # Пересчитаем q для выбранного λ
            q_best = 0
            for x1_val in x1_grid:
                for x2_val in x2_grid:
                    dphi1_dx1_val = abs(1 - lambda_param * df1_dx1(x1_val, x2_val))
                    dphi1_dx2_val = abs(-lambda_param * df1_dx2(x1_val, x2_val))
                    dphi2_dx1_val = abs(-lambda_param * df2_dx1(x1_val, x2_val))
                    dphi2_dx2_val = abs(1 - lambda_param * df2_dx2(x1_val, x2_val))
                    norm = max(dphi1_dx1_val + dphi1_dx2_val, 
                              dphi2_dx1_val + dphi2_dx2_val)
                    q_best = max(q_best, norm)
    
    # Вывод результатов подбора
    print(f"\n📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ УСЛОВИЯ СХОДИМОСТИ:")
    print(f"   Выбран параметр λ = {lambda_param}")
    print(f"   Оценка скорости сходимости q = {q_best:.6f}")
    
    if q_best < 1:
        print(f"   ✅ Условие сходимости выполнено (q = {q_best:.6f} < 1)")
        print(f"   Теоретическая оценка: метод гарантированно сходится")
    else:
        print(f"   ⚠ Условие сходимости не выполнено (q = {q_best:.6f} >= 1)")
        print(f"   Метод может расходиться, но попробуем выполнить итерации")
    
    print("\n" + "=" * 70)
    print("ПРЕОБРАЗОВАНИЕ СИСТЕМЫ К ВИДУ x = φ(x)")
    print("=" * 70)
    print(f"φ₁(x₁, x₂) = x₁ - {lambda_param}·f₁(x₁, x₂)")
    print(f"φ₂(x₁, x₂) = x₂ - {lambda_param}·f₂(x₁, x₂)")
    
    # Функции для итерационного процесса
    def phi1(x1, x2):
        return x1 - lambda_param * f1(x1, x2)
    
    def phi2(x1, x2):
        return x2 - lambda_param * f2(x1, x2)
    
    # Итерационный процесс
    x1, x2 = x1_0, x2_0
    table_data = []
    errors = []
    
    print("\n" + "=" * 70)
    print("ИТЕРАЦИОННЫЙ ПРОЦЕСС МЕТОДА ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 70)
    print(f"{'k':>3} | {'x₁⁽ᵏ⁾':>12} | {'x₂⁽ᵏ⁾':>12} | {'φ₁(x⁽ᵏ⁾)':>14} | {'φ₂(x⁽ᵏ⁾)':>14} | {'Погрешность':>12}")
    print("-" * 85)
    
    for k in range(max_iter):
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)
        
        error = max(abs(x1_new - x1), abs(x2_new - x2))
        errors.append(error)
        
        # Вывод на каждой итерации (первые 10 и через каждые 5)
        if k < 10 or k % 5 == 0 or error < epsilon:
            print(f"{k:>3} | {x1:>12.8f} | {x2:>12.8f} | {x1_new:>14.8f} | {x2_new:>14.8f} | {error:>12.2e}")
        
        table_data.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'phi1': x1_new,
            'phi2': x2_new,
            'error': error
        })
        
        if error < epsilon:
            print("-" * 85)
            print(f"\n✅ МЕТОД ПРОСТОЙ ИТЕРАЦИИ СОШЕЛСЯ за {k+1} итераций")
            print(f"\n📌 РЕШЕНИЕ СИСТЕМЫ:")
            print(f"   x₁ = {x1_new:.10f}")
            print(f"   x₂ = {x2_new:.10f}")
            print(f"\n📌 ПРОВЕРКА:")
            print(f"   f₁(x₁, x₂) = {f1(x1_new, x2_new):.2e}")
            print(f"   f₂(x₁, x₂) = {f2(x1_new, x2_new):.2e}")
            
            # Апостериорная оценка погрешности по формуле (2.26)
            if q_best < 1:
                aposteriori_error = (q_best / (1 - q_best)) * error
                print(f"\n📌 АПОСТЕРИОРНАЯ ОЦЕНКА ПОГРЕШНОСТИ (по формуле 2.26):")
                print(f"   ||x* - x⁽ᵏ⁺¹⁾|| ≤ {q_best:.4f}/(1-{q_best:.4f})·{error:.2e} = {aposteriori_error:.2e}")
            
            return x1_new, x2_new, errors, table_data
        
        x1, x2 = x1_new, x2_new
    
    print("-" * 85)
    print(f"\n⚠ Метод простой итерации не сошелся за {max_iter} итераций")
    return x1, x2, errors, table_data


def print_table_newton_system(table_data: List[Dict]):
    """Вывод таблицы итераций метода Ньютона для системы"""
    if not table_data:
        return
    
    print("\n" + "=" * 130)
    print("ТАБЛИЦА ИТЕРАЦИЙ МЕТОДА НЬЮТОНА")
    print("=" * 130)
    print(f"{'k':>3} | {'x₁⁽ᵏ⁾':>12} | {'x₂⁽ᵏ⁾':>12} | {'f₁':>12} | {'f₂':>12} | "
          f"{'∂f₁/∂x₁':>10} | {'∂f₁/∂x₂':>10} | {'∂f₂/∂x₁':>10} | {'∂f₂/∂x₂':>10} | {'Погрешность':>12}")
    print("-" * 130)
    for row in table_data:
        print(f"{row['k']:>3} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
              f"{row['f1']:>12.6e} | {row['f2']:>12.6e} | "
              f"{row['j11']:>10.6f} | {row['j12']:>10.6f} | "
              f"{row['j21']:>10.6f} | {row['j22']:>10.6f} | {row['error']:>12.2e}")
    print("=" * 130)


def print_table_simple_iteration_system(table_data: List[Dict]):
    """Вывод таблицы итераций метода простой итерации для системы"""
    if not table_data:
        return
    
    print("\n" + "=" * 85)
    print("ТАБЛИЦА ИТЕРАЦИЙ МЕТОДА ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 85)
    print(f"{'k':>3} | {'x₁⁽ᵏ⁾':>12} | {'x₂⁽ᵏ⁾':>12} | {'φ₁(x⁽ᵏ⁾)':>14} | {'φ₂(x⁽ᵏ⁾)':>14} | {'Погрешность':>12}")
    print("-" * 85)
    for row in table_data:
        print(f"{row['k']:>3} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
              f"{row['phi1']:>14.8f} | {row['phi2']:>14.8f} | {row['error']:>12.2e}")
    print("=" * 85)


def plot_convergence_system(errors_si: List[float], errors_newton: List[float], epsilon: float):
    """Анализ зависимости погрешности от количества итераций для системы"""
    plt.figure(figsize=(14, 6))
    
    # Метод простой итерации
    plt.subplot(1, 2, 1)
    if errors_si:
        plt.semilogy(range(1, len(errors_si) + 1), errors_si, 'bo-', 
                     markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность ||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾||', fontsize=12)
    plt.title('Метод простой итерации (линейная сходимость)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    # Метод Ньютона
    plt.subplot(1, 2, 2)
    if errors_newton:
        plt.semilogy(range(1, len(errors_newton) + 1), errors_newton, 'ro-', 
                     markersize=6, linewidth=2, label='Погрешность')
    plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, 
                label=f'Заданная точность ε = {epsilon}')
    plt.xlabel('Номер итерации k', fontsize=12)
    plt.ylabel('Погрешность ||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾||', fontsize=12)
    plt.title('Метод Ньютона (квадратичная сходимость)', fontsize=14, fontweight='bold')
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
    print(f"\n📌 СИСТЕМА УРАВНЕНИЙ (a = {A}):")
    print("   f₁(x₁, x₂) = x₁²/a² + x₂²/(a/2)² - 1 = 0")
    print("   f₂(x₁, x₂) = a·x₂ - e^(x₁) - x₁ = 0")
    print("\n🎯 Задача: найти положительное решение с заданной точностью")
    
    # Этап 1: Отделение корней (графический способ)
    print("\n" + "=" * 80)
    print("ЭТАП 1: ОТДЕЛЕНИЕ КОРНЕЙ (графический способ)")
    print("=" * 80)
    plot_system()
    
    # Параметры
    x1_0 = 1.0
    x2_0 = 1.2
    epsilon_input = input("\nВведите точность вычислений ε (например, 1e-4): ").strip()
    epsilon = float(epsilon_input) if epsilon_input else 1e-4
    
    print(f"\n📌 Начальное приближение: x₁⁽⁰⁾ = {x1_0}, x₂⁽⁰⁾ = {x2_0}")
    print(f"📌 Заданная точность: ε = {epsilon}")
    
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
    
    print(f"\n📊 МЕТОД НЬЮТОНА:")
    print(f"   - Количество итераций: {len(errors_newton)}")
    print(f"   - Найденное решение: x₁ = {x1_n:.10f}, x₂ = {x2_n:.10f}")
    print(f"   - Проверка: f₁ = {f1(x1_n, x2_n):.2e}, f₂ = {f2(x1_n, x2_n):.2e}")
    
    print(f"\n📊 МЕТОД ПРОСТОЙ ИТЕРАЦИИ:")
    print(f"   - Количество итераций: {len(errors_si)}")
    print(f"   - Найденное решение: x₁ = {x1_si:.10f}, x₂ = {x2_si:.10f}")
    print(f"   - Проверка: f₁ = {f1(x1_si, x2_si):.2e}, f₂ = {f2(x1_si, x2_si):.2e}")
    
    if len(errors_newton) > 0 and len(errors_si) > 0:
        print(f"\n📈 ВЫВОД:")
        print(f"   Метод Ньютона сходится быстрее в {len(errors_si) / len(errors_newton):.1f} раза")
        print(f"   (линейная сходимость у метода простой итерации vs квадратичная у метода Ньютона)")
    
    # Этап 5: Визуализация
    print("\n" + "=" * 80)
    print("ЭТАП 5: АНАЛИЗ ЗАВИСИМОСТИ ПОГРЕШНОСТИ ОТ КОЛИЧЕСТВА ИТЕРАЦИЙ")
    print("=" * 80)
    plot_convergence_system(errors_si, errors_newton, epsilon)
    
    print("\n✅ Работа успешно завершена!")
    print("   Графики сохранены: system_plot.png, system_convergence_analysis.png")
    print("=" * 80)


if __name__ == "__main__":
    main()