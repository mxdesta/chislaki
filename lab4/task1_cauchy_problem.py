"""
Вариант 14:
y'' + 2y' + e^x * y = 0
y(1) = 1, y'(1) = 1
x ∈ [1, 2], h = 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


def f_system(x: float, y: np.ndarray) -> np.ndarray:
    """
    Правая часть системы ОДУ
    y'' + 2y' + e^x * y = 0
    
    Замена: y1 = y, y2 = y'
    Система:
        y1' = y2
        y2' = -2*y2 - exp(x)*y1
    """
    y1, y2 = y
    return np.array([y2, -2*y2 - np.exp(x)*y1])


def get_reference_solution(x0: float, x_end: float, y0: List[float], h_ref: float = 0.0001):
    """Получение эталонного ('истинного') решения с очень маленьким шагом"""
    solver_ref = ODESolver(f_system, x0, y0, h_ref)
    x_ref, y_ref, _ = solver_ref.runge_kutta_4(x_end, adaptive=False)
    return x_ref, y_ref


class ODESolver:
    """Решатель задачи Коши для системы ОДУ 1-го порядка"""
    
    def __init__(self, f: Callable, x0: float, y0: List[float], h: float):
        self.f = f
        self.x0 = x0
        self.y0 = np.array(y0, dtype=float)
        self.h = h
    
    def euler(self, x_end: float, y_ref: np.ndarray = None, x_ref: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Явный метод Эйлера
        
        Возвращает:
        - x: массив узлов
        - y: массив решений
        - epsilon_k: глобальные погрешности ε_k = |y_ист - y_k| в каждом узле
        """
        n_steps = int((x_end - self.x0) / self.h) + 1
        x = np.linspace(self.x0, x_end, n_steps)
        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0
        
        # Вычисляем решение методом Эйлера
        for i in range(n_steps - 1):
            y[i+1] = y[i] + self.h * self.f(x[i], y[i])
        
        # Вычисляем глобальные погрешности ε_k = |y_ист - y_k|
        epsilon_k = []
        if y_ref is not None and x_ref is not None:
            for i in range(n_steps):
                y_true_at_x = np.interp(x[i], x_ref, y_ref[:, 0])
                err = abs(y_true_at_x - y[i, 0])
                epsilon_k.append(err)
        
        return x, y, epsilon_k
    
    def runge_kutta_4(self, x_end: float, adaptive: bool = False, 
                      tol: float = 1e-6, h_min: float = 1e-6, h_max: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Метод Рунге-Кутты 4-го порядка с контролем шага по параметру θ
        θ = |K2 - K3| / |K1 - K2|
        - θ < 0.01 => шаг можно увеличить
        - θ > 0.1 => шаг нужно уменьшить
        - 0.01 <= θ <= 0.1 → шаг подходит
        
        Возвращает:
        - x: массив узлов
        - y: массив решений
        - theta_values: локальные погрешности θ на каждом шаге (если adaptive=True)
        """
        if not adaptive:
            # Стандартный РК4 с фиксированным шагом
            n_steps = int((x_end - self.x0) / self.h) + 1
            x_vals = np.linspace(self.x0, x_end, n_steps)
            y_vals = np.zeros((n_steps, len(self.y0)))
            y_vals[0] = self.y0
            
            for i in range(n_steps - 1):
                k1 = self.h * self.f(x_vals[i], y_vals[i])
                k2 = self.h * self.f(x_vals[i] + self.h/2, y_vals[i] + k1/2)
                k3 = self.h * self.f(x_vals[i] + self.h/2, y_vals[i] + k2/2)
                k4 = self.h * self.f(x_vals[i] + self.h, y_vals[i] + k3)
                y_vals[i+1] = y_vals[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            return x_vals, y_vals, []
        
        # Адаптивный РК4 с контролем шага по θ
        points_x = [self.x0]
        points_y = [self.y0.copy()]
        
        x = self.x0
        y = self.y0.copy()
        h = self.h
        
        print(f"\nАдаптивный метод Рунге-Кутты 4-го порядка")
        print(f"Контроль точности по параметру θ = |K2-K3| / |K1-K2|")
        print(f"Начальный шаг: h = {h}")
        print("-" * 70)
        
        step_count = 0
        rejected_steps = 0
        theta_values = []  # локальные погрешности θ на каждом шаге
        
        while x < x_end - 1e-12:
            if h < h_min:
                h = h_min
            
            if x + h > x_end:
                h = x_end - x
            
            # Расчет коэффициентов
            K1 = self.f(x, y)
            K2 = self.f(x + h/2, y + (h/2) * K1)
            K3 = self.f(x + h/2, y + (h/2) * K2)
            K4 = self.f(x + h, y + h * K3)
            
            # Расчет параметра θ (локальная погрешность по методичке стр. 7)
            diff_K2_K3 = np.linalg.norm(K2 - K3)
            diff_K1_K2 = np.linalg.norm(K1 - K2)
            
            if diff_K1_K2 < 1e-15:
                theta = 0.0
            else:
                theta = diff_K2_K3 / diff_K1_K2
            
            # Сохраняем значение локальной погрешности
            theta_values.append(theta)
            
            # Прогноз шага на основе θ
            if theta > 0.1:
                # Шаг слишком большой, уменьшаем
                h = h * 0.7
                rejected_steps += 1
                continue
            else:
                y_new = y + (h/6) * (K1 + 2*K2 + 2*K3 + K4)
                
                if theta < 0.01:
                    # Шаг можно увеличить
                    h = min(h * 1.5, h_max)
                
                y = y_new
                x = x + h
                points_x.append(x)
                points_y.append(y.copy())
                step_count += 1
        
        print(f"Адаптивный расчет завершен:")
        print(f"  Всего шагов: {step_count}")
        print(f"  Отброшенных шагов: {rejected_steps}")
        print(f"  Среднее значение θ: {np.mean(theta_values):.6f}")
        print(f"  Максимальное θ: {np.max(theta_values):.6f}")
        print(f"  Минимальное θ: {np.min(theta_values):.6f}")
        
        return np.array(points_x), np.array(points_y), theta_values
    
    def adams_4(self, x_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """Метод Адамса 4-го порядка (с разгоном методом РК4)"""
        n_steps = int((x_end - self.x0) / self.h) + 1
        x = np.linspace(self.x0, x_end, n_steps)
        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0
        
        # Разгон: первые 3 шага методом РК4
        for i in range(min(3, n_steps - 1)):
            k1 = self.h * self.f(x[i], y[i])
            k2 = self.h * self.f(x[i] + self.h/2, y[i] + k1/2)
            k3 = self.h * self.f(x[i] + self.h/2, y[i] + k2/2)
            k4 = self.h * self.f(x[i] + self.h, y[i] + k3)
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Метод Адамса 4-го порядка
        for i in range(3, n_steps - 1):
            f_vals = [self.f(x[j], y[j]) for j in range(i-3, i+1)]
            y[i+1] = y[i] + self.h/24 * (
                55*f_vals[3] - 59*f_vals[2] + 37*f_vals[1] - 9*f_vals[0]
            )
        
        return x, y


def main():
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА 4. ЗАДАНИЕ 4.1")
    print("Численное решение задачи Коши для ОДУ 2-го порядка")
    print("="*70)
    print("\nВариант 14:")
    print("y'' + 2y' + e^x * y = 0")
    print("y(1) = 1, y'(1) = 1")
    print("x ∈ [1, 2], h = 0.1")
    
    x_start, x_end = 1.0, 2.0
    h_main = 0.1
    
    # Получение эталонного решения
    print("\nВычисление эталонного решения с шагом h = 0.0001...")
    x_ref, y_ref = get_reference_solution(x_start, x_end, [1.0, 1.0], h_ref=0.0001)
    print("Эталонное решение получено")
    
    # ==================== МЕТОД ЭЙЛЕРА ====================
    print("\n" + "="*70)
    print("МЕТОД ЭЙЛЕРА (глобальная погрешность ε_k = |y_ист - y_k|)")
    print("="*70)
    
    solver_euler = ODESolver(f_system, x_start, [1.0, 1.0], h_main)
    x_euler, y_euler, euler_global_errors = solver_euler.euler(x_end, y_ref, x_ref)
    
    print(f"\nРезультаты метода Эйлера с шагом h = {h_main}:")
    print(f"{'k':<3} {'x_k':<12} {'y_k':<14} {'y_ист':<14} {'ε_k = |y_ист - y_k|':<20}")
    print("-" * 65)
    
    for i in range(len(x_euler)):
        y_true_at_x = np.interp(x_euler[i], x_ref, y_ref[:, 0])
        print(f"{i:<3} {x_euler[i]:<12.6f} {y_euler[i,0]:<14.8f} {y_true_at_x:<14.8f} {euler_global_errors[i]:<20.2e}")
    
    print(f"\nМаксимальная глобальная погрешность метода Эйлера: {max(euler_global_errors):.2e}")
    
    # ==================== МЕТОД РК4 (фиксированный шаг) ====================
    print("\n" + "="*70)
    print("МЕТОД РУНГЕ-КУТТЫ 4-го ПОРЯДКА (фиксированный шаг)")
    print("Глобальная погрешность ε_k = |y_ист - y_k|")
    print("="*70)
    
    solver_rk4_fixed = ODESolver(f_system, x_start, [1.0, 1.0], h_main)
    x_rk4_fixed, y_rk4_fixed, _ = solver_rk4_fixed.runge_kutta_4(x_end, adaptive=False)
    
    rk4_fixed_errors = []
    print(f"\nРезультаты метода РК4 с фиксированным шагом h = {h_main}:")
    print(f"{'k':<3} {'x_k':<12} {'y_k':<14} {'y_ист':<14} {'ε_k = |y_ист - y_k|':<20}")
    print("-" * 65)
    
    for i in range(len(x_rk4_fixed)):
        y_true_at_x = np.interp(x_rk4_fixed[i], x_ref, y_ref[:, 0])
        err = abs(y_rk4_fixed[i, 0] - y_true_at_x)
        rk4_fixed_errors.append(err)
        print(f"{i:<3} {x_rk4_fixed[i]:<12.6f} {y_rk4_fixed[i,0]:<14.8f} {y_true_at_x:<14.8f} {err:<20.2e}")
    
    print(f"\nМаксимальная глобальная погрешность метода РК4 (фикс): {max(rk4_fixed_errors):.2e}")
    
    # ==================== МЕТОД РК4 (адаптивный) ====================
    print("\n" + "="*70)
    print("МЕТОД РУНГЕ-КУТТЫ 4-го ПОРЯДКА (адаптивный)")
    print("Локальная погрешность θ = |K2-K3| / |K1-K2|")
    print("="*70)
    
    solver_rk4_adapt = ODESolver(f_system, x_start, [1.0, 1.0], h_main)
    x_rk4_adapt, y_rk4_adapt, theta_values = solver_rk4_adapt.runge_kutta_4(x_end, adaptive=True)
    
    print(f"\nЛокальные погрешности θ на каждом шаге:")
    print(f"{'Шаг':<8} {'θ = |K2-K3|/|K1-K2|':<30} {'Решение (увеличить/уменьшить/норма)':<35}")
    print("-" * 75)
    
    for i, theta in enumerate(theta_values):
        if theta > 0.1:
            status = "→ УМЕНЬШИТЬ шаг"
        elif theta < 0.01:
            status = "→ УВЕЛИЧИТЬ шаг"
        else:
            status = "→ шаг ОПТИМАЛЕН"
        print(f"{i+1:<8} {theta:<30.6e} {status:<35}")
    
    # ==================== МЕТОД АДАМСА ====================
    print("\n" + "="*70)
    print("МЕТОД АДАМСА 4-го ПОРЯДКА")
    print("Глобальная погрешность ε_k = |y_ист - y_k|")
    print("="*70)
    
    solver_adams = ODESolver(f_system, x_start, [1.0, 1.0], h_main)
    x_adams, y_adams = solver_adams.adams_4(x_end)
    
    # Вычисляем глобальные погрешности для метода Адамса
    adams_global_errors = []
    print(f"\nРезультаты метода Адамса с шагом h = {h_main}:")
    print(f"{'k':<3} {'x_k':<12} {'y_k':<14} {'y_ист':<14} {'ε_k = |y_ист - y_k|':<20}")
    print("-" * 65)
    
    for i in range(len(x_adams)):
        y_true_at_x = np.interp(x_adams[i], x_ref, y_ref[:, 0])
        err = abs(y_true_at_x - y_adams[i, 0])
        adams_global_errors.append(err)
        print(f"{i:<3} {x_adams[i]:<12.6f} {y_adams[i,0]:<14.8f} {y_true_at_x:<14.8f} {err:<20.2e}")
    
    print(f"\nМаксимальная глобальная погрешность метода Адамса: {max(adams_global_errors):.2e}")
    
    # ==================== СРАВНИТЕЛЬНАЯ ТАБЛИЦА ====================
    print("\n" + "="*90)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА ГЛОБАЛЬНЫХ ПОГРЕШНОСТЕЙ (h = 0.1)")
    print("="*90)
    print(f"{'x':<12} {'Эйлер ε_k':<18} {'РК4 ε_k':<18} {'Адамс ε_k':<18}")
    print("-" * 66)
    
    for i in range(len(x_rk4_fixed)):
        print(f"{x_rk4_fixed[i]:<12.6f} {euler_global_errors[i]:<18.2e} {rk4_fixed_errors[i]:<18.2e} {adams_global_errors[i]:<18.2e}")
    
    # ==================== ПОСТРОЕНИЕ ГРАФИКОВ ====================
    plot_all_results(x_euler, y_euler, euler_global_errors, 
                     x_rk4_adapt, y_rk4_adapt, theta_values,
                     x_adams, y_adams, adams_global_errors,
                     x_rk4_fixed, y_rk4_fixed, rk4_fixed_errors,
                     x_ref, y_ref)


def plot_all_results(x_euler, y_euler, euler_errors,
                     x_rk4_adapt, y_rk4_adapt, theta_values,
                     x_adams, y_adams, adams_errors,
                     x_rk4_fixed, y_rk4_fixed, rk4_fixed_errors,
                     x_ref, y_ref):
    """Построение всех графиков"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # График 1: Сравнение решений
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_euler, y_euler[:, 0], 'o-', label='Эйлер', markersize=4, linewidth=1.5)
    ax1.plot(x_rk4_fixed, y_rk4_fixed[:, 0], 's-', label='РК4 (фикс)', markersize=4, linewidth=1.5)
    ax1.plot(x_adams, y_adams[:, 0], '^-', label='Адамс 4', markersize=4, linewidth=1.5)
    ax1.plot(x_rk4_adapt, y_rk4_adapt[:, 0], 'd-', label='РК4 (адапт)', markersize=2, linewidth=1, alpha=0.7)
    ax1.plot(x_ref, y_ref[:, 0], 'k-', linewidth=2, label='Эталон', alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Решение y(x)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Глобальные погрешности (все методы)
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(x_euler, euler_errors, 'o-', label='Эйлер', color='red', markersize=4)
    ax2.semilogy(x_rk4_fixed, rk4_fixed_errors, 's-', label='РК4 (фикс)', color='blue', markersize=4)
    ax2.semilogy(x_adams, adams_errors, '^-', label='Адамс 4', color='green', markersize=4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('ε_k = |y_ист - y_k|')
    ax2.set_title('Глобальные погрешности')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Локальная погрешность θ для адаптивного РК4
    ax3 = plt.subplot(2, 3, 3)
    steps = range(1, len(theta_values) + 1)
    ax3.semilogy(steps, theta_values, 'd-', color='purple', markersize=4)
    ax3.axhline(y=0.1, color='r', linestyle='--', label='θ = 0.1 (уменьшить шаг)')
    ax3.axhline(y=0.01, color='b', linestyle='--', label='θ = 0.01 (увеличить шаг)')
    ax3.fill_between(steps, 0.01, 0.1, alpha=0.2, color='green', label='Оптимальная зона')
    ax3.set_xlabel('Номер шага')
    ax3.set_ylabel('θ = |K2-K3| / |K1-K2|')
    ax3.set_title('Локальная погрешность (адаптивный РК4)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Производная y'(x)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x_euler, y_euler[:, 1], 'o-', label='Эйлер', markersize=4, linewidth=1.5)
    ax4.plot(x_rk4_fixed, y_rk4_fixed[:, 1], 's-', label='РК4 (фикс)', markersize=4, linewidth=1.5)
    ax4.plot(x_adams, y_adams[:, 1], '^-', label='Адамс 4', markersize=4, linewidth=1.5)
    ax4.plot(x_ref, y_ref[:, 1], 'k-', linewidth=2, label='Эталон', alpha=0.6)
    ax4.set_xlabel('x')
    ax4.set_ylabel("y'")
    ax4.set_title("Производная y'(x)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # График 5: Фазовый портрет
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(y_euler[:, 0], y_euler[:, 1], 'o-', label='Эйлер', markersize=3, linewidth=1)
    ax5.plot(y_rk4_fixed[:, 0], y_rk4_fixed[:, 1], 's-', label='РК4 (фикс)', markersize=3, linewidth=1)
    ax5.plot(y_adams[:, 0], y_adams[:, 1], '^-', label='Адамс 4', markersize=3, linewidth=1)
    ax5.plot(y_ref[:, 0], y_ref[:, 1], 'k-', linewidth=2, label='Эталон', alpha=0.6)
    ax5.set_xlabel('y')
    ax5.set_ylabel("y'")
    ax5.set_title('Фазовый портрет')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # График 6: Адаптивный шаг (точки РК4)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(x_rk4_adapt, y_rk4_adapt[:, 0], 'd-', color='purple', markersize=4, linewidth=1.5)
    ax6.plot(x_rk4_adapt, y_rk4_adapt[:, 0], 'd', color='purple', markersize=6, label='Узлы РК4 (адапт)')
    ax6.plot(x_ref[::100], y_ref[::100, 0], 'k-', linewidth=1, label='Эталон', alpha=0.5)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Адаптивный РК4 (отмечены узлы)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_all_results.png', dpi=150, bbox_inches='tight')
    print("\nГрафики сохранены в 'task1_all_results.png'")
    plt.show()


if __name__ == "__main__":
    main()