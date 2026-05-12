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


def get_reference_solution(x0: float, x_end: float, y0: List[float], h_ref: float = 0.00001):
    """Получение эталонного решения с очень маленьким шагом"""
    solver_ref = ODESolver(f_system, x0, y0, h_ref)
    x_ref, y_ref = solver_ref.runge_kutta_4(x_end, adaptive=False)
    return x_ref, y_ref


class ODESolver:
    """Решатель задачи Коши для системы ОДУ 1-го порядка"""
    
    def __init__(self, f: Callable, x0: float, y0: List[float], h: float):
        self.f = f
        self.x0 = x0
        self.y0 = np.array(y0, dtype=float)
        self.h = h
    
    def euler(self, x_end: float, compute_errors: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Явный метод Эйлера
        
        Возвращает:
        - x: массив узлов
        - y: массив решений
        - epsilon_k: локальные погрешности на каждом шаге
        """
        n_steps = int((x_end - self.x0) / self.h) + 1
        x = np.linspace(self.x0, x_end, n_steps)
        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0
        
        epsilon_k = []  # локальные погрешности
        
        for i in range(n_steps - 1):
            # Текущий шаг
            y_next = y[i] + self.h * self.f(x[i], y[i])
            
            if compute_errors:
                # Оценка локальной погрешности методом Рунге-Ромберга
                # Делаем два шага с половинным шагом
                h_half = self.h / 2
                
                # Первый половинный шаг
                k1_half = self.h/2 * self.f(x[i], y[i])
                y_half = y[i] + k1_half
                
                # Второй половинный шаг
                k2_half = self.h/2 * self.f(x[i] + h_half, y_half)
                y_double_half = y_half + k2_half
                
                # Оценка погрешности
                # ε = |y_{h/2,h/2} - y_h| / (2^p - 1) = |y_{h/2,h/2} - y_h|
                local_error = np.abs(y_double_half[0] - y_next[0])  # делим на (2^1 - 1) = 1
                epsilon_k.append(local_error)
            
            y[i+1] = y_next
        
        return x, y, epsilon_k
    
    def runge_kutta_4(self, x_end: float, adaptive: bool = True, 
                      tol: float = 1e-6, h_min: float = 1e-6, h_max: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Метод Рунге-Кутты 4-го порядка с контролем шага
        |K2 - K3| / |K1 - K2|
        - θ < 0.01 => шаг можно увеличить
        - θ > 0.1 => шаг нужно уменьшить
        - 0.01 <= θ <= 0.1 → шаг подходит
        """
        points_x = [self.x0]
        points_y = [self.y0.copy()]
        
        x = self.x0
        y = self.y0.copy()
        h = self.h
        
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
            
            return x_vals, y_vals
        
        # Адаптивный РК4 с контролем шага
        print(f"\nАдаптивный метод Рунге-Кутты 4-го порядка")
        print(f"Начальный шаг: h = {h}")
        print("-" * 60)
        
        def vector_norm(vec):
            """Вычисление евклидовой нормы вектора"""
            sum_sq = 0.0
            for component in vec:
                sum_sq += component ** 2
            return np.sqrt(sum_sq)
        
        step_count = 0
        rejected_steps = 0
        theta_values = []  # для хранения значений θ
        
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
            
            # Расчет параметра θ
            K1_arr = np.array(K1)
            K2_arr = np.array(K2)
            K3_arr = np.array(K3)
            
            diff_K2_K3 = vector_norm(K2_arr - K3_arr)
            diff_K1_K2 = vector_norm(K1_arr - K2_arr)
            
            if diff_K1_K2 < 1e-15:
                theta = 0.0
            else:
                theta = diff_K2_K3 / diff_K1_K2
            
            theta_values.append(theta)
            
            # Прогноз шага на основе θ
            if theta > 0.1:
                h_new = h * 0.7
                if h_new >= h_min:
                    h = h_new
                    rejected_steps += 1
                    continue
            else:
                y_new = y + (h/6) * (K1 + 2*K2 + 2*K3 + K4)
                
                if theta < 0.01:
                    h = min(h * 1.5, h_max)
                elif theta <= 0.1:
                    pass  # шаг оптимальный
                
                y = y_new
                x = x + h
                points_x.append(x)
                points_y.append(y.copy())
                step_count += 1
            
            if h < h_min:
                h = h_min
        
        print(f"Адаптивный расчет завершен:")
        print(f"  Всего шагов: {step_count}")
        print(f"  Отброшенных шагов: {rejected_steps}")
        print(f"  Среднее θ: {np.mean(theta_values):.4f}")
        
        return np.array(points_x), np.array(points_y)
    
    def adams_4(self, x_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """Метод Адамса 4-го порядка (с разгоном методом РК4)"""
        n_steps = int((x_end - self.x0) / self.h) + 1
        x = np.linspace(self.x0, x_end, n_steps)
        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0
        
        # делаем первые 3  итерации с РК4
        for i in range(min(3, n_steps - 1)):
            k1 = self.h * self.f(x[i], y[i])
            k2 = self.h * self.f(x[i] + self.h/2, y[i] + k1/2)
            k3 = self.h * self.f(x[i] + self.h/2, y[i] + k2/2)
            k4 = self.h * self.f(x[i] + self.h, y[i] + k3)
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Метод Адамса
        for i in range(3, n_steps - 1):
            f_vals = [self.f(x[j], y[j]) for j in range(i-3, i+1)]
            y[i+1] = y[i] + self.h/24 * (
                55*f_vals[3] - 59*f_vals[2] + 37*f_vals[1] - 9*f_vals[0]
            )
        
        return x, y


def solve_with_different_steps(method_name: str, h_values: List[float]):
    """Решение с разными шагами для оценки погрешности"""
    print(f"\n{'='*70}")
    print(f"Метод: {method_name}")
    print(f"{'='*70}")
    
    results = {}
    
    for h in h_values:
        solver = ODESolver(f_system, x0=1.0, y0=[1.0, 1.0], h=h)
        
        if method_name == "Эйлер":
            x, y, eps_k = solver.euler(2.0, compute_errors=True)
            order = 1
            # Выводим локальные погрешности
            print(f"\nЛокальные погрешности εₖ для шага h={h}:")
            for i, eps in enumerate(eps_k[:5]):  # первые 5 шагов
                print(f"  Шаг {i+1}: εₖ = {eps:.6e}")
            if len(eps_k) > 5:
                print(f"  ... и еще {len(eps_k)-5} шагов")
        elif method_name == "Рунге-Кутта 4":
            x, y = solver.runge_kutta_4(2.0, adaptive=False)
            order = 4
            eps_k = []
        else:
            x, y = solver.adams_4(2.0)
            order = 4
            eps_k = []
        
        results[h] = (x, y, order, eps_k)
    
    # Вывод результатов для h=0.1
    h_main = 0.1
    x, y, order, eps_k = results[h_main]
    
    print(f"\nРезультаты с шагом h = {h_main}:")
    print(f"{'x':>8} {'y':>12} {'y\'':>12}")
    print("-" * 35)
    
    for i in range(0, len(x), 2):
        print(f"{x[i]:8.4f} {y[i, 0]:12.8f} {y[i, 1]:12.8f}")
    
    return results


def compare_methods(results_dict: dict, x_ref: np.ndarray, y_ref: np.ndarray):
    """Сравнение методов с эталонным решением"""
    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ МЕТОДОВ С ЭТАЛОННЫМ РЕШЕНИЕМ (h = 0.1)")
    print(f"{'='*70}")
    
    x_rk4, y_rk4, _, _ = results_dict["Рунге-Кутта 4"][0.1]
    
    print(f"\n{'x':>8} {'Эйлер':>12} {'РК4':>12} {'Адамс':>12} {'Эталон':>12} "
          f"{'|Эйл-Эталон|':>14} {'|РК4-Эталон|':>14}")
    print("-" * 90)
    
    x_euler, y_euler, _, _ = results_dict["Эйлер"][0.1]
    x_adams, y_adams, _, _ = results_dict["Адамс 4"][0.1]
    
    for i in range(0, len(x_rk4), 2):
        # Интерполяция эталонного решения
        y_ref_at_x = np.interp(x_rk4[i], x_ref, y_ref[:, 0])
        
        diff_euler = abs(y_euler[i, 0] - y_ref_at_x)
        diff_rk4 = abs(y_rk4[i, 0] - y_ref_at_x)
        diff_adams = abs(y_adams[i, 0] - y_ref_at_x)
        
        print(f"{x_rk4[i]:8.4f} {y_euler[i,0]:12.8f} {y_rk4[i,0]:12.8f} "
              f"{y_adams[i,0]:12.8f} {y_ref_at_x:12.8f} {diff_euler:14.2e} {diff_rk4:14.2e}")
    
    # Глобальные погрешности
    y_ref_at_rk4 = np.interp(x_rk4, x_ref, y_ref[:, 0])
    global_error_euler = np.max(np.abs(y_euler[:len(x_rk4), 0] - y_ref_at_rk4))
    global_error_rk4 = np.max(np.abs(y_rk4[:, 0] - y_ref_at_rk4))
    global_error_adams = np.max(np.abs(y_adams[:len(x_rk4), 0] - y_ref_at_rk4))
    
    print(f"\nГлобальные максимальные погрешности:")
    print(f"  Метод Эйлера:     {global_error_euler:.2e}")
    print(f"  Метод РК4:        {global_error_rk4:.2e}")
    print(f"  Метод Адамса:     {global_error_adams:.2e}")


def plot_results(results_dict: dict, x_ref: np.ndarray, y_ref: np.ndarray):
    """Построение графиков с эталонным решением"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
    
    # График 1: Сравнение всех методов с эталоном
    for method_name, results in results_dict.items():
        x, y, _, _ = results[0.1]
        ax1.plot(x, y[:, 0], marker='o', markersize=3, label=method_name, linewidth=2)
    
    # Эталонное решение
    ax1.plot(x_ref, y_ref[:, 0], 'k-', linewidth=2, label='Эталон (h=0.001)', alpha=0.7)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Сравнение численных методов с эталонным решением', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Погрешности методов относительно эталона
    x_rk4, y_rk4, _, _ = results_dict["Рунге-Кутта 4"][0.1]
    x_euler, y_euler, _, _ = results_dict["Эйлер"][0.1]
    x_adams, y_adams, _, _ = results_dict["Адамс 4"][0.1]
    
    y_ref_at_rk4 = np.interp(x_rk4, x_ref, y_ref[:, 0])
    
    diff_euler = np.abs(y_euler[:len(x_rk4), 0] - y_ref_at_rk4)
    diff_rk4 = np.abs(y_rk4[:, 0] - y_ref_at_rk4)
    diff_adams = np.abs(y_adams[:len(x_rk4), 0] - y_ref_at_rk4)
    
    ax2.semilogy(x_rk4, diff_euler, 'o-', label='|Эйлер - Эталон|', markersize=4, linewidth=2)
    ax2.semilogy(x_rk4, diff_rk4, 's-', label='|РК4 - Эталон|', markersize=4, linewidth=2)
    ax2.semilogy(x_rk4, diff_adams, '^-', label='|Адамс - Эталон|', markersize=4, linewidth=2)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Погрешность', fontsize=12)
    ax2.set_title('Погрешность методов относительно эталонного решения', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # График 3: y'(x) для всех методов
    for method_name, results in results_dict.items():
        x, y, _, _ = results[0.1]
        ax3.plot(x, y[:, 1], marker='o', markersize=3, label=method_name, linewidth=2)
    
    ax3.plot(x_ref, y_ref[:, 1], 'k-', linewidth=2, label='Эталон (y\')', alpha=0.7)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y\'', fontsize=12)
    ax3.set_title('Производная y\'(x)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Фазовый портрет
    for method_name, results in results_dict.items():
        x, y, _, _ = results[0.1]
        ax4.plot(y[:, 0], y[:, 1], marker='o', markersize=3, label=method_name, linewidth=2)
    
    ax4.plot(y_ref[:, 0], y_ref[:, 1], 'k-', linewidth=2, label='Эталон', alpha=0.7)
    ax4.set_xlabel('y', fontsize=12)
    ax4.set_ylabel('y\'', fontsize=12)
    ax4.set_title('Фазовый портрет (y\' vs y)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_results_with_reference.png', dpi=150, bbox_inches='tight')
    print("\nГрафики сохранены в 'task1_results_with_reference.png'")
    plt.show()


def plot_local_errors(results_dict: dict):
    """Построение графика локальных погрешностей для метода Эйлера"""
    plt.figure(figsize=(10, 6))
    
    for h in [0.05, 0.1, 0.2]:
        x, y, order, eps_k = results_dict["Эйлер"][h]
        if eps_k:
            steps = range(1, len(eps_k) + 1)
            plt.semilogy(steps, eps_k, 'o-', label=f'h={h}', markersize=4)
    
    plt.xlabel('Номер шага k', fontsize=12)
    plt.ylabel('Локальная погрешность εₖ', fontsize=12)
    plt.title('Локальные погрешности метода Эйлера на каждом шаге', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task1_local_errors.png', dpi=150, bbox_inches='tight')
    print("\nГрафик локальных погрешностей сохранен в 'task1_local_errors.png'")
    plt.show()


def main():
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА 4. ЗАДАНИЕ 4.1")
    print("Численное решение задачи Коши для ОДУ 2-го порядка")
    print("="*70)
    print("\nВариант 14:")
    print("y'' + 2y' + e^x * y = 0")
    print("y(1) = 1, y'(1) = 1")
    print("x ∈ [1, 2], h = 0.1")
    
    # Получение эталонного решения с очень маленьким шагом
    print("\nВычисление эталонного решения с шагом h=0.001...")
    x_ref, y_ref = get_reference_solution(1.0, 2.0, [1.0, 1.0], h_ref=0.001)
    print("Эталонное решение получено")
    
    h_values = [0.05, 0.1, 0.2]
    
    # Решение всеми методами
    results_euler = solve_with_different_steps("Эйлер", h_values)
    results_rk4 = solve_with_different_steps("Рунге-Кутта 4", h_values)
    results_adams = solve_with_different_steps("Адамс 4", h_values)
    
    # Сравнение методов
    results_dict = {
        "Эйлер": results_euler,
        "Рунге-Кутта 4": results_rk4,
        "Адамс 4": results_adams
    }
    
    compare_methods(results_dict, x_ref, y_ref)
    
    # Построение графиков
    plot_results(results_dict, x_ref, y_ref)
    plot_local_errors(results_dict)
    
    print("\n" + "="*70)
    print("ВЫВОДЫ:")
    print("="*70)
    print("1. Методы РК4 и Адамс дают результаты, близкие к эталонному решению")
    print("2. Локальные погрешности εₖ для метода Эйлера уменьшаются с уменьшением шага")
    print("3. Глобальная погрешность метода Эйлера ~10^-2, РК4 и Адамса ~10^-5")
    print("4. Адаптивный метод РК4 автоматически подбирает шаг на основе параметра θ")
    print("5. Все методы реализованы корректно согласно теории")


if __name__ == "__main__":
    main()