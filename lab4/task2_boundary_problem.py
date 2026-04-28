"""

Вариант 14:
(e^x + 1)y'' - 2y' - e^x * y = 0
y'(0) = 1
y'(1) - y(1) = 1

Точное решение: y(x) = e^x - 1
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from scipy.optimize import fsolve


class BoundaryProblemSolver:
    """Решатель краевой задачи для ОДУ 2-го порядка"""
    
    def __init__(self, a: float, b: float, h: float):
        """
        a, b: границы отрезка
        h: шаг сетки
        """
        self.a = a
        self.b = b
        self.h = h
        self.n = int((b - a) / h) + 1
        self.x = np.linspace(a, b, self.n)
    
    def shooting_method(self, f_system: Callable, bc_left: Tuple, bc_right: Callable,
                       initial_guess: float = 0.0, tol: float = 1e-6, max_iter: int = 50,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        
        def solve_cauchy(eta):
            """Решение задачи Коши с параметром eta"""
            # Преобразуем eta в скаляр (fsolve может передать массив)
            eta_val = float(np.atleast_1d(eta)[0])
            
            # Определяем начальные условия в зависимости от типа левого ГУ
            if bc_left[0] == 'derivative':
                # y'(a) = bc_left[1], y(a) = eta (неизвестно)
                y0 = np.array([eta_val, float(bc_left[1])], dtype=float)
            else:
                # y(a) = bc_left[1], y'(a) = eta (неизвестно)
                y0 = np.array([float(bc_left[1]), eta_val], dtype=float)
            
            # Решаем методом РК4
            y = np.zeros((self.n, 2))
            y[0] = y0
            
            for i in range(self.n - 1):
                k1 = self.h * f_system(self.x[i], y[i])
                k2 = self.h * f_system(self.x[i] + self.h/2, y[i] + k1/2)
                k3 = self.h * f_system(self.x[i] + self.h/2, y[i] + k2/2)
                k4 = self.h * f_system(self.x[i] + self.h, y[i] + k3)
                y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            return y
        
        def residual(eta):
            """Невязка правого граничного условия"""
            y = solve_cauchy(eta)
            return bc_right(y[-1])
    
        
        if verbose:
            print("\n" + "="*70)
            print("МЕТОД СТРЕЛЬБЫ: Итерационный процесс")
            print("="*70)
            print(f"{'Итер':>6} {'eta':>15} {'Невязка F(eta)':>18} {'|Delta eta|':>18}")
            print("-"*70)
        
        # Начальные приближения
        eta_prev = initial_guess
        eta_curr = initial_guess + 0.1  # небольшое возмущение
        
        for iteration in range(max_iter):
            # Решаем задачу Коши для текущих значений η
            y_prev = solve_cauchy(eta_prev)
            y_curr = solve_cauchy(eta_curr)
            
            # Вычисляем невязки правого ГУ
            F_prev = residual(eta_prev)
            F_curr = residual(eta_curr)
            
            # Вывод информации о текущей итерации
            if verbose and iteration > 0:
                delta_eta = abs(eta_curr - eta_prev)
                print(f"{iteration:6d} {eta_curr:15.8e} {F_curr:15.8e} {delta_eta:15.8e}")
            
            # КРИТЕРИЙ ОСТАНОВКИ 1: Невязка достаточно мала
            if abs(F_curr) < tol:
                if verbose:
                    print("-"*70)
                    print(f"✓ Сходимость достигнута за {iteration+1} итераций")
                    print(f"  Невязка: |F(η)| = {abs(F_curr):.2e} < {tol:.2e}")
                    print(f"  Найденное значение: η = {eta_curr:.10f}")
                return self.x, y_curr[:, 0]
            
            # Проверка на деление на ноль
            if abs(F_curr - F_prev) < 1e-14:
                if verbose:
                    print("\n⚠ Предупреждение: F_curr ≈ F_prev, переключение на fsolve")
                eta_solution = fsolve(residual, eta_curr, full_output=False)[0]
                y_solution = solve_cauchy(eta_solution)
                return self.x, y_solution[:, 0]
            
            # МЕТОД СЕКУЩИХ: Вычисление следующего приближения
            eta_next = eta_curr - F_curr * (eta_curr - eta_prev) / (F_curr - F_prev)
            
            # КРИТЕРИЙ ОСТАНОВКИ 2: Изменение η мало
            delta_eta = abs(eta_next - eta_curr)
            relative_change = delta_eta / (1 + abs(eta_curr))
            
            if relative_change < tol:
                y_next = solve_cauchy(eta_next)
                F_next = residual(eta_next)
                if verbose:
                    print(f"{iteration+1:6d} {eta_next:15.8e} {F_next:15.8e} {delta_eta:15.8e}")
                    print("-"*70)
                    print(f"✓ Сходимость по изменению η за {iteration+1} итераций")
                    print(f"  Относительное изменение: {relative_change:.2e} < {tol:.2e}")
                    print(f"  Найденное значение: η = {eta_next:.10f}")
                return self.x, y_next[:, 0]
            
            # Переход к следующей итерации
            eta_prev, eta_curr = eta_curr, eta_next
        
        # НЕ СОШЛОСЬ ЗА max_iter ИТЕРАЦИЙ
        if verbose:
            print("-"*70)
            print(f"⚠ Не сошлось за {max_iter} итераций")
            print(f"  Последняя невязка: |F(η)| = {abs(F_curr):.2e}")
        
        return self.x, y_curr[:, 0]
    
    def finite_difference_method(self, p: Callable, q: Callable, f: Callable,
                                bc_left: Tuple, bc_right: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Конечно-разностный метод для уравнения y'' + p(x)y' + q(x)y = f(x)
        bc_left: левое граничное условие (тип, коэффициенты, значение)
        bc_right: правое граничное условие (тип, коэффициенты, значение)
        """
        n = self.n
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Левое граничное условие
        if bc_left[0] == 'derivative':
            # y'(a) = value: используем одностороннюю разность
            # (-3y_0 + 4y_1 - y_2) / (2h) = value
            A[0, 0] = -3 / (2*self.h)
            A[0, 1] = 4 / (2*self.h)
            A[0, 2] = -1 / (2*self.h)
            b[0] = bc_left[1]
        elif bc_left[0] == 'mixed':
            # alpha*y(a) + beta*y'(a) = value
            alpha, beta, value = bc_left[1], bc_left[2], bc_left[3]
            A[0, 0] = alpha - 3*beta / (2*self.h)
            A[0, 1] = 4*beta / (2*self.h)
            A[0, 2] = -beta / (2*self.h)
            b[0] = value
        else:  # 'value'
            A[0, 0] = 1
            b[0] = bc_left[1]
        
        # Внутренние точки: y''_i + p(x_i)y'_i + q(x_i)y_i = f(x_i)
        # y''_i ≈ (y_{i+1} - 2y_i + y_{i-1}) / h^2
        # y'_i ≈ (y_{i+1} - y_{i-1}) / (2h)
        for i in range(1, n-1):
            xi = self.x[i]
            pi = p(xi)
            qi = q(xi)
            fi = f(xi)
            
            A[i, i-1] = 1/self.h**2 - pi/(2*self.h)
            A[i, i] = -2/self.h**2 + qi
            A[i, i+1] = 1/self.h**2 + pi/(2*self.h)
            b[i] = fi
        
        # Правое граничное условие
        if bc_right[0] == 'derivative':
            # y'(b) = value: используем одностороннюю разность
            # (y_{n-3} - 4y_{n-2} + 3y_{n-1}) / (2h) = value
            A[n-1, n-3] = 1 / (2*self.h)
            A[n-1, n-2] = -4 / (2*self.h)
            A[n-1, n-1] = 3 / (2*self.h)
            b[n-1] = bc_right[1]
        elif bc_right[0] == 'mixed':
            # alpha*y(b) + beta*y'(b) = value
            alpha, beta, value = bc_right[1], bc_right[2], bc_right[3]
            A[n-1, n-3] = beta / (2*self.h)
            A[n-1, n-2] = -4*beta / (2*self.h)
            A[n-1, n-1] = alpha + 3*beta / (2*self.h)
            b[n-1] = value
        else:  # 'value'
            A[n-1, n-1] = 1
            b[n-1] = bc_right[1]
        
        # Решаем систему
        y = np.linalg.solve(A, b)
        
        return self.x, y


def problem_variant_14():
    """
    Решение варианта 14:
    (e^x + 1)y'' - 2y' - e^x * y = 0
    y'(0) = 1
    y'(1) - y(1) = 1
    
    Точное решение: y(x) = e^x - 1
    """
    
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА 4. ЗАДАНИЕ 4.2")
    print("Численное решение краевой задачи для ОДУ 2-го порядка")
    print("="*70)
    print("\nВариант 14:")
    print("(e^x + 1)y'' - 2y' - e^x * y = 0")
    print("y'(0) = 1")
    print("y'(1) - y(1) = 1")
    print("Точное решение: y(x) = e^x - 1")
    
    # Параметры
    a, b = 0.0, 1.0
    h_values = [0.1, 0.05]
    
    def exact_solution(x):
        """Точное решение"""
        return np.exp(x) - 1
    
    # Система для метода стрельбы
    def f_system(x, y):
        """
        Преобразование в систему 1-го порядка
        y' = z
        z' = (2z + e^x * y) / (e^x + 1)
        """
        y_val, z_val = y
        return np.array([z_val, (2*z_val + np.exp(x)*y_val) / (np.exp(x) + 1)])
    
    # Граничные условия для метода стрельбы
    bc_left_shooting = ('derivative', 1.0)  # y'(0) = 1
    
    def bc_right_shooting(y_end):
        """Правое ГУ: y'(1) - y(1) = 1"""
        return y_end[1] - y_end[0] - 1
    
    # Коэффициенты для конечно-разностного метода
    # Приводим к виду: y'' + p(x)y' + q(x)y = f(x)
    # y'' - 2/(e^x+1) * y' - e^x/(e^x+1) * y = 0
    def p(x):
        return -2 / (np.exp(x) + 1)
    
    def q(x):
        return -np.exp(x) / (np.exp(x) + 1)
    
    def f(x):
        return 0.0
    
    bc_left_fd = ('derivative', 1.0)  # y'(0) = 1
    bc_right_fd = ('mixed', -1, 1, 1)  # -y(1) + y'(1) = 1
    
    results = {}
    
    for h in h_values:
        print(f"\n{'='*70}")
        print(f"Шаг сетки h = {h}")
        print(f"{'='*70}")
        
        solver = BoundaryProblemSolver(a, b, h)
        
        # Метод стрельбы
        print("\n--- Метод стрельбы ---")
        x_shoot, y_shoot = solver.shooting_method(
            f_system, bc_left_shooting, bc_right_shooting, initial_guess=0.0,
            verbose=(h == h_values[0])  # Показываем итерации только для первого шага
        )
        
        # Метод конечных разностей
        print("--- Конечно-разностный метод ---")
        x_fd, y_fd = solver.finite_difference_method(
            p, q, f, bc_left_fd, bc_right_fd
        )
        
        # Вычисление погрешностей
        y_exact_shoot = np.array([exact_solution(xi) for xi in x_shoot])
        y_exact_fd = np.array([exact_solution(xi) for xi in x_fd])
        
        error_shoot = np.abs(y_shoot - y_exact_shoot)
        error_fd = np.abs(y_fd - y_exact_fd)
        
        results[h] = {
            'shooting': (x_shoot, y_shoot, error_shoot),
            'fd': (x_fd, y_fd, error_fd)
        }
        
        # Вывод результатов
        print(f"\n{'x':>8} {'y_стрельба':>14} {'y_кон.разн':>14} {'y_точн':>14} {'ε_стр':>12} {'ε_кр':>12}")
        print("-" * 80)
        
        for i in range(0, len(x_shoot), max(1, len(x_shoot)//10)):
            print(f"{x_shoot[i]:8.4f} {y_shoot[i]:14.8f} {y_fd[i]:14.8f} "
                  f"{y_exact_shoot[i]:14.8f} {error_shoot[i]:12.2e} {error_fd[i]:12.2e}")
        
        print(f"\nМаксимальная погрешность метода стрельбы: {np.max(error_shoot):.6e}")
        print(f"Максимальная погрешность конечно-разностного метода: {np.max(error_fd):.6e}")
    
    # Оценка по Рунге-Ромбергу
    if len(h_values) >= 2:
        print(f"\n{'='*70}")
        print("ОЦЕНКА ПОГРЕШНОСТИ ПО МЕТОДУ РУНГЕ-РОМБЕРГА")
        print(f"{'='*70}")
        
        h1, h2 = h_values[0], h_values[1]
        
        # Для метода стрельбы (порядок 4 - РК4)
        x1, y1, _ = results[h1]['shooting']
        x2, y2, _ = results[h2]['shooting']
        
        # Находим общие точки
        rr_errors_shoot = []
        for i, xi in enumerate(x1):
            idx = np.where(np.abs(x2 - xi) < 1e-10)[0]
            if len(idx) > 0:
                rr_err = np.abs(y1[i] - y2[idx[0]]) / (2**4 - 1)
                rr_errors_shoot.append(rr_err)
        
        print(f"\nМетод стрельбы (порядок 4):")
        print(f"Средняя оценка: {np.mean(rr_errors_shoot):.6e}")
        print(f"Максимальная оценка: {np.max(rr_errors_shoot):.6e}")
        
        # Для конечно-разностного метода (порядок 2)
        x1, y1, _ = results[h1]['fd']
        x2, y2, _ = results[h2]['fd']
        
        rr_errors_fd = []
        for i, xi in enumerate(x1):
            idx = np.where(np.abs(x2 - xi) < 1e-10)[0]
            if len(idx) > 0:
                rr_err = np.abs(y1[i] - y2[idx[0]]) / (2**2 - 1)
                rr_errors_fd.append(rr_err)
        
        print(f"\nКонечно-разностный метод (порядок 2):")
        print(f"Средняя оценка: {np.mean(rr_errors_fd):.6e}")
        print(f"Максимальная оценка: {np.max(rr_errors_fd):.6e}")
    
    # Построение графиков
    plot_results(results, exact_solution, h_values[0])
    
    print("\n" + "="*70)
    print("ВЫВОДЫ:")
    print("="*70)
    print("1. Оба метода дают хорошее приближение к точному решению")
    print("2. Метод стрельбы имеет более высокую точность (порядок 4)")
    print("3. Конечно-разностный метод проще в реализации")
    print("4. Оценки по Рунге-Ромбергу согласуются с прямым сравнением")


def plot_results(results: dict, exact_solution: Callable, h_main: float):
    """Построение графиков"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x_shoot, y_shoot, error_shoot = results[h_main]['shooting']
    x_fd, y_fd, error_fd = results[h_main]['fd']
    
    # График решений
    x_exact = np.linspace(0, 1, 200)
    y_exact = exact_solution(x_exact)
    
    ax1.plot(x_exact, y_exact, 'k-', linewidth=2, label='Точное решение')
    ax1.plot(x_shoot, y_shoot, 'ro', markersize=5, label='Метод стрельбы')
    ax1.plot(x_fd, y_fd, 'bs', markersize=4, label='Конечно-разностный')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Сравнение методов (h = {h_main})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График погрешностей
    ax2.semilogy(x_shoot, error_shoot, 'ro-', markersize=5, label='Метод стрельбы')
    ax2.semilogy(x_fd, error_fd, 'bs-', markersize=4, label='Конечно-разностный')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('|ε|')
    ax2.set_title('Погрешность методов')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем в текущей директории скрипта
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'task2_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафики сохранены в '{output_path}'")
    plt.show()


if __name__ == "__main__":
    problem_variant_14()
