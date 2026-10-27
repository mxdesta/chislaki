import math  # Подключаем математическую библиотеку для числа Пи, косинуса и экспоненты
import matplotlib.pyplot as plt  # Подключаем библиотеку для построения графиков

class Variant3:
    # Класс, хранящий параметры конкретного варианта задачи
    l = math.pi  # Длина отрезка интегрирования (по пространству x)
    T = 1.0      # Конечное время интегрирования
    a = 1.0      # Параметр уравнения (коэффициент температуропроводности), a > 0
    
    @staticmethod
    def exact(x, t):
        # Функция, возвращающая точное (аналитическое) решение задачи
        return math.exp(-Variant3.a * t) * math.cos(x)
        
    @staticmethod
    def psi(x):
        # Начальное условие при t = 0: u(x, 0) = cos(x)
        return math.cos(x)
        
    @staticmethod
    def phi0(t):
        # Левое граничное условие 1-го рода (x = 0): u(0, t) = exp(-at)
        return math.exp(-Variant3.a * t)
        
    @staticmethod
    def phil(t):
        # Правое граничное условие 1-го рода (x = l = pi): u(pi, t) = -exp(-at)
        return -math.exp(-Variant3.a * t)

def solve_parabolic_v3(N, K, scheme='implicit', bc_approx='1st_2pt'):
    # Основная функция численного решения начально-краевой задачи
    h = Variant3.l / N       # Шаг сетки по пространственной координате x
    tau = Variant3.T / K     # Шаг сетки по времени t
    x = [j * h for j in range(N + 1)]  # Массив координат узлов по пространству
    t = [k * tau for k in range(K + 1)] # Массив временных слоев
    
    # Инициализация двумерного массива решения u нулями (размер (K+1) x (N+1))
    u = [[0.0] * (N + 1) for _ in range(K + 1)]
    
    # Заполнение нулевого временного слоя (k=0) начальным условием
    for j in range(N + 1):
        u[0][j] = Variant3.psi(x[j])
        
    # Вычисление сеточного числа Фурье (параметра схемы) sigma = a*tau / h^2
    sigma = Variant3.a * tau / (h ** 2)
    
    if scheme == 'explicit':
        # --- Блок явной конечно-разностной схемы ---
        for k in range(K):  # Цикл по времени (переход со слоя k на слой k+1)
            for j in range(1, N): # Цикл по внутренним пространственным узлам
                # Значение на новом слое считается явно через 3 точки предыдущего слоя
                u[k+1][j] = sigma * u[k][j+1] + (1 - 2*sigma) * u[k][j] + sigma * u[k][j-1]
            
            # Точное задание граничных условий 1-го рода на краях отрезка
            u[k+1][0] = Variant3.phi0(t[k+1]) # На левом конце (j=0)
            u[k+1][N] = Variant3.phil(t[k+1]) # На правом конце (j=N)
            
    else:
        # --- Блок неявной схемы и схемы Кранка-Николсона (через метод прогонки) ---
        # Весовой коэффициент: 1.0 для чисто неявной, 0.5 для Кранка-Николсона
        theta = 1.0 if scheme == 'implicit' else 0.5 
        
        for k in range(K): # Цикл по времени
            # Инициализация диагоналей матрицы СЛАУ и вектора правых частей
            a_mat = [0.0] * (N + 1) # Нижняя побочная диагональ
            b_mat = [0.0] * (N + 1) # Главная диагональ
            c_mat = [0.0] * (N + 1) # Верхняя побочная диагональ
            d_mat = [0.0] * (N + 1) # Вектор правых частей
            
            # Заполнение матрицы для внутренних узлов
            for j in range(1, N):
                a_mat[j] = -theta * Variant3.a / (h ** 2)
                b_mat[j] = (1.0 / tau) + theta * (2.0 * Variant3.a / (h ** 2))
                c_mat[j] = -theta * Variant3.a / (h ** 2)
                
                # Явная часть (влияние предыдущего слоя), зависящая от веса theta
                explicit_part = (1.0 - theta) * Variant3.a * (u[k][j+1] - 2.0*u[k][j] + u[k][j-1]) / (h ** 2)
                d_mat[j] = (1.0 / tau) * u[k][j] + explicit_part
            
            # Коэффициенты для краевых условий 1-го рода на границах (x=0 и x=l)
            b_mat[0] = 1.0; c_mat[0] = 0.0; d_mat[0] = Variant3.phi0(t[k+1])
            a_mat[N] = 0.0; b_mat[N] = 1.0; d_mat[N] = Variant3.phil(t[k+1])
            
            # Прямой ход метода прогонки (вычисление прогоночных коэффициентов A и B)
            A = [0.0] * (N + 1)
            B = [0.0] * (N + 1)
            # Значения A и B для левой границы
            A[0] = -c_mat[0] / b_mat[0] 
            B[0] = d_mat[0] / b_mat[0]
            
            # Рекуррентный расчет A и B для внутренних узлов
            for j in range(1, N):
                denom = b_mat[j] + a_mat[j] * A[j - 1] # Знаменатель (b_j + a_j * A_{j-1})
                A[j] = -c_mat[j] / denom
                B[j] = (d_mat[j] - a_mat[j] * B[j - 1]) / denom
                
            # Обработка правой границы (A_N = 0 для жесткого ГУ)
            denom_N = b_mat[N] + a_mat[N] * A[N - 1]
            A[N] = 0.0 
            B[N] = (d_mat[N] - a_mat[N] * B[N - 1]) / denom_N
            
            # Обратный ход метода прогонки (восстановление искомой функции u на новом слое k+1)
            u[k+1][N] = B[N] # Задаем краевое значение справа
            for j in range(N - 1, -1, -1): # Идем от правого края к левому (j уменьшается)
                u[k+1][j] = A[j] * u[k+1][j+1] + B[j]
                
    # Возвращаем пространственную сетку, временную сетку и найденное численное решение
    return x, t, u

# --- Блок тестирования и визуализации ---
if __name__ == '__main__':
    N, K = 20, 200 # Количество разбиений по пространству и времени
    h, tau = Variant3.l / N, Variant3.T / K # Вычисление шагов
    
    # Списки тестируемых схем
    schemes = [('explicit', 'Явная'), ('implicit', 'Неявная'), ('cn', 'Кранк-Николсон')]
    # (Оставлен список аппроксимаций ГУ, хотя для 1-го рода аппроксимация точная)
    bcs = [('1st_2pt', '2-точ. O(h)'), ('2nd_3pt', '3-точ. O(h^2)'), ('2nd_2pt', '2-точ. O(h^2)')]
    
    print(f"Сетка: N={N}, K={K}, sigma={Variant3.a * tau / h**2:.4f}\n")
    print(f"{'Схема':<15} | {'Аппроксимация ГУ':<15} | {'Макс. погрешность'}")
    print("-" * 55)
    
    results = {}
    # Цикл перебора всех комбинаций схем 
    for sch_code, sch_name in schemes:
        for bc_code, bc_name in bcs:
            # Вызов функции-солвера
            x, t, u_num = solve_parabolic_v3(N, K, scheme=sch_code, bc_approx=bc_code)
            # Вычисление равномерной нормы погрешности (максимальное отклонение) на последнем слое T
            err = max(abs(u_num[-1][j] - Variant3.exact(x[j], t[-1])) for j in range(N+1))
            results[(sch_code, bc_code)] = (u_num, err)
            print(f"{sch_name:<15} | {bc_name:<15} | {err:.6e}")
    print("-" * 55)

    # --- Построение графиков профилей решения и погрешности ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Профили решений и динамика погрешности (все 3 схемы)', fontsize=14)
    
    for i, (sch_code, sch_name) in enumerate(schemes):
        ax_sol, ax_err = axes[i, 0], axes[i, 1]
        
        # Отрисовка точного (аналитического) решения сплошной линией
        u_ex_final = [Variant3.exact(xj, t[-1]) for xj in x]
        ax_sol.plot(x, u_ex_final, 'k-', linewidth=2, label='Точное')
        
        for bc_code, bc_name in bcs:
            u_num, _ = results[(sch_code, bc_code)]
            # Отрисовка численного решения пунктиром
            ax_sol.plot(x, u_num[-1], '--', label=f'Числ. ({bc_name})')
            
            # Расчет динамики максимальной погрешности во времени
            err_t = [max(abs(u_num[k][j] - Variant3.exact(x[j], t[k])) for j in range(N+1)) for k in range(K+1)]
            ax_err.plot(t, err_t, label=f'{bc_name}')
            
        # Настройка графиков
        ax_sol.set_title(f'{sch_name}: Решение при t={Variant3.T}')
        ax_sol.grid(True); ax_sol.legend()
        
        ax_err.set_title(f'{sch_name}: Макс. погрешность ||U-u||')
        ax_err.grid(True); ax_err.legend()
        
    plt.tight_layout()
    plt.show()

    # --- Исследование зависимости погрешности от шагов h и tau ---
    N_list = [10, 20, 40, 80]     # Массив варьируемых узлов по x
    K_list = [100, 200, 400, 800] # Массив варьируемых шагов по t
    
    errs_h = []
    # Исследование порядка точности по пространству (фиксируем время)
    for cur_N in N_list:
        cur_K = 10000 # Фиксируем малый шаг по времени (много узлов K), чтобы исключить ошибку от времени
        x_tmp, t_tmp, u_tmp = solve_parabolic_v3(cur_N, cur_K, scheme='implicit')
        # Сохраняем итоговую макс. погрешность
        errs_h.append(max(abs(u_tmp[-1][j] - Variant3.exact(x_tmp[j], t_tmp[-1])) for j in range(cur_N+1)))
        
    errs_tau = []
    # Исследование порядка точности по времени (фиксируем пространство)
    for cur_K in