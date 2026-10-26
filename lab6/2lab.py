import math
import matplotlib.pyplot as plt

class WaveVariant3:
    l = math.pi
    T = 2.0
    a2 = 1.0
    c = -3.0  # Коэффициент при u в источниковом члене
    
    @staticmethod
    def exact(x, t):
        return math.cos(x) * math.sin(2.0 * t)
        
    @staticmethod
    def psi1(x):
        return 0.0
        
    @staticmethod
    def psi2(x):
        return 2.0 * math.cos(x)
        
    @staticmethod
    def psi1_second_deriv(x):
        # Поскольку psi1(x) = 0, её вторая производная тоже 0
        return 0.0
        
    @staticmethod
    def phi0(t):
        return math.sin(2.0 * t)
        
    @staticmethod
    def phil(t):
        return -math.sin(2.0 * t)

def solve_hyperbolic_v3(N, K, scheme='explicit', ic2_approx='2nd'):
    h = WaveVariant3.l / N
    tau = WaveVariant3.T / K
    x = [j * h for j in range(N + 1)]
    t = [k * tau for k in range(K + 1)]
    
    u = [[0.0] * (N + 1) for _ in range(K + 1)]
    
    # 1. Нулевой слой (t = 0)
    for j in range(N + 1):
        u[0][j] = WaveVariant3.psi1(x[j])
        
    # 2. Первый слой (t = tau)
    for j in range(1, N):
        if ic2_approx == '1st':
            u[1][j] = WaveVariant3.psi1(x[j]) + tau * WaveVariant3.psi2(x[j])
        elif ic2_approx == '2nd':
            # u_tt = a^2 * u_xx + c * u
            u_tt_0 = WaveVariant3.a2 * WaveVariant3.psi1_second_deriv(x[j]) + WaveVariant3.c * WaveVariant3.psi1(x[j])
            u[1][j] = WaveVariant3.psi1(x[j]) + tau * WaveVariant3.psi2(x[j]) + (tau**2 / 2.0) * u_tt_0
            
    u[1][0] = WaveVariant3.phi0(tau)
    u[1][N] = WaveVariant3.phil(tau)
    
    sigma = WaveVariant3.a2 * (tau / h) ** 2
    
    if scheme == 'explicit':
        for k in range(1, K):
            for j in range(1, N):
                # u_tt = u_xx - 3u
                u[k+1][j] = (2.0 * u[k][j] - u[k-1][j] 
                             + sigma * (u[k][j+1] - 2.0 * u[k][j] + u[k][j-1]) 
                             + WaveVariant3.c * tau**2 * u[k][j])
            u[k+1][0] = WaveVariant3.phi0(t[k+1])
            u[k+1][N] = WaveVariant3.phil(t[k+1])
            
    elif scheme == 'implicit':
        for k in range(1, K):
            a_mat = [0.0] * (N + 1)
            b_mat = [0.0] * (N + 1)
            c_mat = [0.0] * (N + 1)
            d_mat = [0.0] * (N + 1)
            
            for j in range(1, N):
                a_mat[j] = -sigma
                # Добавляем вклад члена -3u, аппроксимированного на слое k+1 для абсолютной устойчивости
                b_mat[j] = 1.0 + 2.0 * sigma - WaveVariant3.c * tau**2
                c_mat[j] = -sigma
                d_mat[j] = 2.0 * u[k][j] - u[k-1][j]
                
            b_mat[0] = 1.0; c_mat[0] = 0.0; d_mat[0] = WaveVariant3.phi0(t[k+1])
            a_mat[N] = 0.0; b_mat[N] = 1.0; d_mat[N] = WaveVariant3.phil(t[k+1])
            
            # Прогонка
            A = [0.0] * (N + 1)
            B = [0.0] * (N + 1)
            A[0] = -c_mat[0] / b_mat[0]
            B[0] = d_mat[0] / b_mat[0]
            for j in range(1, N):
                denom = b_mat[j] + a_mat[j] * A[j - 1]
                A[j] = -c_mat[j] / denom
                B[j] = (d_mat[j] - a_mat[j] * B[j - 1]) / denom
                
            denom_N = b_mat[N] + a_mat[N] * A[N - 1]
            A[N] = 0.0
            B[N] = (d_mat[N] - a_mat[N] * B[N - 1]) / denom_N
            
            u[k+1][N] = B[N]
            for j in range(N - 1, -1, -1):
                u[k+1][j] = A[j] * u[k+1][j+1] + B[j]
                
    return x, t, u

if __name__ == '__main__':
    N, K = 40, 400
    h = WaveVariant3.l / N
    tau = WaveVariant3.T / K
    sigma = WaveVariant3.a2 * (tau / h) ** 2
    
    print(f"Сетка: N = {N}, K = {K}, h = {h:.4f}, tau = {tau:.5f}")
    print(f"Число Куранта (sigma) = {sigma:.4f}\n")
    
    configs = [
        ('explicit', '1st', 'Явная (1-й пор. НУ)'),
        ('explicit', '2nd', 'Явная (2-й пор. НУ)'),
        ('implicit', '1st', 'Неявная (1-й пор. НУ)'),
        ('implicit', '2nd', 'Неявная (2-й пор. НУ)')
    ]
    
    results = {}
    print(f"{'Конфигурация':<25} | {'Макс. погрешность'}")
    print("-" * 45)
    for sch, ic2, name in configs:
        x, t, u_num = solve_hyperbolic_v3(N, K, scheme=sch, ic2_approx=ic2)
        err = max(abs(u_num[-1][j] - WaveVariant3.exact(x[j], t[-1])) for j in range(N + 1))
        results[name] = (u_num, err)
        print(f"{name:<25} | {err:.6e}")
    
    # --- Графики ---
    u_ex_final = [WaveVariant3.exact(xj, t[-1]) for xj in x]
    
    plt.figure(figsize=(14, 5))
    
    # График профиля волны
    plt.subplot(1, 2, 1)
    plt.plot(x, u_ex_final, 'k-', linewidth=2, label='Точное решение')
    plt.plot(x, results['Явная (2-й пор. НУ)'][0][-1], 'ro--', markersize=4, label='Явная схема')
    plt.plot(x, results['Неявная (2-й пор. НУ)'][0][-1], 'bs--', markersize=4, label='Неявная схема')
    plt.title(f'Профиль струны u(x, T) при T={WaveVariant3.T}')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.grid(True)
    plt.legend()
    
    # График динамики погрешности
    plt.subplot(1, 2, 2)
    for name in [c[2] for c in configs]:
        u_num = results[name][0]
        err_t = [max(abs(u_num[k][j] - WaveVariant3.exact(x[j], t[k])) for j in range(N + 1)) for k in range(K + 1)]
        style = '-' if 'Явная' in name else '--'
        plt.plot(t, err_t, style, label=name)
        
    plt.title('Максимальная погрешность по времени ||U - u||')
    plt.xlabel('t')
    plt.ylabel('Погрешность')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # --- Исследование зависимости погрешности от параметров сетки ---
    N_list = [10, 20, 40, 80]
    K_list = [100, 200, 400, 800]
    
    errs_h = []
    for cur_N in N_list:
        cur_K = 2000 # Фиксируем малый шаг по времени
        x_tmp, t_tmp, u_tmp = solve_hyperbolic_v3(cur_N, cur_K, scheme='implicit')
        errs_h.append(max(abs(u_tmp[-1][j] - WaveVariant3.exact(x_tmp[j], t_tmp[-1])) for j in range(cur_N+1)))
        
    errs_tau = []
    for cur_K in K_list:
        cur_N = 200 # Фиксируем малый шаг по пространству
        x_tmp, t_tmp, u_tmp = solve_hyperbolic_v3(cur_N, cur_K, scheme='implicit')
        errs_tau.append(max(abs(u_tmp[-1][j] - WaveVariant3.exact(x_tmp[j], t_tmp[-1])) for j in range(cur_N+1)))
        
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog([WaveVariant3.l/n for n in N_list], errs_h, 'o-', color='blue')
    ax1.set_title('Зависимость ошибки от шага h')
    ax1.set_xlabel('h')
    ax1.set_ylabel('Погрешность')
    ax1.grid(True)
    
    ax2.loglog([WaveVariant3.T/k for k in K_list], errs_tau, 's-', color='green')
    ax2.set_title('Зависимость ошибки от шага tau')
    ax2.set_xlabel('tau')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()