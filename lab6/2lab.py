import math
import matplotlib.pyplot as plt

class WaveVariant1:
    l = math.pi
    T = 2.0
    a = 1.0  # Параметр a (для расчетов примем a = 1.0)
    
    @staticmethod
    def exact(x, t):
        return math.sin(x - WaveVariant1.a * t)
        
    @staticmethod
    def psi1(x):
        return math.sin(x)
        
    @staticmethod
    def psi2(x):
        return -WaveVariant1.a * math.cos(x)
        
    @staticmethod
    def psi1_second_deriv(x):
        # Вторая производная от sin(x) это -sin(x)
        return -math.sin(x)
        
    @staticmethod
    def phi0(t):
        return -math.sin(WaveVariant1.a * t)
        
    @staticmethod
    def phil(t):
        return math.sin(WaveVariant1.a * t)

def solve_wave(N, K, scheme='explicit', ic2_approx='2nd'):
    h = WaveVariant1.l / N
    tau = WaveVariant1.T / K
    x = [j * h for j in range(N + 1)]
    t = [k * tau for k in range(K + 1)]
    
    u = [[0.0] * (N + 1) for _ in range(K + 1)]
    
    # 1. Нулевой слой (t = 0)
    for j in range(N + 1):
        u[0][j] = WaveVariant1.psi1(x[j])
        
    # 2. Первый слой (t = tau) - аппроксимация начальной скорости
    for j in range(1, N):
        if ic2_approx == '1st':
            # Первый порядок O(tau)
            u[1][j] = WaveVariant1.psi1(x[j]) + tau * WaveVariant1.psi2(x[j])
        elif ic2_approx == '2nd':
            # Второй порядок O(tau^2) через ряд Тейлора
            u[1][j] = (WaveVariant1.psi1(x[j]) + 
                       tau * WaveVariant1.psi2(x[j]) + 
                       (WaveVariant1.a**2 * WaveVariant1.psi1_second_deriv(x[j])) * (tau**2 / 2.0))
                       
    u[1][0] = WaveVariant1.phi0(tau)
    u[1][N] = WaveVariant1.phil(tau)
    
    sigma = (WaveVariant1.a * tau / h) ** 2
    
    if scheme == 'explicit':
        if sigma > 1.0:
            print(f"[Внимание] Явная схема неустойчива: число Куранта = {sigma:.3f} > 1.0")
            
        for k in range(1, K):
            for j in range(1, N):
                u[k+1][j] = 2.0 * u[k][j] - u[k-1][j] + sigma * (u[k][j+1] - 2.0 * u[k][j] + u[k][j-1])
            # Граничные условия 1-го рода задаются точно
            u[k+1][0] = WaveVariant1.phi0(t[k+1])
            u[k+1][N] = WaveVariant1.phil(t[k+1])
            
    elif scheme == 'implicit':
        for k in range(1, K):
            a_mat = [0.0] * (N + 1)
            b_mat = [0.0] * (N + 1)
            c_mat = [0.0] * (N + 1)
            d_mat = [0.0] * (N + 1)
            
            for j in range(1, N):
                a_mat[j] = -sigma
                b_mat[j] = 1.0 + 2.0 * sigma
                c_mat[j] = -sigma
                d_mat[j] = 2.0 * u[k][j] - u[k-1][j]
                
            b_mat[0] = 1.0; c_mat[0] = 0.0; d_mat[0] = WaveVariant1.phi0(t[k+1])
            a_mat[N] = 0.0; b_mat[N] = 1.0; d_mat[N] = WaveVariant1.phil(t[k+1])
            
            # Метод прогонки
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
    N, K = 30, 300
    h = WaveVariant1.l / N
    tau = WaveVariant1.T / K
    sigma = (WaveVariant1.a * tau / h) ** 2
    
    print(f"Параметры сетки: N = {N}, K = {K}, h = {h:.4f}, tau = {tau:.5f}")
    print(f"Критерий устойчивости (a^2 * tau^2 / h^2) = {sigma:.4f}\n")
    
    configs = [
        ('explicit', '1st', 'Явная схема (Крест) + 1-й порядок НУ'),
        ('explicit', '2nd', 'Явная схема (Крест) + 2-й порядок НУ'),
        ('implicit', '1st', 'Неявная схема + 1-й порядок НУ'),
        ('implicit', '2nd', 'Неявная схема + 2-й порядок НУ')
    ]
    
    print(f"{'Конфигурация':<45} | {'Макс. погрешность ||U - u|| при t=T':<30}")
    print("-" * 80)
    
    results = {}
    for sch, ic2, name in configs:
        x, t, u_num = solve_wave(N, K, scheme=sch, ic2_approx=ic2)
        err = max(abs(u_num[-1][j] - WaveVariant1.exact(x[j], t[-1])) for j in range(N + 1))
        results[name] = (u_num, err)
        print(f"{name:<45} | {err:.6e}")
    print("-" * 80)
    
    # Построение графиков
    u_ex_final = [WaveVariant1.exact(xj, t[-1]) for xj in x]
    
    plt.figure(figsize=(14, 5))
    
    # График 1: Профиль волны при t = T
    plt.subplot(1, 2, 1)
    plt.plot(x, u_ex_final, 'k-', linewidth=2, label='Точное решение')
    
    u_exp_2nd = results['Явная схема (Крест) + 2-й порядок НУ'][0]
    u_imp_2nd = results['Неявная схема + 2-й порядок НУ'][0]
    
    plt.plot(x, u_exp_2nd[-1], 'ro--', markersize=4, label='Явная схема (2-й пор. НУ)')
    plt.plot(x, u_imp_2nd[-1], 'bs--', markersize=4, label='Неявная схема (2-й пор. НУ)')
    plt.title(f'Профиль решения u(x, T) при T={WaveVariant1.T}')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.grid(True)
    plt.legend()
    
    # График 2: Динамика погрешности
    plt.subplot(1, 2, 2)
    
    for name in configs:
        sch, ic2, label = name
        u_num = results[label][0]
        err_t = [max(abs(u_num[k][j] - WaveVariant1.exact(x[j], t[k])) for j in range(N + 1)) for k in range(K + 1)]
        
        style = '-' if sch == 'explicit' else '--'
        plt.plot(t, err_t, style, label=label)
        
    plt.title('Максимальная погрешность по времени')
    plt.xlabel('t')
    plt.ylabel('||U - u||')
    plt.grid(True)
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()