import math
import matplotlib.pyplot as plt

class Variant1:
    l = 1.0
    T = 0.05
    a = 0.02  # Коэффициент a > 0. Выбран 0.02 для наглядности.
    
    @staticmethod
    def exact(x, t):
        # Аналитическое решение: U(x,t) = exp(-4*pi^2*a*t)*sin(2*pi*x)
        return math.exp(-4 * math.pi**2 * Variant1.a * t) * math.sin(2 * math.pi * x)

def solve_variant1(N, K, scheme='implicit'):
    h = Variant1.l / N
    tau = Variant1.T / K
    x = [j * h for j in range(N + 1)]
    t = [k * tau for k in range(K + 1)]
    
    u = [[0.0] * (N + 1) for _ in range(K + 1)]
    
    # Начальное условие u(x, 0) = sin(2*pi*x)
    for j in range(N + 1):
        u[0][j] = math.sin(2 * math.pi * x[j])
        
    sigma = Variant1.a * tau / (h ** 2)
    
    if scheme == 'explicit':
        if sigma > 0.5:
            print(f"[Внимание] Явная схема неустойчива: sigma = {sigma:.3f} > 0.5")
            
        for k in range(K):
            for j in range(1, N):
                u[k + 1][j] = sigma * u[k][j + 1] + (1 - 2 * sigma) * u[k][j] + sigma * u[k][j - 1]
            # Граничные условия 1-го рода (u = 0)
            u[k + 1][0] = 0.0
            u[k + 1][N] = 0.0
            
    else:
        # Неявная схема (theta = 1.0) или Кранка-Николсона (theta = 0.5)
        theta = 1.0 if scheme == 'implicit' else 0.5
        
        for k in range(K):
            a_mat = [0.0] * (N + 1)
            b_mat = [0.0] * (N + 1)
            c_mat = [0.0] * (N + 1)
            d_mat = [0.0] * (N + 1)
            
            for j in range(1, N):
                a_mat[j] = -theta * Variant1.a / (h ** 2)
                b_mat[j] = (1.0 / tau) + theta * (2.0 * Variant1.a / (h ** 2))
                c_mat[j] = -theta * Variant1.a / (h ** 2)
                
                explicit_part = (1.0 - theta) * Variant1.a * (u[k][j + 1] - 2.0 * u[k][j] + u[k][j - 1]) / (h ** 2)
                d_mat[j] = (1.0 / tau) * u[k][j] + explicit_part
            
            # Условия на границах (строго 0)
            b_mat[0] = 1.0; c_mat[0] = 0.0; d_mat[0] = 0.0
            a_mat[N] = 0.0; b_mat[N] = 1.0; d_mat[N] = 0.0
            
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
            
            u[k + 1][N] = B[N]
            for j in range(N - 1, -1, -1):
                u[k + 1][j] = A[j] * u[k + 1][j + 1] + B[j]
                
    return x, t, u

if __name__ == '__main__':
    N, K = 20, 200
    
    x, t, u_exp = solve_variant1(N, K, scheme='explicit')
    _, _, u_imp = solve_variant1(N, K, scheme='implicit')
    _, _, u_cn = solve_variant1(N, K, scheme='cn')
    
    # Расчет максимальной погрешности на каждом временном слое
    err_exp = [max(abs(u_exp[k][j] - Variant1.exact(x[j], t[k])) for j in range(N + 1)) for k in range(K + 1)]
    err_imp = [max(abs(u_imp[k][j] - Variant1.exact(x[j], t[k])) for j in range(N + 1)) for k in range(K + 1)]
    err_cn  = [max(abs(u_cn[k][j]  - Variant1.exact(x[j], t[k])) for j in range(N + 1)) for k in range(K + 1)]
    
    # --- ВЫВОД В КОНСОЛЬ ---
    h = Variant1.l / N
    tau = Variant1.T / K
    sigma = Variant1.a * tau / (h ** 2)
    
    print(f"Параметры сетки: N = {N}, K = {K}, h = {h:.4f}, tau = {tau:.5f}")
    print(f"Параметр устойчивости (sigma) = {sigma:.4f}\n")
    
    print(f"{'Схема':<20} | {'Погрешность ||U - u|| при t=T':<30}")
    print("-" * 55)
    print(f"{'Явная':<20} | {err_exp[-1]:.6e}")
    print(f"{'Неявная':<20} | {err_imp[-1]:.6e}")
    print(f"{'Кранка-Николсона':<20} | {err_cn[-1]:.6e}")
    print("-" * 55)
    
    # --- ПОСТРОЕНИЕ ГРАФИКОВ ---
    u_ex_final = [Variant1.exact(xj, t[-1]) for xj in x]
    
    plt.figure(figsize=(12, 5))
    
    # График 1: Профиль температуры
    plt.subplot(1, 2, 1)
    plt.plot(x, u_ex_final, 'k-', linewidth=2, label='Точное решение')
    plt.plot(x, u_exp[-1], 'ro--', markersize=4, label='Явная схема')
    plt.plot(x, u_imp[-1], 'bs--', markersize=4, label='Неявная схема')
    plt.plot(x, u_cn[-1], 'g^--', markersize=4, label='Кранка-Николсона')
    plt.title(f'Профиль решения u(x, T) при T={Variant1.T}')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.grid(True)
    plt.legend()
    
    # График 2: Погрешность по времени
    plt.subplot(1, 2, 2)
    plt.plot(t, err_exp, 'r-', label='Явная схема')
    plt.plot(t, err_imp, 'b-', label='Неявная схема')
    plt.plot(t, err_cn, 'g-', label='Кранка-Николсона')
    plt.title('Максимальная погрешность по времени')
    plt.xlabel('t')
    plt.ylabel('||U - u||')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()