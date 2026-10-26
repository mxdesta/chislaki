import math
import matplotlib.pyplot as plt

class Variant3:
    l = math.pi
    T = 1.0
    a = 1.0  # Параметр a > 0
    
    @staticmethod
    def exact(x, t):
        return math.exp(-Variant3.a * t) * math.cos(x)
        
    @staticmethod
    def psi(x):
        return math.cos(x)
        
    @staticmethod
    def phi0(t):
        return math.exp(-Variant3.a * t)
        
    @staticmethod
    def phil(t):
        return -math.exp(-Variant3.a * t)

def solve_parabolic_v3(N, K, scheme='implicit', bc_approx='1st_2pt'):
    h = Variant3.l / N
    tau = Variant3.T / K
    x = [j * h for j in range(N + 1)]
    t = [k * tau for k in range(K + 1)]
    
    u = [[0.0] * (N + 1) for _ in range(K + 1)]
    
    for j in range(N + 1):
        u[0][j] = Variant3.psi(x[j])
        
    sigma = Variant3.a * tau / (h ** 2)
    
    if scheme == 'explicit':
        for k in range(K):
            for j in range(1, N):
                u[k+1][j] = sigma * u[k][j+1] + (1 - 2*sigma) * u[k][j] + sigma * u[k][j-1]
            # ГУ 1-го рода - аппроксимация не требуется, задаем точно для всех "методов"
            u[k+1][0] = Variant3.phi0(t[k+1])
            u[k+1][N] = Variant3.phil(t[k+1])
            
    else:
        theta = 1.0 if scheme == 'implicit' else 0.5
        for k in range(K):
            a_mat = [0.0] * (N + 1)
            b_mat = [0.0] * (N + 1)
            c_mat = [0.0] * (N + 1)
            d_mat = [0.0] * (N + 1)
            
            for j in range(1, N):
                a_mat[j] = -theta * Variant3.a / (h ** 2)
                b_mat[j] = (1.0 / tau) + theta * (2.0 * Variant3.a / (h ** 2))
                c_mat[j] = -theta * Variant3.a / (h ** 2)
                
                explicit_part = (1.0 - theta) * Variant3.a * (u[k][j+1] - 2.0*u[k][j] + u[k][j-1]) / (h ** 2)
                d_mat[j] = (1.0 / tau) * u[k][j] + explicit_part
            
            b_mat[0] = 1.0; c_mat[0] = 0.0; d_mat[0] = Variant3.phi0(t[k+1])
            a_mat[N] = 0.0; b_mat[N] = 1.0; d_mat[N] = Variant3.phil(t[k+1])
            
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
    N, K = 20, 200
    h, tau = Variant3.l / N, Variant3.T / K
    
    schemes = [('explicit', 'Явная'), ('implicit', 'Неявная'), ('cn', 'Кранк-Николсон')]
    bcs = [('1st_2pt', '2-точ. O(h)'), ('2nd_3pt', '3-точ. O(h^2)'), ('2nd_2pt', '2-точ. O(h^2)')]
    
    print(f"Сетка: N={N}, K={K}, sigma={Variant3.a * tau / h**2:.4f}\n")
    print(f"{'Схема':<15} | {'Аппроксимация ГУ':<15} | {'Макс. погрешность'}")
    print("-" * 55)
    
    results = {}
    for sch_code, sch_name in schemes:
        for bc_code, bc_name in bcs:
            x, t, u_num = solve_parabolic_v3(N, K, scheme=sch_code, bc_approx=bc_code)
            err = max(abs(u_num[-1][j] - Variant3.exact(x[j], t[-1])) for j in range(N+1))
            results[(sch_code, bc_code)] = (u_num, err)
            print(f"{sch_name:<15} | {bc_name:<15} | {err:.6e}")
    print("-" * 55)

    # --- Построение графиков ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Профили решений и динамика погрешности (все 3 схемы)', fontsize=14)
    
    for i, (sch_code, sch_name) in enumerate(schemes):
        ax_sol, ax_err = axes[i, 0], axes[i, 1]
        
        # Точное решение для профиля
        u_ex_final = [Variant3.exact(xj, t[-1]) for xj in x]
        ax_sol.plot(x, u_ex_final, 'k-', linewidth=2, label='Точное')
        
        for bc_code, bc_name in bcs:
            u_num, _ = results[(sch_code, bc_code)]
            ax_sol.plot(x, u_num[-1], '--', label=f'Числ. ({bc_name})')
            
            err_t = [max(abs(u_num[k][j] - Variant3.exact(x[j], t[k])) for j in range(N+1)) for k in range(K+1)]
            ax_err.plot(t, err_t, label=f'{bc_name}')
            
        ax_sol.set_title(f'{sch_name}: Решение при t={Variant3.T}')
        ax_sol.grid(True); ax_sol.legend()
        
        ax_err.set_title(f'{sch_name}: Макс. погрешность ||U-u||')
        ax_err.grid(True); ax_err.legend()
        
    plt.tight_layout()
    plt.show()

    # --- Исследование зависимости погрешности от h и tau ---
    N_list = [10, 20, 40, 80]
    K_list = [100, 200, 400, 800]
    
    errs_h = []
    for cur_N in N_list:
        cur_K = 10000 # Фиксируем малый tau, чтобы выделить ошибку по пространству
        x_tmp, t_tmp, u_tmp = solve_parabolic_v3(cur_N, cur_K, scheme='implicit')
        errs_h.append(max(abs(u_tmp[-1][j] - Variant3.exact(x_tmp[j], t_tmp[-1])) for j in range(cur_N+1)))
        
    errs_tau = []
    for cur_K in K_list:
        cur_N = 100 # Фиксируем малый h, чтобы выделить ошибку по времени
        x_tmp, t_tmp, u_tmp = solve_parabolic_v3(cur_N, cur_K, scheme='implicit')
        errs_tau.append(max(abs(u_tmp[-1][j] - Variant3.exact(x_tmp[j], t_tmp[-1])) for j in range(cur_N+1)))
        
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog([Variant3.l/n for n in N_list], errs_h, 'o-')
    ax1.set_title('Зависимость погрешности от шага h')
    ax1.set_xlabel('h'); ax1.set_ylabel('Погрешность'); ax1.grid(True)
    
    ax2.loglog([Variant3.T/k for k in K_list], errs_tau, 's-', color='orange')
    ax2.set_title('Зависимость погрешности от шага tau')
    ax2.set_xlabel('tau'); ax2.grid(True)
    
    plt.tight_layout()
    plt.show()