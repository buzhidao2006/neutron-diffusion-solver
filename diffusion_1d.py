"""
一维单群中子扩散方程求解器
"""
import numpy as np
import matplotlib.pyplot as plt

L = 200.0
N = 100
h = L / N
D = 1.0
nu_Sigma_f = 0.01
Sigma_a = 0.01

coeff = D / (h * h)

main_diag = 2 * coeff + Sigma_a * np.ones(N)
off_diag  = -coeff * np.ones(N - 1)

A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
F = nu_Sigma_f * np.eye(N)

phi = np.ones(N)
k_old = 1.0

for _ in range(200):
    source = F @ phi
    phi_new = np.linalg.solve(A, source)
    k_new = k_old * np.sum(phi_new) / np.sum(phi)
    phi = phi_new / np.max(phi_new)
    k_old = k_new

print(f"k_eff = {k_new:.5f}")

x = np.linspace(h/2, L - h/2, N)
plt.figure(figsize=(8, 4))
plt.plot(x, phi, 'b-', linewidth=2)
plt.xlabel('Position (cm)')
plt.ylabel('Neutron Flux (normalized)')
plt.title(f'1D Neutron Diffusion, k_eff = {k_new:.5f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('flux.png', dpi=150)
print("flux saved to flux.png")
plt.show()
