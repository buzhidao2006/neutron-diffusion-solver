#!/usr/bin/env python3
"""
neutron-diffusion-solver — 统一命令行入口

用法:
  python3 main.py 1d              # 一维单群扩散
  python3 main.py 2g              # 一维双群扩散
  python3 main.py critical-scan   # 临界尺寸扫描
  python3 main.py boron-search    # 临界硼浓度搜索
  python3 main.py 2d              # 二维双群扩散
  python3 main.py 3d              # 三维双群扩散
  python3 main.py kinetics        # 点堆动力学演示
  python3 main.py test            # 运行测试套件
  python3 main.py demo            # 运行所有演示
"""

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def cmd_1d(args):
    """一维单群扩散求解。"""
    from diffusion_1d import __doc__ as _  # noqa — 直接 import 就会执行
    print("运行 diffusion_1d.py ...")
    exec(open(PROJECT_DIR / "diffusion_1d.py").read())


def cmd_2g(args):
    """一维双群扩散求解。"""
    from solver import solve_two_group, DEFAULTS

    print("=" * 50)
    print("  一维双群扩散求解")
    print("=" * 50)
    result = solve_two_group(
        L=args.L, N=args.N,
        sections={'D1': args.D1, 'D2': args.D2, 'nu_Sf1': args.nu_Sf1,
                  'nu_Sf2': args.nu_Sf2, 'Sa1': args.Sa1, 'Sa2': args.Sa2,
                  'Ss12': args.Ss12},
    )
    print(f"\n  k_eff = {result['k_eff']:.6f}")
    status = "超临界" if result['k_eff'] > 1.001 else (
        "次临界" if result['k_eff'] < 0.999 else "临界")
    print(f"  状态: {status}")
    print(f"  快群通量峰值: {result['phi1'].max():.4f}")
    print(f"  热群通量峰值: {result['phi2'].max():.4f}")


def cmd_critical_scan(args):
    """临界尺寸扫描。"""
    from solver import scan_critical_size

    print("=" * 50)
    print("  临界尺寸扫描")
    print("=" * 50)
    cs = scan_critical_size(
        L_min=args.L_min, L_max=args.L_max,
        n_points=args.n_pts, N=args.N,
    )
    print(f"\n  k_inf (无穷大介质) = {cs['analytic']['k_inf']:.4f}")
    print(f"  徙动面积 M²       = {cs['analytic']['M2']:.1f} cm²")
    print(f"  临界尺寸 L_crit    = {cs['L_crit']:.0f} cm")
    print(f"  k(L_crit)          = {cs['k_crit']:.6f}")


def cmd_boron_search(args):
    """临界硼浓度搜索。"""
    from solver import search_critical_boron

    print("=" * 50)
    print("  临界硼浓度搜索")
    print("=" * 50)
    cb = search_critical_boron(
        L=args.L, N=args.N,
        alpha=args.alpha,
    )
    print(f"\n  临界硼浓度 C_B* = {cb['C_crit']:.1f} ppm")
    print(f"  硼微分价值       = {cb['boron_worth']:.1f} pcm/ppm")
    print(f"  k(C_B*)          = {cb['k_final']:.8f}")


def cmd_2d(args):
    """二维双群扩散求解。"""
    from solver_2d import solve_two_group_2d

    print("=" * 50)
    print("  二维双群扩散求解")
    print("=" * 50)
    Nx_val = args.Nx or args.N
    Ny_val = args.Ny or args.N
    print(f"  网格: {Nx_val}×{Ny_val} ({2*Nx_val*Ny_val} 未知数)")

    result = solve_two_group_2d(
        Lx=args.Lx, Ly=args.Ly or args.Lx,
        Nx=Nx_val, Ny=Ny_val, method='chebyshev',
    )
    print(f"\n  k_eff     = {result['k_eff']:.6f}")
    print(f"  迭代次数  = {result['n_iter']}")
    print(f"  快群峰值  = {result['phi1'].max():.4f}")
    print(f"  热群峰值  = {result['phi2'].max():.4f}")


def cmd_3d(args):
    """三维双群扩散求解。"""
    from solver_3d import solve_two_group_3d

    print("=" * 50)
    print("  三维双群扩散求解")
    print("=" * 50)
    N_val = args.N
    print(f"  网格: {N_val}³ ({2*N_val**3} 未知数)")

    result = solve_two_group_3d(
        Lx=args.L, Ly=args.L, Lz=args.L,
        Nx=N_val, Ny=N_val, Nz=N_val, method='chebyshev',
    )
    print(f"\n  k_eff     = {result['k_eff']:.6f}")
    print(f"  迭代次数  = {result['n_iter']}")


def cmd_kinetics(args):
    """点堆动力学演示。"""
    print("=" * 50)
    print("  点堆动力学演示")
    print("=" * 50)
    # 直接运行 demo 脚本
    exec(open(PROJECT_DIR / "demo_kinetics.py").read())


def cmd_test(args):
    """运行 pytest 测试套件。"""
    print("=" * 50)
    print("  运行测试套件")
    print("=" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PROJECT_DIR / "tests"), "-v", "--tb=short"],
        cwd=str(PROJECT_DIR),
    )
    sys.exit(result.returncode)


def cmd_demo(args):
    """运行所有演示脚本。"""
    demos = [
        ("一维单群扩散", "diffusion_1d.py"),
        ("一维双群扩散", "diffusion_2g.py"),
        ("临界尺寸扫描", "critical_size.py"),
        ("临界硼搜索", "boron_search.py"),
        ("二维扩散", "diffusion_2d.py"),
        ("三维扩散", "diffusion_3d.py"),
        ("点堆动力学", "demo_kinetics.py"),
    ]
    for name, script in demos:
        print(f"\n{'='*50}")
        print(f"  {name} ({script})")
        print(f"{'='*50}")
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / script)],
            cwd=str(PROJECT_DIR),
        )
        if result.returncode != 0:
            print(f"  ⚠️  {name} 退出码: {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="neutron-diffusion-solver — 反应堆物理数值计算教学工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 main.py 1d
  python3 main.py 2g --L 300 --N 200
  python3 main.py critical-scan --L-min 40 --L-max 400
  python3 main.py 2d --Lx 200 --Nx 60
  python3 main.py 3d --L 160 --N 20
  python3 main.py test
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 1d
    p = subparsers.add_parser("1d", help="一维单群扩散")
    p.set_defaults(func=cmd_1d)

    # 2g
    p = subparsers.add_parser("2g", help="一维双群扩散")
    p.add_argument("--L", type=float, default=200.0, help="平板半厚度 cm")
    p.add_argument("--N", type=int, default=150, help="网格点数")
    p.add_argument("--D1", type=float, default=1.2)
    p.add_argument("--D2", type=float, default=0.4)
    p.add_argument("--nu-Sf1", type=float, default=0.003)
    p.add_argument("--nu-Sf2", type=float, default=0.105)
    p.add_argument("--Sa1", type=float, default=0.008)
    p.add_argument("--Sa2", type=float, default=0.08)
    p.add_argument("--Ss12", type=float, default=0.020)
    p.set_defaults(func=cmd_2g)

    # critical-scan
    p = subparsers.add_parser("critical-scan", help="临界尺寸扫描")
    p.add_argument("--L-min", type=float, default=40.0)
    p.add_argument("--L-max", type=float, default=400.0)
    p.add_argument("--n-pts", type=int, default=30)
    p.add_argument("--N", type=int, default=100)
    p.set_defaults(func=cmd_critical_scan)

    # boron-search
    p = subparsers.add_parser("boron-search", help="临界硼浓度搜索")
    p.add_argument("--L", type=float, default=200.0)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.0e-5, help="硼灵敏度 cm^-1/ppm")
    p.set_defaults(func=cmd_boron_search)

    # 2d
    p = subparsers.add_parser("2d", help="二维双群扩散")
    p.add_argument("--Lx", type=float, default=200.0)
    p.add_argument("--Ly", type=float, default=None)
    p.add_argument("--Nx", type=int, default=None)
    p.add_argument("--Ny", type=int, default=None)
    p.add_argument("--N", type=int, default=60, help="默认网格 (Nx=Ny=N)")
    p.set_defaults(func=cmd_2d)

    # 3d
    p = subparsers.add_parser("3d", help="三维双群扩散")
    p.add_argument("--L", type=float, default=160.0, help="立方体边长 cm")
    p.add_argument("--N", type=int, default=20, help="每方向网格点数")
    p.set_defaults(func=cmd_3d)

    # kinetics
    p = subparsers.add_parser("kinetics", help="点堆动力学演示")
    p.set_defaults(func=cmd_kinetics)

    # test
    p = subparsers.add_parser("test", help="运行测试套件")
    p.set_defaults(func=cmd_test)

    # demo
    p = subparsers.add_parser("demo", help="运行所有演示")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
