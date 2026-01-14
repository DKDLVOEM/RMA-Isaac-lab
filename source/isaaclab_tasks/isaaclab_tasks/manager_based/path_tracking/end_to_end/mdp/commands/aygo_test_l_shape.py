import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms


# -----------------------------
# 1-1) 원래 arc path (부드러운 코너)
# -----------------------------

def generate_arc_waypoints(
    num_waypoints: int = 10,
    std_waypoint_interval: float = 0.15,
    spline_angle_deg: float = 90.0,
):
    """
    단순화된 '원호 path' 생성.
    - 총 길이 ≈ (num_waypoints - 1) * std_waypoint_interval
    - 최종 yaw = spline_angle_deg (deg)
    - 반환: [N, 3] = (x, y, yaw_orig)
    """
    N = num_waypoints
    L = std_waypoint_interval * (N - 1)  # 총 길이
    psi_goal = np.deg2rad(spline_angle_deg)

    # 너무 작은 각 피하기
    if abs(psi_goal) < np.deg2rad(5.0):
        psi_goal = np.sign(psi_goal) * np.deg2rad(5.0) if psi_goal != 0.0 else np.deg2rad(5.0)

    # 원호: R = L / psi_goal
    R = L / psi_goal

    # 0 → psi_goal 까지 균일 분배
    angles = np.linspace(0.0, psi_goal, N)

    # 원호 상의 좌표 (앞 +x, 좌 +y 가정)
    x = R * np.sin(angles)
    y = R * (1.0 - np.cos(angles))

    # 원래 yaw는 path의 접선 방향이라 가정
    yaw = angles.copy()

    waypoints = np.stack([x, y, yaw], axis=-1)
    return waypoints


# -----------------------------
# 1-2) 직각 L자 path 생성
# -----------------------------

def generate_right_angle_waypoints(
    num_waypoints: int = 10,
    std_waypoint_interval: float = 0.15,
    turn_yaw_deg: float = 90.0,
    corner_index: int | None = None,
):
    """
    직각(ㄱ자) path 생성:
    - 시작: x 방향 직진
    - 어떤 index에서 90도 꺾어서 y 방향으로 진행 (L자)
    - yaw는 코너 전까지 0, 이후부터 turn_yaw_deg (기본 90도)

    반환: [N, 3] = (x, y, yaw_orig)
    """
    N = num_waypoints
    if N < 2:
        raise ValueError("num_waypoints must be >= 2")

    if corner_index is None:
        # 중간쯤에서 꺾이게
        corner_index = N // 2

    corner_index = max(1, min(corner_index, N - 1))

    interval = std_waypoint_interval

    x = np.zeros(N)
    y = np.zeros(N)
    yaw = np.zeros(N)

    turn_yaw = np.deg2rad(turn_yaw_deg)

    # 코너 전: x 방향 직진, yaw = 0
    for k in range(corner_index):
        x[k] = interval * k
        y[k] = 0.0
        yaw[k] = 0.0

    # 코너 이후: y 방향 직진, yaw = turn_yaw
    x_corner = interval * (corner_index - 1)
    for k in range(corner_index, N):
        x[k] = x_corner
        y[k] = interval * (k - corner_index + 1)
        yaw[k] = turn_yaw

    waypoints = np.stack([x, y, yaw], axis=-1)
    return waypoints


# -----------------------------
# 2) AYRO 2nd-order yaw shaping
# -----------------------------

def ayro_yaw_profile(
    psi_orig: np.ndarray,
    g: float,
    zeta_min: float = 0.35,
    zeta_max: float = 1.0,
    omega_min: float = 3.0,
    omega_max: float = 8.0,
    blend: bool = False,
):
    """
    IsaacLab `_update_command` 내부 로직과 같은 구조 (1D numpy 버전).

    - psi_orig: 원래 yaw 프로파일 [N]
    - g in [0, 1]
    - 2차 시스템:
      psi'' + 2 zeta omega_n psi' + omega_n^2 (psi - psi_goal) = 0
    - pseudo-time s ∈ [0, 1], ds = 1/(N-1)

    blend=False: psi_cmd = psi_2nd
    blend=True:  psi_cmd = (1-g)*psi_orig + g*psi_2nd
    """
    psi_orig = np.asarray(psi_orig)
    N = psi_orig.shape[0]
    assert N >= 2

    psi_goal = psi_orig[-1]

    # zeta(g), omega_n(g)
    zeta = zeta_max - g * (zeta_max - zeta_min)
    omega_n = omega_min + g * (omega_max - omega_min)

    N_eff = max(N - 1, 1)
    ds = 1.0 / float(N_eff)

    psi = 0.0
    psi_dot = 0.0
    psi_2nd = np.zeros_like(psi_orig)

    for k in range(N):
        psi_2nd[k] = psi
        psi_ddot = -2.0 * zeta * omega_n * psi_dot - (omega_n ** 2) * (psi - psi_goal)
        psi_dot = psi_dot + psi_ddot * ds
        psi = psi + psi_dot * ds

    if blend:
        psi_cmd = (1.0 - g) * psi_orig + g * psi_2nd
    else:
        psi_cmd = psi_2nd

    return psi_cmd


# -----------------------------
# 3) 시각화: 직사각형으로 yaw 표시
# -----------------------------

def draw_pose(ax, x, y, yaw, size=0.08, color="k", alpha=0.8):
    rect = Rectangle(
        (-size * 0.5, -size * 0.25),
        size,
        size * 0.5,
        fill=False,
        edgecolor=color,
        linewidth=1.0,
        alpha=alpha,
    )
    t = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def plot_paths(base_wp, yaw_profiles, title="AYRO yaw-only shaping"):
    """
    base_wp: [N, 3] (x, y, yaw_orig)
    yaw_profiles: dict[label -> yaw_cmd[N]]
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    x = base_wp[:, 0]
    y = base_wp[:, 1]
    yaw_orig = base_wp[:, 2]

    # 원래 path (position + yaw 박스)
    ax.plot(x, y, "k-o", label="original path", linewidth=1.5, markersize=4)
    for k in range(0, len(x), 2):
        draw_pose(ax, x[k], y[k], yaw_orig[k], color="k", alpha=0.7)

    # 각 g에 대해 yaw만 변경 (x,y는 그대로)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (label, psi_cmd) in enumerate(yaw_profiles.items()):
        c = colors[i % len(colors)]
        # x,y는 base_wp 그대로 사용 → 위치 동일, heading만 다름
        ax.plot(x, y, "-", color=c, linewidth=1.2, alpha=0.7, label=label)
        for k in range(0, len(x), 2):
            draw_pose(ax, x[k], y[k], psi_cmd[k], color=c, alpha=0.7)

    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# 4) 메인
# -----------------------------

if __name__ == "__main__":
    num_waypoints = 10
    std_waypoint_interval = 0.15
    spline_angle_deg = 90.0  # 90도 코너 예시

    # "arc" / "right_angle" 중 선택
    path_type = "arc"          # 부드러운 코너
    # path_type = "right_angle"  # 직각 L자 코너

    if path_type == "arc":
        base_wp = generate_arc_waypoints(
            num_waypoints=num_waypoints,
            std_waypoint_interval=std_waypoint_interval,
            spline_angle_deg=spline_angle_deg,
        )
    elif path_type == "right_angle":
        base_wp = generate_right_angle_waypoints(
            num_waypoints=num_waypoints,
            std_waypoint_interval=std_waypoint_interval,
            turn_yaw_deg=spline_angle_deg,
            corner_index=None,  # None이면 가운데쯤에서 꺾임
        )
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    psi_orig = base_wp[:, 2]

    # 여러 g에 대해 yaw warp
    g_list = [0.0, 0.3, 0.7, 1.0]
    yaw_profiles = {}
    for g in g_list:
        psi_cmd = ayro_yaw_profile(
            psi_orig,
            g,
            zeta_min=0.35,
            zeta_max=1.0,
            omega_min=3.0,
            omega_max=8.0,
            blend=False,   # IsaacLab 구현과 맞추려면 False
        )
        yaw_profiles[f"g={g:.1f}"] = psi_cmd

    title = f"{path_type} - AYRO yaw-only shaping (N={num_waypoints}, ds={std_waypoint_interval}, ψ_goal={spline_angle_deg}°)"
    plot_paths(base_wp, yaw_profiles, title=title)
