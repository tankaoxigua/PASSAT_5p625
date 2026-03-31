import torch
from .base import SpaceOperator

# 严格球面有限体积法
class FiniteVolumeOperator(SpaceOperator):
    # d(q * A) / dt = - Σ(face flux) + source

    def __init__(self, mesh, dx=5.625):
        super().__init__()
        self.mesh = mesh
        self.dx_deg = dx
        self.dx_rad = torch.deg2rad(torch.tensor(dx, dtype=torch.float32))

        # 纬度（弧度）
        self.lat = mesh[0].to(torch.float32)
        self.cos_lat = torch.clamp(torch.cos(self.lat), min=1e-6)
        self.sin_lat = torch.sin(self.lat)
        self.tan_lat = torch.tan(self.lat)

        # 地球半径
        self.R = 6731e3 

        #  控制体面积 A = R^2 cos(lat) Δθ Δφ
        self.cell_area = (
            self.R ** 2
            * self.cos_lat
            * self.dx_rad
            * self.dx_rad
        ).view(1, 1, 32, 64)

        # 南北向面（纬向面）
        self.face_lat = self.R * self.dx_rad

        # 东西向面（经向面，依赖 cos(lat)）
        self.face_lon = (
            self.R * self.cos_lat * self.dx_rad
        ).view(1, 1, 32, 64)

    # 通量计算
    def upwind_flux(self, q, u, direction):
        if direction == "lat":
            # j+1/2 面
            q_up = torch.where(u > 0, q, torch.roll(q, -1, dims=2))
            return u * q_up * self.face_lat

        elif direction == "lon":
            # i+1/2 面
            q_up = torch.where(u > 0, q, torch.roll(q, -1, dims=3))
            return u * q_up * self.face_lon

        else:
            raise ValueError("direction must be 'lat' or 'lon'")
        
    # 通量散度
    def divergence(self, F_lat, F_lon):
        # ∇·F = (F_N - F_S + F_E - F_W) / cell_area
        dF_lat = F_lat - torch.roll(F_lat, 1, dims=2)
        dF_lon = F_lon - torch.roll(F_lon, 1, dims=3)
        return (dF_lat + dF_lon) / self.cell_area

    # 状态变量 RHS
    def StateRhs(self, dataStates, velocityCoef):
        """
        ∂(qA)/∂t = -∑F + A·S
        ⇒ ∂q/∂t = -(1/A)∑F + S
        """

        B = dataStates.shape[0]

        # reshape: (B, 32*64, 5) → (B, 5, 32, 64)
        q = dataStates.view(B, 32, 64, 5).permute(0, 3, 1, 2)

        # 速度拆分
        u_lat = velocityCoef[:, :5]
        u_lon = velocityCoef[:, 5:]

        # 面通量
        F_lat = self.upwind_flux(q, u_lat, "lat")
        F_lon = self.upwind_flux(q, u_lon, "lon")

        # 严格有限体积散度
        dqdt = -self.divergence(F_lat, F_lon)

        return dqdt

    # 球面梯度
    def gradient_lat_lon(self, q):
        dq_dlat = (
            torch.roll(q, -1, 2) - torch.roll(q, 1, 2)
        ) / (2 * self.R * self.dx_rad)

        dq_dlon = (
            torch.roll(q, -1, 3) - torch.roll(q, 1, 3)
        ) / (2 * self.R * self.dx_rad * self.cos_lat)

        return dq_dlat, dq_dlon

    # 速度场 RHS
    def VelocityRhs(self, velocityCoef, mesh, lat_presPartial, rectify_lon_presPartial):

        ρ = 1.225
        ω = 0.2618
        μ = 1e-4

        u_lat = velocityCoef[:, :5]
        u_lon = velocityCoef[:, 5:]

        # 梯度
        du_lat_dlat, du_lat_dlon = self.gradient_lat_lon(u_lat)
        du_lon_dlat, du_lon_dlon = self.gradient_lat_lon(u_lon)

        # 对流
        adv_lat = -(u_lat * du_lat_dlat + u_lon * du_lat_dlon)
        adv_lon = -(u_lat * du_lon_dlat + u_lon * du_lon_dlon)

        # 曲率
        curv_lat = -(u_lon ** 2) * self.tan_lat
        curv_lon = +(u_lat * u_lon) * self.tan_lat

        # 压力梯度（使用第 3 个变量）
        p = velocityCoef[:, 2:3].repeat(1, 5, 1, 1)
        dp_dlat, dp_dlon = self.gradient_lat_lon(p)

        pres_lat = -1e-3 * dp_dlat / ρ
        pres_lon = -1e-3 * dp_dlon / ρ

        # 科里奥利
        cor_lat = -2 * ω * u_lon * self.sin_lat
        cor_lon = +2 * ω * u_lat * self.sin_lat

        # 粘性耗散
        diss_lat = -μ * u_lat / (self.cos_lat ** 2)
        diss_lon = -μ * u_lon / (self.cos_lat ** 2)

        du_lat_dt = adv_lat + curv_lat + pres_lat + cor_lat + diss_lat
        du_lon_dt = adv_lon + curv_lon + pres_lon + cor_lon + diss_lon

        return torch.cat([du_lat_dt, du_lon_dt], dim=1)
