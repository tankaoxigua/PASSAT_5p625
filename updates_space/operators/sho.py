import torch
import math
from scipy.special import sph_harm
from .base import SpaceOperator

# 球谐函数作为谱基的谱函数
class SpectralSHOperator(SpaceOperator):

    def __init__(self, mesh, lmax=20, radius=6731e3):
        super().__init__()

        lat, lon = mesh                        # (32, 64)
        device = lat.device
        self.H, self.W = lat.shape
        self.N = self.H * self.W
        self.lmax = lmax
        self.R = radius

        # 球坐标
        self.theta = math.pi / 2 - lat.to(device)        # colatitude
        self.phi = lon.to(device)

        # 构造球谐基
        self._build_basis(device)

    # 构造球谐基及其导数
    def _build_basis(self, device):
        Y, dY_dtheta, m_over_cos = [], [], []

        theta = self.theta.detach().cpu().numpy()
        phi = self.phi.detach().cpu().numpy()
        cos_theta = torch.clamp(torch.cos(self.theta), min=1e-6)

        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                # Y_lm
                Ylm = sph_harm(m, l, phi, theta)
                Y.append(torch.tensor(Ylm.real, dtype=torch.float32, device=device))

                # ∂Y/∂θ（中心差分）
                eps = 1e-6
                Yp = sph_harm(m, l, phi, theta + eps)
                Ym = sph_harm(m, l, phi, theta - eps)
                dY = (Yp - Ym).real / (2 * eps)
                dY_dtheta.append(torch.tensor(dY, dtype=torch.float32, device=device))

                # (1/cosθ) ∂/∂φ → i m / cosθ
                m_over_cos.append(
                    torch.tensor(m, dtype=torch.float32, device=device) / cos_theta
                )

        self.Y = torch.stack(Y, dim=0)                  # (M, H, W)
        self.dY_dtheta = torch.stack(dY_dtheta, dim=0)  # (M, H, W)
        self.m_over_cos = torch.stack(m_over_cos, dim=0)  # (M, H, W)

        self.M = self.Y.shape[0]

    # grid → spectral
    def grid_to_spec(self, f):
        # 计算积分权重：sinθ * dθ * dφ（Riemann近似）
        dtheta = self.theta[1, 0] - self.theta[0, 0]
        dphi = self.phi[0, 1] - self.phi[0, 0]
        sin_theta = torch.sin(self.theta)
        weight = sin_theta * dtheta * dphi  # (H,W)
        
        # 加权内积
        return torch.sum(self.Y * f[None, :, :] * weight[None, :, :], dim=(1, 2))

    # spectral → grid
    def spec_to_grid(self, a):
        return torch.sum(a[:, None, None] * self.Y, dim=0)

    # 谱梯度（严格一阶导）
    def spectral_gradient(self, a):
        dtheta = torch.sum(a[:, None, None] * self.dY_dtheta, dim=0)
        dphi = torch.sum(a[:, None, None] * self.m_over_cos * self.Y, dim=0)

        # ∂/∂lat = -∂/∂theta
        dlat = -dtheta / self.R
        dlon = dphi / self.R

        return dlat, dlon

    # 状态变量 RHS
    def StateRhs(self, dataStates, velocityCoef):
        B, C, H, W = dataStates.shape
        rhs = torch.zeros_like(dataStates)

        lat_u = velocityCoef[:, :C]   # (B, C, H, W)
        lon_u = velocityCoef[:, C:]   # (B, C, H, W)

        for b in range(B):
            for c in range(C):
                q = dataStates[b, c]   

                # 谱展开
                a = self.grid_to_spec(q)

                # 谱梯度
                dq_dlat, dq_dlon = self.spectral_gradient(a)

                # 对流项（严格 fd 形式）
                adv = -(lat_u[b, c] * dq_dlat +
                        lon_u[b, c] * dq_dlon)

                # 维度还原: (H,W) → (N,)
                rhs[b, c] = adv

        return rhs

    # 速度 RHS
    def VelocityRhs(self, velocityCoef, mesh, lat_presPartial, rectify_lon_presPartial):
        """
        与 fd.VelocityRhs 对齐
        仅替换梯度来源
        """
        lat, lon = mesh
        sin_lat = torch.sin(lat)
        tan_lat = torch.tan(lat)
        cos_lat = torch.clamp(torch.cos(lat), min=1e-6)

        lat_u = velocityCoef[:, :5]
        lon_u = velocityCoef[:, 5:]

        # 压力梯度项
        lat_pg = 1e-3 * lat_presPartial[:, None]
        lon_pg = 1e-3 * rectify_lon_presPartial[:, None]

        # 曲率项
        lat_curv = lon_u ** 2 * tan_lat
        lon_curv = -lat_u * lon_u * tan_lat

        # 科氏力
        omega = 0.2618
        lat_cor = -2 * omega * lon_u * sin_lat
        lon_cor = 2 * omega * lat_u * sin_lat

        # 拉普拉斯耗散（保持 fd 的工程形式）
        lat_lap = 1e-4 * lat_u / (cos_lat ** 2)
        lon_lap = 1e-4 * lon_u / (cos_lat ** 2)

        lat_rhs = -lat_pg - lat_curv + lat_cor - lat_lap
        lon_rhs = -lon_pg - lon_curv + lon_cor - lon_lap

        return torch.cat([lat_rhs, lon_rhs], dim=1)
