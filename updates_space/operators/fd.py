import torch
from .base import SpaceOperator

# 有限差分法
class FiniteDifferenceOperator(SpaceOperator):

    def __init__(self, mesh, dx=5.625):
        self.mesh = mesh
        self.dx = dx

    # 计算数据梯度
    def gradient_lat_lon(self, dataStates, mesh):
        lat_dataPartial = torch.gradient(dataStates, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64) 
        lon_dataPartial = torch.gradient(dataStates, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
        rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64))
        return lat_dataPartial, rectify_lon_dataPartial

    # 状态计算
    def StateRhs(self, dataStates, velocityCoef):
        # 从 velocityCoef 中拆分出纬向和经向速度            
        lat_velocity, lon_velocity = velocityCoef[:, :5], velocityCoef[:, 5:] 
        lat_dataPartial, rectify_lon_dataPartial = self.gradient_lat_lon(dataStates, self.mesh)
        # 计算对流项
        advection = -(lat_velocity * lat_dataPartial + lon_velocity * rectify_lon_dataPartial)
        return advection

    # 速度场计算
    def VelocityRhs(self, velocityCoef, mesh, lat_presPartial, rectify_lon_presPartial):
        # 压力梯度在速度各分量上的投影
        lat_presPartial = lat_presPartial[:, None, :, :].repeat(1, 5, 1, 1)
        rectify_lon_presPartial = rectify_lon_presPartial[:, None, :, :].repeat(1, 5, 1, 1)

        # 压力梯度项（单位转换）
        latVelocity_pressureGradient = 1e-3 * lat_presPartial # Change the unit into (6731km)^2 * hour^−2; since we have normalize the geopoential, it just need to multiply 1e-3
        lonVelocity_pressureGradient = 1e-3 * rectify_lon_presPartial
        
        # 速度场在纬度和经度方向的梯度
        lat_velocityPartial = torch.gradient(velocityCoef, dim=2, spacing = 5.625)[0] # (B, 10, 32 ,64)
        # the gradient of lontitude of all velocity component
        lon_velocityPartial = torch.gradient(velocityCoef, dim=3, spacing = 5.625)[0] # (B, 10, 32 ,64)
        # 球面坐标系下的经度梯度需要进行修正
        rectify_lon_velocityPartial = (lon_velocityPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 10, 32, 64)

        # 拆分出纬向和经向速度的梯度部分
        latVelocity_Partial2lat =  lat_velocityPartial[:, :5]   # (B, 5, 32, 64)
        latVelocity_Partial2lon = rectify_lon_velocityPartial[:, :5]    # (B, 5, 32, 64)
        lonVelocity_Partial2lat = lat_velocityPartial[:, 5:]    # (B, 5, 32, 64)
        lonVelocity_Partial2lon = rectify_lon_velocityPartial[:, 5:]    # (B, 5, 32, 64)

        lat_velocityCoef, lon_velocityCoef = velocityCoef[:, :5], velocityCoef[:, 5:] # (B, 5, 32, 64)

        # 计算对流项
        latVelocity_convectivePartial = lat_velocityCoef * latVelocity_Partial2lat + \
                                        lon_velocityCoef * latVelocity_Partial2lon # (B, 5, 32, 64)
        lonVelocity_convectivePartial = lat_velocityCoef * lonVelocity_Partial2lat + \
                                        lon_velocityCoef * lonVelocity_Partial2lon # (B, 5, 32, 64) 
        
        # 球面几何量
        tan_latitude = torch.tan(mesh[0]).view(1, 1, 32, 64)
        sin_latitude = torch.sin(mesh[0]).view(1, 1, 32, 64)
        cos_latitude = torch.cos(mesh[0]).view(1, 1, 32, 64)
        # 曲率项
        latVelocity_curvature = lon_velocityCoef ** 2 * tan_latitude
        lonVelocity_curvature = - lon_velocityCoef * lat_velocityCoef * tan_latitude
        # 科氏力项
        latVelocity_Coriolis = - 2 * 0.2618 * lon_velocityCoef * sin_latitude
        lonVelocity_Coriolis = 2 * 0.2618 * lat_velocityCoef * sin_latitude
        # 拉普拉斯耗散项
        latVelocity_Laplace = 1e-4 * lat_velocityCoef / (cos_latitude ** 2)
        lonVelocity_Laplace = 1e-4 * lon_velocityCoef / (cos_latitude ** 2)
        # 速度变化率的各个分量合成
        latVelocity_localPartial = - latVelocity_pressureGradient \
                                - latVelocity_convectivePartial \
                                - latVelocity_curvature \
                                + latVelocity_Coriolis \
                                - latVelocity_Laplace \

        lonVelocity_localPartial = - lonVelocity_pressureGradient \
                                - lonVelocity_convectivePartial \
                                - lonVelocity_curvature \
                                + lonVelocity_Coriolis \
                                - lonVelocity_Laplace \

        # 返回速度变化率
        return torch.cat([latVelocity_localPartial, lonVelocity_localPartial], dim=1)
