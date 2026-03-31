import torch

# 计算对流项
def AdvectionCore(lat_velocity, lon_velocity, lat_dataPartial, lon_dataPartial):
    # 对流项 = 速度 * 数据梯度
    ConvectiveDerivatives = lat_velocity * lat_dataPartial + lon_velocity * lon_dataPartial
    # 负号来源于物理方程中对流项在右端
    return - ConvectiveDerivatives

# 返回 d(dataStates)/dt
def StateRhs(dataStates, velocityCoef, 
                    lat_dataPartial,
                    rectify_lon_dataPartial,
                    InteractionTendencies):
    # 从 velocityCoef 中拆分出纬向和经向速度            
    lat_velocityCoef, lon_velocityCoef = velocityCoef[:, :5], velocityCoef[:, 5:] # (B, 5, 32, 64)
    # 计算对流项和交互项
    InteractionTendencies = InteractionTendencies.transpose(1, 2).view(-1, 5, 32, 64) # (B, 5, 32, 64)
    AdvectionTendencies = AdvectionCore(lat_velocityCoef, lon_velocityCoef, lat_dataPartial, rectify_lon_dataPartial) # (B, 5, 32, 64)
    # 总变化率 = 对流项 + 神经网络预测的交互项
    localPartial = AdvectionTendencies + InteractionTendencies  # (B, 5, 32, 64)

    return localPartial

# 返回 dv/dt
def VelocityRhs(velocityCoef, mesh, lat_presPartial, rectify_lon_presPartial):
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
