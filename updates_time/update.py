import torch
from .rhs import StateRhs, VelocityRhs
from .integrators import euler_step, rk2_step, rk4_step

# 主更新函数
def update(mesh, constants, dataStates, model, step, integrator, dt):

    # 存储每个时间步长的更新结果
    updatedData, updatedVelocity = [], []
    velocityCoef = None
    B = dataStates.size(0)
    constants = constants[None, :, :, :].repeat(B, 1, 1, 1)

    # 外层：预测step个时间步长
    for t in range(step):
        # 内层：每个时间步长进行6次物理更新
        for t_phy in range(6):
            # 神经网络预测运动场和交互项
            MotionFields, InteractionTendencies = model(dataStates, constants) # (B, 32*64, 10), (B, 32*64, 5)
            # 初始化速度场
            if velocityCoef == None:
                velocityCoef = MotionFields.transpose(1,2).view(-1, 10, 32, 64)
                velocityCoef = torch.where(torch.abs(velocityCoef)<=0.005, velocityCoef, torch.sign(velocityCoef) * 0.005) # Pre-Processing
                updatedVelocity.append(MotionFields.transpose(1,2).view(-1, 10, 32, 64))

            # 多次小时间步长更新物理状态变量
            num_small_steps = int(1 / dt)
            # 防止梯度计算，节省显存
            with torch.no_grad():
                for t_states in range(num_small_steps):

                    # RHS 函数
                    def state_rhs(s):
                        # 计算数据梯度
                        lat_dataPartial = torch.gradient(s, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64)  # The interval of latitude and lontitude is 5.625°
                        lon_dataPartial = torch.gradient(s, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
                        rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 5, 32 ,64) #         
                        # Gradient in Cartesian coordiantes = (1/r)f_{lat}e_{lat}+(1/rcos(lat))f_{lon}e_{phi}
                        return StateRhs(
                            s,
                            velocityCoef,
                            lat_dataPartial,
                            rectify_lon_dataPartial,
                            InteractionTendencies
                        )
                    
                    # 计算数据梯度
                    lat_dataPartial = torch.gradient(dataStates, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64)  # The interval of latitude and lontitude is 5.625°
                    lon_dataPartial = torch.gradient(dataStates, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
                    rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 5, 32 ,64) #         
                    # Gradient in Cartesian coordiantes = (1/r)f_{lat}e_{lat}+(1/rcos(lat))f_{lon}e_{phi}

                    def velocity_rhs(v):
                        return VelocityRhs(
                            v,
                            mesh,
                            lat_dataPartial[:, 2],
                            rectify_lon_dataPartial[:, 2]
                        )

                    # 更新状态变量和速度场
                    dataStates = integrator(dataStates, state_rhs, dt)
                    velocityCoef = integrator(velocityCoef, velocity_rhs, dt)

                    # 限制速度场的变化范围
                    velocityCoef = torch.clamp(velocityCoef, -0.005, 0.005)

            # 计算数据梯度
            lat_dataPartial = torch.gradient(dataStates, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64)  # The interval of latitude and lontitude is 5.625°
            lon_dataPartial = torch.gradient(dataStates, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
            rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 5, 32 ,64) #         
            
            def state_rhs(s):
                return StateRhs(
                    s,
                    velocityCoef,
                    lat_dataPartial,
                    rectify_lon_dataPartial,
                    InteractionTendencies
                )
            def velocity_rhs(v):
                return VelocityRhs(
                    v,
                    mesh,
                    lat_dataPartial[:, 2],
                    rectify_lon_dataPartial[:, 2]
                )
            # 更新状态变量和速度场
            dataStates = integrator(dataStates, state_rhs, dt)
            velocityCoef = integrator(velocityCoef, velocity_rhs, dt)  
            velocityCoef = torch.clamp(velocityCoef, -0.005, 0.005)


        updatedData.append(dataStates)

    return torch.stack(updatedData), torch.stack(updatedVelocity)

