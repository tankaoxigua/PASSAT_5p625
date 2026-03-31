import torch
from .operators import build_space_operator

# 主更新函数
def update(mesh, constants, dataStates, model, step, config):

    # 存储每个时间步长的更新结果
    updatedData, updatedVelocity = [], []
    velocityCoef = None
    B = dataStates.size(0)
    constants = constants[None, :, :, :].repeat(B, 1, 1, 1)

    # 构建空间离散算子
    operator = build_space_operator(config, mesh)

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
            for t_states in range(5):
                # RHS 函数
                def state_rhs(s):
                    # 计算数据梯度       
                    localPartial = operator.StateRhs(s, velocityCoef)
                    # 计算交互项
                    interaction = InteractionTendencies.transpose(1, 2).view(-1, 5, 32, 64)
                    localPartial = localPartial + interaction
                    # 使用 Euler 方法更新状态变量
                    # 1/5 表示时间步长为 1 小时，数据的变化等于速度乘以时间
                    updated_dataStates = s + localPartial * (1/5)  
                    return updated_dataStates
                
                # 计算数据梯度
                lat_dataPartial = torch.gradient(dataStates, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64)  # The interval of latitude and lontitude is 5.625°
                lon_dataPartial = torch.gradient(dataStates, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
                rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 5, 32 ,64) #         
                # Gradient in Cartesian coordiantes = (1/r)f_{lat}e_{lat}+(1/rcos(lat))f_{lon}e_{phi}
                def velocity_rhs(v):
                    localPartial = operator.VelocityRhs(v, mesh, lat_dataPartial[:, 2], rectify_lon_dataPartial[:, 2])
                    # 使用 Euler 方法更新速度场        
                    updated_velocityCoef = v + localPartial * (1/5)    # (B, 10, 32, 64): 1 stands for the time 1 hour. Changes equal to velocity * time
                    return updated_velocityCoef
                
                dataStates = state_rhs(dataStates)
                velocityCoef = velocity_rhs(velocityCoef)
        updatedData.append(dataStates)

    return torch.stack(updatedData), torch.stack(updatedVelocity)