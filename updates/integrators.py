# 时间接口
def euler_step(state, rhs_func, dt):
    return state + dt * rhs_func(state)

def rk2_step(state, rhs_func, dt):
    k1 = rhs_func(state)
    k2 = rhs_func(state + dt * k1)
    return state + 0.5 * dt * (k1 + k2)

def rk4_step(state, rhs_func, dt):
    k1 = rhs_func(state)
    k2 = rhs_func(state + 0.5 * dt * k1)
    k3 = rhs_func(state + 0.5 * dt * k2)
    k4 = rhs_func(state + dt * k3)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
