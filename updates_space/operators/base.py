from abc import ABC, abstractmethod

# 空间离散算子基类
class SpaceOperator(ABC):

    @abstractmethod
    def StateRhs(self, dataStates, velocityCoef, interaction):
        """
        ∂q/∂t = -∇·(u q) + S
        """
        pass

    @abstractmethod
    def VelocityRhs(self, velocity, pressure, mesh):
        """
        ∂u/∂t = momentum equation
        """
        pass
