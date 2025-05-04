import numpy as np
from sedo.optimizer import SEDOptimizer

class SmartHomeEnergyOptimizer:
    """
    基于SED算法的智能家居节能优化系统
    优化目标：在满足用户舒适度的前提下最小化能源消耗
    """
    
    def __init__(self, devices):
        """
        初始化智能家居系统
        :param devices: 家居设备列表，每个设备包含功率、可控范围等信息
        """
        self.devices = devices
        self.user_preferences = {
            'temperature': 22.0,  # 摄氏度
            'humidity': 0.5,      # 相对湿度
            'light_level': 300    # 勒克斯
        }
        self.energy_prices = self._get_energy_prices()
        
        # 初始化优化器 (6维问题：温度、湿度、3个设备功率、1个时间权重)
        self.optimizer = SEDOptimizer(
            objective_func=self._calculate_cost,
            problem_dim=6,
            n_particles=20,
            bounds=[(18, 26), (0.3, 0.7)] + [(0, 1)]*4,
            barrier_height=0.7,
            use_parallel=True
        )
    
    def _get_energy_prices(self):
        """模拟分时电价数据"""
        return {
            'peak': 0.8,    # 高峰时段价格
            'normal': 0.5,  # 平时段价格
            'off': 0.3      # 低谷时段价格
        }
    
    def _calculate_cost(self, x):
        """
        计算总成本函数（适应度函数）
        :param x: [温度设定, 湿度设定, 空调功率比, 加湿器功率比, 照明功率比, 时间权重]
        """
        temp_set, hum_set, ac_ratio, hum_ratio, light_ratio, time_weight = x
        
        # 1. 计算舒适度偏离惩罚
        temp_diff = abs(temp_set - self.user_preferences['temperature'])
        hum_diff = abs(hum_set - self.user_preferences['humidity'])
        comfort_penalty = 0.6*temp_diff + 0.4*hum_diff
        
        # 2. 计算能源成本
        current_hour = 14  # 模拟下午2点（高峰时段）
        price = self.energy_prices['peak'] if 8 <= current_hour < 22 else self.energy_prices['off']
        
        energy_cost = 0
        for device, ratio in zip(self.devices[:3], [ac_ratio, hum_ratio, light_ratio]):
            energy_cost += device['power'] * ratio * price
        
        # 3. 时间权重因子（用户对舒适度与成本的权衡）
        total_cost = time_weight * comfort_penalty + (1-time_weight) * energy_cost
        
        return total_cost
    
    def optimize_schedule(self, max_iter=50):
        """运行优化过程"""
        self.optimizer.optimize(max_iter)
        best_solution = self.optimizer.get_best_solution()
        
        # 解释最优解
        optimal_temp = best_solution[0]
        optimal_hum = best_solution[1]
        device_ratios = best_solution[2:5]
        time_weight = best_solution[5]
        
        print(f"\n最优温度设定: {optimal_temp:.1f}°C")
        print(f"最优湿度设定: {optimal_hum*100:.1f}%")
        print("设备运行比例:")
        for device, ratio in zip(self.devices, device_ratios):
            print(f"  {device['name']}: {ratio*100:.1f}%")
        print(f"舒适度权重: {time_weight*100:.1f}%")
        
        return best_solution

# 示例使用
if __name__ == "__main__":
    # 定义家居设备
    devices = [
        {'name': '空调', 'power': 1500},  # 功率单位：瓦
        {'name': '加湿器', 'power': 50},
        {'name': '客厅照明', 'power': 100},
        {'name': '冰箱', 'power': 200}  # 不可调节设备
    ]
    
    optimizer = SmartHomeEnergyOptimizer(devices)
    best_settings = optimizer.optimize_schedule()
    
    # 可视化结果
    import matplotlib.pyplot as plt
    history = optimizer.optimizer.history
    plt.plot([h['best_fitness'] for h in history])
    plt.title("Smart Home Energy Optimization Result")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()