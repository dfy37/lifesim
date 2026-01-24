import numpy as np
import random

# 随机生成一个M x M的矩阵，填充原子
def generate_target_state(M, max_atoms=None):
    state = np.zeros((M, M), dtype=int)
    
    center = M // 2  # 中心的坐标
    state[center-1:center+2, center-1:center+2] = 1  # 设置中心3x3为1

    total_atoms = np.sum(state)  # 当前已经有的原子数量（中心部分）

    if max_atoms is None:
        max_atoms = (M * M) // 2  # 总原子数限制为50%

    remaining_atoms = max_atoms - total_atoms
    remaining_atoms = random.randint(0, remaining_atoms)

    possible_positions = [(i, j) for i in range(M) for j in range(M) if state[i, j] == 0]
    selected_positions = random.sample(possible_positions, remaining_atoms)  # 随机选择剩余位置
    for i, j in selected_positions:
        state[i, j] = 1

    return state

# 显示状态
def print_state(state):
    print("\n".join(" ".join(str(cell) for cell in row) for row in state))

# 倒推操作：根据当前状态生成前一状态
def reverse_operation(current_state, x, y1, y2, d):
    previous_state = current_state.copy()  # 深拷贝当前状态
    
    # 横向操作（行操作）
    if d == 0:
        previous_state[x, y1:y2+1] = 1 - previous_state[x, y1:y2+1]  # 反转行中y1到y2的位置
    # 纵向操作（列操作）
    elif d == 1:
        previous_state[y1:y2+1, x] = 1 - previous_state[y1:y2+1, x]  # 反转列中y1到y2的位置
    
    return previous_state

# 从目标状态倒推步骤，直到生成初始状态
def generate_data(target_M, steps):
    # 生成目标状态
    target_state = generate_target_state(target_M, max_atoms=(target_M * target_M) // 2)
    print("目标状态 (5x5):")
    print_state(target_state)
    
    # 存储步骤
    steps_data = []

    current_state = target_state.copy()  # 从目标状态开始

    # 倒推步骤
    for step in range(steps):
        # 随机生成一个操作 {x, y1, y2, d}
        nx = random.randint(1, target_M)
        x = random.sample(range(target_M), nx)  # 随机选择一个行或列
        
        y1 = random.randint(0, target_M - 1)  # 随机选择起始位置
        y2 = random.randint(y1, target_M - 1)  # 终止位置大于等于起始位置，确保单增
        d = random.choice([0, 1])  # 随机选择方向，0是行，1是列

        # 根据 {x, y1, y2, d} 反推 previous state
        previous_state = reverse_operation(current_state, x, y1, y2, d)
        
        # 记录操作和状态
        steps_data.append({
            'step': step + 1,
            'x': x,        # 选择的行或列索引
            'y1': y1,      # 起始位置
            'y2': y2,      # 终止位置
            'd': d,        # 操作方向
            'previous_state': previous_state.copy(),  # 上一步的状态
            'current_state': current_state.copy()     # 当前状态
        })

        # 更新current_state为previous_state，准备进行下一步倒推
        current_state = previous_state.copy()

        # 输出每一步倒推的状态
        print(f"倒推步骤 {step+1}: x={x}, y1={y1}, y2={y2}, d={d}")
        print_state(previous_state)
    
    return steps_data

if __name__ == '__main__':
    steps_data = generate_data(5, steps=5)

    for step_info in steps_data:
        print(f"\nStep {step_info['step']}: {step_info['direction']} {step_info['rows_or_cols']}")
        print_state(step_info['state'])
