import matplotlib.pyplot as plt

loss_path = 'saved-loss/Pers_h128_loss.txt'
loss_values = []
with open(loss_path, 'r', newline='', encoding='utf-8') as f:
    for line in f:
        loss_values.append(eval(line.strip()))
f.close()

# 定义颜色列表
colors = ['b', 'g', 'r', 'c', 'm']

# 绘制折线图
plt.figure(figsize=(10, 6))
for i, values in enumerate(loss_values, 1):
    start_index = (i - 1) * len(values)
    end_index = i * len(values)
    plt.plot(range(start_index, end_index), values, color=colors[i-1], label=f'Epoch {i}')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve with Different Colors for Each Epoch')
plt.legend()
plt.grid(True)
plt.show()
