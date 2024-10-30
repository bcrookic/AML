import matplotlib.pyplot as plt

# 横坐标的值
x_values = ['8', '16', '32', '64']

# 三组纵坐标的值
filename = 'line_chart_gcn'
y1_values = [0.79653368, 0.8764, 0.862069469, 0.863707402]  #p
y2_values = [0.796660066, 0.7676, 0.75734612, 0.749310673] #r
y3_values = [0.794966716, 0.8114, 0.797793392, 0.793591848] #f

# 绘制折线图
plt.plot(x_values, y1_values, marker='o', label='precision', color='green')
plt.plot(x_values, y2_values, marker='o', label='recall', color='blue')
plt.plot(x_values, y3_values, marker='o', label='f1-score', color='red')

# 设置横坐标名称
plt.xlabel('d', ha='right', x=1)

# 设置纵坐标起始值
# plt.ylim(0.7, max(max(y1_values), max(y2_values), max(y3_values)) + 0.05)
plt.ylim(0.65, max(max(y1_values), max(y2_values), max(y3_values)) + 0.03)

# 设置横坐标间距和值
virtual_ticks = [0,1,2,3]
plt.xticks(virtual_ticks, x_values)

# 去掉右框线和上框线
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# 显示网格线
# plt.grid(True)

# 显示图例
plt.legend(loc='upper right')

# 保存图形为PDF格式
plt.savefig('png_pdf/' + filename + '.pdf', format='pdf')
# plt.show()

# 显示保存成功的消息
print(f"图形已保存为 {filename}")