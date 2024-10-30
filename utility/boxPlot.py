filename = 'macro-mlp'
data1 = [0.697936298,
         0.692867838,
         0.693856642,
         0.697241777,
         0.687714191

         ]
data2 = [0.69398392,
         0.648451108,
         0.670751159,
         0.694873942,
         0.690444307

         ]
data3 = [0.692888234,
         0.691473907,
         0.69622755,
         0.698457009,
         0.695241341

         ]
data4 = [0.696795278,
         0.692668878,
         0.695558052,
         0.679361339,
         0.69478905

         ]

data = [data1, data2, data3, data4]

style.use("ggplot")
colors = ["#948cc5", "#969696", "#c55040", "#6699cc"]

fig, ax = plt.subplots()  # 创建画布和子图
sns.boxplot(data=data, ax=ax, palette=sns.color_palette(colors))
ax.set_xticklabels(['AF', 'LF', 'AF-RPGE', 'LF-RPGE'])
ax.set_ylabel('F1 Score', fontsize=20)
plt.tight_layout()
# 保存图形为PDF格式
plt.savefig('png_pdf/' + filename + '.pdf', format='pdf')
# plt.show()


print(f"图形已保存为 {filename}")
