import pickle
import csv
import os

with open('../data/reduced_dict.pkl', 'rb') as file:  # 128/32的pkl
    data = pickle.load(file)

with open('./raw_data/elliptic_txs_features.csv', 'r') as csvfile, open('temp.csv', 'w', newline='') as temp:  # 166的pkl
    reader = csv.reader(csvfile)
    writer = csv.writer(temp)

    for row, array in zip(reader, data.values()):
        # 展开数组并添加到行的末尾
        flattened_array = list(array)
        row.extend(flattened_array)

        writer.writerow(row)

csvfile.close()
temp.close()

os.rename('temp.csv', 'data.csv')
