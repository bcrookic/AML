import pickle
import csv
import os

with open('../data/reduced_dict.pkl', 'rb') as file:  
    data = pickle.load(file)

with open('../raw_data/elliptic_txs_features.csv', 'r') as csvfile, open('../raw_data/new_features.csv', 'w', newline='') as temp:  
    reader = csv.reader(csvfile)
    writer = csv.writer(temp)

    for row, array in zip(reader, data.values()):
        flattened_array = list(array)
        row.extend(flattened_array)

        writer.writerow(row)

csvfile.close()
temp.close()
