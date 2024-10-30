import pandas as pd
from sklearn.preprocessing import normalize

data = pd.read_csv('data.csv')
label_columns = data.iloc[:, :2]
feature_columns = data.iloc[:, 2:]
normalized_features = normalize(feature_columns, norm='l2', axis=1)
normalized_data = pd.concat([label_columns, pd.DataFrame(normalized_features, columns=feature_columns.columns)], axis=1)
normalized_data.to_csv('normalized_data.csv', index=False)