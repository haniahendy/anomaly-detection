from anom.detect import iforest_flags, make_data, precision_recall, zscore_flags

X, y = make_data()
print("z-score  (precision, recall):", precision_recall(y, zscore_flags(X)))
print("iforest  (precision, recall):", precision_recall(y, iforest_flags(X)))
