import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 100)

from utils import *

df = load_dataset()

print(df.describe())

# plot correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.savefig("correlation_matrix.png")
plt.show()

hist = df.hist(bins = 10)
plt.show()

print(df)
print(df.loc[:, "cover_type"])
hist =  df.cover_type.hist()
plt.show()
