import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 读取第一个Excel文件
df1 = pd.read_csv('./Results.csv')

# 读取第二个Excel文件
df2 = pd.read_csv('./real_Results.csv')

# 提取IntDen列数据
intden1 = df1['IntDen']
intden2 = df2['IntDen']

# 计算皮尔逊系数
pearson_coef, pearson_p_value = pearsonr(intden1, intden2)

# 计算斯皮尔曼系数
spearman_coef, spearman_p_value = spearmanr(intden1, intden2)

print(f"pearson_coef:{pearson_coef}  pearson_p_value:{pearson_p_value}")

print(f"spearman_coef:{spearman_coef}  spearman_p_value:{spearman_p_value}")
