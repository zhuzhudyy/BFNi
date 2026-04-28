import pandas as pd
df = pd.read_csv('Optimized_Training_Data.csv')
print('产率异常值检查:')
print('  产率=0:', len(df[df['TARGET_Yield'] == 0]), '行')
print('  产率=1:', len(df[df['TARGET_Yield'] == 1]), '行')
print('  产率>0.9:', len(df[df['TARGET_Yield'] > 0.9]), '行')
print()
print('所有数据点的产率和温度:')
for i, row in df.iterrows():
    print('  行%2d: T=%.0f, t=%.1f, yield=%.4f' % (i, row['Process_Temp'], row['Process_Time'], row['TARGET_Yield']))
