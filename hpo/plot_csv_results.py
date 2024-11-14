import pandas as pd
from matplotlib import pyplot as plt  #
import seaborn as sns

df = pd.read_csv('hpo/result_lambda_heatmap_seg.csv')
# clean the dataframe
df = df[df['status'] == 'completed']
df = df.drop_duplicates(subset=['Args/lambda_loss'])

sns.scatterplot(data=df, x='TRE [mm]/val', y='Dice/val')
for idx, (tre, dice) in df[['TRE [mm]/val', 'Dice/val']].iterrows():
    plt.text(tre, dice, idx, ha='left', weight='bold')
best_idx = 28
print(df.iloc[best_idx], df.iloc[best_idx, 0])
plt.show()
