import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('./data/train.csv', index_col=None)
user_df = pd.read_csv("./data/user.csv", index_col=0)
item_df = pd.read_csv("./data/business.csv", index_col=0)
user_df = user_df.rename(index=str, columns={t: 'user_' + t for t in user_df.columns if t != 'user_id'})
item_df = item_df.rename(index=str, columns={t: 'business_' + t for t in item_df.columns if t != 'business_id'})
data_merged = pd.merge(pd.merge(data, user_df, on='user_id'), item_df, on='business_id').reset_index(
    drop=True).drop(['user_yelping_since', 'user_elite', 'business_attributes', 'business_hours'],
                    axis=1).reset_index(drop=True)
data_merged = data_merged.drop(
    ['user_id', 'business_id', 'user_name', 'business_name', 'business_address', 'business_city',
     'business_state', 'business_postal_code', 'business_categories'], axis=1)
print(data_merged.dtypes)
# calculate the correlation matrix
corr = data_merged.corr()

# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig('./heatmap.png', bbox_inches='tight')
plt.show()
