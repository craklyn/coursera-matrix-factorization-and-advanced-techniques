import pandas as pd
import numpy as np

#df = pd.read_excel("../data/assignment-6-edited.xlsx", sheet_name=None, header=1)
#df_items = df['Items']
#df_weights = df['Weights']
#df_users = df['Users']
#del df

df_items = pd.read_csv("../data/assignment-6-edited_items.csv")
df_users = pd.read_csv("../data/assignment-6-edited_users.csv")
df_weights = pd.read_csv("../data/assignment-6-edited_weights.csv")

print("Top five most relevant for feature 1:")
first_dim_argsort = np.argsort(-df_items[["1"]].iloc[:,0])
print(df_items.iloc[first_dim_argsort][["Movie ID", "Title"]].iloc[0:5])

print("Top five most relevant for feature 2:")
second_dim_argsort = np.argsort(-df_items[["2"]].iloc[:,0])
print(df_items.iloc[second_dim_argsort][["Movie ID", "Title"]].iloc[0:5])


user_4469 = df_users[df_users['User']==4469]

df_items['user_4469_score'] = 0

for index, item_row in df_items.iterrows():
#    print("Index:")
#    print(index)

    score = np.sum([(item_row[[str(i)]] * df_weights[[str(i)]].iloc[0] * user_4469[[str(i)]].iloc[0]).iloc[0] for i in range(1,16)])
    df_items.loc[index, 'user_4469_score'] = score


print("Top five most recommended items for user 4469:")
user_4469_itemsort = np.argsort(-df_items[["user_4469_score"]].loc[:,"user_4469_score"])
print(df_items.iloc[user_4469_itemsort][["Movie ID", "Title", "user_4469_score"]])



