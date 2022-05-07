import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE
def rmse(pred, actual):
	# Ignore nonzero terms.
	pred = pred[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()].flatten()
	return sqrt(mean_squared_error(pred, actual))

if __name__ == "__main__":

	val_df = pd.read_csv("../data/valid.csv")

	pred_df = pd.read_csv('val_pre.csv')

	df = pd.merge(val_df, pred_df, how="left", left_on=['user_id',
		'business_id'], right_on = ['user_id','business_id'])

	df.fillna(0, inplace=True)


	print("VALIDATION RMSE: ", rmse(df['stars_y'].values, df['stars_x'].values))

