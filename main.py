import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import numpy as np

d = pd.read_csv('numerai_datasets/numerai_training_data.csv')
test_d = pd.read_csv('numerai_datasets/numerai_tournament_data.csv')

# Look for validation tag
iv = d.validation == 1

# Get the training and validation data separated
val = d[iv].copy()
train = d[~iv].copy()

# Drop out the validation; data has been split
train.drop('validation', axis = 1, inplace = True)
val.drop('validation', axis = 1, inplace = True)

train_dummies = pd.get_dummies(train.c1)
train_num = pd.concat((train.drop('c1', axis = 1), train_dummies), axis = 1)

val_dummies = pd.get_dummies(val.c1)
val_num = pd.concat((val.drop('c1', axis = 1), val_dummies), axis = 1)

test_dummies = pd.get_dummies(test_d.c1)
test_num = pd.concat((test_d.drop('c1', axis = 1), test_dummies), axis = 1)

train_targets = train_num["target"]
val_targets = val_num["target"]
test_ids = test_num["t_id"]

train_num.drop("target", axis = 1, inplace = True)
val_num.drop("target", axis = 1, inplace = True)
test_num.drop("t_id", axis = 1, inplace = True)

train_targets.to_csv('numerai_datasets/train_v_targets.csv', index=False)
val_targets.to_csv('numerai_datasets/val_v_targets.csv', index=False)
train_num.to_csv('numerai_datasets/train_v_num.csv', index=False)
val_num.to_csv('numerai_datasets/val_v_num.csv', index=False)
test_num.to_csv('numerai_datasets/test_v_num.csv', index=False)
test_ids.to_csv('numerai_datasets/test_v_ids.csv', index=False)
