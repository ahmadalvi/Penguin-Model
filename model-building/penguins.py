import pandas as pd

#reading data into a dataframe
penguins = pd.read_csv('penguins_cleaned.csv')

#setting target and feature variables for the dataset
df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

#encoding the sex and island columns
for col in encode:
	dummy = pd.get_dummies(df[col], prefix = col)
	df = pd.concat([df, dummy], axis = 1)
	del df[col]

#this block of code will encode the target species
mapper = {'Adelie': 0, 'Chinstrap':1, 'Gentoo': 2}

def target_encode(val):
	return mapper[val]

#This line of code will apply the custom function in order to perform the encoding
df['species'] = df['species'].apply(target_encode)

#These 2 lines of code will seperate the dataset into X and y into data matrices to use in scikit learn
X = df.drop('species', axis = 1)
y = df['species']

#Building random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

#Saving the model with the pickle library
import pickle
pickle.dump(clf, open('penguin_clf.pkl', 'wb'))
