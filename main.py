import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt #So we can make figures
from sklearn.decomposition import PCA #This will allow us to do the PCA efficiently
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error




random = 16924188

# Begin by importing the data
art = pd.read_csv('theArt.csv')
data = pd.read_csv('theData.csv', index_col=False, names=range(1,222))
print(art.dtypes)
print(data.shape)

# Preprocessing: 
art.dropna(inplace=True)
# data.dropna(inplace=True)

# questions to conduct PCA on
preferenceRating = data.iloc[:,:91]
energyRating = data.iloc[:,91:182]
personality = data.iloc[:,182:194]
preferences = data.iloc[:,194:205]
esteem = data.iloc[:,205:214]

age = data.iloc[:,215]
gender = data.iloc[:,216]
political = data.iloc[:, 217]
education = data.iloc[:,218]
sophistication = data.iloc[:,219]
amArtist = data.iloc[:,220]

z_personality = stats.zscore(personality.dropna().to_numpy())
z_preferences = stats.zscore(preferences.dropna().to_numpy())
z_esteem = stats.zscore(esteem.dropna().to_numpy())

pca_personality = PCA().fit(z_personality)
pca_preferences = PCA().fit(z_preferences)
pca_esteem = PCA().fit(z_esteem)

# print(pca_personality.explained_variance_)
# numPersonality = len(personality.columns)
# x = np.linspace(1,numPersonality,numPersonality)
# plt.bar(x, pca_personality.explained_variance_, color='gray')
# plt.plot([0,numPersonality],[1,1],color='orange') # Orange Kaiser criterion line for the fox
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.show()
# # for personality, pick 3 main components 
# print(pca_personality.components_)

# whichPrincipalComponent = 1 # Select and look at one factor at a time, in Python indexing
# plt.bar(x,pca_personality.components_[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
# #and Python reliably picks the wrong one. So we flip it.
# plt.xlabel('Question')
# plt.ylabel('Loading')
# plt.show() # Show bar plot

# 1) Is classical art more well liked than modern art? 

# 5) Build a regression model to predict art preference ratings from energy ratings only
# reformat our data into a single dimension 
# check if response lengths are the same 
print('--------------- Question 5 ----------------')
print(preferenceRating.dropna().shape)
print(energyRating.dropna().shape)

# no need to z-score or standardlize as the scales are the same
x = energyRating.to_numpy().flatten()
x = x.reshape(len(x),1) 
y = preferenceRating.to_numpy().flatten()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)

linReg = LinearRegression()
linReg.fit(x_train, y_train)

y_pred = linReg.predict(x_test)
r_sq = linReg.score(x, y)
print(f'score: {r_sq}')
r2 = r2_score(y_test, y_pred)
print(f'cross-validation r^2 score: {r2}')

# r squared is a very poor predictor, ploting a visualization to see how they're actually related

plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.xlabel('Prediction from model') 
plt.ylabel('Actual val') 
plt.show()

# 6) 
print('-----------Question 6-----------------')
# buliding a similar model, but this time including age, gender, education and artistic status
# political affiliation is not considered a demographic characteristics in this case
df6 = pd.concat([age, gender, education], axis=1)
df6 = pd.merge(pd.merge(df6, energyRating, left_index=True, right_index=True), preferenceRating, left_index=True, right_index=True).dropna()

scaler = StandardScaler()
scaler.fit(df6)
df6 = scaler.transform(df6) # df6 actually gets transformed into a ndarray
x = df6[:,:94]
# print(x.shape)
# x_2 = np.average(df6[:,3:94], axis=1)
# print(x_2.shape)
# x = np.concatenate((x, x_2[:, None]), axis=1)
# print(x.shape)

y = df6[:,94:]
y = np.average(y, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)

multiReg = LinearRegression()
multiReg.fit(x_train, y_train)
r_sq = multiReg.score(x_train, y_train)
y_pred = multiReg.predict(x_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred)}')
print(f'score: {r_sq}')

# for part 2 try to regularize our previous model
# df6 = pd.concat([age, gender, education], axis=1)
alpha = 10
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)
y_ridge = ridge_reg.predict(x_test)
regular_rsq = ridge_reg.score(x_train, y_train)
print(f'regularized RMSE: {mean_squared_error(y_test, y_ridge)}')
print(f'regularized score: {regular_rsq}')

# as alpha increases, the RMSE decreases, which indicates that the model is becoming more adaptable to new data

