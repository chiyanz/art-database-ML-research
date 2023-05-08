#%% 0) Setups
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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import statsmodels.api as sm
import math

random = 16924188
np.random.seed(random)

# Begin by importing the data
art = pd.read_csv('theArt.csv')
art.rename(columns={'Number ': "Number", 'Artist ': "Artist", 'Title' : "Title", 'Style': "Style", 'Year': "Year", 'Source (1 = classical, 2 = modern, 3 = nonhuman)': "Type", 'computerOrAnimal (0 = human, 1 = computer, 2 = animal)': "Computer",'Intent (0 = no, 1 = yes)': "Intent" }, inplace=True)
print(art.columns)

data = pd.read_csv('theData.csv', index_col=False, names=range(1,222))
print(art.dtypes)
print(data.shape)

# Preprocessing: 
art.dropna(inplace=True)

# questions to conduct PCA on
preferenceRating = data.iloc[:,:91]
energyRating = data.iloc[:,91:182]
personality = data.iloc[:,182:194]
preferences = data.iloc[:,194:205]
esteem = data.iloc[:,205:215]

age = data.iloc[:,215]
gender = data.iloc[:,216]
political = data.iloc[:, 217]
education = data.iloc[:,218]
sophistication = data.iloc[:,219]
amArtist = data.iloc[:,220]

# 1) Is classical art more well liked than modern art? 

# 5) Build a regression model to predict art preference ratings from energy ratings only
# %% 
# 5a)  reformat our data into a single dimension 
# check if response lengths are the same 
print('--------------- Question 5 ----------------')
# method 1
print('Method 1:')
x = energyRating.to_numpy().flatten()
x = x.reshape(len(x),1) 
y = preferenceRating.to_numpy().flatten()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)

linReg = LinearRegression()
linReg.fit(x_train, y_train)

y_pred = linReg.predict(x_test)
r_sq = linReg.score(x, y)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'R^2: {r_sq}')

plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.title('Individual preference rating predictions from individual energy rating')
plt.yticks(np.arange(1,7))
plt.xlabel('Energy Rating') 
plt.ylabel('Preference Rating Prediction') 
plt.show()

# method 2
print('Method 2:')
x1 = np.average(energyRating.to_numpy(), axis=1)
x1 = x1.reshape(len(x1),1) 
y1 = np.average(preferenceRating.to_numpy(), axis=1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=random)

linReg = LinearRegression()
linReg.fit(x1_train, y1_train)

y1_pred = linReg.predict(x1_test)
r_sq = linReg.score(x1, y1)
print(f'RMSE: {mean_squared_error(y1_test, y1_pred, squared=False)}')
print(f'R^2: {r_sq}')

plt.plot(x1_test, y1_pred, color="blue", linewidth=3)
plt.title('Average preference rating predictions from average energy rating')
plt.xlabel('Average Energy Rating') 
plt.ylabel('Average Preference Rating Prediction') 
plt.show()

#%%
# 5b) regularization


alpha = 10
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)
y_ridge = ridge_reg.predict(x_test)
regular_rsq = ridge_reg.score(x_train, y_train)
print(f'Alpha = {alpha}')
print(f'regularized Ridge RMSE: {mean_squared_error(y_test, y_ridge, squared=False)- mean_squared_error(y_test, y_pred, squared=False)}')
print(f'regularized Ridge R^2: {regular_rsq}')

lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(x_train, y_train)
y_lasso = lasso_reg.predict(x_test)
regular_rsq = lasso_reg.score(x_train, y_train)
print(f'Regularized Lasso RMSE: {mean_squared_error(y_test, y_lasso, squared=False)}')
print(f'Regularized Lasso R^2: {regular_rsq}')

alpha = 4
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x1_train, y1_train)
y1_ridge = ridge_reg.predict(x1_test)
regular_rsq = ridge_reg.score(x1_train, y1_train)
print(f'Alpha = {alpha}')
print(f'regularized Ridge RMSE: {mean_squared_error(y1_test, y1_ridge, squared=False)}')
print(f'regularized Ridge R^2: {regular_rsq}')

lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(x1_train, y1_train)
y1_lasso = lasso_reg.predict(x1_test)
regular_rsq = lasso_reg.score(x1_train, y1_train)
print(f'Regularized Lasso RMSE: {mean_squared_error(y1_test, y1_lasso, squared=False)}')
print(f'Regularized Lasso R^2: {regular_rsq}')

# 6) 
# %% 
print('-----------Question 6-----------------')
# 6a) buliding a similar model, but this time including age, gender, education and artistic status
# political affiliation is not considered a demographic characteristics in this case
df6 = pd.concat([age, gender], axis=1)
df6 = pd.merge(pd.merge(df6, energyRating, left_index=True, right_index=True), preferenceRating, left_index=True, right_index=True).dropna()
x = stats.zscore(df6.iloc[:,:2].to_numpy())
x_2 = np.average(df6.iloc[:,2:93].to_numpy(), axis=1) # reduce the energy ratings into a single predictor to make it weigh less to fairly consider other predictors
x = np.concatenate((x, x_2[:, None]), axis=1)

y = df6.iloc[:,93:].to_numpy()
y = np.average(y, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)

multiReg = LinearRegression()
multiReg.fit(x_train, y_train)
r_sq = multiReg.score(x_train, y_train)
y_pred = multiReg.predict(x_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'R^2: {r_sq}')



# for part 2 try to regularize our previous model
# df6 = pd.concat([age, gender, education], axis=1)
alpha = 9
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)
y_ridge = ridge_reg.predict(x_test)
regular_rsq = ridge_reg.score(x_train, y_train)
print(f'regularized RMSE: {mean_squared_error(y_test, y_ridge, squared=False)}')
print(f'regularized score: {regular_rsq}')

lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(x_train, y_train)
y_lasso = lasso_reg.predict(x_test)
regular_rsq = lasso_reg.score(x_train, y_train)
print(lasso_reg.coef_)
print(f'Regularized Lasso RMSE: {mean_squared_error(y_test, y_lasso, squared=False)}')
print(f'Regularized Lasso R^2: {regular_rsq}')
# as alpha increases, the RMSE decreases, which indicates that the model is becoming more adaptable to new data


# 7) 
#%% 
# 7A) average both preference rating and energy rating 
print('-----------Question 7-----------------')
avg_preference = np.average(preferenceRating.to_numpy(), axis=0)
avg_energy = np.average(energyRating.to_numpy(), axis=0)

print(avg_preference.shape) 
x = np.column_stack((avg_preference, avg_energy)) # input data to be classified

# code adapted from code session 13
numClusters = 9 
sSum = np.empty([numClusters,1])*np.NaN # init container to store sums
# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,20)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot   
plt.show()

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.title('Silhouette scores')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

#%% 
# 7 B) print out the clusters and see if their identity can be inferred 
num_clusters = 4 # try 3 clusters and see how the data is distributed 
kmeans = KMeans(n_clusters = num_clusters).fit(x)
cId = kmeans.labels_
cCoords = kmeans.cluster_centers_

for i in range(num_clusters):
    print('Type of art in each cluser:')
    print(f'{cCoords[int(i-1),0]}, {cCoords[int(i-1),1]}')
    plotIndex = np.argwhere(cId == int(i))
    print(art.iloc[plotIndex.flatten()]['Style'])
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(i-1),0],cCoords[int(i-1),1],'o',markersize=5,color='black')  
    plt.title('Art piece clusters')
    plt.xlabel('Preference')
    plt.ylabel('Energy')
plt.show()

# Clusters from left to right: 
# Left: 1 and 2
# Middle Top: 1 and 2
# Middle Bottom: 1
# Right: 3

# Relating each cluster with types of art, it seems that while classical art contains more variability both in rating and in energy, modern art is noticeably more likely to be rated high for energy
# while computer generated art is significantly high in preference rating and comparively low in energy rating

# the clusters could be interpreted as: 
# left: low preference and slightly agitating: mostly characterized by abstract, symbolic modern pieces and dark classical pieces that leave no strong impressions and is relatively agonizing
# upper: strong in energy with bold colors and messages, and appealing to the common eye
# lower: the "classics": mona lisa, michalangelo, etc. Soothing, pretty, and delicate
# right: very pretty images generated by a computer, not much more

# 8) Considering only the first principal component of the self-image ratings
# %%
#  8a) Getting the first principle component of self-image ratings
# only need a single component
# clean data for nulls first 
df8 = pd.merge(preferenceRating, esteem, left_index=True, right_index=True).dropna()
esteem8 = df8.iloc[:,91:]
# scaler.fit(esteem8)
# esteem8 = scaler.transform(esteem8)
esteem8 = stats.zscore(esteem8.to_numpy())
y  = np.average(df8.iloc[:,:91].to_numpy(), axis=1)

#PCA on self-image
pca_esteem = PCA(n_components=1).fit(esteem8)
x = np.linspace(1,10,10)
plt.bar(x,pca_esteem.components_[0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot
x = pca_esteem.fit_transform(esteem8).reshape(len(esteem8),1) # 1-dim principle component
print(f'Explained variance ratio: {pca_esteem.explained_variance_ratio_}')

#%% 
# 8b) Linear Regression with the first principle component
# build a linear regression model with this 1st principle component
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)

esteemReg = LinearRegression()
esteemReg.fit(x_train, y_train)
r_sq = esteemReg.score(x, y)
y_pred = esteemReg.predict(x_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'R^2: {r_sq}')
# plt.scatter(x_test, y_test, color="black")
plt.plot(y_pred, y_test,'o',markersize=4)
plt.title('Predicted preference rating vs actual preference rating (average)')
plt.xlabel('Prediction from model') 
plt.ylabel('Actual val') 
plt.show()


# %%
# 8c) Regularization
alpha = 1000
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)
y_ridge = ridge_reg.predict(x_test)
regular_rsq = ridge_reg.score(x_train, y_train)
print(f'regularized Ridge RMSE: {mean_squared_error(y_test, y_ridge, squared=False)}')
print(f'regularized Ridge R^2: {regular_rsq}')

lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(x_train, y_train)
y_lasso = lasso_reg.predict(x_test)
regular_rsq = lasso_reg.score(x_train, y_train)
print(f'Regularized Lasso RMSE: {mean_squared_error(y_test, y_lasso, squared=False)}')
print(f'Regularized Lasso R^2: {regular_rsq}')
plt.scatter(x_test, y_test, color = 'g', label="actual")
plt.plot(x_test, y_lasso, color='k', label="prediction")
plt.title('Visualizing regularized regression model')
plt.xlabel('Predictor') 
plt.ylabel('Predicted Rating')
plt.legend(loc="upper left")
plt.show()


# 9) Consider the first 3 principal components of the “dark personality” traits –
# %% 
# 9a) conduct the principle component extraction with PCA
df9 = pd.merge(personality, preferenceRating, left_index=True, right_index=True).dropna()
personality9 = df9.iloc[:,:12]
personality9 = stats.zscore(personality9.to_numpy())
y = np.average(df9.iloc[:,12:].to_numpy(), axis=1)

#PCA on self-image
pca_personality = PCA(n_components=3).fit(personality9)
print(f'Explained variance ratio: {pca_personality.explained_variance_ratio_}')

# Take a look at the eigenvalues and see that there is indeed 3 under kaiser
# plt.bar(x, pca_personality.explained_variance_, color='gray')
# plt.plot([0,12],[1,1],color='orange') # Orange Kaiser criterion line for the fox
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.show()
# print(personality.columns)

# actually transform the inputs to 3 dimensions
x = pca_personality.fit_transform(personality9).reshape(len(personality9),3)


# %% 
# 9b) building the linear regression model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random)
personalityReg = LinearRegression()
personalityReg.fit(x_train, y_train)
r_sq = personalityReg.score(x_train, y_train)
y_pred = personalityReg.predict(x_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'score: {r_sq}')
# plt.scatter(x_test, y_test, color="black")
plt.plot(y_pred, y_test,'o',markersize=4)
plt.title('Prediction using all 3 principle components')
plt.xlabel('Prediction from model') 
plt.ylabel('Actual val') 
plt.show()


X2 = sm.add_constant(x_train)
print(X2.shape)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
# PRINCIPLE COMPONENT 2 IS SIGNIFICANT

# %% 
# 9c) Look at the principle components separately
x_scale = np.linspace(1,12,12)
plt.subplot(3, 1, 1)
plt.bar(x_scale,pca_personality.components_[0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show()
# 1 I tend to manipulate others to get my way
# 4 I tend to exploit others towards my own end
# = SELFLESSNESS(?), KINDESS

plt.subplot(3, 1, 2)
plt.bar(x_scale,pca_personality.components_[1,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show()
# 9 I tend to want others to admire me
# 10 I tend to want others to pay attention to me
# = ATTENTION SEEKING

plt.subplot(3, 1, 3)
plt.bar(x_scale,pca_personality.components_[2,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show()
# 7 I can be callous or insensitive
# 8 I tend to be cynical
# = CYNICAL

x2 = x[:,1].reshape(len(x), 1)
x_train, x_test, y_train, y_test = train_test_split(x2, y, random_state=random)
personalityReg = LinearRegression()
personalityReg.fit(x_train, y_train)
r_sq = personalityReg.score(x_train, y_train)
y_pred = personalityReg.predict(x_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'score: {r_sq}')
# plt.scatter(x_test, y_test, color="black")
plt.plot(y_pred, y_test,'o',markersize=4)
plt.title('Prediction using principle component 2')
plt.xlabel('Prediction from model') 
plt.ylabel('Actual val') 
plt.show()

# %% 
# 9d) regularization
alpha = 1
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(x_train, y_train)
y_ridge = ridge_reg.predict(x_test)
regular_rsq = ridge_reg.score(x_train, y_train)
print(f'regularized Ridge RMSE: {mean_squared_error(y_test, y_ridge, squared=False)}')
print(f'regularized Ridge R^2: {regular_rsq}')

lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(x_train, y_train)
y_lasso = lasso_reg.predict(x_test)
regular_rsq = lasso_reg.score(x_train, y_train)
print(f'Regularized Lasso RMSE: {mean_squared_error(y_test, y_lasso, squared=False)}')
print(f'Regularized Lasso R^2: {regular_rsq}')

# 10) 
# %%
# 10a) determine which factors to chose for building the model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


poli_education = political.corr(education)
print(poli_education)

poli_age = political.corr(age)
print(poli_age)

poli_soph = political.corr(sophistication)
print(poli_soph)

poli_artist = political.corr(amArtist)
print(poli_artist)

personality['Political'] = political
personality.dropna(inplace=True)
print(personality.corr())

# 6 I tend to be unconcerned with the morality of my actions
# 11 I tend to seek prestige and status

df = pd.concat([amArtist, sophistication, personality.iloc[:,5], personality.iloc[:,10], political], axis=1).dropna()
x = stats.zscore(df.iloc[:,:4].to_numpy())
x = x.reshape(len(x), 4)
# categorize: 1 = 1 or 2 (left), 0 = all else 
y = df.iloc[:,4].apply(lambda x: 1 if x <= 2 else 0).to_numpy()

# %% 
# 10b) conduct the multiregression and check significance of predictors
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
model = LogisticRegression().fit(x_train,y_train)
print(model.coef_)
y_pred = model.predict(x_test)
r_sq = model.score(x_test, y_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}') # accuracy of 0.7
scores = cross_val_score(model, x_train, y_train, cv=10)
scores = pd.Series(scores)
print(f'Min accuracy: {"{:.2f}".format(scores.min())}, Avg accuracy:{"{:.2f}".format(scores.mean())}, Max Accuracy: {"{:.2f}".format(scores.max())}')
logit_model1 = sm.Logit(y_train, x_train) 
result = logit_model1.fit()
print(result.summary())

# %%
