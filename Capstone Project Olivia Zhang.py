## import the libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# random seed
random=16924188

## read the data and form the dataframes
a1=pd.read_csv('E:\\NYU\\07 2023 Spring\\ds112\\captstone project\\theArt.csv')
a1.columns=['Number','Artist','Title','Style','Year','Source','CorA','Intent']
a1[['Number','Year','Source','CorA','Intent']]=a1[['Number','Year','Source','CorA','Intent']].astype(int)
d1=pd.read_csv('E:\\NYU\\07 2023 Spring\\ds112\\captstone project\\theData.csv', names=range(1,222), index_col=False).replace('NaN',np.nan)

#### 1) Is classical art more well liked than modern art?
# set up dataframes to save results for this quesions
df1a=pd.DataFrame(columns=['Name','Mean','Median','Variance'])
df1b=pd.DataFrame(columns=['Name',"t-test",'Mann-Whitney U rank test'])

# classical art rating data
ca_index=a1.loc[a1.Source==1,'Number']
c1=d1.loc[:,ca_index]
ca=np.array(list(filter(lambda i:i==i, c1.values.reshape(c1.values.size))))
plt.hist(x=ca,bins=7)
plt.title('Histogram for Classical Art Preference Ratings')
plt.show()
# mean
ca_mean=ca.mean()
# median
ca_median=np.median(ca)
# variance
ca_var=ca.var()

df1a.loc[len(df1a)]=["Classical Art Ratings",ca_mean,ca_median,ca_var]

# modern art rating data
ma_index=a1.loc[a1.Source==2,'Number']
m1=d1.loc[:,ma_index]
ma=np.array(list(filter(lambda i:i==i, m1.values.reshape(m1.values.size))))
plt.hist(x=ma,bins=7)
plt.title('Histogram for Mordern Art Preference Ratings')
plt.show()
# mean
ma_mean=ma.mean()
# median
ma_median=np.median(ma)
# variance
ma_var=ma.var()

df1a.loc[len(df1a)]=["Modern Art Ratings",ma_mean,ma_median,ma_var]

# independent t-test
cm_indt=stats.ttest_ind(ca,ma,equal_var=False)

# Mann-Whitney U rank test
cm_mw=stats.mannwhitneyu(ca,ma,alternative='greater')

df1b.loc[len(df1b)]=["Classical Art and Modern Art",cm_indt,cm_mw]

## testing this by getting the mean rating of each category from each person
# classical
classical_mean=[]
for i in range(300):
    r=d1.loc[i,ca_index]
    classical_mean.append(r.mean())
plt.hist(x=classical_mean,bins=7)
plt.title('Histogram for Mean of Classical Art Preference Ratings')
plt.show()
# mean
meanca_mean=np.array(classical_mean).mean()
# median
meanca_median=np.median(np.array(classical_mean))
# variance
meanca_var=np.array(classical_mean).var()

df1a.loc[len(df1a)]=["Mean of Classical Art Ratings",meanca_mean,meanca_median,meanca_var]

# modern
modern_mean=[]
for i in range(300):
    r=d1.loc[i,ma_index]
    modern_mean.append(r.mean())
plt.hist(x=modern_mean,bins=7)
plt.title('Histogram for Mean of Mordern Art Preference Ratings')
plt.show()
# mean
meanma_mean=np.array(modern_mean).mean()
# median
meanma_median=np.median(np.array(modern_mean))
# variance
meanma_var=np.array(modern_mean).var()

df1a.loc[len(df1a)]=["Mean of Modern Art Ratings",meanma_mean,meanma_median,meanma_var]

# paired t-test
cmmean_pairt=stats.ttest_rel(classical_mean,modern_mean)

# Mann-Whitney U rank test
cmmean_mw=stats.mannwhitneyu(classical_mean,modern_mean,alternative='greater')

df1b.loc[len(df1b)]=["Mean of Classical Art and Modern Art",cmmean_pairt,cmmean_mw]

#######################################################################################

#### 2) Is there a difference in the preference ratings for 
####    modern art vs. non-human (animals and computers) generated art?
# set up a dataframe to save results for this quesions
df2=pd.DataFrame(columns=['Name',"t-test",'Mann-Whitney U rank test'])

# non-human generated art rating data
na_index=a1.loc[a1.Source==3,'Number']
n1=d1.loc[:,na_index]
na=np.array(list(filter(lambda i:i==i, n1.values.reshape(n1.values.size))))
plt.hist(x=na,bins=7)
plt.title('Histogram for Non-human Art Preference Ratings')
plt.show()
# mean
na_mean=na.mean()
# median
na_median=np.median(na)
# variance
na_var=na.var()

df1a.loc[len(df1a)]=["Non-human Art Ratings",na_mean,na_median,na_var]

# independent t-test
mn_indt=stats.ttest_ind(ma,na,equal_var=False)

# Mann-Whitney U rank test
mn_mw=stats.mannwhitneyu(ma,na)

df2.loc[len(df2)]=["Modern Art and Non-human Art",mn_indt,mn_mw]

## testing this by getting the mean rating of each category from each person
# non-human
nonhuman_mean=[]
for i in range(300):
    r=d1.loc[i,na_index]
    nonhuman_mean.append(r.mean())
plt.hist(x=nonhuman_mean,bins=7)
plt.title('Histogram for Mean of Non-human Art Preference Ratings')
plt.show()
# mean
meanna_mean=np.array(nonhuman_mean).mean()
# median
meanna_median=np.median(np.array(nonhuman_mean))
# variance
meanna_var=np.array(nonhuman_mean).var()

df1a.loc[len(df1a)]=["Mean of Non-human Art Ratings",meanna_mean,meanna_median,meanna_var]
# paired t-test
mnmean_pairt=stats.ttest_rel(modern_mean,nonhuman_mean)

# Mann-Whitney U rank test
mnmean_mw=stats.mannwhitneyu(modern_mean,nonhuman_mean)

df2.loc[len(df2)]=["Mean of Modern Art and Non-human Art",mnmean_pairt,mnmean_mw]

###########################################################################

#### 3) Do women give higher art preference ratings than men?
# set up dataframes to save results for this quesions
df3a=pd.DataFrame(columns=['Name','Mean','Median','Variance'])
df3b=pd.DataFrame(columns=['Name','t-test','Mann-Whitney U rank test'])

# women's rating data
wr=d1.loc[d1[217]==2,range(1,92)]
woman_r=np.array(list(filter(lambda i:i==i, wr.values.reshape(wr.values.size))))
plt.hist(x=woman_r,bins=7)
plt.title("Histogram of Women's Ratings")
plt.show()
# mean
wr_mean=woman_r.mean()
# median
wr_median=np.median(woman_r)
# variance
wr_var=woman_r.var()

df3a.loc[len(df3a)]=["Women's All Ratings",wr_mean,wr_median,wr_var]

# men's rating data
mr=d1.loc[d1[217]==1,range(1,92)]
man_r=np.array(list(filter(lambda i:i==i, mr.values.reshape(mr.values.size))))
plt.hist(x=man_r,bins=7)
plt.title("Histogram of Men's Ratings")
plt.show()
# mean
mr_mean=man_r.mean()
# median
mr_median=np.median(man_r)
# variance
mr_var=man_r.var()

df3a.loc[len(df3a)]=["Men's All Ratings",mr_mean,mr_median,mr_var]

# independent t-test
wm_indt=stats.ttest_ind(woman_r, man_r,equal_var=False)

# Mann-Whitney U rank test
wm_mw=stats.mannwhitneyu(woman_r, man_r,alternative='greater')

df3b.loc[len(df3b)]=["Women and Men(all)",wm_indt,wm_mw]

## group by art type

# classical
# woman
c_wr=wr.loc[:,ca_index]
classical_wr=np.array(list(filter(lambda i:i==i, c_wr.values.reshape(c_wr.values.size))))
plt.hist(x=classical_wr,bins=7)
plt.title("Histogram of Women's Classical Art Ratings")
plt.show()
# mean
cwr_mean=classical_wr.mean()
# median
cwr_median=np.median(classical_wr)
# variance
cwr_var=classical_wr.var()


df3a.loc[len(df3a)]=["Women's Classical Ratings",cwr_mean,cwr_median,cwr_var]

# man
c_mr=mr.loc[:,ca_index]
classical_mr=np.array(list(filter(lambda i:i==i, c_mr.values.reshape(c_mr.values.size))))
plt.hist(x=classical_mr,bins=7)
plt.title("Histogram of Men's Classical Art Ratings")
plt.show()
# mean
cmr_mean=classical_mr.mean()
# median
cmr_median=np.median(classical_mr)
# variance
cmr_var=classical_mr.var()

df3a.loc[len(df3a)]=["Men's Classical Ratings",cmr_mean,cmr_median,cmr_var]

# independent t-test
cwm_indt=stats.ttest_ind(classical_wr,classical_mr,equal_var=False)

# Mann-Whitney U rank test
cwm_mw=stats.mannwhitneyu(classical_wr,classical_mr,alternative='greater')

df3b.loc[len(df3b)]=["Women and Men(classical)",cwm_indt,cwm_mw]

# modern
# woman
m_wr=wr.loc[:,ma_index]
modern_wr=np.array(list(filter(lambda i:i==i, m_wr.values.reshape(m_wr.values.size))))
plt.hist(x=modern_wr,bins=7)
plt.title("Histogram of Women's Modern Art Ratings")
plt.show()
# mean
mwr_mean=modern_wr.mean()
# median
mwr_median=np.median(modern_wr)
# variance
mwr_var=modern_wr.var()

df3a.loc[len(df3a)]=["Women's Modern Ratings",mwr_mean,mwr_median,mwr_var]

# man
m_mr=mr.loc[:,ma_index]
modern_mr=np.array(list(filter(lambda i:i==i, m_mr.values.reshape(m_mr.values.size))))
plt.hist(x=modern_mr,bins=7)
plt.title("Histogram of Men's Modern Art Ratings")
plt.show()
# mean
mmr_mean=modern_mr.mean()
# median
mmr_median=np.median(modern_mr)
# variance
mmr_var=modern_mr.var()

df3a.loc[len(df3a)]=["Men's Modern Ratings",mmr_mean,mmr_median,mmr_var]

# independent t-test
mwm_indt=stats.ttest_ind(modern_wr, modern_mr,equal_var=False)

# Mann-Whitney U rank test
mwm_mw=stats.mannwhitneyu(modern_wr, modern_mr,alternative='greater')

df3b.loc[len(df3b)]=["Women and Men(modern)",mwm_indt,mwm_mw]

# non-human
# woman
n_wr=wr.loc[:,na_index]
nonhuman_wr=np.array(list(filter(lambda i:i==i, n_wr.values.reshape(n_wr.values.size))))
plt.hist(x=nonhuman_wr,bins=7)
plt.title("Histogram of Women's Non-human Art Ratings")
plt.show()
# mean
nwr_mean=nonhuman_wr.mean()
# median
nwr_median=np.median(nonhuman_wr)
# variance
nwr_var=nonhuman_wr.var()

df3a.loc[len(df3a)]=["Women's Non-human Ratings",nwr_mean,nwr_median,nwr_var]

# man
n_mr=mr.loc[:,na_index]
nonhuman_mr=np.array(list(filter(lambda i:i==i, n_mr.values.reshape(n_mr.values.size))))
plt.hist(x=nonhuman_mr,bins=7)
plt.title("Histogram of Men's Nonhuman Art Ratings")
plt.show()
# mean
nmr_mean=nonhuman_mr.mean()
# median
nmr_median=np.median(nonhuman_mr)
# variance
nmr_var=nonhuman_mr.var()

df3a.loc[len(df3a)]=["Men's Non-human Ratings",nmr_mean,nmr_median,nmr_var]

# independent t-test
nwm_indt=stats.ttest_ind(nonhuman_wr, nonhuman_mr,equal_var=False)

# Mann-Whitney U rank test
nwm_mw=stats.mannwhitneyu(nonhuman_wr, nonhuman_mr,alternative='greater')

df3b.loc[len(df3b)]=["Women and Men(non-human)",nwm_indt,nwm_mw]

#####################################################################################################

#### 4) Is there a difference in the preference ratings of users with some art background (some art
####    education) vs. none?
# set up a dataframe to save results for this quesions
df4a=pd.DataFrame(columns=['Art Education','Mean','Median','Variance'])
df4b=pd.DataFrame(columns=['Art Education','t-test','Mann-Whitney U rank test'])

# people with non art background (art education=0)
edu0=d1.loc[d1[219]==0,range(1,92)]
edu0_r=np.array(list(filter(lambda i:i==i, edu0.values.reshape(edu0.values.size))))
plt.hist(x=edu0_r,bins=7)
plt.title("Histogram of Art Education=0 Ratings")
plt.show()
# mean
edu0_mean=edu0_r.mean()
# median
edu0_median=np.median(edu0_r)
# variance
edu0_var=edu0_r.var()

df4a.loc[len(df4a)]=[0,edu0_mean,edu0_median,edu0_var]

# people with some art background (art education=1)
edu1=d1.loc[d1[219]==1,range(1,92)]
edu1_r=np.array(list(filter(lambda i:i==i, edu1.values.reshape(edu1.values.size))))
plt.hist(x=edu1_r,bins=7)
plt.title("Histogram of Art Education=1 Ratings")
plt.show()
# mean
edu1_mean=edu1_r.mean()
# median
edu1_median=np.median(edu1_r)
# variance
edu1_var=edu1_r.var()

df4a.loc[len(df4a)]=[1,edu1_mean,edu1_median,edu1_var]

# people with some art background (art education=2)
edu2=d1.loc[d1[219]==2,range(1,92)]
edu2_r=np.array(list(filter(lambda i:i==i, edu2.values.reshape(edu2.values.size))))
plt.hist(x=edu2_r,bins=7)
plt.title("Histogram of Art Education=2 Ratings")
plt.show()
# mean
edu2_mean=edu2_r.mean()
# median
edu2_median=np.median(edu2_r)
# variance
edu2_var=edu2_r.var()

df4a.loc[len(df4a)]=[2,edu2_mean,edu2_median,edu2_var]

# between art education 0&1
# independent t-test
art01_indt=stats.ttest_ind(edu0_r, edu1_r,equal_var=False)

# Mann-Whitney U rank test
art01_mw=stats.mannwhitneyu(edu0_r, edu1_r)

df4b.loc[len(df4b)]=["0 and 1",art01_indt,art01_mw]

# between art education 0&2
# independent t-test
art02_indt=stats.ttest_ind(edu0_r, edu2_r,equal_var=False)

# Mann-Whitney U rank test
art02_mw=stats.mannwhitneyu(edu0_r, edu2_r)

df4b.loc[len(df4b)]=["0 and 2",art02_indt,art02_mw]

# between art education 1&2
# independent t-test
art12_indt=stats.ttest_ind(edu1_r, edu2_r,equal_var=False)

# Mann-Whitney U rank test
art12_mw=stats.mannwhitneyu(edu1_r, edu2_r)

df4b.loc[len(df4b)]=["1 and 2",art12_indt,art12_mw]

# %%
#### 5) Build a regression model to predict art preference ratings from energy ratings only
# part 1
d5=pd.DataFrame({'Preference Ratings':d1.iloc[:,0:91].to_numpy().flatten(),'Energy Ratings':d1.iloc[:,91:182].to_numpy().flatten()}).dropna()
X=np.array(d5['Energy Ratings']).reshape(-1,1)
y=d5['Preference Ratings']
ene_train,ene_test,pre_train,pre_test=train_test_split(X,y,test_size=0.25,random_state=16924188)

model5=LinearRegression().fit(ene_train,pre_train)
y_hat=model5.predict(ene_test)
rmse=mean_squared_error(pre_test, y_hat, squared=False)
r2=model5.score(ene_train,pre_train)
print(f'Model RMSE: {rmse}')
print(f'Model R^2: {r2}')

plt.plot(ene_test, y_hat, color="blue", linewidth=3)
plt.title('preference rating predicted by individual energy ratings')
plt.yticks(np.arange(1,7))
plt.xlabel('Energy Rating') 
plt.ylabel('Predicted Preference Rating') 
plt.show()
#regularization
alpha=10
model5_ridge=Ridge(alpha=alpha).fit(ene_train, pre_train)
y_ridge=model5_ridge.predict(ene_test)
rmse_ridge=mean_squared_error(pre_test, y_ridge, squared=False)
r2_ridge=model5_ridge.score(ene_train,pre_train)
r2_cross=sklearn.metrics.r2_score(pre_test, y_ridge)

model5_lasso=Lasso(alpha=alpha).fit(ene_train, pre_train)
y_lasso=model5_lasso.predict(ene_test)
rmse_lasso=mean_squared_error(pre_test, y_lasso, squared=False)
r2_lasso=model5_lasso.score(ene_train,pre_train)
r2_cross=sklearn.metrics.r2_score(pre_test, y_lasso)
print(f'Alpha = {alpha}')
print(f'Regularized Ridge RMSE: {rmse_ridge}')
print(f'Regularized Ridge R^2: {r2_ridge}')
print(f'Regularized Lasso RMSE: {rmse_lasso}')
print(f'Regularized Lasso R^2: {r2_lasso}\n')

#%%
# part 2
preference_df=d1.iloc[:,0:91].dropna()
energy_df=d1.iloc[:,91:182].dropna()
p_avg=np.average(preference_df.to_numpy(), axis=1)
e_avg=np.average(energy_df.to_numpy(), axis=1)

ene_train,ene_test,pre_train,pre_test=train_test_split(e_avg.reshape(len(e_avg), 1),p_avg,test_size=0.25,random_state=random)
model5_2=LinearRegression().fit(ene_train,pre_train)
y_hat=model5_2.predict(ene_test)
rmse=mean_squared_error(pre_test, y_hat, squared=False)
r2=model5_2.score(ene_train,pre_train)
print(f'Model 2 RMSE: {rmse}')
print(f'Model 2 R^2: {r2}')



plt.plot(ene_test, y_hat, color="blue", linewidth=3)
plt.title('avg preference rating predicted by avg energy rating')
plt.yticks(np.arange(1,7))
plt.xlabel('Avg Energy Rating') 
plt.ylabel('Predicted Avg Preference Rating') 
plt.show()
#regularization
alpha=4
model5_ridge=Ridge(alpha=alpha).fit(ene_train, pre_train)
y_ridge=model5_ridge.predict(ene_test)
rmse_ridge=mean_squared_error(pre_test, y_ridge, squared=False)
r2_ridge=model5_ridge.score(ene_train,pre_train)

model5_lasso=Lasso(alpha=alpha).fit(ene_train, pre_train)
y_lasso=model5_lasso.predict(ene_test)
rmse_lasso=mean_squared_error(pre_test, y_lasso, squared=False)
r2_lasso=model5_lasso.score(ene_train,pre_train)
print(f'Alpha = {alpha}')
print(f'Regularized Ridge RMSE: {rmse_ridge}')
print(f'Regularized Ridge R^2: {r2_ridge}')
print(f'Regularized Lasso RMSE: {rmse_lasso}')
print(f'Regularized Lasso R^2: {r2_lasso}\n')

# %% 
#####################################################################################################
##### 6)
age = d1.iloc[:,215]
gender = d1.iloc[:,216]
df6 = pd.concat([age, gender], axis=1)
df6 = pd.merge(pd.merge(df6, d1.iloc[:,91:182], left_index=True, right_index=True), d1.iloc[:,:91], left_index=True, right_index=True).dropna()
ene_avg = np.average(df6.iloc[:,2:93].to_numpy(), axis=1) # reduce the energy ratings into a single predictor to make it weigh less to fairly consider other predictors
X = np.concatenate((df6.iloc[:, :2].to_numpy(), ene_avg[:, None]), axis=1)
X = stats.zscore(X) 
y = df6.iloc[:,93:].to_numpy()
y = np.average(y, axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=random)

multi_reg = LinearRegression().fit(x_train, y_train)
r2 = multi_reg.score(x_train, y_train)
y_hat = multi_reg.predict(x_test)
rmse = mean_squared_error(y_test, y_hat, squared=False)
r2 = multi_reg.score(x_train, y_train)
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

alpha=4
model6_ridge=Ridge(alpha=alpha).fit(x_train, y_train)
y_ridge=model6_ridge.predict(x_test)
rmse_ridge=mean_squared_error(y_test, y_ridge, squared=False)
r2_ridge=model6_ridge.score(x_train,y_train)

model6_lasso=Lasso(alpha=alpha).fit(x_train, y_train)
y_lasso=model6_lasso.predict(x_test)
rmse_lasso=mean_squared_error(y_test, y_lasso, squared=False)
r2_lasso=model6_lasso.score(x_train,y_train)
print(f'Alpha Level: {alpha}')
print(f'Regularized Ridge RMSE: {rmse_ridge}')
print(f'Regularized Ridge R^2: {r2_ridge}')
print(f'Regularized Lasso RMSE: {rmse_lasso}')
print(f'Regularized Lasso R^2: {r2_lasso}\n')














