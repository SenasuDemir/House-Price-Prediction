######################################################
# House Price Prediction
######################################################

# Duty
# Using a dataset containing the features and prices of each house,
# a machine learning project is intended to be carried out regarding the prices of different types of houses.
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation


# Requirements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from  xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.float_format',lambda x:'%.3f' % x)

######################################################
# Read and combine the Train and Test data sets. Proceed on the data we combined..
######################################################

# Combining train and test sets
train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')
df=pd.concat([train,test],axis=0,ignore_index=True)

df.head()
df.tail()

######################################################
# General Picture
######################################################
def check_df(dataframe,head=5):
    print('########################### Shape ###########################')
    print(dataframe.shape)
    print('########################### Types ###########################')
    print(dataframe.dtypes)
    print('########################### Head ###########################')
    print(dataframe.head(head))
    print('########################### Tail ###########################')
    print(dataframe.tail(head))
    print('########################### NA ###########################')
    print(dataframe.isnull().sum())
    print('########################### Quantiles ###########################')
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10,car_th=20):
    """
    Define the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables also include those that appear numeric but are actually categorical

    Parameters
    ----------
    dataframe: dataframe
                dataframe from which variable names are to be extracted
    cat_th: int, optional
                Threshold value for the number of classes for variables that are numerical but actually categorical
    car_th: int,optional
                Threshold value for the number of classes for variables that are categorical but cardinal

    Returns
    -------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of cardinal variables that appear categorical


    Examples
    --------
        import seaborn as sns
        df=sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = Total number of variables
        num_but_cat is within cat_cols'
    """

    #cat_cols, cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()< cat_th and dataframe[col].dtypes != 'O']
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes =='O']
    cat_cols=cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols {len(cat_cols)}')
    print(f'num_cols {len(num_cols)}')
    print(f'cat_but_car {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)


###########################################
# Analysis of Categorical Variables
###########################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        '`Ratio':100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print('**********************************************')
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col)
###########################################
# Analysis of Numerical Variables
###########################################

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col)

###########################################
# Analysis of Target Variable
###########################################

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({'TARGET_MEAN':dataframe.groupby(categorical_col)[target].mean()}),end='\n\n\n')

for col in cat_cols:
    target_summary_with_cat(df,'SalePrice',col)

# TRANSFORMATION
# Examining the Dependent Variable
df['SalePrice'].hist(bins=100)
plt.show(block=True)

# Examining the logarithm of the dependent variable
np.log1p(df['SalePrice']).hist(bins=50)
plt.show(block=True)

###########################################
# Analysis of Correlation
###########################################

corr=df[num_cols].corr()
corr

# Showing correlation
sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap='RdBu')
plt.show(block=True)

def high_correlated_cols(dataframe,plot=False,corr_th=0.70):
    corr=dataframe.corr()
    cor_matrix=corr.abs()
    upper_triangle_matrix=cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    drop_list=[col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr,cmap='RdBu')
        plt.show(block=True)
    return drop_list

high_correlated_cols(df[num_cols])
high_correlated_cols(df[num_cols],plot=True)

###############################################################
# Feature Engineering
###############################################################

###########################################
# Outlier Analysis
###########################################

def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe,col_name):
    low_limit, up_limit =outlier_threshold(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != 'SalePrice':
        print(col, (check_outlier(df,col)))

def replace_with_threshold(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_threshold(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != 'SalePrice':
        print(col, (check_outlier(df,col)))



###########################################
# Missing Value Analysis
###########################################
def missing_values_table(dataframe,na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss=dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df=pd.concat([n_miss,np.round(ratio,2)],axis=1,keys=['n_miss','ratio'])
    print(missing_df,end='\n')
    if na_name:
        return na_columns

missing_values_table(df)

df['Alley'].value_counts()

# Blank values in some variables indicate that the house does not have that feature.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]


for col in no_cols:
    df[col].fillna('No',inplace=True)

missing_values_table(df)

# This function fills in missing values with median or mean.

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Variables with missing values are listed

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Number of missing values of variables before implementation

    # If the variable object and class count is equal to or less than cat_length, fill the empty values with mode
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # If num_method is mean, empty values of variables that are not of type object are filled with mean.
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # If num_method is median, empty values of variables that are not of type object are filled with the mean.
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df=quick_missing_imp(df,num_method='median',cat_length=17)

######################################
# RARE Analysis
######################################

# Examining the distribution of categorical columns
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

# Inclusion of classes into other classes according to their ratios

# ExterCond : 5
#     COUNT  RATIO  TARGET_MEAN
# Ex     12  0.004   201333.333
# Fa     67  0.023   102595.143
# Gd    299  0.102   168897.568
# Po      3  0.001    76500.000
# TA   2538  0.869   184034.896


df["ExterCond"] = np.where(df.ExterCond.isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df.ExterCond.isin(["Ex", "Gd"]), "Ex", df["ExterCond"])



# LotShape : 4
#      COUNT  RATIO  TARGET_MEAN
# IR1    968  0.332   206101.665
# IR2     76  0.026   239833.366
# IR3     16  0.005   216036.500
# Reg   1859  0.637   164754.818

df["LotShape"] = np.where(df.LotShape.isin(["IR1", "IR2", "IR3"]), "IR", df["LotShape"])



# GarageQual : 5
#     COUNT  RATIO  TARGET_MEAN
# Ex      3  0.001   241000.000
# Fa    124  0.042   123573.354
# Gd     24  0.008   215860.714
# Po      5  0.002   100166.667
# TA   2763  0.892   187489.836

df["GarageQual"] = np.where(df.GarageQual.isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["Ex", "Gd", "TA"]), "ExGd", df["GarageQual"])


# BsmtFinType2 : 6
#      COUNT  RATIO  TARGET_MEAN
# ALQ     52  0.018   209942.105
# BLQ     68  0.023   151101.000
# GLQ     34  0.012   180982.143
# LwQ     87  0.030   164364.130
# Rec    105  0.036   164917.130
# Unf   2493  0.854   184694.690

df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])


# Identification of rare classes
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df,0.01)

######################################
# Create new variables and add 'NEW' to the beginning of the new variables you create.
######################################

df["NEW_1st*GrLiv"] = (df["1stFlrSF"]*df["GrLivArea"])

df["NEW_Garage*GrLiv"] = (df["GarageArea"]*df["GrLivArea"])

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].apply(pd.to_numeric, errors='coerce').sum(axis = 1)

df["TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis = 1)

df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis = 1)

df["Exter"] = df[["ExterQual", "ExterCond"]].sum(axis = 1)

df["Qual"] = df[["OverallQual", "ExterQual", "GarageQual", "Fence", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu"]].apply(pd.to_numeric, errors='coerce').sum(axis = 1)

df["Cond"] = df[["OverallCond", "ExterCond", "GarageCond", "BsmtCond", "HeatingQC", "Functional"]].apply(pd.to_numeric, errors='coerce').sum(axis = 1)

# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

df["NEW_TotalFullBath"] = df.BsmtFullBath + df.FullBath
df["NEW_TotalHalfBath"] = df.BsmtHalfBath + df.HalfBath

df["NEW_TotalBath"] = df["NEW_TotalFullBath"] + (df["NEW_TotalHalfBath"]*0.5)

# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)

# LowQualFinSF
df["NEW_LowQualFinSFRatio"] = df.LowQualFinSF / df.NEW_TotalHouseArea

df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]

# Overall kitchen score
df["NEW_KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
# Overall fireplace score
df["NEW_FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt



drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# Dropping variables in drop_list
df.drop(drop_list, axis=1, inplace=True)

######################################
# Apply Label Encoding & One-Hot Encoding operations.
######################################

cat_cols,num_cols,cat_but_car=grab_col_names(df)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes =='O' and len(df[col].unique())==2]

for col in binary_cols:
    df=label_encoder(df,col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape

######################################
# Modeling
######################################

train_df=df[df['SalePrice'].notnull()]
test_df=df[df['SalePrice'].isnull()]

y=train_df['SalePrice']
X=train_df.drop(['Id','SalePrice'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 43013.196 (LR)
# RMSE: 32719.1862 (Ridge)
# RMSE: 42937.5864 (Lasso)
# RMSE: 34473.2607 (ElasticNet)
# RMSE: 47372.2042 (KNN)
# RMSE: 40805.4634 (CART)
# RMSE: 29039.7766 (RF)
# RMSE: 25746.642 (GBM)
# RMSE: 28424.8469 (XGBoost)
# RMSE: 28361.3714 (LightGBM)

##################
# Hyperparameter Optimization
##################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               #"colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=1,
                            verbose=False).fit(X_train, y_train)
lgbm_gs_best.best_params_
final_model=lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X,y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

###########################################
# Feature Importance
###########################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X,30)

