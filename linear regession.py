import numpy  as np
import pandas as pd

df = pd.read_csv("Medical_Cost_Personal_Datasets.csv")


# # # Data vandalism for repair training

# Nan
df.loc[df.sample(frac=0.05).index , "age"] = np.nan
df.loc[df.sample(frac=0.04).index , "bmi"] = np.nan

# Outliers
df.loc[df.sample(n=5).index , "age"] = 250
df.loc[df.sample(n=5).index , "bmi"] = -10


# Duplicate Rows
Duplicate = df.sample(n=15)
df = pd.concat([df , Duplicate] , ignore_index=True)

# Inconsistent Data
male = df[df["sex"] == "male"].sample(n=10).index
df.loc[male , "sex"] = "m"

# # # clean data with pd

# clean Duplicate
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)

# transformation from m to male
df["sex"] = df["sex"].replace({"m" : "male"})

# solve the Outliers
df.loc[(df["age"] < 0) | (df["age"] > 110), "age"]   = np.nan
df.loc[(df["bmi"] < 0) | (df["bmi"] > 90)  , "bmi" ] = np.nan

# NAN = mean 
df["age"] = df["age"].fillna(df["age"].mean())
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())



# transformation from str to int
df["sex"]    = df["sex"].map({"male" : 1 , "female" : 0})
df["smoker"] = df["smoker"].map({"yes" : 1 , "no" : 0})
df           = pd.get_dummies(df , columns=["region"],drop_first=True)

# Reveal strange numbers
print(f"NAN {df.isnull().sum()}")



x     = df.drop(columns=["charges"])
print(f"std {x.std()}")
print(f"head {df.head()}")
y     = df["charges"].values.reshape(-1 , 1)
x     = (x - x.mean()) / x.std()
print(x)
alpha = 0.01
loop  = 100000
m , n = x.shape
print(m ,n)
w     = np.zeros((n , 1)) 
b     = 0
for z in range(loop):
    y_pre = np.dot(x , w) + b
    Error = y_pre - y
    cost  = np.mean(Error ** 2) / 2
    dw    = np.dot(x.T , Error) / m
    db    = Error.sum() / m
    w    -= alpha * dw
    b    -= alpha * db
    if z % 100 == 0 :
        print(cost)
for i in range(10):
    print(f"predicted: {(np.dot(x,w) + b)[i][0]:.2f} | actual {y[i][0]:.2f}")