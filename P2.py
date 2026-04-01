import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

def evaluate_regression(file_path, dims):
    #Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(file_path, header = None).dropna()

    X = df.iloc[:,:dims]
    Y = df.iloc[:,dims]

    #Fit
    linreg = LinearRegression()
    linreg.fit(X,Y)
    y_pred = linreg.predict(X)

    #Calculate R2
    r2 = r2_score(Y,y_pred)

    #Calculate Residual Standard Error(RSE)
    n = len(Y)                      #number of samples 
    p = X.shape[1]                  #number of predictors
    rss = np.sum((Y - y_pred)**2)   #Residual Sum of Squares
    rse = np.sqrt(rss / (n - p - 1))

    #MSE
    mse = mean_squared_error(Y,y_pred)
   
    print(f"Results for {file_path}")
    print(f"RSE: {rse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MSE: {mse:.4f}\n")

#Execution
datasets = [
    ("1D L.csv", 1), ("1D M.csv", 1), ("1D H.csv", 1),
    ("2D L.csv", 2), ("2D M.csv", 2), ("2D H.csv", 2),
    ("3D L.csv", 3), ("3D M.csv", 3), ("3D H.csv", 3),
]

for file, dims in datasets:
    evaluate_regression(file, dims)
