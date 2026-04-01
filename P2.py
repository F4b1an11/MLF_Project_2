import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

def evaluate_regression(file_path, dims, file_name):
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

    # Retrieve parameters
    intercept = linreg.intercept_
    coefficients = linreg.coef_

    # Build fit equation
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f})b{i+1}"
   
    print(f"Results for {file_name}")
    print(f"RSE: {rse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Equation: {equation}\n")



    if dims == 1:

        plt.scatter(X, Y)
        plt.plot(X, y_pred, color='black')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{file_name} Regression Line")

        plt.savefig(f"{file_name}/{file_name}_data.png")
        plt.clf()


    if dims == 2:

        x1 = X.iloc[:, 0]
        x2 = X.iloc[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, x2, Y)

        # Create a grid to get the plane to be smooth
        x1_range = np.linspace(x1.min(), x1.max(), 20)
        x2_range = np.linspace(x2.min(), x2.max(), 20)

        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

        # Predict the values for the plane
        grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        y_grid = linreg.predict(grid_points).reshape(x1_grid.shape)

        # Plot the regression
        ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        plt.title(f"{file_name} Regression Plane")
        #ax.view_init(elev=20, azim=45)

        plt.savefig(f"{file_name}/{file_name}_data.png")
        plt.clf()


    if dims == 3:
        # For 3D regression, hold x3 constant at several values
        # and make a 3D plot for each of those

        x1 = X.iloc[:, 0]
        x2 = X.iloc[:, 1]
        x3 = X.iloc[:, 2]

        # Get slice values for x3
        x3_slices = np.linspace(x3.min(), x3.max(), 3)

        for i, val in enumerate(x3_slices):

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x1, x2, Y, alpha=0.6)

            # Create a grid of points to make the predicted fit plane smooth
            x1_range = np.linspace(x1.min(), x1.max(), 20)
            x2_range = np.linspace(x2.min(), x2.max(), 20)

            x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
            grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel(), np.full(x1_grid.size, val)]
            y_grid = linreg.predict(grid_points).reshape(x1_grid.shape)

            # Plot the regression plane slice
            ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4)

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")

            plt.title(f"{file_name} slice at x3={val:.2f}")
            plt.savefig(f"{file_name}/{file_name}_slice_{i}.png")
            plt.clf()



    # Predicted vs Actual values
    # This shows how accurate the model is
    plt.scatter(Y, y_pred)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="black")

    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title(f"{file_name} Predicted vs Actual")

    plt.savefig(f"{file_name}/{file_name}_PvA.png")
    plt.clf()


    # Residual Plot
    plt.scatter(y_pred, Y - y_pred)
    plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), color="black")
    
    plt.xlabel("Predicted Y")
    plt.ylabel("Residuals")
    plt.title(f"{file_name} Residuals")

    plt.savefig(f"{file_name}/{file_name}_Residual.png")
    plt.clf()

#Execution
datasets = [
    ("1D L.csv", 1, "1DL"), ("1D M.csv", 1, "1DM"), ("1D H.csv", 1, "1DH"),
    ("2D L.csv", 2, "2DL"), ("2D M.csv", 2, "2DM"), ("2D H.csv", 2, "2DH"),
    ("3D L.csv", 3, "3DL"), ("3D M.csv", 3, "3DM"), ("3D H.csv", 3, "3DH"),
]

for file, dims, name in datasets:
    evaluate_regression(file, dims, name)
