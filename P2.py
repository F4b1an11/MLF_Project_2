import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_regression(file_path, dims, file_name):
    #Load dataset, dropping NaNs to handle gaps
    df = pd.read_csv(file_path, header = None).dropna()

    X = df.iloc[:,:dims]
    Y = df.iloc[:,dims]

    #Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Fit
    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    #Calculate R2
    train_r2 = r2_score(Y_train, y_train_pred)
    test_r2 = r2_score(Y_test, y_test_pred)

    #Calculate Residual Standard Error(RSE)
    n_train = len(Y_train)                             #number of samples 
    p = X_train.shape[1]                              #number of predictors
    train_rss = np.sum((Y_train - y_train_pred)**2)   #Residual Sum of Squares
    train_rse = np.sqrt(train_rss / (n_train - p - 1))

    n_test = len(Y_test)
    test_rss = np.sum((Y_test - y_test_pred)**2)
    test_rse = np.sqrt(test_rss / (n_test - p - 1))

    #MSE
    train_mse = mean_squared_error(Y_train, y_train_pred)
    test_mse = mean_squared_error(Y_test, y_test_pred)

    # Retrieve parameters
    intercept = linreg.intercept_
    coefficients = linreg.coef_

    # Build fit equation
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f})b{i+1}"
   

    print(f"Results for {file_name}")
    print(f"Train RSE: {train_rse:.4f}")
    print(f"Test RSE: {test_rse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Equation: {equation}\n")


    # Save data so it can be written to a cvs later
    results.append({
        "dataset": file_name,
        "train_RSE": train_rse,
        "test_RSE": test_rse,
        "train_R2": train_r2,
        "test_R2": test_r2,
        "train_MSE": train_mse,
        "test_MES": test_mse,
        "equation": equation
    })


    if dims == 1:

        plt.scatter(X_train, Y_train)

        # Create a line so the predicted fit will look smooth
        x_line = np.linspace(X.min(), X.max(), 20).reshape(-1, 1)
        y_line = linreg.predict(x_line)

        plt.plot(x_line, y_line, color='black')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{file_name} Regression Line")

        plt.savefig(f"{file_name}/{file_name}_data.png")
        plt.clf()


    if dims == 2:

        x1 = X_train.iloc[:, 0]
        x2 = X_train.iloc[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, x2, Y_train)

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

        x1 = X_train.iloc[:, 0]
        x2 = X_train.iloc[:, 1]
        x3 = X_train.iloc[:, 2]

        # Get slice values for x3
        x3_slices = np.linspace(x3.min(), x3.max(), 3)

        for i, val in enumerate(x3_slices):

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x1, x2, Y_train, alpha=0.6)

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
    plt.scatter(Y_test, y_test_pred)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="black")

    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title(f"{file_name} Predicted vs Actual")

    plt.savefig(f"{file_name}/{file_name}_PvA.png")
    plt.clf()


    # Residual Plot
    plt.scatter(y_test_pred, Y_test - y_test_pred)
    plt.hlines(0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), color="black")
    
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

results = []

for file, dims, name in datasets:
    evaluate_regression(file, dims, name)

results_df = pd.DataFrame(results)
results_df.to_csv("Project2_data.csv", index=False)
