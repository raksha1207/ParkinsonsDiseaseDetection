import preprocessing
import svr

def main():
    #load and preprocess data
    X_train, y_train, X_test, y_test = preprocessing.load_data('./parkinsons_updrs.csv')
    X_train, X_test = preprocessing.extract_features(X_train, X_test)
    
    #use SVR to predict UPDRS scores
    r_squared_svr, rmse_svr = svr.svr_model(X_train, y_train, X_test, y_test)
    print("R-squared value for SVR: " + str(round(r_squared_svr,3)))
    print("RMSE for SVR: " + str(round(rmse_svr, 3)))

if __name__ == "__main__":
    main()
