import preprocessing


def main():

    X_train, y_train, X_test, y_test = preprocessing.load_data('./parkinsons_updrs.csv')
    X_train, X_test = preprocessing.extract_features(X_train, y_train)


if __name__ == "__main__":
    main()
