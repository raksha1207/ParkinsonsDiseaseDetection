import preprocessing


def main():
    X_train, X_test, y_train, y_test = preprocessing.load_data('./parkinsons_updrs.csv')
    X_train, X_test = preprocessing.extract_features(X_train, X_test)


if __name__ == "__main__":
    main()
