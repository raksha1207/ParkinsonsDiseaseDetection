import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(path):
    parkinsons_data = pd.read_csv(path)

    X = parkinsons_data.drop(['total_UPDRS', 'motor_UPDRS'], axis=1)
    y = parkinsons_data.loc[:, 'total_UPDRS']

    # y2 is also a response variable but right now we are predicting only total_UPDRS score
    y2 = parkinsons_data.loc[:, 'motor_UPDRS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, y_train, X_test, y_test


def extract_features(X_train, X_test):

    scalar = StandardScaler()
    scalar.fit(X_train)

    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    pca = PCA(0.95)
    pca.fit(X_train)

    print(pca.n_components_)
    # print(pca.components_)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

