from src.features.preprocessor import impute_features, encode_features


def run_preprocessing(x_train, x_test, num_features: list, cat_features: list):
    x_train, x_test = impute_features(x_train, x_test, num_features, cat_features)
    x_train, x_test = encode_features(x_train, x_test, cat_features)
    return x_train, x_test
