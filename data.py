import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold


class Data:

    data = None

    def __init__(self, path, scaler):
        """
        Preprocess the data as requested
        :param path: the path to the file from which we take the data
        """
        col_list = ["age", "workclass", "education", "education-num", "martial-status", "occupation", "relationship",
                    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

        # Read only the wanted columns from the file
        df = pandas.read_csv(path, usecols=col_list)

        # Delete all rows in which there's a question mar
        df.replace(r'\s*\?\s*', pandas.NaT, regex=True, inplace=True)
        df.dropna(inplace=True)

        # Turn the categorical columns to vectors
        categorical_list = ["workclass", "education", "martial-status", "occupation", "relationship", "race",
                            "native-country", "sex"]
        encoded = pandas.get_dummies(df, columns=categorical_list)

        for categorical in categorical_list:
            df = df.drop(categorical, axis=1)
        for col in col_list:
            if col in encoded.keys():
                encoded = encoded.drop(col, axis = 1)
        df = pandas.concat([df, encoded], axis=1)

        # Turn the salary values to 0 or 1
        le = LabelEncoder()
        df['salary'] = le.fit_transform(df['salary'])

        # Normalize the continuous columns
        scaled_columns = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        for scaled_column in scaled_columns:
            df[[scaled_column]] = scaler.fit_transform(df[[scaled_column]])
        self.data = df

    def split_to_k_folds(self, k):
        """
        Split the data to K folds
        :param k: the number of wanted folds
        :return: KFold object
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=10)
        return kf







