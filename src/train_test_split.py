# from sklearn.model_selection import train_test_split
# import json
# from config.path_config import (
#     ALL_DATA_FILE_PATH ,
#     TRAIN_DATA_FILE_PATH,
#     TEST_DATA_FILE_PATH , 
#     VAL_DATA_FILE_PATH,
#     SAMPLE_DATA_FILE_PATH,
# )
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer


# class Train_Test_Val:

#     @staticmethod
#     def split_data_to_json(data, test_size=0.2, val_size=0.1, random_state=42):

#         # train + temp split
#         train, temp = train_test_split(
#             data,
#             test_size=(test_size + val_size),
#             random_state=random_state
#         )

#         # val + test split
#         val, test = train_test_split(
#             temp,
#             test_size=test_size / (test_size + val_size),
#             random_state=random_state
#         )

#         return {
#             "train": train,
#             "val": val,
#             "test": test
#         }


# with open(SAMPLE_DATA_FILE_PATH, "r") as f:
#     data = json.load(f)

# splits = Train_Test_Val.split_data_to_json(data)

# with open(TRAIN_DATA_FILE_PATH, "w") as f:
#     json.dump(splits["train"], f, indent=2)

# with open(TEST_DATA_FILE_PATH, "w") as f:
#     json.dump(splits["val"], f, indent=2)

# with open(VAL_DATA_FILE_PATH, "w") as f:
#     json.dump(splits["test"], f, indent=2)

# class Vectorised_data:

#     @staticmethod
#     def vectorise_data(data):
#         df = pd.read_json(data)
#         vectorizer = TfidfVectorizer()
#         tfiddf_matrix = vectorizer.fit_transform


# if __name__ =="__main__":
#     data_obj = DataNormalization()
#     data_obj.load_data()