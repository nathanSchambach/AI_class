import math
import numpy as np
import pandas as pd
import os
from sklearn import neighbors, metrics, tree, model_selection

# folder to view

database_to_view = 'Face Database'


# distance function


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# get_features function


def get_features(in_df):
    features = [feature1(in_df), feature2(in_df), feature3(in_df), feature4(in_df),
             feature5(in_df), feature6(in_df), feature7(in_df)]
    return features


# feature 1 


def feature1(in_df):
    eye1 = distance(in_df.iloc[11], in_df.iloc[12])
    eye2 = distance(in_df.iloc[9], in_df.iloc[10])
    large_eye = max(eye2, eye1)
    face_width = distance(in_df.iloc[8], in_df.iloc[13])
    return large_eye / face_width


# feature 2 


def feature2(in_df):
    face_width = distance(in_df.iloc[8], in_df.iloc[13])
    center_eye_width = distance(in_df.iloc[0], in_df.iloc[1])
    return center_eye_width / face_width


# feature 3 


def feature3(in_df):
    nose_width = distance(in_df.iloc[15], in_df.iloc[16])
    face_width = distance(in_df.iloc[20], in_df.iloc[21])
    return nose_width / face_width


# feature 4 


def feature4(in_df):
    width = distance(in_df.iloc[2], in_df.iloc[3])
    height = distance(in_df.iloc[17], in_df.iloc[18])
    return width / height


# feature 5 


def feature5(in_df):
    lip_width = distance(in_df.iloc[2], in_df.iloc[3])
    face_width = distance(in_df.iloc[20], in_df.iloc[21])
    return lip_width / face_width


# feature 6 


def feature6(in_df):
    eyebrow1 = distance(in_df.iloc[4], in_df.iloc[5])
    eyebrow2 = distance(in_df.iloc[6], in_df.iloc[7])
    long_brow = max(eyebrow1, eyebrow2)
    face_width = distance(in_df.iloc[8], in_df.iloc[13])
    return long_brow / face_width


# feature 7 


def feature7(in_df):
    face_width = distance(in_df.iloc[20], in_df.iloc[21])
    aggressive = distance(in_df.iloc[10], in_df.iloc[19])
    return aggressive / face_width


# Blank list to make list of lists to turn into a pandas df


rows = []

for folder in os.listdir(database_to_view):
    folder_to_view = folder
    for file in os.listdir(database_to_view + "/" + folder_to_view):
        file_dir = database_to_view + "/" + folder_to_view + "/" + file  # get full file directory

        # Open the file into a pandas df skipping rows 1-3 and last since useless characters
        # Plus renaming columns to X Y since they are coordinates

        df = pd.read_csv(file_dir, sep=" ", skiprows=3, skipfooter=1, names=['x', 'y'], engine='python')
        file_features = get_features(df)  # Plug data into function to get features
        file_features.insert(0, file.replace(".pts", ""))  # Add file name as identifier
        if 'm' in file:  # Used to create the target column
            file_features.append("m")
        else:
            file_features.append("w")
        rows.append(file_features)

df_features = pd.DataFrame(rows, columns=["sample_ID", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "Target"])

df_features.sort_values("sample_ID", inplace=True, ignore_index=True)  # Sort into M and W to make test/train split
# easier

print(df_features)
print(df_features[["F4"]])
#set up KNN

m_features = df_features.query("Target == 'm'")
w_features = df_features.query("Target == 'w'")

m_data = m_features[["F1", "F2", "F3", "F4", "F5", "F6", "F7"]]
m_targets = m_features[["Target"]]

w_data = w_features[["F1", "F2", "F3", "F4", "F5", "F6", "F7"]]
w_targets = w_features[["Target"]]

x_m_train, x_m_test, y_m_train, y_m_test = model_selection.train_test_split(m_data, m_targets, test_size=.2)
x_w_train, x_w_test, y_w_train, y_w_test = model_selection.train_test_split(w_data, w_targets, test_size=.2)

train_data_m = df_features.iloc[0:16, 1:7]
train_data_w = df_features.iloc[20:36, 1:7]

test_data_m = df_features.iloc[16:20, 1:7]
test_data_w = df_features.iloc[36:40, 1:7]

train_target_m = df_features.iloc[0:12, 8]
train_target_w = df_features.iloc[20:36, 8]

test_target_m = df_features.iloc[12:20, 8]
test_target_w = df_features.iloc[36:40, 8]

train_data = pd.concat((x_m_train, x_w_train), ignore_index=True)
train_target = pd.concat((y_m_train, y_w_train), ignore_index=True)
test_data = pd.concat((x_m_test, x_w_test), ignore_index=True)
test_target = pd.concat((y_m_test, y_w_test), ignore_index=True)

#print(train_data)
#print(train_target)
#print(test_data)
#print(test_target)

error_rate = []

for a in range(1, 16):
    knn = neighbors.KNeighborsClassifier(n_neighbors=a)
    knn.fit(train_data, train_target)
    preds = knn.predict(test_data)
    error_rate.append(np.mean(preds != list(test_target["Target"])))

n_neighbors = error_rate.index(min(error_rate))

nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(train_data, train_target)
nn_pr = nn.predict(test_data)

print(nn_pr)
print(test_target)

# Print based on K value 15

print(metrics.confusion_matrix(test_target, nn_pr))
print(metrics.classification_report(test_target, nn_pr))

# Now use sklearn decision tree

face_classifier = tree.DecisionTreeClassifier(criterion='entropy')
face_classifier.fit(train_data, train_target)
tree_pr = face_classifier.predict(test_data)

print(metrics.confusion_matrix(test_target, tree_pr))
print(metrics.classification_report(test_target, tree_pr))
