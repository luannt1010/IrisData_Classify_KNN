import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")
X = df.drop(["Species", "Id"], axis=1)
y = np.array(df["Species"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
# Convert to numpy array
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train.shape)

# Pie chart for count each species
species_count = df["Species"].value_counts().to_list()
species_list = df["Species"].unique()
plt.pie(x=species_count, labels=species_list, autopct='%1.1f%%')
plt.show()

# Histogram for distribution for each feature
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for feature in features:
    sns.histplot(data=df, x=feature, hue="Species", kde=True, palette="Set1", bins=15)
    plt.show()

def euclid_distance(training_point, test_point):
    distance = np.sqrt((np.sum(np.square(training_point - test_point))))
    return distance

def knn_predict(training_data, training_label, test_point, k):
    distances = []
    predictions = []
    for i in range(len(training_data)):
        dist = euclid_distance(training_point=training_data[i], test_point=test_point)
        distances.append((dist, training_label[0, i]))
    distances.sort(key=lambda x: x[0])
    for i in range(k):
        predictions.append(distances[i][-1])
    return predictions

def get_final_answer(predictions):
    frequency = {}
    for label in predictions:
        if label not in frequency:
            frequency[label] = 1
        else:
            frequency[label] += 1
    return max(frequency, key=lambda x: frequency[x])

def predict(X_train, X_test, y_train, k):
    predicts = []
    for i in range(len(X_test)):
        test_point = X_test[i]
        predictions = knn_predict(X_train, y_train, test_point, k=k)
        predict = get_final_answer(predictions)
        predicts.append(predict)
    return predicts

def accuracy(y_true, predicts):
    return accuracy_score(y_true=y_true, y_pred=predicts)

def optimize_k(X_train, X_test, y_train, y_test, loss=False):
    ks = [i for i in range(1, X_train.shape[0] + 1)]
    y_true = np.squeeze(y_test)
    best_k = {"k": [], "score": []}
    for k in ks:
        predicts = predict(X_train, X_test, y_train, k)
        if not loss:
            score = accuracy(y_true, predicts)
        else:
            score = np.mean(y_true != predicts)
        best_k["k"].append(k)
        best_k["score"].append(score)
    return best_k

loss = True
dic = optimize_k(X_train, X_test, y_train, y_test, loss)
k = dic["k"]
score = dic["score"]
label = None
if not loss:
    idx_best = np.argmax(score)
    label = "Score"
else:
    idx_best = np.argmin(score)
    label = "Loss"
k_best = k[idx_best]
score_best = score[idx_best]
plt.plot(k, score, color="blue", linestyle="-", linewidth=2, label=label)
plt.scatter(k_best, score_best, color='red', s=50, label=f'Best K={k_best}, Score={score_best}')
plt.annotate(f"Best K={k_best}\nScore={score_best}",
             xy=(k_best, score_best), xytext=(k_best+0.5, score_best-0.02))
plt.title(f"{label} at value K")
plt.xlabel("K")
plt.ylabel(label)
plt.legend()
plt.tight_layout()
plt.show()

