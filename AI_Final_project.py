import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

# 資料處理
loan_dataframe = pd.read_json("loan_approval_dataset.json")
print(loan_dataframe.shape)
print(loan_dataframe.info())
print(loan_dataframe.describe())
print(loan_dataframe.isna().sum()) #檢查有沒有missing value

# 職業分析
plt.rc("font", family="Microsoft JhengHei")
# plt.figure(figsize=(10,20))
# sns.countplot(y=loan_dataframe["Profession"])
# plt.title("不同職業的頻率")
# plt.ylabel("職業")
# plt.xlabel("Count")
# plt.show()

#  Risk分析
loan_change_risk_flag = loan_dataframe["Risk_Flag"].replace({0: "Yes", 1: "No"}).value_counts()
plt.figure()
plt.pie(loan_change_risk_flag.values,
        labels=loan_change_risk_flag.index,
        startangle=90,
        autopct="%1.2f%%",
        labeldistance=None
        )
plt.legend()
plt.title("透過用戶行為決定是否借貸")
plt.tight_layout()
plt.show()

#會製熱圖觀察各欄位間的相關性
correlation_matrix = loan_dataframe.corr(numeric_only=True)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("各屬性相關性熱圖")
plt.show()

# Data analysis
strong_correlations = correlation_matrix[abs(correlation_matrix) > 0.2].stack().reset_index()
strong_correlations.columns = ["Attribute1", "Attribute2", "Correlation"]
strong_correlations = strong_correlations[strong_correlations["Attribute1"] != strong_correlations["Attribute2"]]
print("強相關性的屬性:")
print(strong_correlations)

#Machine Learning with random forest classfication

object = loan_dataframe.select_dtypes(include="object").columns
print("object type columns: ", object)

loan_dataframe["Married/Single"] = loan_dataframe["Married/Single"].replace({"single": 1, "married": 0})
loan_dataframe["House_Ownership"] = loan_dataframe["House_Ownership"].replace({"rented": 2, "norent_noown": 1,"owned":0})
loan_dataframe["Car_Ownership"] = loan_dataframe["Car_Ownership"].replace({"yes":1, "no":0})

features =[ 'Profession','CITY', 'STATE']
le = LabelEncoder()
for col in features:
    loan_dataframe[col]= le.fit_transform(loan_dataframe[col])

loan_dataframe = loan_dataframe.drop(["Id"],axis=1)   
# print(loan_dataframe.head())

X = loan_dataframe.drop(["Risk_Flag"],axis=1)

y= loan_dataframe["Risk_Flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Prediction Results of Loan DataFrame")
df_loan = pd.DataFrame({"Actual Values":y_test, "Random Forest":prediction})
print(df_loan.head())


#model accuracy

model_accuracy = accuracy_score(y_test, prediction)

# 定義圖型
cmap = sns.cubehelix_palette(dark=0.4, light=0.8, as_cmap=True)

# 計算混淆矩陣

confusion_metrix = metrics.confusion_matrix(y_test, prediction)

# model figure
plt.figure(figsize=(20, 20))
sns.heatmap(confusion_metrix, annot=True, fmt="d", cmap=cmap, cbar=False, square=True, xticklabels=False, yticklabels=False)
plt.xlabel("Prediction")
plt.ylabel("True")
plt.title("Confusion Metrix")
plt.tight_layout()
plt.show()

print("Accuracy of Random Forest Model", model_accuracy*100)
print("\n模型評估:\n")
print(metrics.classification_report(y_test, prediction))


