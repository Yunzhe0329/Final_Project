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
print(loan_dataframe.shape)   #252000行13列
print(loan_dataframe.info())   #概要資訊，count non-null代表非空值
print(loan_dataframe.describe())  #describe只會顯示數值
print(loan_dataframe.isna().sum()) #檢查有沒有missing value


# 職業分析
plt.rc("font", family="Microsoft JhengHei")  #設定字體為 Microsoft JhengHei
# plt.figure(figsize=(10,20))
# sns.countplot(y=loan_dataframe["Profession"])
# plt.title("不同職業的頻率")
# plt.ylabel("職業")
# plt.xlabel("Count")
# plt.show()

#  Risk_Flag分析
loan_change_risk_flag = loan_dataframe["Risk_Flag"].replace({0: "Yes", 1: "No"}).value_counts()
plt.figure()
plt.pie(loan_change_risk_flag.values,
        labels=loan_change_risk_flag.index,
        startangle=90,
        autopct="%1.2f%%", #autopct 設定餅圖每個區域的百分比值
        labeldistance=None #label和餅圖中心的距離 Default 為 1.1
        )
plt.legend()
plt.title("透過用戶行為決定是否借貸")
plt.tight_layout()
plt.show()

#繪製熱圖觀察各欄位間的相關性
correlation_matrix = loan_dataframe.corr(numeric_only=True)  #使用corr方法計算彼此之間的相關係數，numeric_only=True只考慮數值型的欄位
plt.figure(figsize=(10, 10))  #10*10的格式
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")  #圖形的呈現樣式
plt.title("各屬性相關性熱圖")  
plt.show()

# Data analysis(進行數值化)
strong_correlations = correlation_matrix[abs(correlation_matrix) > 0.2].stack().reset_index()
strong_correlations.columns = ["Attribute1", "Attribute2", "Correlation"]
strong_correlations = strong_correlations[strong_correlations["Attribute1"] != strong_correlations["Attribute2"]]
print("強相關性的屬性:")
print(strong_correlations)

#Machine Learning with random forest classfication

object = loan_dataframe.select_dtypes(include="object").columns  #變數儲存
print("object type columns: ", object)

loan_dataframe["Married/Single"] = loan_dataframe["Married/Single"].replace({"single": 1, "married": 0})  #資料正規化，用數字代替文字
loan_dataframe["House_Ownership"] = loan_dataframe["House_Ownership"].replace({"rented": 2, "norent_noown": 1,"owned":0})  #資料正規化，用數字代替文字  
loan_dataframe["Car_Ownership"] = loan_dataframe["Car_Ownership"].replace({"yes":1, "no":0})  #資料正規化，用數字代替文字

features =[ 'Profession','CITY', 'STATE']   #定義標準
le = LabelEncoder()  #對欄位進行標籤編碼
for col in features:
    loan_dataframe[col]= le.fit_transform(loan_dataframe[col])

loan_dataframe = loan_dataframe.drop(["Id"],axis=1)   
# print(loan_dataframe.head())

X = loan_dataframe.drop(["Risk_Flag"],axis=1)

y= loan_dataframe["Risk_Flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  #將資料分成訓練集和測試集，訓練集占 70%，測試集占 30%

model = RandomForestClassifier()  #創建了一個隨機森林分類器模型
model.fit(X_train, y_train)  #使用訓練集進行模型訓練

prediction = model.predict(X_test)  #使用訓練好的模型對測試集進行預測

print("Prediction Results of Loan DataFrame") 
df_loan = pd.DataFrame({"Actual Values":y_test, "Random Forest":prediction})  #將預測結果和實際值組成一個 DataFrame df_loan，用於展示預測結果
print(df_loan.head())  #輸出預測結果的前幾行，顯示實際值和隨機森林模型預測值的對比
#print(df_loan)

#model accuracy

model_accuracy = accuracy_score(y_test, prediction)


# 定義圖型
cmap = sns.cubehelix_palette(dark=0.4, light=0.8, as_cmap=True)  #定義顏色

# 計算混淆矩陣

confusion_metrix = metrics.confusion_matrix(y_test, prediction)

# model figure
plt.figure(figsize=(20, 20)) 
sns.heatmap(confusion_metrix, annot=True, fmt="d", cmap=cmap, cbar=False, square=True, xticklabels=False, yticklabels=False)
plt.xlabel("Prediction")   #設置 x 軸和 y 軸的標籤
plt.ylabel("True")  #設置 x 軸和 y 軸的標籤
plt.title("Confusion Metrix")
plt.tight_layout()
plt.show()

print("Accuracy of Random Forest Model", model_accuracy*100)  #準確率89.91%(when test_size=0.3, random_state=42)
print("\n模型評估:\n")
print(metrics.classification_report(y_test, prediction))  #做總結

