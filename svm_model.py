import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 读取训练集、测试集和提交文件
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# 数据预处理
# 1. 找到所有类别变量列（排除目标列'NObeyesdad'）
categorical_columns = train_df.select_dtypes(
    include=['object']).columns.tolist()
categorical_columns = [
    col for col in categorical_columns if col != 'NObeyesdad']

# 合并训练集和测试集进行统一编码
df_combined = pd.concat(
    [train_df[categorical_columns], test_df[categorical_columns]], axis=0)

# 2. 对类别变量进行编码
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    # 对训练集和测试集进行统一编码
    df_combined[column] = le.fit_transform(df_combined[column])
    label_encoders[column] = le  # 保存编码器以便后续使用

# 将合并后的数据重新分割回训练集和测试集
train_df[categorical_columns] = df_combined.iloc[:len(train_df), :]
test_df[categorical_columns] = df_combined.iloc[len(train_df):, :]

# 3. 处理目标变量 NObeyesdad（标签编码）
y_train = train_df['NObeyesdad']
le_target = LabelEncoder()
y_train = le_target.fit_transform(y_train)

# 4. 特征选择（去除目标列NObeyesdad和id列）
X_train = train_df.drop(['id', 'NObeyesdad'], axis=1)
X_test = test_df.drop(['id'], axis=1)  # 测试集没有NObeyesdad

# 5. 初始化支持向量机分类器
# 使用线性核函数，你可以根据需要选择 'linear'、'rbf'、'poly' 等
svm_model = SVC(kernel='linear', random_state=42)

# 6. 训练模型
svm_model.fit(X_train, y_train)

# 7. 在测试集上进行预测
y_pred = svm_model.predict(X_test)

# 将预测结果还原为类别标签
y_pred_label = le_target.inverse_transform(y_pred)

# 8. 保存预测结果到 sample_submission.csv
sample_submission_df['NObeyesdad'] = y_pred_label
sample_submission_df.to_csv('sample_submission.csv', index=False)

# 输出一些信息以确认
print("Prediction completed and saved to sample_submission.csv.")
