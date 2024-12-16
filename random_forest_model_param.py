import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

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

# 合并训练集和测试集进行编码
df_combined = pd.concat(
    [train_df[categorical_columns], test_df[categorical_columns]], axis=0)

# 2. 对类别变量进行编码
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df_combined[column] = le.fit_transform(df_combined[column])
    label_encoders[column] = le

# 将合并后的数据重新分割回训练集和测试集
train_df[categorical_columns] = df_combined.iloc[:len(train_df), :]
test_df[categorical_columns] = df_combined.iloc[len(train_df):, :]

# 3. 处理目标变量 NObeyesdad
y = train_df['NObeyesdad']
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 4. 特征选择
X = train_df.drop(['id', 'NObeyesdad'], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 5. 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择：使用 SelectKBest 选择最重要的特征
selector = SelectKBest(f_classif, k=10)  # 选择前10个最重要的特征
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

# 7. 交叉验证和参数调优
param_grid = {
    'n_estimators': [100, 200, 500],  # 森林中的树数量
    'max_depth': [5, 10, 20, None],   # 最大深度
    'min_samples_split': [2, 5, 10],  # 节点分裂所需最小样本数
    'min_samples_leaf': [1, 2, 4],    # 叶节点所需最小样本数
    'max_features': ['sqrt', 'log2']  # 每次分裂时的最大特征数
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    verbose=2,
    n_jobs=1  # 不使用多线程
)

# 拟合并输出最佳参数
grid_search.fit(X_selected, y)
best_params = grid_search.best_params_
print("最佳参数:", best_params)

# 使用最佳参数的模型
best_model = grid_search.best_estimator_

# 交叉验证评分
cv_scores = cross_val_score(best_model, X_selected, y, cv=5)
print("\n交叉验证得分:", cv_scores)
print(
    "平均交叉验证准确率: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std() * 2))

# 在整个训练集上训练最佳模型
best_model.fit(X_selected, y)

# 在测试集上进行预测
y_pred = best_model.predict(X_test_selected)

# 将预测结果还原为类别标签
y_pred_label = le_target.inverse_transform(y_pred)

# 保存预测结果到 sample_submission.csv
sample_submission_df['NObeyesdad'] = y_pred_label
sample_submission_df.to_csv('sample_submission.csv', index=False)

print("\n预测完成，结果已保存到 sample_submission.csv")
