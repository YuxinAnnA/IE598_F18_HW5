import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns


# import dataset and split it into training and testing
df_wine = pd.read_csv('/Users/yuxin/Desktop/course/2018Fall/IE598/HW/HW5/wine.csv')
X, y = df_wine.iloc[:, 0:13].values, df_wine['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# standardize X_train and X_test
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Part 1: EDA
cols = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
        'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity','Hue',
        'OD280/OD315 of diluted wines','Proline']
sns.pairplot(df_wine[cols])
plt.tight_layout()
plt.show()
cm = np.corrcoef(df_wine[cols].values.T)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 6},
                 yticklabels = cols,
                 xticklabels = cols)
plt.show()

# Part 2: logistic regression baseline
lr = LogisticRegression(C=0.5, random_state=42)
lr.fit(X_train_std, y_train)
train_pred_lr = lr.predict(X_train_std)
test_pred_lr = lr.predict(X_test_std)
print accuracy_score(y_train, train_pred_lr)
print accuracy_score(y_test, test_pred_lr)

# Part 2: SVM baseline
svm = SVC(kernel='linear', random_state=42, C=0.5)
svm.fit(X_train_std, y_train)
train_pred_svm = svm.predict(X_train_std)
test_pred_svm = svm.predict(X_test_std)
print accuracy_score(y_train, train_pred_svm)
print accuracy_score(y_test, test_pred_svm)

# Part 3: PCA then logistic regression
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
train_pred_lr_pca = lr.predict(X_train_pca)
test_pred_lr_pca = lr.predict(X_test_pca)
print accuracy_score(y_train, train_pred_lr_pca)
print accuracy_score(y_test, test_pred_lr_pca)

# Part 3: PCA then SVM
svm = SVC(kernel='linear', random_state=42, C=0.5)
svm.fit(X_train_pca, y_train)
train_pred_svm_pca = svm.predict(X_train_pca)
test_pred_svm_pca = svm.predict(X_test_pca)
print accuracy_score(y_train, train_pred_svm_pca)
print accuracy_score(y_test, test_pred_svm_pca)

# Part 4: LDA then logistic regression
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda, y_train)
train_pred_lr_lda = lr.predict(X_train_lda)
test_pred_lr_lda = lr.predict(X_test_lda)
print accuracy_score(y_train, train_pred_lr_lda)
print accuracy_score(y_test, test_pred_lr_lda)

# Part 4: LDA then SVM
svm = SVC(kernel='linear', random_state=42, C=0.5)
svm.fit(X_train_lda, y_train)
train_pred_svm_lda = svm.predict(X_train_lda)
test_pred_svm_lda = svm.predict(X_test_lda)
print accuracy_score(y_train, train_pred_svm_lda)
print accuracy_score(y_test, test_pred_svm_lda)

# Part 5: KPCA
gamma = np.arange(0.1, 4, 0.05)
score_lr_train = np.empty(len(gamma))
score_svm_train = np.empty(len(gamma))
score_lr_test = np.empty(len(gamma))
score_svm_test = np.empty(len(gamma))

for i,k in enumerate(gamma):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=k)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    # logistic regression
    lr = LogisticRegression(C=0.5, random_state=42)
    lr.fit(X_train_kpca, y_train)
    train_pred_lr_kpca = lr.predict(X_train_kpca)
    test_pred_lr_kpca = lr.predict(X_test_kpca)
    # svm
    svm = SVC(kernel='linear', random_state=42, C=0.5)
    svm.fit(X_train_kpca, y_train)
    train_pred_svm_kpca = svm.predict(X_train_kpca)
    test_pred_svm_kpca = svm.predict(X_test_kpca)
    # accuracy score
    score_lr_train[i]=accuracy_score(y_train, train_pred_lr_kpca)
    score_lr_test[i] = accuracy_score(y_test, test_pred_lr_kpca)
    score_svm_train[i] = accuracy_score(y_train, train_pred_svm_kpca)
    score_svm_test[i]= accuracy_score(y_test, test_pred_svm_kpca)
plt.plot(gamma, score_lr_train, label='Logistic regression-Training')
plt.plot(gamma, score_lr_test, label='Logistic regression-Testing')
plt.legend()
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.title("Accuracy score for Logistic regression")
plt.show()
plt.plot(gamma, score_svm_train, label='SVM Training')
plt.plot(gamma, score_svm_test, label='SVM Testing')
plt.legend()
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.title("Accuracy score for SVM")
plt.show()

print max(score_lr_train)
print max(score_lr_test)
print max(score_svm_train)
print max(score_svm_test)

print("My name is Yuxin Sun")
print("My NetID is: yuxins5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
