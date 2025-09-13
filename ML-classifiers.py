# Avoiding warning
import warnings


def warn(*args, **kwargs): pass


warnings.warn = warn

# Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    f1_score,
    recall_score,
    matthews_corrcoef,
    auc,
    cohen_kappa_score
)

# ML Classifiers
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
import lightgbm as lgb

# Step 1: Load dataset
iRec = 'BiPSSM_Drug.csv'
D = pd.read_csv(iRec, header=None)

# Step 2: Separate features and labels
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

# Shuffle dataset
X, y = shuffle(X, y)

# Step 3: Feature scaling
scale = StandardScaler()
X = scale.fit_transform(X)

# Step 4: Encode labels
y = LabelEncoder().fit_transform(y)

# Step 5: Define classifiers and names
Names = ['RNN', 'CNN', 'TCN', 'SN-TCN']
Classifiers = [
    ExtraTreesClassifier(n_estimators=200),
    RandomForestClassifier(n_estimators=200),
    #AdaBoostClassifier(),
    XGBClassifier(n_estimators=100),
    lgb.LGBMClassifier(n_estimators=300)
]


# Step 6: Classification function
def runClassifiers():
    Results = []
    cv = KFold(n_splits=10, shuffle=True)

    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC = []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([[0, 0], [0, 0]], dtype=int)

        print(classifier.__class__.__name__)
        model = classifier

        for (train_index, test_index) in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0

            roc_auc = auc(FPR, TPR)
            y_pred = model.predict(X_test)

            auROC.append(roc_auc_score(y_test, y_proba))
            accuray.append(accuracy_score(y_test, y_pred) * 100)
            avePrecision.append(average_precision_score(y_test, y_proba))
            F1_Score.append(f1_score(y_test, y_pred))
            MCC.append(matthews_corrcoef(y_test, y_pred))
            Recall.append(recall_score(y_test, y_pred))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_test, y_pred)

        Results.append(accuray)
        mean_TPR /= cv.get_n_splits(X, y)
        mean_TPR[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_TPR)

        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:.3f})'.format(name, mean_auc),
            lw=2.0
        )

        TN, FP, FN, TP = CM.ravel()
        print('Accuracy: {:.4f} %'.format(np.mean(accuray)))
        print('Sensitivity (+): {:.4f} %'.format((TP / (TP + FN)) * 100))
        print('Specificity (-): {:.4f} %'.format((TN / (TN + FP)) * 100))
        print('auROC: {:.6f}'.format(mean_auc))
        print('F1-score: {:.4f}'.format(np.mean(F1_Score)))
        print('MCC: {:.4f}'.format(np.mean(MCC)))
        print('Recall: {:.4f}'.format(np.mean(Recall)))
        print("Kappa: {:.4f}".format(cohen_kappa_score(y_test, y_pred)))
        print('auPR: {:.4f}'.format(np.mean(avePrecision)))
        print('_______________________________________')

    auROCplot()


# Step 7: Plot functions
def boxPlot(Results, Names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert=True, whis=True, showbox=True)
    ax.set_xticklabels(Names)
    plt.xlabel('\nName of Classifiers')
    plt.ylabel('\nAccuracy (%)')
    plt.savefig('AccuracyBoxPlot.png', dpi=100)
    plt.show()


def auROCplot():
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('auROC.png', dpi=100)
    plt.show()


# Run the classifiers
if __name__ == '__main__':
    runClassifiers()
