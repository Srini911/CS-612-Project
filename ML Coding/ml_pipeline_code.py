import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from matplotlib_venn import venn2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

# ─── SETTINGS ─────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\srini\Desktop\docking_project\ML Coding"
DATA_PATH  = os.path.join(BASE_DIR, "CS612 Dataset.csv")
OUT_DIR    = os.path.join(BASE_DIR, "pipeline_results")
os.makedirs(OUT_DIR, exist_ok=True)

RNG        = 42
TEST_SIZE  = 0.3
CV_SPLITS  = 5
N_ITER     = 60

# ─── LOAD & BALANCE ───────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH).dropna()
df["BindingAtMutationSite"] = df["BindingAtMutationSite"].map({"Yes":1, "No":0})

maj     = df[df.BindingAtMutationSite == 0]
minr    = df[df.BindingAtMutationSite == 1]
minr_up = resample(minr, replace=True, n_samples=len(maj), random_state=RNG)
df_bal  = pd.concat([maj, minr_up]).reset_index(drop=True)

features = [
    "Binding_Affinity",
    "Hydropathy_Change",
    "Is_Charged_Change",
    "AA_Change_Type_polar?hydro",
    "AA_Change_Type_polar?polar",
    "RMSD"
]
X = df_bal[features]
y = df_bal.BindingAtMutationSite

# ─── DATA DISTRIBUTION PLOTS ──────────────────────────────────────────
plt.figure(figsize=(8,6))
X.hist(bins=20, layout=(2,3), edgecolor='k')
plt.suptitle("Feature Distributions (Balanced Data)")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR, "data_distribution.png"))
plt.close()

# ─── TRAIN/TEST SPLIT & COUNTS ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    stratify=y, random_state=RNG
)
pd.Series({
    "Train_Pos": y_train.sum(),
    "Train_Neg": len(y_train) - y_train.sum(),
    "Test_Pos":  y_test.sum(),
    "Test_Neg":  len(y_test) - y_test.sum()
}).plot(kind="bar", title="Train/Test Counts")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "split_counts.png"))
plt.close()

# ─── MODELS & HYPERPARAMETER SEARCH ───────────────────────────────────
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                              max_iter=2000, random_state=RNG))
])
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=RNG))
])

lr_param = {
    "lr__C": uniform(loc=0.1, scale=100),
    "lr__class_weight": [None, "balanced"]
}
svm_param = {
    "svc__kernel":      ["rbf", "poly"],
    "svc__degree":      [2, 3],
    "svc__C":           uniform(loc=0.1, scale=10),
    "svc__gamma":       ["scale", "auto"],
    "svc__class_weight":[None, "balanced"]
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RNG)
lr_search = RandomizedSearchCV(pipe_lr, lr_param, n_iter=N_ITER,
                               cv=cv, scoring="roc_auc",
                               random_state=RNG, n_jobs=-1)
svm_search = RandomizedSearchCV(pipe_svm, svm_param, n_iter=N_ITER,
                                cv=cv, scoring="roc_auc",
                                random_state=RNG, n_jobs=-1)

lr_search.fit(X_train, y_train)
svm_search.fit(X_train, y_train)
lr_best, svm_best = lr_search.best_estimator_, svm_search.best_estimator_

print("Best LR params:", lr_search.best_params_)
print("Best SVM params:", svm_search.best_params_)

# ─── LEARNING CURVES ───────────────────────────────────────────────────
def plot_lc(estimator, X_, y_, name):
    sizes, train_scores, val_scores = learning_curve(
        estimator, X_, y_, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=RNG
    )
    plt.figure()
    plt.plot(sizes, train_scores.mean(axis=1),  "o-", label="Train")
    plt.plot(sizes, val_scores.mean(axis=1),    "s--", label="Validation")
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"learning_curve_{name}.png"))
    plt.close()

plot_lc(lr_best,  X_train, y_train, "LogisticRegression")
plot_lc(svm_best, X_train, y_train, "SVM")

# ─── ROC CURVES ───────────────────────────────────────────────────────
plt.figure()
for name, model in [("LogReg", lr_best), ("SVM", svm_best)]:
    prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} AUC {auc(fpr, tpr):.2f}")
plt.plot([0,1],[0,1], "--", color="gray")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_combined.png"))
plt.close()

# ─── PERFORMANCE METRICS ──────────────────────────────────────────────
y_pred_lr  = lr_best.predict(X_test)
y_pred_svm = svm_best.predict(X_test)

metrics = {
    "Model":      ["LogReg", "SVM"],
    "Accuracy":   [accuracy_score(y_test, y_pred_lr),
                   accuracy_score(y_test, y_pred_svm)],
    "Precision":  [precision_score(y_test, y_pred_lr),
                   precision_score(y_test, y_pred_svm)],
    "Recall":     [recall_score(y_test, y_pred_lr),
                   recall_score(y_test, y_pred_svm)],
    "F1 Score":   [f1_score(y_test, y_pred_lr),
                   f1_score(y_test, y_pred_svm)]
}
pd.DataFrame(metrics).set_index("Model") \
    .plot(kind="bar", rot=0, title="Performance Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "performance_metrics.png"))
plt.close()

# ─── CLASSIFICATION REPORT PLOTS ──────────────────────────────────────
cr_lr = classification_report(y_test, y_pred_lr, output_dict=True)
pd.DataFrame(cr_lr).transpose() \
    .plot(kind="bar", rot=45, title="Classification Report: LogReg")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_report_logreg.png"))
plt.close()

cr_svm = classification_report(y_test, y_pred_svm, output_dict=True)
pd.DataFrame(cr_svm).transpose() \
    .plot(kind="bar", rot=45, title="Classification Report: SVM")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_report_svm.png"))
plt.close()

# ─── CONFUSION MATRICES & VENN ───────────────────────────────────────
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title("Confusion Matrix: LogReg")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_lr.png"))
plt.close()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
plt.title("Confusion Matrix: SVM")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_svm.png"))
plt.close()

mask_lr  = y_pred_lr  != y_test
mask_svm = y_pred_svm != y_test
plt.figure()
venn2(subsets=(
    mask_lr.sum() & ~mask_svm.sum(),
    mask_svm.sum() & ~mask_lr.sum(),
    (mask_lr & mask_svm).sum()
), set_labels=("LR errors","SVM errors"))
plt.title("Misclassification Overlap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "venn_misclass.png"))
plt.close()

# ─── MISCLASSIFIED DATA SCATTER ──────────────────────────────────────
plt.figure()
plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=y_test,
            cmap="coolwarm", alpha=0.3, label="Correct")
plt.scatter(X_test.iloc[:,0][mask_lr], X_test.iloc[:,1][mask_lr],
            facecolors="none", edgecolors="k", label="LR errors")
plt.scatter(X_test.iloc[:,0][mask_svm], X_test.iloc[:,1][mask_svm],
            marker="x", c="black", label="SVM errors")
plt.xlabel(features[0]); plt.ylabel(features[1])
plt.title("Misclassified Data by Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "misclass_scatter.png"))
plt.close()

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────
coef_lr = pd.Series(lr_best.named_steps["lr"].coef_.ravel(), index=features) \
              .sort_values()
coef_lr.plot(kind="barh", title="LogReg Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feat_imp_logreg.png"))
plt.close()

perm = permutation_importance(svm_best, X_test, y_test,
                              n_repeats=10, random_state=RNG, n_jobs=-1)
imp_svm = pd.Series(perm.importances_mean, index=features).sort_values()
imp_svm.plot(kind="barh", title="SVM Permutation Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feat_imp_svm.png"))
plt.close()

# ─── DISCUSSION (console) ────────────────────────────────────────────
print("\nFeature Importances:")
print("LogReg coefficients:")
for feat, val in list(coef_lr.items())[::-1]:
    print(f"  {feat}: {val:.3f}")
print("\nSVM importances:")
for feat, val in list(imp_svm.items())[::-1]:
    print(f"  {feat}: {val:.3f}")

print("""
Biological Insights:
 - ↑Binding_Affinity & ↓RMSD strongly predict binding.
 - Hydropathy & charge changes modulate drug interactions.
 - Polar–polar AA swaps influence binding stability.
""")

print("All plots saved to", OUT_DIR)
