import numpy as np

from scipy.stats import spearmanr

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    LogisticRegression,
)

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.svm import SVR, SVC

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)

from sklearn.neural_network import (
    MLPRegressor,
    MLPClassifier,
)

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import (
    XGBRegressor,
    XGBClassifier,
)

from lightgbm import (
    LGBMRegressor,
    LGBMClassifier,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


############################################################
# EVALUATION FUNCTIONS
############################################################


def continuous_evaluation(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    target_std = np.std(actual, ddof=1)

    if not np.isfinite(target_std) or target_std == 0:
        nrmse = np.nan

    else:
        nrmse = rmse / target_std

    mae = mean_absolute_error(actual, predicted)

    r2 = r2_score(actual, predicted)

    if len(actual) < 2 or len(np.unique(actual)) < 2 or len(np.unique(predicted)) < 2:
        rank_ic = np.nan

    else:
        rank_ic = spearmanr(actual, predicted).statistic

    return {"RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2, "Rank IC": rank_ic}


def binary_evaluation(actual, predicted, probabilities, classes):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)

    if len(classes) != 2:
        raise ValueError(f"Binary evaluation expected 2 classes, got: {classes}")

    positive_class = classes[-1]

    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError("Binary probability matrix must have two columns.")

        positive_probabilities = probabilities[:, 1]

    else:
        positive_probabilities = probabilities

    binary_actual = (actual == positive_class).astype(int)

    if len(np.unique(binary_actual)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan

    else:
        roc_auc = roc_auc_score(binary_actual, positive_probabilities)

        pr_auc = average_precision_score(binary_actual, positive_probabilities)

    loss = log_loss(actual, probabilities, labels=classes)

    f1 = f1_score(actual, predicted, pos_label=positive_class, zero_division=0)

    precision = precision_score(actual, predicted, pos_label=positive_class, zero_division=0)

    recall = recall_score(actual, predicted, pos_label=positive_class, zero_division=0)

    balanced_accuracy = balanced_accuracy_score(actual, predicted)

    return {
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Log Loss": loss,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Balanced Accuracy": balanced_accuracy,
    }


def multiclass_evaluation(actual, predicted, probabilities, classes):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)

    loss = log_loss(actual, probabilities, labels=classes)

    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)

    balanced_accuracy = balanced_accuracy_score(actual, predicted)

    return {"Log Loss": loss, "Macro F1": macro_f1, "Balanced Accuracy": balanced_accuracy}


############################################################
# CONTINUOUS MODELS
############################################################


def fit_mean_baseline(x_train, y_train, x_validation, y_validation):

    predicted = np.full(len(y_validation), np.mean(y_train))

    return continuous_evaluation(y_validation, predicted)


def fit_ols(x_train, y_train, x_validation, y_validation):

    model = LinearRegression()

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_ridge(x_train, y_train, x_validation, y_validation, alpha):

    model = Ridge(alpha=alpha)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_lasso(x_train, y_train, x_validation, y_validation, alpha):

    model = Lasso(alpha=alpha, max_iter=10000)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_elastic_net(x_train, y_train, x_validation, y_validation, alpha, l1_ratio):

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_huber(x_train, y_train, x_validation, y_validation, epsilon, alpha):

    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_hist_gradient_boosting_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    max_iter,
    max_leaf_nodes,
    max_depth,
    min_samples_leaf,
    l2_regularization,
):
    if max_depth is not None:
        if np.isnan(max_depth):
            max_depth = None

    max_depth = int(max_depth) if max_depth is not None else None

    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_gradient_boosting_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    n_estimators,
    max_depth,
    min_samples_leaf,
    subsample,
    max_features,
):

    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_random_forest_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    max_features,
    bootstrap,
):

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_xgboost_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    reg_alpha,
    reg_lambda,
):

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_lightgbm_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    num_leaves,
    max_depth,
    min_child_samples,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda,
):

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_svr(x_train, y_train, x_validation, y_validation, kernel, C, epsilon, gamma="scale"):

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_knn_regressor(x_train, y_train, x_validation, y_validation, n_neighbors, weights, p):

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


def fit_mlp_regressor(
    x_train,
    y_train,
    x_validation,
    y_validation,
    hidden_layer_sizes,
    activation,
    alpha,
    learning_rate_init,
    batch_size,
):

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    return continuous_evaluation(y_validation, predicted)


############################################################
# BINARY MODELS
############################################################


def fit_binary_baseline(x_train, y_train, x_validation, y_validation):

    classes, counts = np.unique(y_train, return_counts=True)

    if len(classes) != 2:
        raise ValueError(f"Binary baseline expected 2 classes, got: {classes}")

    class_probabilities = counts / counts.sum()

    most_common_class = classes[np.argmax(counts)]

    predicted = np.full(len(y_validation), most_common_class)

    probabilities = np.tile(class_probabilities, (len(y_validation), 1))

    return binary_evaluation(y_validation, predicted, probabilities, classes)


def fit_logistic_regression(x_train, y_train, x_validation, y_validation, class_weight):

    model = LogisticRegression(
        C=np.inf, class_weight=class_weight, solver="lbfgs", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_l2_logistic_regression(x_train, y_train, x_validation, y_validation, C, class_weight):

    model = LogisticRegression(
        C=C, l1_ratio=0, class_weight=class_weight, solver="lbfgs", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_l1_logistic_regression(x_train, y_train, x_validation, y_validation, C, class_weight):

    model = LogisticRegression(
        C=C, l1_ratio=1, class_weight=class_weight, solver="saga", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_elastic_net_logistic_regression(
    x_train, y_train, x_validation, y_validation, C, l1_ratio, class_weight
):

    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        solver="saga",
        max_iter=5000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_hist_gradient_boosting_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    max_iter,
    max_leaf_nodes,
    max_depth,
    min_samples_leaf,
    l2_regularization,
    class_weight,
):

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        class_weight=class_weight,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_gradient_boosting_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    n_estimators,
    max_depth,
    min_samples_leaf,
    subsample,
    max_features,
):

    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_random_forest_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    max_features,
    bootstrap,
    class_weight,
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_xgboost_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    reg_alpha,
    reg_lambda,
    class_weight,
):

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    sample_weight = None

    if class_weight == "balanced":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train_encoded, sample_weight=sample_weight)

    predicted_encoded = model.predict(x_validation).astype(int)

    predicted = label_encoder.inverse_transform(predicted_encoded)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, label_encoder.classes_)


def fit_lightgbm_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    num_leaves,
    max_depth,
    min_child_samples,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda,
    class_weight,
):

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_svm_classifier(
    x_train, y_train, x_validation, y_validation, kernel, C, class_weight, gamma="scale"
):

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_knn_classifier(x_train, y_train, x_validation, y_validation, n_neighbors, weights, p):

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_naive_bayes(x_train, y_train, x_validation, y_validation, var_smoothing):

    model = GaussianNB(var_smoothing=var_smoothing)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_mlp_classifier(
    x_train,
    y_train,
    x_validation,
    y_validation,
    hidden_layer_sizes,
    activation,
    alpha,
    learning_rate_init,
    batch_size,
):

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return binary_evaluation(y_validation, predicted, probabilities, model.classes_)


############################################################
# MULTICLASS MODELS
############################################################


def fit_multiclass_baseline(x_train, y_train, x_validation, y_validation):

    classes, counts = np.unique(y_train, return_counts=True)

    class_probabilities = counts / counts.sum()

    most_common_class = classes[np.argmax(counts)]

    predicted = np.full(len(y_validation), most_common_class)

    probabilities = np.tile(class_probabilities, (len(y_validation), 1))

    return multiclass_evaluation(y_validation, predicted, probabilities, classes)


def fit_multinomial_logistic_regression(x_train, y_train, x_validation, y_validation, class_weight):

    model = LogisticRegression(
        C=np.inf, class_weight=class_weight, solver="lbfgs", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_l2_multinomial_logistic_regression(
    x_train, y_train, x_validation, y_validation, C, class_weight
):

    model = LogisticRegression(
        C=C, l1_ratio=0, class_weight=class_weight, solver="lbfgs", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_l1_multinomial_logistic_regression(
    x_train, y_train, x_validation, y_validation, C, class_weight
):

    model = LogisticRegression(
        C=C, l1_ratio=1, class_weight=class_weight, solver="saga", max_iter=5000, random_state=42
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_elastic_net_multinomial_logistic_regression(
    x_train, y_train, x_validation, y_validation, C, l1_ratio, class_weight
):

    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        solver="saga",
        max_iter=5000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_lda(x_train, y_train, x_validation, y_validation, solver, shrinkage=None):

    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_qda(x_train, y_train, x_validation, y_validation, reg_param):

    model = QuadraticDiscriminantAnalysis(reg_param=reg_param)

    try:
        model.fit(x_train, y_train)

    except Exception:
        return None

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_hist_gradient_boosting_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    max_iter,
    max_leaf_nodes,
    max_depth,
    min_samples_leaf,
    l2_regularization,
    class_weight,
):

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        class_weight=class_weight,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_gradient_boosting_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    learning_rate,
    n_estimators,
    max_depth,
    min_samples_leaf,
    subsample,
    max_features,
):

    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_random_forest_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    max_features,
    bootstrap,
    class_weight,
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_xgboost_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    reg_alpha,
    reg_lambda,
    class_weight,
):

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    sample_weight = None

    if class_weight == "balanced":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train_encoded, sample_weight=sample_weight)

    predicted_encoded = model.predict(x_validation).astype(int)

    predicted = label_encoder.inverse_transform(predicted_encoded)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, label_encoder.classes_)


def fit_lightgbm_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    n_estimators,
    learning_rate,
    num_leaves,
    max_depth,
    min_child_samples,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda,
    class_weight,
):

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_svm_multiclass(
    x_train, y_train, x_validation, y_validation, kernel, C, class_weight, gamma="scale"
):

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_mlp_multiclass(
    x_train,
    y_train,
    x_validation,
    y_validation,
    hidden_layer_sizes,
    activation,
    alpha,
    learning_rate_init,
    batch_size,
):

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)

    return multiclass_evaluation(y_validation, predicted, probabilities, model.classes_)


def fit_ordinal_regression(x_train, y_train, x_validation, y_validation, alpha):

    raise NotImplementedError("Ordinal regression has no implementation added yet.")
