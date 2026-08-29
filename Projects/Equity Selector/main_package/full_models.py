RIDGE_ALPHAS = [
     1e-5,
     3e-5,
     1e-4,
     3e-4,
     1e-3,
     3e-3,
     1e-2,
     3e-2,
    0.1,
     0.3,
    1,
     3,
     10,
     30,
     100,
     300,
     1000,
     3000,
     10000,
]


SPARSE_ALPHAS = [
     1e-8,
     3e-8,
     1e-7,
     3e-7,
     1e-6,
     3e-6,
     1e-5,
     3e-5,
     1e-4,
     3e-4,
     1e-3,
     3e-3,
    1e-2,
     3e-2,
    0.1,
     0.3,
     1,
     3,
     10,
]


C_VALUES = [
     1e-5,
     3e-5,
     1e-4,
     3e-4,
     1e-3,
     3e-3,
     1e-2,
     3e-2,
    0.1,
     0.3,
     1,
     3,
     10,
     30,
     100,
     300,
     1000,
]


L1_RATIOS = [
     0.01,
     0.05,
     0.10,
     0.25,
    0.50,
     0.75,
     0.90,
     0.95,
     0.99,
]


LEARNING_RATES = [
     0.005,
     0.01,
     0.03,
    0.05,
     0.10,
     0.20,
]


CLASS_WEIGHTS = [
    None,
    "balanced",
]

HIST_GRADIENT_PARAMS = {

    "learning_rate": LEARNING_RATES,

    "max_iter": [
         100,
        200,
         300,
         600,
         1000,
         1500,
    ],

    "max_leaf_nodes": [
         7,
         15,
        31,
         63,
         127,
    ],

    "max_depth": [
         None,
         3,
        5,
         7,
         10,
    ],

    "min_samples_leaf": [
         5,
         10,
        20,
         50,
         100,
         200,
    ],

    "l2_regularization": [
         0,
         1e-4,
         1e-3,
         1e-2,
        0.1,
         1,
         10,
         100,
    ],
}


GRADIENT_BOOSTING_PARAMS = {

    "learning_rate": LEARNING_RATES,

    "n_estimators": [
         100,
        200,
         300,
         600,
         1000,
    ],

    "max_depth": [
         2,
        3,
         4,
         5,
         8,
    ],

    "min_samples_leaf": [
         5,
         10,
        20,
         50,
         100,
    ],

    "subsample": [
         0.5,
         0.6,
        0.8,
         1.0,
    ],

    "max_features": [
         None,
        "sqrt",
         0.3,
         0.5,
         0.8,
    ],
}


RANDOM_FOREST_PARAMS = {

    "n_estimators": [
        200,
         500,
         1000,
         1500,
    ],

    "max_depth": [
         None,
         5,
        10,
         15,
         20,
         30,
    ],

    "min_samples_leaf": [
         1,
         2,
        5,
         10,
         20,
         50,
         100,
    ],

    "min_samples_split": [
         2,
        5,
         10,
         20,
         50,
    ],

    "max_features": [
        "sqrt",
         0.2,
         0.3,
         0.5,
         0.75,
         1.0,
    ],

    "bootstrap": [
        True,
         False,
    ],
}


XGBOOST_PARAMS = {

    "n_estimators": [
        200,
         300,
         500,
         750,
         1000,
         1500,
    ],

    "learning_rate": LEARNING_RATES,

    "max_depth": [
         2,
        3,
         4,
         5,
         7,
         10,
    ],

    "min_child_weight": [
        1,
         2,
         3,
         5,
         10,
         20,
         50,
    ],

    "subsample": [
         0.5,
         0.6,
        0.8,
         1.0,
    ],

    "colsample_bytree": [
         0.4,
         0.5,
        0.75,
         1.0,
    ],

    "gamma": [
        0,
         0.001,
         0.01,
         0.1,
         0.5,
         1,
         5,
    ],

    "reg_alpha": [
        0,
         1e-5,
         1e-4,
         1e-3,
         1e-2,
         0.1,
         1,
         10,
    ],

    "reg_lambda": [
         0,
         0.01,
         0.1,
        1,
         10,
         100,
    ],
}


LIGHTGBM_PARAMS = {

    "n_estimators": [
        200,
         300,
         500,
         750,
         1000,
         1500,
    ],

    "learning_rate": LEARNING_RATES,

    "num_leaves": [
         7,
         15,
        31,
         63,
         127,
         255,
    ],

    "max_depth": [
        -1,
         3,
         5,
         8,
         12,
         16,
    ],

    "min_child_samples": [
         5,
         10,
        20,
         50,
         100,
         200,
    ],

    "subsample": [
         0.5,
         0.6,
        0.8,
         1.0,
    ],

    "colsample_bytree": [
         0.4,
         0.5,
        0.75,
         1.0,
    ],

    "reg_alpha": [
        0,
         1e-5,
         1e-4,
         1e-3,
         1e-2,
         0.1,
         1,
         10,
    ],

    "reg_lambda": [
         0,
         0.01,
         0.1,
        1,
         10,
         100,
    ],
}


MLP_PARAMS = {

    "hidden_layer_sizes": [
         (32,),
        (64,),
         (128,),
         (256,),
         (64, 32),
         (128, 64),
         (256, 128),
         (128, 64, 32),
         (256, 128, 64),
    ],

    "activation": [
        "relu",
         "tanh",
    ],

    "alpha": [
         1e-7,
         1e-6,
         1e-5,
        1e-4,
         1e-3,
         1e-2,
         0.1,
    ],

    "learning_rate_init": [
         1e-5,
         3e-5,
         1e-4,
         3e-4,
          1e-3,
         3e-3,
         1e-2,
    ],

    "batch_size": [
         32,
         64,
        128,
         256,
         512,
         "auto",
    ],
}

KNN_PARAMS = {

    "n_neighbors": [
         3,
        5,
        10,
        20,
        40,
         80,
         150,
         250,
    ],

    "weights": [
        "uniform",
        "distance",
    ],

    "p": [
        1,
        2,
    ],
}

def full_model_source(type):
    if type == "continuous":
        return [            

            {
                "name": "Mean Baseline",
                "function": "fit_mean_baseline",
                "scaled": False,
                "params": {},
            },

            {
                "name": "OLS",
                "function": "fit_ols",
                "scaled": True,
                "params": {},
            },

            {
                "name": "Ridge",
                "function": "fit_ridge",
                "scaled": True,
                "params": {
                    "alpha": RIDGE_ALPHAS,
                },
            },

            {
                "name": "Lasso",
                "function": "fit_lasso",
                "scaled": True,
                "params": {
                    "alpha": SPARSE_ALPHAS,
                },
            },

            {
                "name": "Elastic Net",
                "function": "fit_elastic_net",
                "scaled": True,
                "params": {
                    "alpha": SPARSE_ALPHAS,
                    "l1_ratio": L1_RATIOS,
                },
            },



            {
                "name": "Huber",
                "function": "fit_huber",
                "scaled": True,
                "params": {

                    "epsilon": [
                         1.05,
                         1.15,
                         1.25,
                        1.35,
                         1.50,
                         1.75,
                         2.00,
                         2.50,
                    ],

                    "alpha": [
                         0,
                         1e-7,
                         1e-6,
                         1e-5,
                        1e-4,
                         1e-3,
                         1e-2,
                         0.1,
                         1,
                    ],
                },
            },


            {
               "name": "Hist Gradient Boosting",
               "function": "fit_hist_gradient_boosting_regressor",
               "scaled": False,
               "params": HIST_GRADIENT_PARAMS,
            },


            {
                "name": "Gradient Boosting",
                "function": "fit_gradient_boosting_regressor",
                "scaled": False,
                "params": GRADIENT_BOOSTING_PARAMS,
            },


            {
                "name": "Random Forest",
                "function": "fit_random_forest_regressor",
                "scaled": False,
                "params": RANDOM_FOREST_PARAMS,
            },

            {
                "name": "XGBoost",
                "function": "fit_xgboost_regressor",
                "scaled": False,
                "params": XGBOOST_PARAMS,
            },

            {
                "name": "LightGBM",
                "function": "fit_lightgbm_regressor",
                "scaled": False,
                "params": LIGHTGBM_PARAMS,
            },



            {
                "name": "SVR Linear",
                "function": "fit_svr",
                "scaled": True,
                "params": {

                    "kernel": ["linear"],
                    "C": C_VALUES,

                    "epsilon": [
                         1e-4,
                         1e-3,
                         1e-2,
                         0.05,
                        0.1,
                         0.25,
                         0.5,
                    ],
                },
            },

            {
                "name": "SVR RBF",
                "function": "fit_svr",
                "scaled": True,
                "params": {

                    "kernel": ["rbf"],
                    "C": C_VALUES,

                    "epsilon": [
                         1e-4,
                         1e-3,
                         1e-2,
                         0.05,
                        0.1,
                         0.25,
                         0.5,
                    ],

                    "gamma": [
                        "scale",
                         "auto",
                         1e-4,
                         1e-3,
                         1e-2,
                         0.1,
                         1,
                    ],
                },
            },


            {
                "name": "kNN",
                "function": "fit_knn_regressor",
                "scaled": True,
                "params": KNN_PARAMS,
            },


            {
                "name": "MLP",
                "function": "fit_mlp_regressor",
                "scaled": True,
                "params": MLP_PARAMS,
            },
        ]

    elif type == "binary":
        return [


            {
                "name": "Binary Baseline",
                "function": "fit_binary_baseline",
                "scaled": False,
                "params": {},
            },

            {
                "name": "Logistic Regression",
                "function": "fit_logistic_regression",
                "scaled": True,
                "params": {
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "L2 Logistic Regression",
                "function": "fit_l2_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "L1 Logistic Regression",
                "function": "fit_l1_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "Elastic Net Logistic Regression",
                "function": "fit_elastic_net_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "l1_ratio": L1_RATIOS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
               "name": "Hist Gradient Boosting",
               "function": "fit_hist_gradient_boosting_classifier",
               "scaled": False,
               "params": {
                   **HIST_GRADIENT_PARAMS,
                   "class_weight": CLASS_WEIGHTS,
               },
            },


            {
                "name": "Gradient Boosting",
                "function": "fit_gradient_boosting_classifier",
                "scaled": False,
                "params": GRADIENT_BOOSTING_PARAMS,
            },


            {
                "name": "Random Forest",
                "function": "fit_random_forest_classifier",
                "scaled": False,
                "params": {
                    **RANDOM_FOREST_PARAMS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
            "name": "XGBoost",
            "function": "fit_xgboost_classifier",
            "scaled": False,
            "params": {
                **XGBOOST_PARAMS,
                "class_weight": CLASS_WEIGHTS,
            },
            },

            {
                "name": "LightGBM",
                "function": "fit_lightgbm_classifier",
                "scaled": False,
                "params": {
                    **LIGHTGBM_PARAMS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
                "name": "SVM Linear",
                "function": "fit_svm_classifier",
                "scaled": True,
                "params": {

                    "kernel": ["linear"],
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "SVM RBF",
                "function": "fit_svm_classifier",
                "scaled": True,
                "params": {

                    "kernel": ["rbf"],
                    "C": C_VALUES,

                    "gamma": [
                        "scale",
                         "auto",
                         1e-4,
                         1e-3,
                         1e-2,
                         0.1,
                         1,
                    ],

                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
                "name": "kNN",
                "function": "fit_knn_classifier",
                "scaled": True,
                "params": KNN_PARAMS,
            },


            {
                "name": "Naive Bayes",
                "function": "fit_naive_bayes",
                "scaled": False,
                "params": {

                    "var_smoothing": [
                         1e-13,
                         1e-12,
                         1e-11,
                         1e-10,
                        1e-9,
                         1e-8,
                         1e-7,
                        1e-6,
                         1e-5,
                    ],
                },
            },


            {
                "name": "MLP",
                "function": "fit_mlp_classifier",
                "scaled": True,
                "params": MLP_PARAMS,
            },
        ]


    elif type == "multiclass":
        return [


            {
                "name": "Multiclass Baseline",
                "function": "fit_multiclass_baseline",
                "scaled": False,
                "params": {},
            },

            {
                "name": "Multinomial Logistic Regression",
                "function": "fit_multinomial_logistic_regression",
                "scaled": True,
                "params": {
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "L2 Multinomial Logistic Regression",
                "function": "fit_l2_multinomial_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "L1 Multinomial Logistic Regression",
                "function": "fit_l1_multinomial_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "function": "fit_elastic_net_multinomial_logistic_regression",
                "scaled": True,
                "params": {
                    "C": C_VALUES,
                    "l1_ratio": L1_RATIOS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
                "name": "LDA SVD",
                "function": "fit_lda",
                "scaled": True,
                "params": {

                    "solver": ["svd"],
                },
            },

            {
                "name": "LDA LSQR/Eigen",
                "function": "fit_lda",
                "scaled": True,
                "params": {

                    "solver": [
                        "lsqr",
                        "eigen",
                    ],

                    "shrinkage": [
                         None,
                        "auto",
                         0.05,
                         0.1,
                         0.25,
                         0.5,
                         0.75,
                         0.9,
                         0.95,
                    ],
                },
            },


            {
                "name": "QDA",
                "function": "fit_qda",
                "scaled": True,
                "params": {

                    "reg_param": [
                        0,
                         0.0001,
                         0.001,
                         0.01,
                         0.05,
                         0.1,
                         0.25,
                        0.5,
                         0.75,
                         1.0,
                    ],
                },
            },

            {
               "name": "Hist Gradient Boosting",
               "function": "fit_hist_gradient_boosting_multiclass",
               "scaled": False,
               "params": {
                   **HIST_GRADIENT_PARAMS,
                   "class_weight": CLASS_WEIGHTS,
               },
            },


            {
                "name": "Gradient Boosting",
                "function": "fit_gradient_boosting_multiclass",
                "scaled": False,
                "params": GRADIENT_BOOSTING_PARAMS,
            },


            {
                "name": "Random Forest",
                "function": "fit_random_forest_multiclass",
                "scaled": False,
                "params": {
                    **RANDOM_FOREST_PARAMS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "XGBoost",
                "function": "fit_xgboost_multiclass",
                "scaled": False,
                "params": {
                    **XGBOOST_PARAMS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "LightGBM",
                "function": "fit_lightgbm_multiclass",
                "scaled": False,
                "params": {
                    **LIGHTGBM_PARAMS,
                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
                "name": "SVM Linear",
                "function": "fit_svm_multiclass",
                "scaled": True,
                "params": {

                    "kernel": ["linear"],
                    "C": C_VALUES,
                    "class_weight": CLASS_WEIGHTS,
                },
            },

            {
                "name": "SVM RBF",
                "function": "fit_svm_multiclass",
                "scaled": True,
                "params": {

                    "kernel": ["rbf"],
                    "C": C_VALUES,

                    "gamma": [
                        "scale",
                         "auto",
                         1e-4,
                         1e-3,
                         1e-2,
                         0.1,
                         1,
                    ],

                    "class_weight": CLASS_WEIGHTS,
                },
            },


            {
                "name": "MLP",
                "function": "fit_mlp_multiclass",
                "scaled": True,
                "params": MLP_PARAMS,
            },

             {
                 "name": "Ordinal Regression",
                 "function": "fit_ordinal_regression",
                 "scaled": True,
                 "params": {
            
                     "alpha": [
                          1e-5,
                          3e-5,
                          1e-4,
                          3e-4,
                          1e-3,
                          3e-3,
                          1e-2,
                          3e-2,
                         0.1,
                          0.3,
                          1,
                          3,
                          10,
                          30,
                          100,
                     ],
                 },
             },
        ]
