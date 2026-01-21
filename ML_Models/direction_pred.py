# import warnings
# warnings.filterwarnings("ignore")

from typing import Any, List, Tuple, Dict, Optional, Union, Callable, Iterable, Sequence, TypeVar
import numpy as np
import pandas as pd
import re
import json

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras as tfk
import keras as k
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.layers import LSTM, Masking # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

#The baseline prediction. Any model with a lower accuacy than this is useless.
class Baseline_Pred():
    def __init__(self,features: pd.DataFrame, y_direction: pd.Series):
        self.features = features
        self.y_direction = y_direction
        print("Training the baseline logistic regression model for direction prediction")
        self.logi = self.logistic_reg_train()

    def logistic_reg_train(self) -> LogisticRegression: 
        '''Baseline model'''
        logi = LogisticRegression()
        logi.fit(self.features,self.y_direction)
        return logi

#Extracting rules using decision tree
class Extract_Rules():
    def __init__(self,features: pd.DataFrame, y_direction: pd.Series):
        self.dtree = None
        self.features = features
        self.y_direction = y_direction
        self.rule_text = None
        self.f_names = list(self.features.columns)
        print("Training the decision tree model and extracting rules wrt direction of market")
        self.rules_json = self.get_rules()

    def dec_tree(self):
        self.dtree = DecisionTreeClassifier(
            criterion='gini',  # The function to measure the quality of a split.
            splitter='best',  # The strategy used to choose the split at each node.
            max_depth=5,  # The maximum depth of the tree.
            min_samples_split=2,  # The minimum number of samples required to split an internal node.
            min_samples_leaf=1,  # The minimum number of samples required to be at a leaf node.
            min_weight_fraction_leaf=0.0,  # The minimum weighted fraction of the sum total of weights required to be at a leaf node.
            max_features=None,  # The number of features to consider when looking for the best split.
            random_state=None,  # Controls the randomness of the estimator.
            max_leaf_nodes=None,  # Grow a tree with max_leaf_nodes in best-first fashion.
            min_impurity_decrease=0.0,  # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            class_weight=None,  # Weights associated with classes in the form `{class_label: weight}`.
            ccp_alpha=0.0  # Complexity parameter used for Minimal Cost-Complexity Pruning.
        )
        self.dtree.fit(self.features,self.y_direction)
        
    def extract_store_in_json(self) -> json:
            """
            Generates the tree text, parses it into structured JSON-compatible dictionaries,
            and returns the list of rules.
            """
            if self.dtree is None:
                raise ValueError("Model not trained! Call dec_tree() first.")

            # 1. Generate the raw text representation
            self.rule_text = export_text(self.dtree, feature_names=self.f_names)

            # 2. Parse the text
            structured_rules = self._parse_text_to_structure(self.rule_text)
            
            json_output = json.dumps(structured_rules, indent=4)

            return json_output

    def _parse_text_to_structure(self, text_output):
        """
        Internal helper method: Converts raw export_text string into a list of rule dicts.
        """
        lines = text_output.split('\n')
        rules = []
        path = {}  # Stores current path: {depth: condition}
        
        # Regex to handle indentation and split condition/class
        # Matches: "|   |--- " (indent) and the rest of the string
        line_regex = re.compile(r'^(\|   )*\|--- (.+)$')
        
        for line in lines:
            if not line.strip():
                continue
                
            # A. Calculate Depth based on pipes (each "|   " is one level)
            indent_matches = line.count('|   ')
            depth = indent_matches
            
            # B. Clean the line to get the raw condition or class
            cleaned_line = line.replace('|   ', '').replace('|--- ', '')
            
            # C. Check if this line is a Leaf (Prediction) or Node (Condition)
            if "class:" in cleaned_line:
                # It is a leaf! Extract prediction.
                prediction = cleaned_line.split("class:")[1].strip()
                
                # Construct the full logic path from root to this leaf
                rule_conditions = [path[i] for i in range(depth)]
                
                # Create the rule object
                structured_rule = {
                    "logic": " AND ".join(rule_conditions),
                    "prediction": prediction,
                    "complexity": len(rule_conditions) # Depth of logic
                }
                rules.append(structured_rule)
                
            else:
                # It is a decision node. Store the condition at this depth.
                path[depth] = cleaned_line.strip()

        return rules
    def get_rules(self):
        self.dec_tree()
        rules_json = self.extract_store_in_json()
        return rules_json

#Ranking feature importance using Random Forest Classifier
class Feature_importance():
    def __init__(self,features: pd.DataFrame, y_direction: pd.Series):
        self.features = features
        self.y_direction = y_direction
        self.forest_model = None
        self.feature_ranking = None
        self.top_30_features_list = None
        print("Training the random forest model and extracting feature importance wrt direction of market")
        self._get_rankings()

    def rando_forest(self):
        self.forest_model = RandomForestClassifier(
            n_estimators=100,  # The number of trees in the forest.
            criterion='gini',  # The function to measure the quality of a split.
            max_depth=5,  # The maximum depth of the tree.
            min_samples_split=2,  # The minimum number of samples required to split an internal node.
            min_samples_leaf=1,  # The minimum number of samples required to be at a leaf node.
            min_weight_fraction_leaf=0.0,  # The minimum weighted fraction of the sum total of weights required to be at a leaf node.
            max_features='sqrt',  # The number of features to consider when looking for the best split.
            max_leaf_nodes=None,  # Grow a tree with max_leaf_nodes in best-first fashion.
            min_impurity_decrease=0.0,  # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            bootstrap=True,  # Whether bootstrap samples are used when building trees.
            oob_score=False,  # Whether to use out-of-bag samples to estimate the generalization accuracy.
            n_jobs=None,  # The number of jobs to run in parallel.
            random_state=None,  # Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
            verbose=0,  # Controls the verbosity when fitting and predicting.
            warm_start=False,  # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
            class_weight=None,  # Weights associated with classes in the form `{class_label: weight}`.
            ccp_alpha=0.0,  # Complexity parameter used for Minimal Cost-Complexity Pruning.
            max_samples=None  # If bootstrap is True, the number of samples to draw from X to train each base estimator.
        )
        self.forest_model.fit(self.features, self.y_direction)

    def rank_features(self, n_top=30):
        """
        Extracts feature importance, ranks them, and returns the names of the top N features.
        """
        if self.forest_model is None:
            raise ValueError("Model not trained! Call rando_forest() first.")

        # 1. Get the raw importance scores
        importances = self.forest_model.feature_importances_

        # 2. Create a DataFrame for easy sorting
        feature_ranking = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })

        # 3. Sort by Importance (Highest first)
        self.feature_ranking = feature_ranking.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        # 4. Extract the top N feature names
        self.top_30_features_list = feature_ranking.head(n_top)['Feature'].tolist()

    def _get_rankings(self):
        self.rando_forest()#trains the model
        self.rank_features()#gets the ranking of each feature
        #all stored in the class variables.
    
#This should be the model with the best accuracy for direction prediction
class XGBoost_Pred():
    def __init__(self,features, y_direction):
        self.features = features
        self.y_direction = y_direction
        print("Training the XGBoost model for direction prediction")
        self.xgbc = self.xgbc_train()

    def xgbc_train(self):
        self.xgbc = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgbc.fit(self.features, self.y_direction)
        return self.xgbc
    
#Feed forward neural network: For Learning the non linear relations between the features and the target variable
class FF_NN():
    def __init__(self,features,y_direction):
        self.features = features
        self.y_direction = y_direction
        print("Training the feed forward neural network for direction prediction")
        self.ffnn = self.ffnn_train()

    def ffnn_train(self):
        self.ffnn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.features.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.ffnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.ffnn.fit(self.features, self.y_direction, epochs=10, batch_size=32)
        return self.ffnn

#Long short term memory: For Learning the temporal dependencies between the features and the target variable
class LSTM_Pred():
    '''Code may need attention'''
    def __init__(self,features,y_direction):
        self.features = features
        self.y_direction = y_direction
        print("Training the LSTM model for direction prediction")
        self.lstm = self.lstm_train()

    def lstm_train(self):
        self.lstm = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(self.features.shape[1], 1), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm.fit(self.features, self.y_direction, epochs=10, batch_size=32)
        return self.lstm

class Trainiing_Direction_Classificaiton():
    def __init__(self):
        pass
