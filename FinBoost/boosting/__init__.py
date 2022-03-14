"""
The :mod:`FinBoost.boosting` module includes boosting-based methods for
classification and regression.

The customizability is pursued to enable to improve the model development

"""
from ._base import BaseEnsemble

from ._bagging import BaggingClassifier
from ._bagging import BaggingRegressor

## AdaBoost
from ._weight_boosting import AdaBoostClassifier
from ._weight_boosting import AdaBoostRegressor

## Gradient Boosting
from ._gb import GradientBoostingClassifier
from ._gb import GradientBoostingRegressor

## Voting Classifier
#from ._voting import VotingClassifier
#from ._voting import VotingRegressor

## Stacking Classifier
from ._stacking import StackingClassifier
from ._stacking import StackingRegressor


__all__ = [
    "BaseEnsemble",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "IsolationForest",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor"
]
