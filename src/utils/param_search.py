from sklearn.model_selection import ParameterGrid


def param_search_f1(function, params):
    grid = ParameterGrid(params)
