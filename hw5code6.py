import numpy as np
from collections import Counter



def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    target_vector = target_vector[feature_vector.argsort()]
    feature_vector = np.sort(feature_vector)



    # Найдем индексы для разбиений 
    ids = np.unique(feature_vector[::-1], return_index=True)[1]
    indices = np.array([i for i in range(target_vector.shape[0] - 1)])
    indices = indices[feature_vector.shape[0] - ids[:-1] - 1]

    # Найдем пороги 
    cumsum = np.cumsum(feature_vector)
    thresholds = np.hstack((cumsum[1], cumsum[2:] - cumsum[:-2]))[indices] / 2

    # Посчитаем джини 
    p_left = np.cumsum(target_vector)[indices] / (indices + 1)
    p_right = np.cumsum(target_vector[::-1])[::-1][indices + 1] / (target_vector.shape[0] - indices - 1)

    H_right = 1 - p_right**2 - (1 - p_right)**2
    H_left = 1 - p_left**2 - (1 - p_left)**2

    ginis =  -((indices + 1) * H_left + (target_vector.shape[0] - (indices + 1)) * H_right) / target_vector.shape[0]
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = max(ginis)
    
    return thresholds, ginis, threshold_best, gini_best
    pass


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
    
    def get_params(self, deep=True):
        return {'min_samples_split': self._min_samples_leaf,
                'min_samples_leaf': self._min_samples_split,
                'feature_types': self._feature_types,
                'max_depth': self._max_depth}



#     def _fit_node(self, sub_X, sub_y, node, dp): # добавим параметр depth для бонусной задачи
    def _fit_node(self, sub_X, sub_y, node, dp):
        sub_X, sub_y = np.array(sub_X), np.array(sub_y)

        # if np.all(sub_y != sub_y[0]):
        if np.all(sub_y == sub_y[0]) or self._min_samples_split and sub_y.size < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return


        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] =  current_click / current_count 
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(categories_map) == 1 or np.all(feature_vector[0] == feature_vector):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None or (self._max_depth != None and dp == self._max_depth) or self._min_samples_leaf != None\
        and (sub_y[split].size < self._min_samples_leaf or sub_y[~split].size < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], dp + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], dp + 1)

    def _predict_node(self, x, node):
        while node['type'] == 'nonterminal':
            ft = node['feature_split']
            if self._feature_types[ft] == 'real':
                flag = x[ft] < node['threshold']
            else:
                flag = x[ft] in node['categories_split']
            if flag:
                node = node['left_child']
            else:
                node = node['right_child']
        return node['class']
#         pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 1)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)