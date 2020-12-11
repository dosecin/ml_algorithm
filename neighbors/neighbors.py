import numpy as np


class KDNode:
    def __init__(self, value: np.ndarray, label: np.ndarray, deep: int):
        self.value = value
        self.label = label
        self.dimension = int(deep % value.shape[0])
        self.left = None
        self.right = None

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right


class KDTree:
    def __init__(self, dataIn: np.ndarray, dataLabel: np.ndarray):
        self.root, left, right, left_l, right_l = KDTree._split(
            dataIn, dataLabel, 0)
        self._create(self.root, left, left_l, right, right_l, 0)

    def _split(dataIn: np.ndarray, dataLabel: np.ndarray, deep: int):
        n_sample, n_feature = dataIn.shape
        if n_sample == 0:
            return None, None, None, None, None
        dataIn = dataIn[dataIn[:, int(deep % n_feature)].argsort()]
        mid = n_sample//2
        return KDNode(dataIn[mid], dataLabel[mid], deep), dataIn[:mid], dataIn[mid+1:], dataLabel[:mid], dataLabel[mid+1:]

    def _create(self,
                parent: KDNode,
                left: np.ndarray,
                left_l: np.ndarray,
                right: np.ndarray,
                right_l: np.ndarray,
                deep: int):
        l_root, l_left, l_right, l_left_l, l_right_l = KDTree._split(
            left, left_l, deep+1)
        if l_root is not None:
            parent.set_left(l_root)
            self._create(l_root, l_left, l_left_l, l_right, l_right_l, deep+1)

        r_root, r_left, r_right, r_left_l, r_right_l = KDTree._split(
            right, right_l, deep+1)
        if r_root is not None:
            parent.set_right(r_root)
            self._create(r_root, r_left, r_left_l, r_right, r_right_l, deep+1)


def distance2(x, y):
    z = x-y
    return np.dot(z, z.T)


class KNearest:
    def __init__(self, x: np.ndarray, k: int):
        self.k = k
        self.target = x
        self.nodes = None
        self.dis = None

    def append(self, node: KDNode):
        if self.nodes is None:
            self.nodes = [node]
            self.dis = [distance2(node.value, self.target)]
        else:
            self.nodes.append(node)
            self.dis.append(distance2(node.value, self.target))

    def full(self):
        return len(self.nodes) >= self.k

    def max(self):
        maxIdx, maxDis = None, 0
        for i, dis in enumerate(self.dis):
            if dis > maxDis:
                maxIdx = i
                maxDis = dis
        return maxIdx

    def check_and_replace(self, node: KDNode):
        maxIdx = self.max()
        if maxIdx is None:
            return
        dis = distance2(node.value, self.target)
        if dis < self.dis[maxIdx]:
            self.dis[maxIdx] = dis
            self.nodes[maxIdx] = node

    def search_in_child(self, node: KDNode):
        if node is None:
            return
        self.check_and_replace(node)
        maxIdx = self.max()
        if maxIdx is None:
            return
        if np.abs(node.value[node.dimension]-self.target[node.dimension]) >= self.dis[maxIdx]:
            return
        if self.target[node.dimension] < node.value[node.dimension]:
            self.search_in_child(node.right)
        else:
            self.search_in_child(node.left)

    def search(self, node: KDNode):
        if node is None:
            return None

        if self.target[node.dimension] < node.value[node.dimension]:
            self.search(node.left)
        else:
            self.search(node.right)

        if self.nodes is None:
            return self.append(node)

        if not self.full():
            return self.append(node)

        self.search_in_child(node)


class KNearestNeightbors:
    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.kd_tree = KDTree(X, y)

    def predict(self, X: np.ndarray):
        y = []
        for x in X:
            nearest = KNearest(x, self.k)
            nearest.search(self.kd_tree.root)
            lables = {}
            for n in nearest.nodes:
                if n.label in lables:
                    lables[n.label] += 1
                else:
                    lables[n.label] = 1
            maxNum, maxLable = 0, None
            for (l, num) in lables.items():
                if num > maxNum:
                    maxNum = num
                    maxLable = l
            y.append(maxLable)
        return y

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        n_sample = X.shape[0]
        return np.sum(y_pred == y)/n_sample
