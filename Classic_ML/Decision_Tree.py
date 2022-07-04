
import numpy as np
from tqdm import tqdm

# Info gain is used to find the best split (use entropy or gini index)
def info_gain(parent, left_ch, right_ch, method):
    p_size = parent.size
    l_size = left_ch.size
    r_size = right_ch.size

    ig = method(parent) - \
         (l_size / p_size) * method(left_ch) - \
         (r_size / p_size) * method(right_ch)
    return ig


# Entropy formula
def entropy(Y):
    total_obj = Y.size
    unique, counts = np.unique(Y, return_counts=True)
    entropy = 0
    for elem in counts:
        class_prob = elem / total_obj
        entropy -= class_prob * np.log2(class_prob)
    return entropy

# Gini index formula
def gini(Y):
    total_obj = Y.size
    unique, counts = np.unique(Y, return_counts=True)
    gini = 0
    for elem in counts:
        class_prob = elem / total_obj
        gini += class_prob ** 2
    return 1 - gini


# Check all the possible combinations of features
#  and it's values and get the best possible
#  split using information gain.
def get_best_split(X, Y):
    curr_IG = 0
    l_ch = None
    r_ch = None
    condition = None

    # Iterate for each i-individ and j-feature
    for i in tqdm(range(0, X.shape[0])):
        for j in range(0, X.shape[1]):
            elem = X[i, j]

            # Introduce arrays to store the data values
            #  for left and right child
            itr = 0
            l_Y_arr = np.array([])
            r_Y_arr = np.array([])
            l_X_arr = np.array([])
            r_X_arr = np.array([])
            for comp_elem in X[:, j]:
                corresp_Y = Y[itr][np.newaxis]
                corresp_X = X[itr]

                if elem >= comp_elem:
                    if l_Y_arr.size == 0 and l_X_arr.size == 0:
                        l_Y_arr = corresp_Y
                        l_X_arr = corresp_X
                    else:
                        l_X_arr = np.vstack((l_X_arr, corresp_X))
                        l_Y_arr = np.vstack((l_Y_arr, corresp_Y))

                else:
                    if r_Y_arr.size == 0 and r_X_arr.size == 0:
                        r_Y_arr = corresp_Y
                        r_X_arr = corresp_X
                    else:
                        r_X_arr = np.vstack((r_X_arr, corresp_X))
                        r_Y_arr = np.vstack((r_Y_arr, corresp_Y))
                itr += 1

            # Check information gain and save max in left and right node.
            IG = info_gain(Y, l_Y_arr, r_Y_arr, gini)
            if curr_IG < IG:
                l_ch = Node(None, l_X_arr, l_Y_arr, leaf=True)
                r_ch = Node(None, r_X_arr, r_Y_arr, leaf=True)


                # Set prediction if there is only one class in node.
                l_unique, l_counts = np.unique(l_Y_arr, return_counts=True)
                r_unique, r_counts = np.unique(r_Y_arr, return_counts=True)
                if len(l_unique) == 1:
                    l_ch.prediction = l_unique[0]
                if len(r_unique) == 1:
                    r_ch.prediction = r_unique[0]

                curr_IG = IG
                condition = [i, j]
            pass

    return [condition, l_ch, r_ch]




"""
    Node class for tree.
"""
class Node:
    def __init__(self, condition, X, Y, left_ch=None, right_ch=None, root=False, leaf=False, prediction=None):
        self.condition = condition
        self.X = X
        self.Y = Y
        self.left_ch = left_ch
        self.right_ch = right_ch
        self.root = root
        self.leaf = leaf
        self.prediction = prediction


"""
    Main tree class.
"""
class DecisionTree():
    def __init__(self):
        self.root = None
        self.depth = 0
        self.nodes = 0


    # Return all leafs that could be modified.
    def get_leafs(self, root, leaf_list=None):
        if leaf_list is None:
            leaf_list = []

        l_ch = root.left_ch
        r_ch = root.right_ch

        if l_ch.leaf and l_ch.prediction is None:
            leaf_list.append(l_ch)
        elif not l_ch.leaf:
            leaf_list = self.get_leafs(l_ch, leaf_list=leaf_list)

        if r_ch.leaf and r_ch.prediction is None:
            leaf_list.append(r_ch)
        elif not r_ch.leaf:
            leaf_list = self.get_leafs(r_ch, leaf_list=leaf_list)

        return leaf_list


    # Save data to variables.
    def fit(self, X, Y):
        self.X = X
        self.Y = Y


    # Train model.
    def train(self, max_depth=4):
        # Create initial root node before the main loop
        cond, l_ch, r_ch, = get_best_split(self.X, self.Y)
        self.root = Node(cond, self.X, self.Y, left_ch=l_ch, right_ch=r_ch, root=True)
        self.depth += 1

        while True:
            leaf_modif_list = self.get_leafs(self.root)

            # Set predictions if the tree is not
            #  completed but have maximal depth
            if self.depth == max_depth:
                for leaf in leaf_modif_list:
                    leaf.prediction = np.mean(leaf.Y)
                    leaf.leaf = True
                break

            # Break if the tree is complete
            if not leaf_modif_list:
                break

            for leaf in leaf_modif_list:
                cond, l_ch, r_ch, = get_best_split(leaf.X, leaf.Y)
                leaf.left_ch = l_ch
                leaf.right_ch = r_ch
                leaf.condition = cond
                leaf.leaf = False
                self.nodes += 1
            self.depth += 1


    # Predict single individ (if X is a vector).
    def predict_single(self, root, X):
        pred = root.prediction
        cond = root.condition
        out = None

        # Base case when there is prediction.
        if pred is not None:
            return pred

        elif cond is not None:
            cond_i, cond_j = cond
            if self.X[cond_i, cond_j] >= X[cond_j]:
                out = self.predict_single(root.left_ch, X)
            else:
                out = self.predict_single(root.right_ch, X)
        return out


    # Predict multiple individs using 'predict_single' method.
    def predict_many(self, X):
        out = []
        for elem in X:
            pred = self.predict_single(self.root, elem)
            out.append(pred)
        return np.array(out)[np.newaxis].T








if __name__ == '__main__':
    X_train = np.array([[1, 1], [1, 3], [2, 3], [-10, -1], [2, 1], [3, 1], [3, 3], [20, 10]])
    Y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    D_T = DecisionTree()
    D_T.fit(X_train, Y_train)
    D_T.train(max_depth=5)
    c = D_T.predict_many(np.array([[-100, 1], [100, -1]]))
    print(c)