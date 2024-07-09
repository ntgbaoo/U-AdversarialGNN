from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
import numpy as np

class ISplitter:
    def split(X,y):
        pass

class Splitter(ISplitter):
    def __init__(self,
                 stratified:bool,
                 n_split:int,
                 train_ratio:float,
                 val_ratio:float,
                 test_ratio:float,
                 seed:int
                 ) -> None:
        self.stratified = stratified
        self.n_split = n_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.outer_splitter = None
        self.inner_splitter = None
        self._init_splitter()

    def _init_splitter(self):
        if self.stratified:
            self.outer_splitter = StratifiedShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = StratifiedShuffleSplit(1,test_size=self.val_ratio,random_state=self.seed)
        else:
            self.outer_splitter = ShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = ShuffleSplit(1,test_size=self.val_ratio,random_state=self.seed)

    def split(self,X:np.ndarray,
              y: np.ndarray = None):
        # print(y)
        if self.stratified:
            assert y is not None, "Labels y are missing for stratified split."
            assert X.shape[0] == len(y)
        outer_idx =np.array(range(X.shape[0]))

        for i, (train_index_outer, test_index) in enumerate(self.outer_splitter.split(outer_idx, y)):
            fold_test_index = outer_idx[test_index]
            

            inner_idx = outer_idx[train_index_outer]
            inner_y = y[train_index_outer]
            for j, (train_index_inner, val_index) in enumerate(self.inner_splitter.split(inner_idx, inner_y)):
                fold_train_index = inner_idx[train_index_inner]
                fold_val_index = inner_idx[val_index]

            
           
            assert len(set(fold_train_index.tolist() + fold_val_index.tolist() + fold_test_index.tolist())) == X.shape[0]
        



if __name__ == '__main__':
    X = np.random.rand(10, 1433)
    y = np.random.randint(1, size=10)
    # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.8, random_state=0)
    
    
    # sss= StratifiedKFold(n_splits=5)
    # sss.get_n_splits(X, y)
    # # 5
    # print(sss)
    # # StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    # for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    #     print(len(train_index))
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")

    split = Splitter(True,2,0.2,0.2,0.6,0)
    split.split(X,y)
