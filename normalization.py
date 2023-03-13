class MinMaxScaler(object):
    def __init__(self, feature_range=(0, 1)):
        self.low, self.high = feature_range

    def fit_transform(self, X):
        min = X.min(axis=0)
        max = X.max(axis=0)

        x = (X - min) / (max - min)
        return x * (self.high - self.low) + self.low

class MinMaxScalerList(list):
    def __init__(self,feature_range=(0,1)):
        self.low, self.high = feature_range
        
    def fit_transform(self, X):
        min_v = min(X)
        max_v = max(X)
        
        x = (X - min_v) / ( max_v-min_v)
        return x * (self.high - self.low) + self.low