from sentence_transformer import WordFeatureExtractor


class Tagger:

    def __init__(self, estimator=None, **kwargs):
        if estimator is None:
            estimator = DecisionTreeClassifier(criterion='gini')
        self._pipe = Pipeline([
            ('extractor',WordFeatureExtractor(**kwargs)),
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', estimator)
            ])


    def __getitem__(self, key):
        return self._pipe.named_steps[key]

    def fit(self, X, y)
        self._pipe.fit(X, y)

    @property
    def extractor(self):
        return self.named_steps['extractor']
    
    @property
    def vectorizer(self):
        return self.named_steps['vectorizer']

    @property
    def classifier(self):
        return self.named_steps['classifier']



def get_tagger_pipeline(estimator=None, **kwargs):
    if estimator is None:
        estimator = DecisionTreeClassifier(criterion='gini')
        
    return Pipeline([
            ('extractor',WordFeatureExtractor(**kwargs)),
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', estimator)
            ])
    
