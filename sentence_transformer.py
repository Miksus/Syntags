from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import featurizers




class WordFeatureExtractor(BaseEstimator, TransformerMixin):




    def __init__(self, columns=None, extract='extraction_syllables'):
        'Transforms pandas series to list of word features'
        columns = [columns] if isinstance(columns, (str, type(None))) else columns
        extract = [extract] if isinstance(extract, (str, type(None))) else extract

        
        extract = [getattr(featurizers, func_str) for func_str in extract]
        self.tasks = list(
            zip(columns, extract)
            )


        
    def fit(self, X, y=None):
        return self

    
    def transform(self, X):
        'Transform a pandas series of words to list of dictionaries'
        keys, _ = zip(*self.tasks)

        X_to_transform, X_other = self._to_sentences(X, list(keys))

        
        X_transformed = []
        for key, func in self.tasks:
            X_sents = X_to_transform[key] if key is not None \
                      else X_to_transform
            X_transformed.append(self._transform_sentences(X_sents, extractor_func=func))

        return self._merge_output(*X_transformed, X_other)

    
    def _transform_sentences(self, X, extractor_func):
        'Transforms list of sentences to list of dictionaries'
        
        X_transformed = []
        for sentence in X:
            for word_index in range(len(sentence)):
                x_dict = extractor_func(sentence, word_index)
                X_transformed.append(x_dict)
                
        return X_transformed

    def _to_sentences(self, obj, columns=None):
        'Return list of sentences'

        def series_to_sentences(series):
            # Inner function
            if series.apply(lambda x: isinstance(x, list)).all():
                return series.tolist()
            else:
                return series.groupby(obj.index, sort=False).apply(lambda sent: sent.values.tolist()).tolist()


        if isinstance(obj, (list, tuple)):
            sents = obj
            other = None
            
        elif isinstance(obj, pd.Series):
            sents = series_to_sentences(obj)
            other = None
            
        elif isinstance(obj, pd.DataFrame):
            sents = {col:series_to_sentences(obj[col])
                     for col in columns}
            other = obj.drop(columns=columns).to_dict('records')

        return sents, other

  
        

    def _merge_output(self, *args, suffix=False):
        args = [arg for arg in args if arg is not None]
        if len(args) == 1:
            return args[0]
            
        main = []
        max_len = max((len(arg) for arg in args))
        for index in range(max_len):
            master_dict = {}
            for n, arg in enumerate(args):
                if suffix is True:
                    arg[index] = {key+'_{}'.format(chr(n+65)):arg[index][key] for key in arg[index]}
                master_dict = {**master_dict, **arg[index]}
            main.append(master_dict)
        
        return main




