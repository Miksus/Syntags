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
        '''Transforms list of sentences to list of dictionaries
        >>> [['Word','word'], ['Word','word'], ...]
            return [{...}, {...}, {...}, {...}, ...]
        '''
        
        X_transformed = []
        for sentence in X:
            for word_index in range(len(sentence)):
                x_dict = extractor_func(sentence, word_index)
                X_transformed.append(x_dict)
                
        return X_transformed

    def _to_sentences(self, obj, columns=None):
        '''Return list of sentences


        >>> obj = [['Word','word'], ['Word','word']]
        OR
        >>> obj = pd.Series([['Word','word'], ['Word','word']])
        OR
        >>> obj = pd.Series(['Word','word', 'Word','word'], index=[1,1,2,2])
        return [['Word','word'], ['Word','word']]


        >>> obj = pd.DataFrame({
                'Text':['Word', 'word', 'Word', 'word'],
                'Pretag':[tag,tag,tag,tag],
                'feat1':[11,12,13,14], 'feat2':[21,22,23,24]},
                index=[1,1,2,2])
            columns = ['Text', 'Pretag']
        return {'Text'[['Word','word'], ['Word','word']], 'Pretag'[[tag,tag],[tag,tag]]}, 
                [{'feat1':11, 'feat2':21}, {'feat1':12, 'feat2':22},{'feat1':13, 'feat2':23},{'feat1':14, 'feat2':24}]

                
        '''
        def is_series_of_lists(series):
            # series = pd.Series([[W,w,w],[W,w,w]...])
            return series.apply(lambda x: isinstance(x, list)).any()

        def is_sentence_id_index(index):
            return (not isinstance(index, pd.MultiIndex) and(index >= 0).all()
                    and index.duplicated(keep=False).mean() >0.5)

        def series_to_sentences(series):
            if is_series_of_lists(series):
                return series.tolist()
            
            elif is_sentence_id_index(series.index):
                grouped_sentences = series.groupby(obj.index, sort=False)
                return grouped_sentences.apply(lambda sent: sent.values.tolist()).tolist()
            
            else:
                raise NotImplementedError('No conversion for series structure')

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
        else:
            raise TypeError('Cannot turn {} to sentences'.format(type(obj)))

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




