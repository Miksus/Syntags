# Syntags
Lightweight Part of Speech tagger with support for custom word featuring. Development of the default featuring functions are focused on Finnish grammar.

## Getting Started




### Requirements

```
Python 3
Pandas
Numpy
Scikit-learn
```


### Train the tagger

The tagger does not come pretrained thus requires pretagged (labeled) data. 
The input data to the transformer can take various forms:
- list of sentences
```
>>> [["This", "is", "first", "example"], ["This", "is", "another", "example"]]
```
- pandas Series with index indicating sentence number
```
>>> pd.Series(["This", "is", "first", "example", "This", "is", "another", "example"],
              index=[1,1,1,1,2,2,2,2])
```
- pandas Series containing lists
```
>>> pd.Series([["This", "is", "first", "example"], ["This", "is", "another", "example"]])
```
- pandas DataFrame with the text column in same format as above Series examples. The other columns are considered as additional features and passed to the estimator.
