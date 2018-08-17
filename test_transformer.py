
import pandas as pd

import unittest

from sentence_transformer import WordFeatureExtractor


class TestExtract(unittest.TestCase):
    
    
    def setUp(self):
        self.extr_simpl = WordFeatureExtractor(columns=None, extract='extraction_simple')
        self.extr_syl = WordFeatureExtractor(columns=None, extract='extraction_syllables')
        self.extr_test = WordFeatureExtractor(columns=None, extract='_extraction_test')

        self.extr_test_df = WordFeatureExtractor(columns='Text', extract='_extraction_test')
        

    def teadDown(self):
        pass
    
    def test_extractor_errors(self):
        with self.assertRaises(AttributeError):
            self.extr_err = WordFeatureExtractor(columns=None, extract='fit')
            self.extr_err = WordFeatureExtractor(columns=None, extract='NOTFOUND')
    
    def test_transform(self):

        X_in = [['Foo','and','bar'],
                ['Bar','and','foo']]

        X_in_s = pd.Series(X_in)
        X_in_df = pd.DataFrame(
            {'Text':[word for sent in X_in for word in sent],
             'Nontext':[99]*sum([len(sent) for sent in X_in])},
            index=[1,1,1,2,2,2]
            )
        
        X_out = [{'Word': 'Foo', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'bar', 'Index': 2},
                 {'Word': 'Bar', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'foo', 'Index': 2}]

        X_out_df = [{'Word': 'Foo', 'Index': 0, 'Nontext':99}, {'Word': 'and', 'Index': 1, 'Nontext':99}, {'Word': 'bar', 'Index': 2, 'Nontext':99},
                    {'Word': 'Bar', 'Index': 0, 'Nontext':99}, {'Word': 'and', 'Index': 1, 'Nontext':99}, {'Word': 'foo', 'Index': 2, 'Nontext':99}]

        self.assertEqual(self.extr_test.transform(X=X_in), X_out)
        self.assertEqual(self.extr_test.transform(X=X_in_s), X_out)
        
        self.assertEqual(self.extr_test_df.transform(X=X_in_df), X_out_df)

    def test_featurizers_syllables(self):
        x_in_syllables = [['Tarra-arkki', 'on', 'kauan', 'kestänyt','5:lle','henkilölle']]

        X_out = [
            {'word':'Tarra-arkki','suffix_3':'ra','suffix_2':'ark','suffix_1':'ki'},
            {'word':'on','suffix_3':'','suffix_2':'','suffix_1':''},
            {'word':'kauan','suffix_3':'','suffix_2':'','suffix_1':'an'},
            {'word':'kestänyt','suffix_3':'','suffix_2':'tä','suffix_1':'nyt'},
            {'word':'5:lle','suffix_3':'','suffix_2':'','suffix_1':'lle'},
            {'word':'henkilölle','suffix_3':'ki','suffix_2':'löl','suffix_1':'le'},
                 ]
        X_real = [{'word':word_dict['word'], 'suffix_3':word_dict['suffix_3'], 'suffix_2':word_dict['suffix_2'],'suffix_1':word_dict['suffix_1']}
                  for word_dict in self.extr_syl.transform(x_in_syllables)]

        self.maxDiff = None
        self.assertEqual(X_out, X_real)

                           


if __name__ == '__main__':
    unittest.main()
