
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
    
    def test_transform_with_list(self):

        X_in = [['Foo','and','bar'],
                ['Bar','and','foo']]

        X_out = [{'Word': 'Foo', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'bar', 'Index': 2},
                 {'Word': 'Bar', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'foo', 'Index': 2}]

        self.assertEqual(self.extr_test.transform(X=X_in), X_out)
        
    def test_transform_with_series(self):
        X_in = pd.Series([['Foo','and','bar'],
                            ['Bar','and','foo']])
        
        X_out = [{'Word': 'Foo', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'bar', 'Index': 2},
                 {'Word': 'Bar', 'Index': 0}, {'Word': 'and', 'Index': 1}, {'Word': 'foo', 'Index': 2}]
        
        self.assertEqual(self.extr_test.transform(X=X_in), X_out)

    def test_transform_with_dataframe(self):
        X_in = pd.DataFrame(
            {'Text':['Foo','and','bar', 'Bar','and','foo'],
             'Nontext':[11,12,13,14,15,16]},
            index=[1,1,1,2,2,2]
            )

        X_out = [{'Word': 'Foo', 'Index': 0, 'Nontext':11}, {'Word': 'and', 'Index': 1, 'Nontext':12}, {'Word': 'bar', 'Index': 2, 'Nontext':13},
                    {'Word': 'Bar', 'Index': 0, 'Nontext':14}, {'Word': 'and', 'Index': 1, 'Nontext':15}, {'Word': 'foo', 'Index': 2, 'Nontext':16}]
        self.assertEqual(self.extr_test_df.transform(X=X_in), X_out)

    def test_featurizers_syllables(self):
        x_in_syllables = [['Pörssi-iltaa', 'on', 'kauan', 'suunniteltu'],['Ruokapurkkeja','oli', '5:lle']]

        X_out = [
            {'word':'Pörssi-iltaa','suffix_3':'si','suffix_2':'il','suffix_1':'taa'},
            {'word':'on','suffix_3':'','suffix_2':'','suffix_1':''},
            {'word':'kauan','suffix_3':'','suffix_2':'','suffix_1':'an'},
            {'word':'suunniteltu','suffix_3':'ni','suffix_2':'tel','suffix_1':'tu'},
            {'word':'Ruokapurkkeja','suffix_3':'purk','suffix_2':'ke','suffix_1':'ja'},
            {'word':'oli','suffix_3':'','suffix_2':'','suffix_1':'li'},
            {'word':'5:lle','suffix_3':'','suffix_2':'','suffix_1':''},
                 ]
        X_real = [{'word':word_dict['word'], 'suffix_3':word_dict['suffix_3'], 'suffix_2':word_dict['suffix_2'],'suffix_1':word_dict['suffix_1']}
                  for word_dict in self.extr_syl.transform(x_in_syllables)]

        self.maxDiff = None
        self.assertEqual(X_out, X_real)

                           


if __name__ == '__main__':
    unittest.main()
