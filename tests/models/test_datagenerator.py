from src.models.datagenerator import DataGenerator
import pandas as pd
import unittest



class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        nouns = ['dog', 'cat']
        genders = ['male', 'female']
        self.df = pd.DataFrame(zip(nouns, genders))
        
        self.generator = DataGenerator(self.df)
    def tearDown(self):
        pass

    def test_pad_sequence(self):
        self.assertEqual(self.generator.pad_sequence(self.df, 3), [0, 1, '<pad>'])