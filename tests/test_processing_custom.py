import os
import unittest
from processing import CustomNLPProcessor


class CustomProcessingTestcase(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        os.chdir('..')
        cls.processor = CustomNLPProcessor()

    def test_extract_noun(self):
        nouns = self.processor.extract_noun_phrases('print invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_noun_phrases('create invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_noun_phrases('start invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_noun_phrases('start at the beginning')
        self.assertListEqual(nouns, ['beginning'])
        nouns = self.processor.extract_noun_phrases('create from scratch')
        self.assertListEqual(nouns, ['scratch'])
        nouns = self.processor.extract_noun_phrases("box's office")
        self.assertListEqual(nouns, ["box' s office"])
        nouns = self.processor.extract_noun_phrases("sign contract for customer")
        self.assertListEqual(nouns, ["contract for customer"])
        nouns = self.processor.extract_noun_phrases("sign contract of customer")
        self.assertListEqual(nouns, ["contract of customer"])
        nouns = self.processor.extract_noun_phrases("return invoice for reentry into SAP")
        self.assertListEqual(nouns, ["invoice for reentry"])
        nouns = self.processor.extract_noun_phrases("Return invoice for reentry into SAP")
        self.assertListEqual(nouns, ["reentry"])  # Failure!
        nouns = self.processor.extract_noun_phrases("property of the customer")
        self.assertListEqual(nouns, ["property of the customer"])

    def test_extract_verb(self):
        verb = self.processor.extract_verb_phrase('print invoice entry')
        self.assertEquals(verb, 'print')
        verb = self.processor.extract_verb_phrase('create invoice entry')
        self.assertEquals(verb, 'create')
        verb = self.processor.extract_verb_phrase('start invoice entry')
        self.assertEquals(verb, 'start')
        verb = self.processor.extract_verb_phrase('start at the beginning')
        self.assertEquals(verb, 'start at')
        verb = self.processor.extract_verb_phrase('create from scratch')
        self.assertEquals(verb, 'create from')

    def test_extract_proper_nouns(self):
        concepts = self.processor.extract_proper_nouns('visit America')
        self.assertListEqual(concepts, ['America'])
        concepts = self.processor.extract_proper_nouns('create BR')
        self.assertListEqual(concepts, ['BR'])



if __name__ == '__main__':
    unittest.main()
