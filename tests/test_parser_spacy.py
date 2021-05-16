import os
import spacy
import unittest
from parsing import ConjunctiveParser


class SpacyParserTestCase(unittest.TestCase):
    nlp, parser = None, None

    @classmethod
    def setUpClass(cls):
        os.chdir('..')
        cls.nlp = spacy.load("spacy/en_core_web_lg/en_core_web_lg/en_core_web_lg")
        cls.parser = ConjunctiveParser(cls.nlp)

    def test_case1(self):
        phrase = "Initiate, produce and test software"  # "test" is failed to be recognized as a verb
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc), {('produce', 'test software'), ('Initiate', 'test software')})

    def test_case2(self):
        phrase = "Implement specification, software and tests"
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Implement', 'specification'), ('Implement', 'software'), ('Implement', 'tests')})

    def test_case3(self):
        doc = self.nlp("Implement initial specification, required software or test items")
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Implement', 'initial specification'), ('Implement', 'required software'), ('Implement', 'test items')})

    def test_case4(self):
        phrase = "Write specification, produce software and create tests"
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Write', 'specification'), ('produce', 'software'), ('create', 'tests')})

    def test_case5(self):
        phrase = 'Implement initial specification, required software or test items'
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Implement', 'initial specification'), ('Implement', 'required software'), ('Implement', 'test items')})

    def test_case5a(self):
        # "required" is recognized as a verb
        phrase = 'Implement specification, required software or test items'
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Implement', 'specification'), ('Implement', 'required software'), ('Implement', 'test items')})

    def test_case6(self):
        phrase = 'Create specification and develop software, as well as improve tools'
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Create', 'specification'), ('develop', 'software'), ('improve', 'tools')})

    def test_case7(self):
        phrase = 'Create specification and develop software, start testing and deploying'
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Create', 'specification'), ('develop', 'software')})

    def test_case8(self):
        phrase = 'Create specification and develop software, start tests, integration and deployment'
        doc = self.nlp(phrase)
        self.assertSetEqual(self.parser.process_parsed(doc),
                            {('Create', 'specification'), ('develop', 'software'), ('start', 'tests'), ('start', 'integration'), ('start', 'deployment')})

    def test_leftmost(self):
        phrase = 'Create specification and develop software, start tests, integration and deployment'
        doc = self.nlp(phrase)
        self.assertEquals(doc[9].text, 'integration')
        self.assertEquals(self.parser.leftmost_verb(doc[9]).text, 'start')

    def test_leftmost_case2(self):
        phrase = "Implement specification, software and tests"
        doc = self.nlp(phrase)
        self.assertEquals(doc[5].text, 'tests')
        self.assertEquals(self.parser.leftmost_verb(doc[5]).text, 'Implement')

    def test_leftmost_case3(self):
        doc = self.nlp("Implement initial specification, required software or test items")
        self.assertEquals(doc[8].text, 'items')
        self.assertEquals(self.parser.leftmost_verb(doc[8]).text, 'Implement')

    def test_leftmost_case4(self):
        doc = self.nlp("Write specification, produce software and create tests")
        self.assertEquals(doc[7].text, 'tests')
        self.assertEquals(self.parser.leftmost_verb(doc[7]).text, 'create')
        self.assertEquals(doc[4].text, 'software')
        self.assertEquals(self.parser.leftmost_verb(doc[4]).text, 'produce')


if __name__ == '__main__':
    unittest.main()


