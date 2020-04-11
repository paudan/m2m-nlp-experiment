import unittest
from processing import SpacyNLPProcessor


class ProcessingTestCase(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        cls.processor = SpacyNLPProcessor()

    def test_is_hypernym(self):
        self.assertFalse(self.processor.is_hypernym('cat', 'dog'))
        self.assertTrue(self.processor.is_hypernym('domestic animal', 'dog'))
        self.assertTrue(self.processor.is_hypernym('animal', 'domestic animal'))
        # If full hierarchy is not processed, such relationships are asserted as False
        self.assertFalse(self.processor.is_hypernym('entity', 'dog'))
        self.assertFalse(self.processor.is_hypernym('organism', 'dog'))

    def test_is_hypernym2(self):
        self.assertTrue(self.processor.is_hypernym('domestic animal', 'dog', full_hierarchy=True))
        self.assertTrue(self.processor.is_hypernym('object', 'dog', full_hierarchy=True))
        self.assertTrue(self.processor.is_hypernym('entity', 'dog', full_hierarchy=True))
        self.assertTrue(self.processor.is_hypernym('organism', 'dog', full_hierarchy=True))
        self.assertTrue(self.processor.is_hypernym('physical entity', 'dog', full_hierarchy=True))

    def test_is_hyponym(self):
        self.assertTrue(self.processor.is_hyponym('poodle', 'dog'))
        self.assertTrue(self.processor.is_hyponym('dog', 'domestic animal'))

    def test_is_holonym(self):
        self.assertTrue(self.processor.is_holonym('human face', 'eye'))
        self.assertTrue(self.processor.is_holonym('computer', 'keyboard'))

    def test_is_meronym(self):  # Part-of
        self.assertTrue(self.processor.is_meronym('eye', 'face'))
        self.assertTrue(self.processor.is_meronym('keyboard', 'computer'))

    def test_is_synonym(self):
        self.assertTrue(self.processor.is_synonym('president', 'chair'))
        self.assertTrue(self.processor.is_synonym('chair', 'president'))

    def test_extract_named_entities(self):
        entities = self.processor.extract_named_entities('visit United States')
        self.assertIsNotNone(entities)
        self.assertEquals(1, len(entities))
        self.assertEquals('United States', entities[0])

    def test_get_named_entity(self):
        entity = self.processor.get_named_entity('visit United States')
        self.assertIsNotNone(entity)
        self.assertEquals('United States', entity)

    def test_get_named_entity_type(self):
        self.assertEquals('LOCATION', self.processor.get_named_entity_type('visit United States'))
        self.assertEquals('ORGANIZATION', self.processor.get_named_entity_type('report Microsoft'))
        self.assertEquals('ORGANIZATION', self.processor.get_named_entity_type('report IBM'))
        # Spacy does not recognize person names
        self.assertIsNone(self.processor.get_named_entity_type('meet John Adams'))

    def test_extract_noun(self):
        nouns = self.processor.extract_nouns('print invoice entry')
        self.assertListEqual(nouns, ['print', 'entry'])  # Failure!
        nouns = self.processor.extract_nouns('create invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_nouns('start invoice entry')
        self.assertListEqual(nouns, ['entry'])   # Failure!
        nouns = self.processor.extract_nouns('start at the beginning')
        self.assertListEqual(nouns, ['the beginning'])
        nouns = self.processor.extract_nouns('create from scratch')
        self.assertListEqual(nouns, ['scratch'])

    def test_extract_verb(self):
        verb = self.processor.extract_verb('print invoice entry')
        self.assertEquals(verb, 'invoice')  # Failure!
        verb = self.processor.extract_verb('create invoice entry')
        self.assertEquals(verb, 'create')
        verb = self.processor.extract_verb('start invoice entry')
        self.assertEquals(verb, 'start invoice')  # Failure!
        verb = self.processor.extract_verb('start at the beginning')
        self.assertEquals(verb, 'start at')
        verb = self.processor.extract_verb('create from scratch')
        self.assertEquals(verb, 'create from')


if __name__ == '__main__':
    unittest.main()
