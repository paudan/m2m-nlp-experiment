import os
import unittest
from processing import SpacyNLPProcessor


class ProcessingTestCase(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        os.chdir('..')
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

    def test_normalize_verb(self):
        # Single verb
        self.assertTrue(self.processor.normalize_verb('created'), 'creates')
        self.assertTrue(self.processor.normalize_verb('create'), 'creates')
        self.assertTrue(self.processor.normalize_verb('creating'), 'creates')
        # Verb phrase
        self.assertTrue(self.processor.normalize_verb('started with'), 'starts with')
        self.assertTrue(self.processor.normalize_verb('starting with'), 'starts with')

    def test_extract_named_entities(self):
        entities = self.processor.extract_named_entities('visit United States')
        self.assertIsNotNone(entities)
        self.assertEquals(1, len(entities))
        self.assertEquals('United States', entities[0])
        entities = self.processor.extract_named_entities('iTunes system')
        self.assertEquals(0, len(entities))  # Does mpt recognize iTunes

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
        nouns = self.processor.extract_noun_phrases('print invoice entry')
        self.assertListEqual(nouns, ['print', 'entry'])  # Failure!
        nouns = self.processor.extract_noun_phrases('create invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_noun_phrases('start invoice entry')
        self.assertListEqual(nouns, ['entry'])   # Failure!
        nouns = self.processor.extract_noun_phrases('start at the beginning')
        self.assertListEqual(nouns, ['beginning'])
        nouns = self.processor.extract_noun_phrases('create from scratch')
        self.assertListEqual(nouns, ['scratch'])
        nouns = self.processor.extract_noun_phrases("box's office")
        self.assertListEqual(nouns, ["office"])  # Failure: box is recognized as proper noun PNP
        nouns = self.processor.extract_noun_phrases("sign contract for customer")
        self.assertListEqual(nouns, ["contract for customer"])
        nouns = self.processor.extract_noun_phrases("sign contract of customer")
        self.assertListEqual(nouns, ["sign contract of customer"])  # Failure!
        nouns = self.processor.extract_noun_phrases("return invoice for reentry into SAP")
        self.assertListEqual(nouns, ["invoice for reentry into SAP"])
        nouns = self.processor.extract_noun_phrases("Return invoice for reentry into SAP")
        self.assertListEqual(nouns, ["Return invoice for reentry into SAP"])
        nouns = self.processor.extract_noun_phrases("property of the customer")
        self.assertListEqual(nouns, ["property of the customer"])

    def test_extract_verb(self):
        verb = self.processor.extract_verb_phrase('print invoice entry')
        self.assertEquals(verb, 'invoice')  # Failure!
        verb = self.processor.extract_verb_phrase('create invoice entry')
        self.assertEquals(verb, 'create')
        verb = self.processor.extract_verb_phrase('start invoice entry')
        self.assertEquals(verb, 'start invoice')  # Failure!
        verb = self.processor.extract_verb_phrase('start at the beginning')
        self.assertEquals(verb, 'start at')
        verb = self.processor.extract_verb_phrase('create from scratch')
        self.assertEquals(verb, 'create from')
        verb = self.processor.extract_verb_phrase('un-mute speakers')
        self.assertIsNone(verb)  # Failure

    def test_extract_verb_phrases(self):
        verbs = self.processor.extract_verb_phrases('Get M2 Center Position')
        self.assertListEqual(verbs, ["Get"])


    def test_extract_proper_nouns(self):
        concepts = self.processor.extract_proper_nouns('visit America')
        self.assertListEqual(concepts, ['America'])
        concepts = self.processor.extract_proper_nouns('create BR')
        self.assertListEqual(concepts, ['BR'])


if __name__ == '__main__':
    unittest.main()
