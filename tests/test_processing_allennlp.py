import os
import unittest
from processing import AllenNLPProcessor


class AllenProcessingTestCase(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        os.chdir('..')
        cls.processor = AllenNLPProcessor()

    def test_extract_named_entities(self):
        entities = self.processor.extract_named_entities('visit United States')
        self.assertIsNotNone(entities)
        self.assertEquals(1, len(entities))
        self.assertEquals('United States', entities[0])

    def test_get_named_entity(self):
        entity = self.processor.get_named_entity('visit United States')
        self.assertIsNotNone(entity)
        self.assertEquals('United States', entity)
        entity = self.processor.get_named_entity('visit united states')
        self.assertIsNone(entity)   # Cannot deal with normalized lowercase entities!

    def test_get_named_entity_type(self):
        self.assertEquals('LOCATION', self.processor.get_named_entity_type('visit United States'))
        self.assertEquals('ORGANIZATION', self.processor.get_named_entity_type('report Microsoft'))
        self.assertEquals('ORGANIZATION', self.processor.get_named_entity_type('report IBM'))
        self.assertEquals('PERSON', self.processor.get_named_entity_type('meet John Adams'))

    def test_extract_noun(self):
        nouns = self.processor.extract_noun_phrases('print invoice entry')
        self.assertListEqual(nouns, ['print', 'entry'])
        nouns = self.processor.extract_noun_phrases('create invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_noun_phrases('start invoice entry')
        self.assertListEqual(nouns, ['entry'])
        nouns = self.processor.extract_noun_phrases('start at the beginning')
        self.assertListEqual(nouns, ['beginning'])
        nouns = self.processor.extract_noun_phrases('create from scratch')
        self.assertListEqual(nouns, ['scratch'])

    def test_extract_verb(self):
        verb = self.processor.extract_verb_phrase('print invoice entry')
        self.assertEquals(verb, 'invoice')
        verb = self.processor.extract_verb_phrase('create invoice entry')
        self.assertEquals(verb, 'create')
        verb = self.processor.extract_verb_phrase('start invoice entry')
        self.assertEquals(verb, 'start invoice')
        verb = self.processor.extract_verb_phrase('start at the beginning')
        self.assertEquals(verb, 'start at')
        verb = self.processor.extract_verb_phrase('create from scratch')
        self.assertEquals(verb, 'create from')

    def test_extract_individual_concepts(self):
        concepts = self.processor.extract_proper_nouns('visit America')
        self.assertListEqual(concepts, ['America'])

    def test_corner_case(self):
        verb = self.processor.extract_verb_phrase('Review & Escalate')
        self.assertIsNone(verb)
        verb = self.processor.extract_verb_phrase('Review and Escalate')
        self.assertIsNone(verb)

if __name__ == '__main__':
    unittest.main()
