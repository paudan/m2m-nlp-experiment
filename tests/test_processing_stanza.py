import os
import unittest
from processing import StanzaNLPProcessor


class StanzaProcessingTestcase(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        os.chdir('..')
        cls.processor = StanzaNLPProcessor()

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
        nouns = self.processor.extract_nouns('print invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_nouns('create invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_nouns('start invoice entry')
        self.assertListEqual(nouns, ['invoice entry'])
        nouns = self.processor.extract_nouns('start at the beginning')
        self.assertListEqual(nouns, ['beginning'])
        nouns = self.processor.extract_nouns('create from scratch')
        self.assertListEqual(nouns, ['scratch'])

    def test_extract_verb(self):
        verb = self.processor.extract_verb('print invoice entry')
        self.assertEquals(verb, 'print')
        verb = self.processor.extract_verb('create invoice entry')
        self.assertEquals(verb, 'create')
        verb = self.processor.extract_verb('start invoice entry')
        self.assertEquals(verb, 'start')
        verb = self.processor.extract_verb('start at the beginning')
        self.assertEquals(verb, 'start at')
        verb = self.processor.extract_verb('create from scratch')
        self.assertEquals(verb, 'create from')

    def test_extract_individual_concepts(self):
        concepts = self.processor.extract_individual_concepts('visit America')
        self.assertListEqual(concepts, ['America'])

    def test_corner_case(self):
        verb = self.processor.extract_verb('Review & Escalate')
        self.assertEqual(verb, 'Escalate')
        verb = self.processor.extract_verb('Review and Escalate')
        self.assertEqual(verb, 'Escalate')


if __name__ == '__main__':
    unittest.main()