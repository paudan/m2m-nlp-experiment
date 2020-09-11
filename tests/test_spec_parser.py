import unittest
from parser import *

TESTCASE_DIR = 'testcases'

class SpecificationTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        os.chdir('..')

    def validate_instance_spec(self, instanceSpec):
        self.assertIsNotNone(instanceSpec.get('source'))
        self.assertEqual(len(instanceSpec.get('source')), 1)

    def validate_instance_with_pattern_spec(self, instanceSpec):
        self.validate_instance_spec(instanceSpec)
        self.assertIsNotNone(instanceSpec.get('transformationPatternSpecification'))

    def validate_synonymy_instance_spec(self, instanceSpec):
        self.validate_instance_spec(instanceSpec)
        self.assertIsInstance(instanceSpec.get('output'), list)
        if instanceSpec.get('existingElements') is not None:
            self.assertIsInstance(instanceSpec.get('existingElements'), list)

    def validate_mapping(self, mapping, src_type=dict):
        self.assertIsNotNone(mapping.get('target'))
        self.assertIsNotNone(mapping.get('sourceElement'))
        self.assertIsInstance(mapping.get('sourceElement'), src_type)
        self.assertIsNotNone(mapping.get('targetElement'))
        self.assertIsInstance(mapping.get('targetElement'), dict)
        self.assertIsNotNone(mapping.get('sourceInstances'))
        self.assertIsInstance(mapping.get('sourceInstances'), list)
        self.assertIsNotNone(mapping.get('targetInstances'))
        self.assertIsInstance(mapping.get('targetInstances'), list)

    def validate_processed_spec(self, spec, src_type=dict):
        self.assertIsNotNone(spec.get("conditionedMappingPattern"))
        self.assertIsInstance(spec.get("conditionedMappingPattern"), list)
        for mappingPattern in spec.get("conditionedMappingPattern"):
            self.assertIsNotNone(mappingPattern.get("mappings"))
            for mapping in mappingPattern.get("mappings"):
                self.validate_mapping(mapping, src_type=src_type)
        if spec.get("defaultMappingPattern") is not None:
            for mapping in spec.get("defaultMappingPattern"):
                self.validate_mapping(mapping)

    def process_specification(self, filename):
        with open(filename, 'r') as f:
            instanceSpec = json.load(f)
        return instanceSpec, process_instance_specification(instanceSpec)

    def test_cond_spec_1(self):
        spec_file = os.path.join(TESTCASE_DIR, 'cond_invoice_class.json')
        instanceSpec, spec = self.process_specification(spec_file)
        print(json.dumps(spec, indent=2))
        self.validate_instance_with_pattern_spec(instanceSpec)
        self.assertIsNotNone(spec.get("conditionedMappingPattern"))
        self.assertEquals(len(spec.get("conditionedMappingPattern")), 3)
        self.assertIsNotNone(spec.get("defaultMappingPattern"))
        self.assertEquals(len(spec.get("defaultMappingPattern")), 1)
        self.validate_processed_spec(spec)

    def test_cond_spec_2(self):
        spec_file = os.path.join(TESTCASE_DIR, 'cond_invoice_payment.json')
        instanceSpec, spec = self.process_specification(spec_file)
        print(json.dumps(spec, indent=2))
        self.assertIsNotNone(spec.get("conditionedMappingPattern"))
        self.assertEquals(len(spec.get("conditionedMappingPattern")), 1)
        self.assertIsNotNone(spec.get("defaultMappingPattern"))
        self.assertEquals(len(spec.get("defaultMappingPattern")), 1)
        self.validate_instance_with_pattern_spec(instanceSpec)
        self.validate_processed_spec(spec, src_type=list)

    def test_ne_spec_1(self):
        spec_file = os.path.join(TESTCASE_DIR, 'ne_receive_notification.json')
        instanceSpec, spec = self.process_specification(spec_file)
        self.validate_instance_with_pattern_spec(instanceSpec)
        self.assertIsInstance(spec, list)
        self.assertEquals(len(spec), 1)
        self.validate_mapping(spec[0])

    def test_syn_spec_1(self):
        spec_file = os.path.join(TESTCASE_DIR, 'syn_employee.json')
        instanceSpec, spec = self.process_specification(spec_file)
        print(json.dumps(spec, indent=2))
        self.validate_synonymy_instance_spec(instanceSpec)


if __name__ == '__main__':
    unittest.main()
