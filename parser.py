import os
import json
import re
from processing import AbstractNLPProcessor


def get_external_specification(mappingPattern):
    if mappingPattern is None or not isinstance(mappingPattern, dict):
        return None
    if mappingPattern.get('$ref') is None:
        return None
    patternPath = mappingPattern.get('$ref')
    if not os.path.isfile(patternPath):
        return None
    with open(patternPath, 'r') as f:
        patternSpec = json.load(f)
    return patternSpec

def process_instance_specification(instanceSpec):
    patternSpec = instanceSpec.get('transformationPatternSpecification')
    if patternSpec is None:
        # We have simple type of specification for synonymy relation processing
        return instanceSpec
    mappingSpec = patternSpec.get('mappingPattern')
    parentSpec = mappingSpec or patternSpec
    patternSpec = get_external_specification(parentSpec)
    if mappingSpec is not None:
        mappings = process_mapping_pattern(patternSpec, instanceSpec)
    else:
        # Check conditioned pattern property (in both external and local property cases)
        externalPatternSpec = patternSpec.get('transformationPatternSpecification')
        if externalPatternSpec is not None and isinstance(externalPatternSpec, dict):
            conditioned = externalPatternSpec.get("isConditioned")
        else:
            conditioned = patternSpec.get("isConditioned")
        if conditioned is None or conditioned.lower() == 'false':
            mappings = process_mapping_pattern(patternSpec, instanceSpec)
        else:
            mappings = process_conditioned_specification(patternSpec, instanceSpec)
    return mappings


def check_source(srcInfo, srcInst):
    inherited = srcInfo.get('inherited')
    if inherited is not None:
        for entry in inherited:
            if srcInst.get('type') == entry.get('type') and srcInst.get('notation') == entry.get('notation'):
                return True
    return srcInst.get('type') == srcInfo.get('type') and srcInst.get('notation') == srcInfo.get('notation')

def check_target(targetInfo, targetInst):
    return targetInst.get('type') == targetInfo.get('type') and targetInst.get('notation') == targetInfo.get('notation')

def process_mapping(mapping, patternSpec, instanceSpec):
    sourceInstances = instanceSpec.get('source')
    outputs = instanceSpec.get('output')
    mappingPattern = patternSpec.get('mappingPattern')
    mappingSrc = mappingPattern.get('source')
    mappingTarget = mappingPattern.get('target')
    source_id = mapping.get('source')
    if source_id is None:
        print('Mapping source element is invalid, skipping mapping processing')
        return mapping
    srcInfo = [map_ for map_ in mappingSrc if map_.get('id') is not None and map_.get('id') == source_id]
    srcInfo = srcInfo[0] if len(srcInfo) > 0 else None
    mapping['sourceElement'] = srcInfo
    target_id = mapping.get('target')
    if target_id is None:
        print('Mapping target element is invalid, skipping mapping processing')
        return mapping
    targetInfo = [map_ for map_ in mappingTarget if map_.get('id') is not None and map_.get('id') == target_id]
    targetInfo = targetInfo[0] if len(targetInfo) > 0 else None
    mapping['targetElement'] = targetInfo
    if srcInfo is None or targetInfo is None:
        print('Parsed mapping is invalid, skipping')
        return mapping
    mapping['sourceInstances'] = []
    for srcInst in sourceInstances:
        nameInst = srcInst.get('name')
        if check_source(srcInfo, srcInst) and nameInst is not None:
            mapping['sourceInstances'].append(nameInst)
    mapping['targetInstances'] = []
    for targetInst in outputs:
        nameInst = targetInst.get('name')
        if check_target(targetInfo, targetInst) and nameInst is not None:
            mapping['targetInstances'].append(nameInst)
    return mapping

def process_join_mapping(mapping, patternSpec, instanceSpec):
    sourceInstances = instanceSpec.get('source')
    outputs = instanceSpec.get('output')
    mappingPattern = patternSpec.get('mappingPattern')
    mappingSrc = mappingPattern.get('source')
    mappingTarget = mappingPattern.get('target')
    inputs = mapping.get('input')
    mapping['sourceElement'] = list()
    for input in inputs:
        source_id = input.get('source')
        if source_id is None:
            print('Mapping source element is invalid, skipping mapping processing')
            continue
        srcInfo = [map_ for map_ in mappingSrc if map_.get('id') is not None and map_.get('id') == source_id]
        srcInfo = srcInfo[0] if len(srcInfo) > 0 else None
        srcInfo['condition'] = input['condition']
        mapping['sourceElement'].append(srcInfo)
        mapping['sourceInstances'] = []
        for srcInst in sourceInstances:
            nameInst = srcInst.get('name')
            if check_source(srcInfo, srcInst) and nameInst is not None:
                mapping['sourceInstances'].append(nameInst)
    targetSpec = mapping.get('target')
    if targetSpec is None:
        print('Join Mapping target element is missing, skipping mapping processing')
        return mapping
    target_id = targetSpec.get('id')
    if target_id is None:
        print('Join Mapping target element is invalid, skipping mapping processing')
        return mapping
    targetInfo = [map_ for map_ in mappingTarget if map_.get('id') is not None and map_.get('id') == target_id]
    targetInfo = targetInfo[0] if len(targetInfo) > 0 else None
    mapping['targetElement'] = targetInfo
    mapping['targetInstances'] = []
    for targetInst in outputs:
        nameInst = targetInst.get('name')
        if check_target(targetInfo, targetInst) and nameInst is not None:
            mapping['targetInstances'].append(nameInst)
    return mapping


def process_mapping_pattern(patternSpec, instanceSpec):
    if None in (patternSpec, instanceSpec):
        return None
    mappingPattern = patternSpec.get('mappingPattern')
    mappings = mappingPattern.get('mapping')
    processed = list()
    if mappings is not None:
        processed.extend([process_mapping(mapping, patternSpec, instanceSpec) for mapping in mappings])
    # Process merge type of patterns
    mappingJoin = mappingPattern.get('mappingJoin')
    if mappingJoin is not None:
        mappingJoin = process_join_mapping(mappingJoin, patternSpec, instanceSpec)
        processed.append(mappingJoin)
    return processed


def process_conditioned_specification(patternSpec, instanceSpec):
    patternSpec = patternSpec.get('transformationPatternSpecification')
    if patternSpec is None:
        return None
    conditionedPatterns = patternSpec.get('conditionedMappingPattern')
    conditionedMappings = list()
    if conditionedPatterns is not None:
        for condPattern in conditionedPatterns:
            condMappingPattern = condPattern.get('mappingPattern')
            if condMappingPattern is None: continue
            condPatternSpec = get_external_specification(condMappingPattern)
            if condPatternSpec is None: continue
            conditionedMappings.append({
                'id': condPattern.get('id'),
                'mappings': process_mapping_pattern(condPatternSpec, instanceSpec),
                'conditionalPredicate': condPattern.get("conditionalPredicate")
            })
    defaultPattern = patternSpec.get('defaultMappingPattern')
    defaultId = defaultPattern.get('id')
    defaultPattern = defaultPattern.get('mappingPattern')
    if defaultPattern is not None:
        if isinstance(defaultPattern, list):
            if len(defaultPattern) >= 1:
                defaultPattern = defaultPattern[0]  # Single transformation pattern can be specified as default
            else:
                defaultPattern = None
    defaultMappingSpec = get_external_specification(defaultPattern) if defaultPattern is not None else None
    defaultMapping = process_mapping_pattern(defaultMappingSpec, instanceSpec)
    mappings = {'id': defaultId, 'conditionedMappingPattern': conditionedMappings, 'defaultMappingPattern': defaultMapping}
    return mappings

def process_condition(source, condition, processor: AbstractNLPProcessor):
    if condition is None:
        return source
    if condition.upper() == 'EXTRACTNE':
        return processor.extract_named_entities(source)
    # Parse condition like: TYPENE(@source.name)[1].type in ('Person', 'Organization')
    if condition.upper().startswith('TYPENE'):
        targets = re.split(r'TYPENE\(@source.name\)\[1\].type in ', condition, re.IGNORECASE)
        if targets and len(targets) > 1:
            targets = list(map(lambda x: x.strip("()[]'\" "), targets[1].split(',')))
        return processor.get_named_entity_type(source) in targets
    # Parse semantic operators like ISHYPONYM(@source.name, 'Person')
    if condition.upper().startswith('ISHYPONYM'):
        target = re.findall(r"'(\w+)'", source)
        return processor.is_hyponym(source, target)
    if condition.upper().startswith('ISHYPERNYM'):
        target = re.findall(r"'(\w+)'", source)
        return processor.is_hypernym(source, target)
    if condition.upper().startswith('ISMERONYM'):
        target = re.findall(r"'(\w+)'", source)
        return processor.is_meronym(source, target)
    if condition.upper().startswith('ISHOLONYM'):
        target = re.findall(r"'(\w+)'", source)
        return processor.is_holonym(source, target)
    return source


def perform_mappings_test(mappings, instanceSpec, processor: AbstractNLPProcessor):

    def process_mappings(mappings):
        for mapping in mappings:
            source = mapping.get('sourceInstances')
            if source is None: continue
            if isinstance(source, list):
                if len(source) == 0:
                    continue
                source = source[0]
            target = mapping.get('targetInstances')
            if target is None or (isinstance(target, list) and len(target) == 0):
                continue
            resultsProcessed = process_condition(source, mapping.get('conditionalPredicate'), processor)
            for processed in resultsProcessed:
                if processed in target:
                    print('Match: processed =', processed, ', actual =', processed)
                else:
                    print('Match: processed =', processed, ', actual =', target)

    if mappings is None:
        return None
    # Simple 1-to-1" type of mapping with (or without) condition
    if isinstance(mappings, list):
        process_mappings(mappings)
    # Process conditional mappings
    elif isinstance(mappings, dict):
        # Ensure that correct matching name is generated
        process_mappings(mappings.get('conditionedMappingPattern'))
        process_mappings(mappings.get('defaultMappingPattern'))
        # Test that mapping with correct output type is selected for execution
        outputs = instanceSpec.get('output')
        if outputs is None: return None
        allOutputsGenerated = [False] * len(outputs)
        for i in range(len(outputs)):
            output = outputs[i]
            foundOutput = False
            for mapping in mappings:
                if check_target(mapping, output) is True:
                    foundOutput = True
                    break
            allOutputsGenerated[i] = foundOutput
        if not False in allOutputsGenerated:
            print('All outputs generated successfully')


if __name__ == '__main__':
    spec_path = os.path.join('testcases', 'cond_invoice_payment.json')
    with open(spec_path, 'r') as f:
        instanceSpec = json.load(f)
    print(process_instance_specification(instanceSpec))




