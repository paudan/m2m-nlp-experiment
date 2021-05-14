from spacy.symbols import conj, dobj, obj, NOUN, VERB

def leftmost_verb(parsed):
    if parsed.pos == VERB:
        return parsed
    current = parsed
    while current.head != current:
        if current.dep in (dobj, obj):
            return current.head
        current = current.head
    print(current, current.dep_)
    if current.head == current and current.pos == VERB: # Check for ROOT
        return current
    return None

def process_parsed(doc):
    np_ind = {npf[-1]:npf for npf in list(doc.noun_chunks)}
    dobj_ind = {parsed.head: np_ind.get(parsed) for parsed in doc if parsed.dep in (dobj, obj)}
    results = set()
    for parsed in doc:
        # print(parsed.text, parsed.dep_, parsed.conjuncts, parsed.pos_)
        if parsed.head == parsed:   # Check for ROOT
            conjunct = parsed.conjuncts
            subject_np = dobj_ind.get(parsed)
            if parsed.pos == VERB and subject_np:
                results.add((parsed.text, subject_np.text))
            # Pattern <VERB>, <NOUN>, ..., and <NOUN>
            elif len(dobj_ind) == 1:
                results.add((list(dobj_ind.keys())[0].text, np_ind.get(conjunct[0]).text))
            # Pattern <VERB>, <VERB> and <VERB> <NOUN>
            elif len(np_ind) == 1 and len(conjunct) > 1:
                results.add((conjunct[0].text, np_ind.get(conjunct[1]).text))
        elif parsed.dep == conj:
            conjunct = parsed.conjuncts
            # Invalid case; process if single verb is present in the doc
            if len(conjunct) == 1:
                if len(dobj_ind) == 1:
                    results.add((list(dobj_ind.keys())[0].text, np_ind.get(conjunct[0]).text))
                else:
                    verb, noun = leftmost_verb(parsed), np_ind.get(conjunct[0])
                    if None not in (verb, noun):
                        results.add((verb.text, noun.text))
            elif len(conjunct) == 2:
                if conjunct[0].pos == VERB:
                    subject_np = np_ind.get(conjunct[1])
                    if subject_np:
                        results.add((conjunct[0].text, subject_np.text))
                    if conjunct[1].pos == VERB:
                        # Add another verb/noun pair
                        subject_np = dobj_ind.get(conjunct[1])
                        if subject_np:
                            results.add((conjunct[1].text, subject_np.text))
                    elif conjunct[1].pos == NOUN:
                        subject_np = np_ind.get(conjunct[1])
                        if subject_np and len(dobj_ind) == 1:
                            results.add((conjunct[0].text, subject_np.text))
                elif conjunct[0].pos == NOUN:
                    # Check pattern <VERB> <NOUN>, <NOUN> and <NOUN>
                    subject_np = None
                    if conjunct[1].pos == VERB:
                        subject_np = np_ind.get(conjunct[1])
                    elif conjunct[1].pos == NOUN:
                        subject_np = np_ind.get(conjunct[1])
                    if subject_np:
                        verb = leftmost_verb(parsed)
                        if verb:
                            results.add((verb.text, subject_np.text))
    return results