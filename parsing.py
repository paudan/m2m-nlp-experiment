from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.symbols import conj, dobj, obj, NOUN, VERB, ADP, AUX
from spikex.pipes import PhraseX


VP_PATTERNS = [[
    {"POS": {"IN": ["AUX", "PART", "VERB"]}, "OP": "*"},
    {"POS": {"IN": ["AUX", "VERB"]}}
]]

class VerbPhrase(PhraseX):

    def __init__(self, vocab):
        super(VerbPhrase, self).__init__(vocab, "verb_phrases", VP_PATTERNS)


class ConjunctiveParser():

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def leftmost_verb(self, parsed: Token):
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

    def process_parsed(self, doc: Doc):
        phrasex = VerbPhrase(self.nlp.vocab)
        doc = phrasex(doc)
        fix_noun_phrase = lambda span: span[1:] if span[0].pos in (ADP, AUX) else span
        np_ind = {npf[-1]:fix_noun_phrase(npf) for npf in list(doc.noun_chunks)}
        vp_ind = {vpf[0]:vpf for vpf in list(doc._.verb_phrases)}
        dobj_ind = {parsed.head: np_ind.get(parsed) for parsed in doc if parsed.dep in (dobj, obj)}
        get_noun_phrase = lambda x: np_ind.get(x).text if np_ind.get(x) is not None else None
        get_verb_phrase = lambda x: vp_ind.get(x).text if vp_ind.get(x) is not None else None
        print(dobj_ind)
        print(np_ind)
        print(vp_ind)
        results = set()
        for parsed in doc:
            # print(parsed.text, parsed, parsed.dep_, parsed.conjuncts, parsed.pos_)
            if parsed.head == parsed:   # Check for ROOT
                conjunct = parsed.conjuncts
                subject_np = dobj_ind.get(parsed)
                if parsed.pos == VERB and subject_np:
                    results.add((get_verb_phrase(parsed), subject_np.text))
                # Pattern <VERB>, <NOUN>, ..., and <NOUN>
                elif len(dobj_ind) == 1:
                    results.add((get_verb_phrase(list(dobj_ind.keys())[0]), get_noun_phrase(conjunct[0])))
                # Pattern <VERB>, <VERB> and <VERB> <NOUN>
                elif len(np_ind) == 1 and len(conjunct) > 1:
                    results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(conjunct[1])))
            elif parsed.dep == conj:
                conjunct = parsed.conjuncts
                # Invalid case; process if single verb is present in the doc
                if len(conjunct) == 1:
                    if len(dobj_ind) == 1:
                        results.add((get_verb_phrase(list(dobj_ind.keys())[0]), get_noun_phrase(conjunct[0])))
                    else:
                        verb = self.leftmost_verb(parsed)
                        if verb is not None:
                            results.add((get_verb_phrase(verb), get_noun_phrase(conjunct[0])))
                elif len(conjunct) == 2:
                    if conjunct[0].pos == VERB:
                        results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(conjunct[1])))
                        if conjunct[1].pos == VERB:
                            # Add another verb/noun pair
                            subject_np = dobj_ind.get(conjunct[1])
                            if subject_np:
                                results.add((get_verb_phrase(conjunct[1]), subject_np.text))
                        elif conjunct[1].pos == NOUN:
                            if len(dobj_ind) == 1:
                                results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(conjunct[1])))
                    elif conjunct[0].pos == NOUN:
                        # Check pattern <VERB> <NOUN>, <NOUN> and <NOUN>
                        verb = self.leftmost_verb(parsed)
                        if verb is not None:
                            results.add((get_verb_phrase(verb), get_noun_phrase(conjunct[1])))
        results = set(filter(lambda x: x[0] is not None and x[1] is not None, results))
        return results