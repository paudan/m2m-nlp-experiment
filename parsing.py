from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.symbols import conj, dobj, obj, NOUN, VERB, ADP, AUX
from spikex.pipes import PhraseX


VP_PATTERNS = [[
    {"POS": {"IN": ["AUX", "PART", "VERB", "ADP"]}, "OP": "*"},
    {"POS": {"IN": ["AUX", "VERB", "ADP"]}}
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
        if current.head == current and current.pos == VERB: # Check for ROOT
            return current
        return None

    def process_parsed(self, doc: Doc):
        phrasex = VerbPhrase(self.nlp.vocab)
        doc = phrasex(doc)
        fix_noun_phrase = lambda span: span[1:] if span[0].pos in (ADP, AUX) else span
        np_ind = dict()
        for npf in doc.noun_chunks:
            np_ind.update({tok:fix_noun_phrase(npf) for tok in npf})
        vp_ind = {vpf[0]:vpf for vpf in list(doc._.verb_phrases)}
        dobj_ind = {parsed.head: parsed for parsed in doc if parsed.dep in (dobj, obj)}
        get_noun_phrase = lambda x: np_ind.get(x).text if np_ind.get(x) is not None else None
        get_verb_phrase = lambda x: vp_ind.get(x).text if vp_ind.get(x) is not None else None
        results = set()
        for parsed in doc:
            # print(parsed.text, parsed, parsed.dep_, parsed.conjuncts, parsed.pos_)
            if parsed.head == parsed:   # Check for ROOT
                conjunct = parsed.conjuncts
                subject_np = get_noun_phrase(dobj_ind.get(parsed))
                if parsed.pos == VERB and subject_np:
                    results.add((get_verb_phrase(parsed), subject_np))
                # Pattern <VERB>, <NOUN>, ..., and <NOUN>
                elif len(dobj_ind) == 1:
                    if len(conjunct) > 0:
                        ntok = dobj_ind.get(conjunct[0]) if conjunct[0].pos == VERB else conjunct[0]
                        subject_np = get_noun_phrase(ntok)
                    else:
                        subject_np = list(dobj_ind.values())[0].text
                    results.add((get_verb_phrase(list(dobj_ind.keys())[0]), subject_np))
                # Pattern <VERB>, <VERB> and <VERB> <NOUN>
                elif len(set(np_ind.values())) == 1 and len(conjunct) > 1:
                    results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(conjunct[1])))
            elif parsed.dep == conj:
                conjunct = parsed.conjuncts
                # Invalid case
                if len(conjunct) == 1:
                    verb = self.leftmost_verb(conjunct[0])
                    if verb is not None:
                        results.add((get_verb_phrase(verb), get_noun_phrase(conjunct[0])))
                    else:
                        # process if single verb is present in the doc
                        if len(dobj_ind) == 1:
                            results.add((get_verb_phrase(list(dobj_ind.keys())[0]), get_noun_phrase(conjunct[0])))
                elif len(conjunct) == 2:
                    if conjunct[0].pos == VERB:
                        if len(dobj_ind) > 0:
                            results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(dobj_ind.get(conjunct[0]))))
                        if conjunct[1].pos == VERB:
                            # Add another verb/noun pair
                            if len(dobj_ind) > 0:
                                results.add((get_verb_phrase(conjunct[1]), get_noun_phrase(dobj_ind.get(conjunct[1]))))
                        elif conjunct[1].pos == NOUN:
                            results.add((get_verb_phrase(conjunct[0]), get_noun_phrase(conjunct[1])))
                    elif conjunct[0].pos == NOUN:
                        # Check pattern <VERB> <NOUN>, <NOUN> and <NOUN>
                        verb = self.leftmost_verb(parsed)
                        if verb is not None:
                            results.add((get_verb_phrase(verb), get_noun_phrase(conjunct[1])))
        results = set(filter(lambda x: x[0] is not None, results))
        return results