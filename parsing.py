from collections import defaultdict
from spacy.language import Language
from spacy.tokens import Token, Doc, Span
from spacy.symbols import conj, dobj, obj, pobj, NOUN, VERB, ADP, AUX, DET, CCONJ
from spikex.pipes import PhraseX, NounPhraseX


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

    def fix_verb_phrase(self, span: Span):
        # Strip ADP, etc. at the beginning
        try:
            first = [x.pos for x in span].index(VERB)
        except ValueError:
            first = None
        if first is None:
            return None
        return span[first:]

    def fix_noun_phrase(self, span: Span):
        poslist = [x.pos for x in span]
        first = None
        for i, item in enumerate(poslist):
            if item not in (ADP, AUX, DET, CCONJ):
                first = i
                break
        return span[first:] if first is not None else span

    def process_parsed(self, doc: Doc):
        phrasex = NounPhraseX(self.nlp.vocab)
        doc = phrasex(doc)
        phrasex = VerbPhrase(self.nlp.vocab)
        doc = phrasex(doc)
        np_ind = dict()
        for npf in list(doc._.noun_phrases):
            fixed = self.fix_noun_phrase(npf)
            if fixed is not None:
                np_ind.update({tok:fixed for tok in npf})
        vp_ind = dict()
        for vpf in list(doc._.verb_phrases):
            if vpf[0] not in np_ind.keys():
                fixed = self.fix_verb_phrase(vpf)
                if fixed is not None:
                    vp_ind.update({tok:fixed for tok in vpf})
        dobj_ind = {parsed.head: parsed for parsed in doc
                    if parsed.dep in (dobj, obj) or (parsed.dep == pobj and parsed.head in vp_ind.keys())}
        conj_ind = {parsed.head: parsed for parsed in doc if parsed.dep == conj}
        get_noun_phrase = lambda x: np_ind.get(x).text if np_ind.get(x) is not None else None
        get_verb_phrase = lambda x: vp_ind.get(x).text if vp_ind.get(x) is not None else None
        results = set()
        # for parsed in doc:
        #     print(parsed.head.text, parsed.text, parsed, parsed.dep_, parsed.conjuncts, parsed.pos_)tok.
        # Add pairs from dobj relation
        for vp, np in dobj_ind.items():
            results.add((get_verb_phrase(vp), get_noun_phrase(np)))
        for start, end in conj_ind.items():
            conj_pos = [(t.pos, t) for t in end.conjuncts]
            pos_dict = defaultdict(list)
            for pos, t in conj_pos:
                pos_dict[pos].append(t)
            pos_dict[end.pos].append(end)
            # Assume pattern <VERB>, <VERB> and <VERB> <NOUN>
            if not None in (pos_dict.get(NOUN), pos_dict.get(VERB)) and len(pos_dict.get(NOUN)) == 1 and len(pos_dict.get(VERB)) >= 1:
                np = pos_dict.get(NOUN)[0]
                for vb in pos_dict.get(VERB):
                    results.add((get_verb_phrase(vb), get_noun_phrase(np)))
            elif pos_dict.get(NOUN) is not None and len(pos_dict.get(NOUN)) >= 1:
                # Find leftmost verb
                if pos_dict.get(VERB) is None:
                    for np in pos_dict.get(NOUN):
                        vb = self.leftmost_verb(np)
                        if vb is not None:
                            results.add((get_verb_phrase(vb), get_noun_phrase(np)))
                # Assume pattern <NOUN> <VERB>, <VERB> and <VERB>
                elif len(pos_dict.get(VERB)) == 1:
                    vb = pos_dict.get(VERB)[0]
                    for np in pos_dict.get(NOUN):
                        results.add((get_verb_phrase(vb), get_noun_phrase(np)))
        return results
