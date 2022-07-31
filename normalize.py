#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from simplenlg.features import Feature, Tense
from simplenlg.framework import InflectedWordElement
from simplenlg.lexicon import Lexicon, LexicalCategory
from simplenlg.realiser.english import Realiser

lexicon = Lexicon.getDefaultLexicon()
realiser = Realiser(lexicon)

def normalize_verb(verb):
    if verb is None or pd.isnull(verb):
        return verb

    def normalize(vb):
        word = lexicon.getWord(vb, LexicalCategory.VERB)
        infl = InflectedWordElement(word)
        infl.setFeature(Feature.TENSE, Tense.PRESENT)
        return realiser.realise(infl).getRealisation()

    return ' '.join(normalize(t) if ind == 0 else t for ind, t in enumerate(verb.split()))
