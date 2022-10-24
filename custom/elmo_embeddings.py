import logging
import re
from typing import List
import torch
import flair
from flair.data import Sentence, Token
from flair.embeddings.token import TokenEmbeddings
from elmo_embedder import DEFAULT_OPTIONS_FILE, DEFAULT_WEIGHT_FILE, ElmoEmbedder

log = logging.getLogger("flair")


class ELMoEmbeddings(TokenEmbeddings):
    """Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018.
    ELMo word vectors can be constructed by combining layers in different ways.
    Default is to concatene the top 3 layers in the LM."""

    def __init__(
            self, model: str = "original", options_file: str = None, weight_file: str = None,
            embedding_mode: str = "all"
    ):
        super().__init__()

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        assert embedding_mode in ["all", "top", "average"]

        self.name = f"elmo-{model}-{embedding_mode}"
        self.static_embeddings = True

        if not options_file or not weight_file:
            # the default model for ELMo is the 'original' model, which is very large
            options_file = DEFAULT_OPTIONS_FILE
            weight_file = DEFAULT_WEIGHT_FILE
            # alternatively, a small, medium or portuguese model can be selected by passing the appropriate mode name
            if model == "small":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            if model == "medium":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            if model in ["large", "5.5B"]:
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            if model == "pt" or model == "portuguese":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5"
            if model == "pubmed":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

        if embedding_mode == "all":
            self.embedding_mode_fn = self.use_layers_all
        elif embedding_mode == "top":
            self.embedding_mode_fn = self.use_layers_top
        elif embedding_mode == "average":
            self.embedding_mode_fn = self.use_layers_average

            # put on Cuda if available
        from flair import device

        if re.fullmatch(r"cuda:[0-9]+", str(device)):
            cuda_device = int(str(device).split(":")[-1])
        elif str(device) == "cpu":
            cuda_device = -1
        else:
            cuda_device = 0

        self.ee = ElmoEmbedder(
            options_file=options_file, weight_file=weight_file, cuda_device=cuda_device
        )

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def use_layers_all(self, x):
        return torch.cat(x, 0)

    def use_layers_top(self, x):
        return x[-1]

    def use_layers_average(self, x):
        return torch.mean(torch.stack(x), 0)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        # ELMoEmbeddings before Release 0.5 did not set self.embedding_mode_fn
        if not getattr(self, "embedding_mode_fn", None):
            self.embedding_mode_fn = self.use_layers_all

        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])

        embeddings = self.ee.embed_batch(sentence_words)

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                elmo_embedding_layers = [
                    torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                    torch.FloatTensor(sentence_embeddings[2, token_idx, :])
                ]
                word_embedding = self.embedding_mode_fn(elmo_embedding_layers)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name

    def __setstate__(self, state):
        self.__dict__ = state

        if re.fullmatch(r"cuda:[0-9]+", str(flair.device)):
            cuda_device = int(str(flair.device).split(":")[-1])
        elif str(flair.device) == "cpu":
            cuda_device = -1
        else:
            cuda_device = 0

        self.ee.cuda_device = cuda_device

        self.ee.elmo_bilm.to(device=flair.device)
        self.ee.elmo_bilm._elmo_lstm._states = tuple(
            [state.to(flair.device) for state in self.ee.elmo_bilm._elmo_lstm._states])