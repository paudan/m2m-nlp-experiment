import json
import logging
from typing import IO, List, Iterable, Tuple
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy
import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.common.checks import ConfigurationError
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


def empty_embedding() -> numpy.ndarray:
    return numpy.zeros((3, 0, 1024))


class ElmoEmbedder():
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)

        self.cuda_device = cuda_device

    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask

    def embed_sentence(self, sentence: List[str]) -> numpy.ndarray:
        """
        Computes the ELMo embeddings for a single tokenized sentence.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        sentence : ``List[str]``, required
            A tokenized sentence.
        Returns
        -------
        A tensor containing the ELMo vectors.
        """

        return self.embed_batch([sentence])[0]

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a batch of tokenized sentences.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        elmo_embeddings = []

        # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
        # and return an empty embedding instead.
        if batch == [[]]:
            elmo_embeddings.append(empty_embedding())
        else:
            embeddings, mask = self.batch_to_embeddings(batch)
            for i in range(len(batch)):
                length = int(mask[i, :].sum())
                # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
                if length == 0:
                    elmo_embeddings.append(empty_embedding())
                else:
                    elmo_embeddings.append(embeddings[i, :, :length, :].detach().cpu().numpy())

        return elmo_embeddings

    def embed_sentences(self,
                        sentences: Iterable[List[str]],
                        batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a iterable of sentences.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.
        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)

    def embed_file(self,
                   input_file: IO,
                   output_file_path: str,
                   output_format: str = "all",
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   forget_sentences: bool = False,
                   use_sentence_keys: bool = False) -> None:
        """
        Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
        The ELMo embeddings are written out in HDF5 format, where each sentence embedding
        is saved in a dataset with the line number in the original file as the key.
        Parameters
        ----------
        input_file : ``IO``, required
            A file with one tokenized sentence per line.
        output_file_path : ``str``, required
            A path to the output hdf5 file.
        output_format : ``str``, optional, (default = "all")
            The embeddings to output.  Must be one of "all", "top", or "average".
        batch_size : ``int``, optional, (default = 64)
            The number of sentences to process in ELMo at one time.
        forget_sentences : ``bool``, optional, (default = False).
            If use_sentence_keys is False, whether or not to include a string
            serialized JSON dictionary that associates sentences with their
            line number (its HDF5 key). The mapping is placed in the
            "sentence_to_index" HDF5 key. This is useful if
            you want to use the embeddings without keeping the original file
            of sentences around.
        use_sentence_keys : ``bool``, optional, (default = False).
            Whether or not to use full sentences as keys. By default,
            the line numbers of the input file are used as ids, which is more robust.
        """

        assert output_format in ["all", "top", "average"]

        # Tokenizes the sentences.
        sentences = [line.strip() for line in input_file]

        blank_lines = [i for (i, line) in enumerate(sentences) if line == ""]
        if blank_lines:
            raise ConfigurationError(f"Your input file contains empty lines at indexes "
                                     f"{blank_lines}. Please remove them.")
        split_sentences = [sentence.split() for sentence in sentences]
        # Uses the sentence index as the key.

        if use_sentence_keys:
            logger.warning("Using sentences as keys can fail if sentences "
                           "contain forward slashes or colons. Use with caution.")
            embedded_sentences = zip(sentences, self.embed_sentences(split_sentences, batch_size))
        else:
            embedded_sentences = ((str(i), x) for i, x in
                                  enumerate(self.embed_sentences(split_sentences, batch_size)))

        sentence_to_index = {}
        logger.info("Processing sentences.")
        with h5py.File(output_file_path, 'w') as fout:
            for key, embeddings in Tqdm.tqdm(embedded_sentences):
                if use_sentence_keys and key in fout.keys():
                    raise ConfigurationError(f"Key already exists in {output_file_path}. "
                                             f"To encode duplicate sentences, do not pass "
                                             f"the --use-sentence-keys flag.")

                if not forget_sentences and not use_sentence_keys:
                    sentence = sentences[int(key)]
                    sentence_to_index[sentence] = key

                if output_format == "all":
                    output = embeddings
                elif output_format == "top":
                    output = embeddings[-1]
                elif output_format == "average":
                    output = numpy.average(embeddings, axis=0)

                fout.create_dataset(
                        str(key),
                        output.shape, dtype='float32',
                        data=output
                )
            if not forget_sentences and not use_sentence_keys:
                sentence_index_dataset = fout.create_dataset(
                        "sentence_to_index",
                        (1,),
                        dtype=h5py.special_dtype(vlen=str))
                sentence_index_dataset[0] = json.dumps(sentence_to_index)

        input_file.close()
