import numpy as np
import torch
from torch import nn

from rlkit.util.misc_functions import l2_unit_normalize

LONGEST_SENTENCE_LEN = 20

"""
Functionals are not nn.Modules and are only meant to be used in eval mode.
"""


class DistilBERTFunctional():
    out_dim = 768

    def __init__(self, freeze=True, l2_unit_normalize=True, gpu=0):
        super().__init__()
        from pytorch_transformers import DistilBertTokenizer, DistilBertModel
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.freeze = freeze
        self.longest_sentence_len = LONGEST_SENTENCE_LEN
        self.l2_unit_normalize = l2_unit_normalize
        self.model.to(f"cuda:{gpu}")

        # WARNING: This gets modified in training where all the networks
        # become trainable anyway.
        if self.freeze:
            print("Freezing DistilBERT")
            self.model.eval()
        else:
            print("Making DistilBERT finetunable")
            self.model.train()

    def __call__(self, input_ids):
        """
        input_ids is a matrix tensor of token IDs.
        """
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()

        with torch.set_grad_enabled(not self.freeze):
            embs = self.model(input_ids)[0]
            sentence_emb = torch.mean(embs, axis=1)

        if self.l2_unit_normalize:
            # Unit-norm
            sentence_emb = l2_unit_normalize(sentence_emb)

        return sentence_emb

    def tokenize_strs(self, input_str_list):
        def pad_token_id_batch(token_id_batch):
            """token_id_batch is a 2D list that is potentially a ragged array.
            This function adds padding token_id to the end of each sentence
            so that the array is no longer ragged.
            """
            token_id_batch_padded = []
            for sentence in token_id_batch:
                padding = [self.tokenizer.pad_token_id] * (
                    self.longest_sentence_len - len(sentence))
                padded_sentence = sentence + padding
                token_id_batch_padded.append(padded_sentence)
            return token_id_batch_padded

        input_ids_list = []
        for input_str in input_str_list:
            input_ids = self.tokenizer.encode(input_str)
            input_ids_list.append(input_ids)
        input_ids_padded_list = pad_token_id_batch(input_ids_list)
        input_ids = np.array(input_ids_padded_list)
        return input_ids


class DistilBERT(DistilBERTFunctional, nn.Module):
    def __init__(self, **kwargs):
        return super().__init__(freeze=False, **kwargs)

    def forward(self, input_ids):
        return self.__call__(input_ids)


class SBERTWrapper():
    def __init__(self, sbert_model_name, l2_unit_normalize=True, gpu=0):
        from sentence_transformers import SentenceTransformer
        self.sbert_model_name = sbert_model_name
        self.l2_unit_normalize = l2_unit_normalize
        self.model = SentenceTransformer(sbert_model_name)
        self.device = f"cuda:{gpu}"

    def __call__(self, input_str_list):
        return self.model.encode(
            sentences=input_str_list,
            output_value='sentence_embedding',
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.l2_unit_normalize)


class DistilRoBERTaSBERTFunctional(SBERTWrapper):
    out_dim = 768

    def __init__(self, **kwargs):
        return super().__init__(
            sbert_model_name="all-distilroberta-v1", **kwargs)


class MiniLM_L3v2SBERTFunctional(SBERTWrapper):
    out_dim = 384

    def __init__(self, **kwargs):
        return super().__init__(
            sbert_model_name="paraphrase-MiniLM-L3-v2", **kwargs)


# Functionals map
LM_STR_TO_FN_CLASS_MAP = {
    "distilbert": DistilBERTFunctional,
    "distilroberta": DistilRoBERTaSBERTFunctional,
    "minilm": MiniLM_L3v2SBERTFunctional,
}

# Finetunable (nn.Module) map
LM_STR_TO_CLASS_MAP = {
    "distilbert": DistilBERT,
}
