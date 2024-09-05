import torch
import numpy as np

from torch import nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""Functionals are not nn.Modules and are only meant to be used models in eval mode."""

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


LM_STR_TO_FN_CLASS_MAP = { # Functionals map
    "distilroberta": DistilRoBERTaSBERTFunctional,
    "minilm": MiniLM_L3v2SBERTFunctional,
}


if __name__ == "__main__":
    lang_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
    lang_model = lang_model_class()
    embs = lang_model(["pick and place the milk", "grasp the can"])
    print(embs.shape)
