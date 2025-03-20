import sys

sys.path.append(sys.path[0] + "/..")
import torch
from lib.tune_abc import parse_abc_tune, TuneABC
import models.abc_rnn as abcrnn
from typing import Optional


def parse_tune(tune_text: str) -> TuneABC:
    return parse_abc_tune(tune_text.splitlines())


def tune_extension(tune_body: str, key: str, bit_len: str) -> TuneABC:
    model = ModelHandler.get()
    in_tune = parse_abc_tune(tune_body.splitlines())
    in_tune.K = key
    in_tune.L = bit_len
    model_out = model.generate_tune(encode_tune(in_tune), max_len=800)
    return parse_abc_tune(model_out.splitlines())


def tune_generation() -> TuneABC:
    model = ModelHandler.get()
    model_out = model.generate_tune(max_len=800)
    return parse_abc_tune(model_out.splitlines())


def encode_tune(tune: TuneABC) -> torch.Tensor:
    model = ModelHandler.get()
    tt = tune.tokenize()

    encoded: list[torch.Tensor] = []
    for token in [f"K: {tune.K}\n"] + [f"L: {tune.L}\n"] + tt:
        encoded.append(torch.tensor([model.codec.token2code[token]]))
    return torch.concat(encoded)


class ModelHandler:
    MODEL: Optional[abcrnn.Model] = None

    def get() -> abcrnn.Model:
        if ModelHandler.MODEL is None:
            ModelHandler.MODEL = torch.jit.load("models/serialized/abcrnn-h512-e25.pt")
        return ModelHandler.MODEL
