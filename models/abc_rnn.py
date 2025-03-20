from __future__ import annotations
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import torch.utils.data as tdata
import numpy as np
import torch

import sys

sys.path.append(f"{sys.path[0]}/..")
from lib.tune_abc import DatasetTuneABC


class Codec(torch.nn.Module):
    def __init__(self, code: list[str]):
        super().__init__()
        self.code = code
        self.token2code = {t: i for i, t in enumerate(code)}

    def encode(self, tokens: list[str]) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        for token in tokens:
            encoded.append(torch.tensor([self.token2code[token]]))
        return torch.concat(encoded)

    def decode(self, encoded: torch.Tensor) -> list[str]:
        decoded: list[str] = []
        for idx in encoded:
            decoded.append(self.code[idx.item()])
        return decoded


def swaddle(code: list[str], what: str) -> list[str]:
    return [f"{what}: {c}\n" for c in code]


class Dataset(tdata.Dataset):
    def __init__(self, dataset: DatasetTuneABC):
        tunes = [torch.from_numpy(t) for t in dataset.tune_data]
        keys = torch.from_numpy(dataset.key_data)
        bit_lens = torch.from_numpy(dataset.bit_data)

        token_count = len(dataset.tune_codec.code2raw)
        keys += token_count
        token_count += len(dataset.key_codec.code2raw)
        bit_lens += token_count
        token_count += len(dataset.bit_codec.code2raw)

        self.tunes = self.prepare_tunes(tunes, keys, bit_lens)

        self.code_begin = torch.tensor([token_count])
        self.code_end = self.code_begin + 1
        self.code_padding = self.code_end + 1

        self.codec = self.prepare_codec(dataset)

    def prepare_codec(self, dataset: DatasetTuneABC) -> Codec:
        code: list[str] = dataset.tune_codec.code2raw.tolist()
        code.extend(swaddle(dataset.key_codec.code2raw.tolist(), "K"))
        code.extend(swaddle(dataset.bit_codec.code2raw.tolist(), "L"))
        code.extend(["<s>", "<e>", "<p>"])
        return Codec(code)

    def prepare_tunes(
        self, tunes: list[torch.Tensor], keys: torch.Tensor, bit_lens: torch.Tensor
    ) -> list[torch.Tensor]:
        new_tunes = []
        for bit, key, tune in zip(bit_lens, keys, tunes):
            new_tunes.append(
                torch.concat([bit.unsqueeze(dim=0), key.unsqueeze(dim=0), tune])
            )
        return new_tunes

    def __len__(self) -> int:
        return len(self.tunes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.concat([self.code_begin, self.tunes[idx]])
        y = torch.concat([self.tunes[idx], self.code_end])
        return x, y

    def padding(
        self, batch_list: tuple[list[torch.Tensor], list[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_list, y_list = list(zip(*batch_list))
        lens = [len(x) for x in x_list]
        x_padded = pad_sequence(
            x_list, batch_first=True, padding_value=float(self.code_padding)
        )
        y_padded = pad_sequence(
            y_list, batch_first=True, padding_value=float(self.code_padding)
        )
        return x_padded, y_padded, torch.tensor(lens)

    @property
    def tune_token_count(self) -> int:
        return int(self.code_padding) + 1


class Model(torch.nn.Module):
    LSTMState = tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        codec: Codec,
        token_count: int,
        start_code: int,
        end_code: int,
        padding_code: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: int = 0,
    ):
        super().__init__()

        self.codec = codec

        self.start_code = start_code
        self.end_code = end_code
        self.padding_code = padding_code

        self.tune_embedding = torch.nn.Embedding(
            num_embeddings=token_count,
            embedding_dim=embedding_dim,
            padding_idx=padding_code,
        )

        self.rnn = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.dense = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=token_count,
        )

    def initial_state_batched(self, batch_size: int):
        shape = (self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        return torch.zeros(*shape), torch.zeros(*shape)

    def initial_state_unbatched(self):
        shape = (self.rnn.num_layers, self.rnn.hidden_size)
        return torch.zeros(*shape), torch.zeros(*shape)

    def forward(
        self, tunes: torch.Tensor, lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tunes: Tensor [batches, tokens]
        lens: List [tune lengths]
        """

        batch_size = tunes.shape[0]
        embedded_tunes = self.tune_embedding(tunes)
        # embedded_tunes: Tensor [batches, token, embedding]
        packed_tunes = pack_padded_sequence(
            embedded_tunes, lens, batch_first=True, enforce_sorted=False
        )
        # packed_tunes: Tensor [batches x tokens, embedding] Tensor [batch_sizes]
        hc = self.initial_state_batched(batch_size)
        packed_seq, _ = self.rnn(packed_tunes, hc)
        output, batch_sizes = packed_seq[0], packed_seq[1]
        # output: Tensor [batches x tokens, rnn output (size of hidden dim)]

        pred = self.dense(output)
        # pred: Tensor [batches x tokens, out_features]
        pred = torch.nn.utils.rnn.PackedSequence(pred, batch_sizes, None, None)
        return pred[0], pred[1]  # packed_output, batch_sizes

    def forward_token(
        self, token: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            embedded_token = self.tune_embedding(token)
            output, new_state = self.rnn(embedded_token, state)
            pred = self.dense(output)
            return pred, new_state

    @torch.jit.export
    def generate(
        self,
        prompt_sequence: torch.Tensor = torch.tensor([], dtype=torch.int),
        max_len: int = 100,
        random: bool = True,
    ) -> torch.Tensor:
        tune = []
        token = torch.concat([torch.tensor([self.start_code]), prompt_sequence])
        state = self.initial_state_unbatched()
        for _ in range(max_len):
            pred, state = self.forward_token(token, state)
            pred = pred[-1]

            if random:
                distr = torch.softmax(pred, dim=0)
                choices = torch.arange(self.padding_code + 1)
                idx = distr.multinomial(num_samples=1, replacement=True)
                choice = choices[idx]
            else:
                choice = pred.topk(1)[1]

            if choice == self.end_code:
                break
            if choice == self.start_code or choice == self.padding_code:
                continue

            token = choice
            tune.append(choice)
        if len(tune) == 0:
            return prompt_sequence
        return torch.concat([prompt_sequence, torch.concat(tune)])

    @torch.jit.export
    def generate_tune(
        self,
        prompt_sequence: torch.Tensor = torch.tensor([], dtype=torch.int),
        max_len: int = 100,
        random: bool = True,
    ) -> str:
        encoded_tune = self.generate(prompt_sequence, max_len, random)
        return "".join(self.codec.decode(encoded_tune))


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_on_batch(
        self, x: torch.Tensor, y: torch.Tensor, lens: list[int]
    ) -> float:
        y_packed = pack_padded_sequence(y, lens, batch_first=True, enforce_sorted=False)

        self.model.train()
        self.model.zero_grad()

        pred, _ = self.model.forward(x, lens)
        loss: torch.Tensor = self.criterion(pred, y_packed[0])
        loss.backward()
        self.optimizer.step()

        return loss.item()


def _main():
    tune_ds = DatasetTuneABC().load("./data/dataset_abctunes.json")
    ds = Dataset(tune_ds)
    dataloader = tdata.DataLoader(
        ds, batch_size=20, collate_fn=ds.padding, shuffle=True
    )

    model = Model(
        token_count=ds.tune_token_count,
        start_code=int(ds.code_begin),
        end_code=int(ds.code_end),
        padding_code=int(ds.code_padding),
        embedding_dim=ds.tune_token_count,
        hidden_dim=125,
        num_layers=3,
    )

    # state = model.initial_state(unbatched=True)
    # print(state[0].shape)
    # pred, state = model.forward_token(torch.tensor([1]), state)
    # print(torch.softmax(pred, dim=1))

    print(model.generate([], 10))

    # trainer = Trainer(
    #     model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.002)
    # )

    # for epoch in tqdm.tqdm(range(5), desc=" epochs", position=0):
    #     for x, y, lens in tqdm.tqdm(
    #         dataloader, desc="batches", position=1, leave=False
    #     ):
    #         loss = trainer.train_on_batch(x, y, lens)


if __name__ == "__main__":
    _main()
