#!/usr/bin/python3
import hydra, tqdm, logging
from omegaconf import DictConfig

import torch
import torch.utils.data as tdata
import torch.utils.tensorboard as tb
import models.abc_rnn as abcrnn
from lib.tune_abc import DatasetTuneABC, parse_abc_tune, TuneABC
import time, os, shutil, math


@hydra.main(config_path="config", config_name="abcrnn.yaml", version_base="1.1")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ["gen", "generate"]:
        generate(cfg)
    else:
        print(f"unknown mode: {cfg.mode}")
        print("available modes: train, gen, generate")


def generate(cfg: DictConfig):
    model: abcrnn.Model = torch.jit.load(hydra.utils.to_absolute_path(cfg.model.path))

    prompt = parse_prompt(cfg.prompt, model.codec.code)

    if cfg.generate.eval:
        model.eval()
    else:
        model.train()

    print(os.getcwd())

    start = time.perf_counter()
    tune: str = model.generate_tune(
        prompt, max_len=cfg.generate.max_len, random=cfg.generate.random
    )
    end = time.perf_counter()

    print(f"time generating: {end-start:.2f} s")

    shutil.rmtree(os.getcwd())
    # because muspy doesn't work while changing directory
    os.chdir(hydra.utils.get_original_cwd())

    tune_abc = parse_abc_tune(tune.split("\n"))

    print(list(tune_abc.metrics.bar_lengths()))
    print(tune_abc.metrics.pitch_class_entropy())
    print(tune_abc.metrics.scale_consistency())

    print(tune_abc)
    tune_abc.play()


def parse_prompt(prompt, code: list[str]) -> torch.Tensor:
    tune = TuneABC().tokenize(prompt.tune) if prompt.tune else []

    if prompt.bit_len:
        bit_len = [f"L: {prompt.bit_len}\n"]
    elif tune or prompt.key:
        bit_len = [f"L: {prompt.default_bit_len}\n"]
    else:
        bit_len = []

    if prompt.key:
        key = [f"K: {prompt.key}\n"]
    elif tune or bit_len:
        key = [f"K: {prompt.default_key}\n"]
    else:
        key = []

    prompt = bit_len + key + tune
    print("prompt:", prompt)
    encoder = {c: i for i, c in enumerate(code)}
    return torch.tensor([encoder[t] for t in prompt], dtype=torch.int)


def train(cfg: DictConfig):
    log = logging.getLogger(__name__)

    log.info("loading dataset")
    logging.FileHandler("abcrnn.py")

    tune_ds = DatasetTuneABC().load(hydra.utils.to_absolute_path(cfg.dataset.path))
    ds = abcrnn.Dataset(tune_ds)

    dataloader = tdata.DataLoader(
        ds, batch_size=cfg.training.batch_size, collate_fn=ds.padding, shuffle=True
    )

    if cfg.model.path is None:
        model = abcrnn.Model(
            codec=ds.codec,
            token_count=ds.tune_token_count,
            start_code=int(ds.code_begin),
            end_code=int(ds.code_end),
            padding_code=int(ds.code_padding),
            embedding_dim=ds.tune_token_count,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.training.dropout,
        )
    else:
        model: abcrnn.Model = torch.jit.load(
            hydra.utils.to_absolute_path(cfg.model.path)
        )

    print(model)

    torch.jit.script(model)  # compile model to verify whether it can be serialized

    trainer = abcrnn.Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg.training.lr),
    )

    writer = tb.SummaryWriter("runs")
    global_step = 0
    for epoch in tqdm.tqdm(range(cfg.training.epochs), desc=" epochs", position=0):
        log.info(f"beginning epoch: {epoch}")
        for x, y, lens in tqdm.tqdm(
            dataloader, desc="batches", position=1, leave=False
        ):
            loss = trainer.train_on_batch(x, y, lens)

            writer.add_scalar("loss/loss", loss, global_step)
            global_step += 1

            log_melodies(log, model, global_step)

            if global_step % 1 == 0 and epoch > 0:
                pce, sc, nbl, sr = evaluate(model, hydra_dir_swap=True)
                if pce:
                    writer.add_scalar("eval/pitch_class_entropy", pce, global_step)
                if sc:
                    writer.add_scalar("eval/scale_consistency", sc, global_step)
                if nbl:
                    writer.add_scalar("eval/n_bar_lengts", nbl, global_step)
                if sr:
                    writer.add_scalar("eval/success_rate", sr, global_step)
                print(pce, sc, nbl, sr)

        model_scripted = torch.jit.script(model)
        model_scripted.save("model_scripted.pt")


def avg(sth: list) -> float | None:
    if len(sth) == 0:
        return None
    return sum(sth) / len(sth)


def remove_nones(sth: list) -> list:
    return [x for x in sth if x is not None and not math.isnan(x)]


def evaluate(model: abcrnn.Model, size=100, hydra_dir_swap=False):
    if hydra_dir_swap:
        cwd = os.getcwd()
        owd = hydra.utils.get_original_cwd()
        os.chdir(owd)

    print("generating")
    tunes = [model.generate_tune(max_len=1000) for _ in range(size)]
    print("parsing")
    tunes = [parse_abc_tune(tune.split("\n")) for tune in tunes]
    print(len(tunes))

    pitch_class_entropy = remove_nones(
        [tune.metrics.pitch_class_entropy() for tune in tunes]
    )
    scale_consistency = remove_nones(
        [tune.metrics.scale_consistency() for tune in tunes]
    )
    n_bar_lengths = remove_nones([tune.metrics.n_bar_lengths() for tune in tunes])

    successful_rate = (len(pitch_class_entropy) + len(scale_consistency)) / (2 * size)

    if hydra_dir_swap:
        os.chdir(cwd)

    return (
        avg(pitch_class_entropy),
        avg(scale_consistency),
        avg(n_bar_lengths),
        successful_rate,
    )


def log_melodies(
    log: logging.Logger,
    model: abcrnn.Model,
    global_step: int,
    count=5,
):
    if global_step % 100 == 0:
        melodies = generate_melodies(model, count)
        info = "\n"
        for i, melody in enumerate(melodies):
            info += f"melody {1} #################\n{melody}\n\n"
        log.info(info)


def generate_melodies(model: abcrnn.Model, count: int):
    model.eval()
    melodies = []
    for _ in range(count):
        try:
            tune = model.generate_tune(max_len=500)
        except Exception as ex:
            melodies.append(ex)
        else:
            melodies.append(tune)
    return melodies


if __name__ == "__main__":
    main(None)
