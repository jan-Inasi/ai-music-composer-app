import numpy as np
import torch as pt
import muspy as mp
import copy as cp
import torch


REST_CODE: int = 128


def copy_torch_numpy(array_or_tensor: pt.Tensor | np.ndarray):
    if isinstance(array_or_tensor, np.ndarray):
        return array_or_tensor.copy()
    return pt.clone(array_or_tensor)


def transpose(prd: pt.Tensor | np.ndarray, interval: int = 0):
    pitch = prd[:, 0]
    to_keep = (pitch > 127) | (pitch < 0)
    new_prd = copy_torch_numpy(prd)
    new_prd[:, 0] += interval
    # don't change values out of midi range [0, 127]
    new_prd[to_keep, 0] = pitch[to_keep]
    return new_prd


def from_muspy(music: mp.Music) -> np.ndarray:
    # tpdv - time, pitch, duration, velocity
    # pdr - pitch/rest, duration
    notes_tpdv = mp.to_representation(music, "note")

    notes_prd = []
    t = notes_tpdv[0][0]
    for tpdv in notes_tpdv:
        dt = tpdv[0] - t
        if dt > 0:
            notes_prd.append([int(REST_CODE), tpdv[0] - t])
            t += dt
        elif dt < 0:
            raise Exception(
                f"should not happen: t > tpdv[0] (sth has gone terribly wrong :( ))"
            )

        notes_prd.append([tpdv[1], tpdv[2]])
        t += tpdv[2]

    return np.array(notes_prd)


def to_muspy(
    notes_prd: np.ndarray | list[int],
    *,
    title: str | None = None,
    resolution: int = 24,
    tempo_qpm: float = 120.0,
    program: int = 0,
    start_time: int = 0,
    velocity: int = 64,
) -> mp.Music:
    t = start_time
    track = mp.Track(program=program)
    for note in notes_prd:
        if note[0] != REST_CODE:
            nt = mp.Note(
                time=int(t),
                pitch=int(note[0]),
                duration=int(note[1]),
                velocity=velocity,
            )
            track.notes.append(nt)
        t += note[1]

    return mp.Music(
        metadata=mp.Metadata(title=title, source_format="notes prd"),
        resolution=resolution,
        tempos=[mp.Tempo(0, tempo_qpm)],
        tracks=[track],
    )


class DatasetPRD(torch.utils.data.Dataset):

    PADDING = -3  # unencoded (0 as code)
    SOT_DURATION = -2  # unencoded
    EOT_DURATION = -1  # unencoded
    SOT_DUR_CODE = 1
    EOT_DUR_CODE = 2

    def __init__(
        self,
        prd_tunes: list[torch.Tensor],
        *,
        xy_split: float | list[float] = 0.5,
        sot_code: int = 129,
        eot_code: int = 130,
        trnasposition_augment_range: tuple[int] | None = None,
    ):
        self.xy_split = xy_split
        self.sot_code = sot_code  # sot <=> start of tune
        self.eot_code = eot_code  # eot <=> end of tune
        self.transposition_range = trnasposition_augment_range
        self.duration2code = {}
        self.code2duration = []
        self.create_code_translation(prd_tunes)
        self.tunes = self.prepare_tunes(prd_tunes)

    def random_interval(self):
        lowest, highest = self.transposition_range
        return np.random.randint(lowest, highest + 1)

    def transpose(self, tune):
        if self.transposition_range is None:
            return tune

        interval = self.random_interval()
        return transpose(tune, interval)

    def create_code_translation(self, tunes: list[torch.Tensor]):
        set_of_durations = set(
            int(duration) for tune in tunes for duration in tune[:, 1]
        )
        padding_sot_eot = [self.PADDING, self.SOT_DURATION, self.EOT_DURATION]
        self.code2duration = padding_sot_eot + sorted(list(set_of_durations))
        self.duration2code = {dur: i for i, dur in enumerate(self.code2duration)}

    def encode_durations(self, prd_tunes: list[torch.Tensor]) -> list[torch.Tensor]:
        encoded_tunes = []
        for tune in cp.deepcopy(prd_tunes):
            tune[:, 1] = tune[:, 1].apply_(lambda x: self.duration2code[x])
            encoded_tunes.append(tune)
        return encoded_tunes

    def decode_durations(self, prd_tunes: list[torch.Tensor]) -> list[torch.Tensor]:
        decoded_tunes = []
        for tune in prd_tunes:
            tune[:, 1] = tune[:, 1].apply_(lambda x: self.code2duration[x])
            decoded_tunes.append(tune)
        return decoded_tunes

    def mark_start_end(self, tunes: list[torch.Tensor]) -> list[torch.Tensor]:
        marked_tunes = []
        for tune in tunes:
            pitch = tune[:, 0]
            duration = tune[:, 1]
            pitch = tensor_prepend_append(pitch, self.sot_code, self.eot_code)
            duration = tensor_prepend_append(
                duration, self.SOT_DUR_CODE, self.EOT_DUR_CODE
            )
            marked_tunes.append(torch.stack([pitch, duration]).T)
        return marked_tunes

    def prepare_tunes(self, tunes: list[torch.Tensor]) -> list[torch.Tensor]:
        tunes = cp.deepcopy(tunes)
        tunes = self.encode_durations(tunes)
        tunes = self.mark_start_end(tunes)
        return tunes

    @property
    def pitch_dim(self):
        return 128 + 1 + 2  # padding + midi_range\{0} + rest + start_stop

    @property
    def duration_dim(self):
        return len(self.code2duration) + 1  # nmb_of_durations + start_end + padding

    @property
    def xy_split_ratio(self) -> float:
        if isinstance(self.xy_split, float):
            return self.xy_split
        return random_uniform_float(*self.xy_split)

    def __len__(self):
        return len(self.tunes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tune = self.tunes[index]
        tune = self.transpose(tune)
        split_idx = round(len(tune) * self.xy_split_ratio)
        x = tune[:split_idx]
        y = tune[split_idx:]
        return x, y


def random_uniform_float(a: float = 0, b: float = 1) -> float:
    rnd = np.random.random()
    return rnd * (b - a) + a


def tensor_prepend_append(tensor, pre, ap):
    return torch.concat((torch.tensor([pre]), tensor, torch.tensor([ap])), dim=0)
