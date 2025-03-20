import numpy as np
import muspy as mp


class ContinuesPitchDurationConverter:
    def __init__(self, dataset: list[mp.Music]):
        self.min_pitch = np.Inf
        self.max_pitch = -1
        self.duration_map = []

        durset = set()

        for music in dataset:
            tpd = mp.to_note_representation(music, encode_velocity=False)
            pitches = tpd[:, 1]
            durations = tpd[:, 2]

            self.max_pitch = max(np.max(pitches), self.max_pitch)
            self.min_pitch = min(np.min(pitches), self.min_pitch)

            durset |= set(durations)

        ord_durset = sorted(durset)
        self.duration_map = {
            d: map_range_to_range(i, (0, len(ord_durset) - 1), (-1, 1))
            for i, d in enumerate(ord_durset)
        }

        self.encode_durations = np.vectorize(lambda x: self.duration_map[x])

        decode_map = list(self.duration_map.keys())
        self.decode_durations = np.vectorize(lambda x: decode_map[x])

    @property
    def pitch_range(self):
        return self.min_pitch, self.max_pitch

    @property
    def duration_range(self):
        return 0, len(self.duration_map) - 1

    def to_representation(self, music):
        tpd = mp.to_note_representation(music, encode_velocity=False)
        pitches = tpd[:, 1]
        durations = tpd[:, 2]

        new_pitches = map_range_to_range(
            pitches, (self.min_pitch, self.max_pitch), (-1, 1)
        )
        new_durations = self.encode_durations(durations)
        return np.stack([new_pitches, new_durations]).T

    def from_representation(self, pd: np.ndarray):
        encoded_pitches = pd[:, 0]
        encoded_durations = pd[:, 1]
        pitches = map_range_to_range(encoded_pitches, (-1, 1), self.pitch_range)
        decoded_pitches = np.round(pitches).astype(int)

        durations = map_range_to_range(encoded_durations, (-1, 1), self.duration_range)
        idx_durations = np.round(durations).astype(int)
        decoded_durations = self.decode_durations(idx_durations)

        return np.stack([decoded_pitches, decoded_durations]).T


def pd2tpd(pd: np.ndarray) -> np.ndarray:
    """pitch-duration to time-pitch_duration convertion"""
    time = 0
    tpd = []
    for note in pd:
        tpd.append([time, note[0], note[1]])
        time += note[1]
    return np.stack(tpd).T


def pd2muspy(pd: np.ndarray, **kwargs) -> mp.Music:
    return mp.from_note_representation(pd2tpd(pd).T, encode_velocity=False, **kwargs)


def map_range_to_range(
    value: float | np.ndarray,
    range_in: tuple[float, float],
    range_out: tuple[float, float],
) -> float | np.ndarray:
    a, b = range_in
    c, d = range_out

    value = value - a
    value = value * (d - c) / (b - a)
    value = value + c

    return value
