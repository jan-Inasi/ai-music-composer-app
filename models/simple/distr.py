import muspy as mp
import numpy as np


class DistrModel:
    def __init__(self):
        self.is_trained = False
        self.len_distr = PrevalenceSampler([0])
        self.pitch_distr = PrevalenceSampler([0])
        self.duration_distr = PrevalenceSampler([0])

    def train(self, dataset: list[mp.Music]):
        len_list = []
        pitch_list = []
        duration_list = []
        for music in dataset:
            len_list.append(len(music.tracks[0]))
            for note in music.tracks[0].notes:
                pitch_list.append(note.pitch)
                duration_list.append(note.duration)
        self.len_distr = PrevalenceSampler(len_list)
        self.pitch_distr = PrevalenceSampler(pitch_list)
        self.duration_distr = PrevalenceSampler(duration_list)
        self.is_trained = True

    def generate_music(self) -> mp.Music:
        if not self.is_trained:
            print("WARNING: untrained model")
        t = 0
        notes = []
        for _ in range(self.len_distr.sample()):
            pitch = self.pitch_distr.sample()
            duration = self.duration_distr.sample()
            notes.append(mp.Note(t, pitch, duration))
            t += duration
        return mp.Music(tracks=[mp.Track(notes=notes)])


class PrevalenceSampler:
    def __init__(self, element_list: list[int], precision: int = 1000):
        num, count = np.unique(element_list, return_counts=True)
        count = count // (max(1, np.sum(count) // precision))
        elems = []
        for n, c in zip(num, count):
            elems.extend([n] * c)
        self.elems = np.array(elems)

    def sample(self) -> int:
        return np.random.choice(self.elems)
