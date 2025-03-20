import numpy as np
import muspy as mp

from const import MAJOR_SCALE


def make_simple_music(notes):
    return mp.Music(
        time_signatures=[mp.TimeSignature(0, 4, 4)],
        tracks=[mp.Track(program=0, notes=notes)],
    )


def compose_music_scale(note_duration=96, velocity=64, scale=MAJOR_SCALE):
    notes = []
    t = 0
    for pitch in scale:
        note = mp.Note(time=t, pitch=pitch, duration=note_duration, velocity=velocity)
        notes.append(note)
        t += note_duration

    return make_simple_music(notes)


def compose_music_random_scale(
    duration_in_bars=5, velocity=64, scale=MAJOR_SCALE, bar_duration=96
):
    notes = []
    t = 0
    while t < duration_in_bars * bar_duration:
        new_note = random_note(t, bar_duration, scale, velocity=velocity)
        notes.append(new_note)
        t += new_note.duration

    return make_simple_music(notes)


def random_note_duration(bar=96):
    return bar // 2 ** np.random.randint(0, 4)


def random_pitch(scale=MAJOR_SCALE):
    return np.random.choice(scale)


def random_note(time=0, bar=96, scale=MAJOR_SCALE, **kwargs):
    return mp.Note(time, random_pitch(scale), random_note_duration(bar), **kwargs)
