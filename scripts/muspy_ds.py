from fractions import Fraction
from tkinter import W
import muspy as mp
import numpy as np


# ds = mp.datasets.EssenFolkSongDatabase("data", download_and_extract=True, convert=True)
ds = mp.datasets.EssenFolkSongDatabase("data")


def music_to_notes_prd(music: mp.Music):
    # tpdv - time, pitch, duration, velocity
    # pdr - pitch/rest, duration
    REST = -1
    notes_tpdv = mp.to_representation(music, "note")

    notes_prd = []
    t = notes_tpdv[0][0]
    for tpdv in notes_tpdv:
        dt = tpdv[0] - t
        if dt > 0:
            notes_prd.append([REST, tpdv[0] - t])
            t += dt
        elif dt < 0:
            raise Exception(
                f"should not happen: t > tpdv[0] (sth has gone terribly wrong :( ))"
            )

        notes_prd.append([tpdv[1], tpdv[2]])
        t += tpdv[2]

    return np.array(notes_prd)


def music_from_notes_prd(
    notes_prd: np.ndarray | list[int],
    *,
    title: str | None = None,
    resolution: int = 24,
    tempo_qpm: float = 120.0,
    program: int = 0,
    start_time: int = 0,
    velocity: int = 64,
):
    REST = -1
    t = start_time
    track = mp.Track(program=program)
    for note in notes_prd:
        if note[0] != REST:
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


print(len(ds))

durations = set()
resolutions = set()
tempos = set()
key_signatures = set()

for i, music in enumerate(ds):
    # music.print()
    # print(i)

    if i != 2002:
        continue

    notes = mp.to_representation(music, "note")
    try:
        notes = music_to_notes_prd(music)
    except Exception:
        print(f"convertion error {i}")
    durs = np.unique(notes[:, 1])
    # durations.add(durs.tolist())
    durations |= set(durs.flatten())
    resolutions.add(music.resolution)
    tempos.add(music.tempos[0])
    ks = music.key_signatures[0]
    key_signatures.add((ks.root, ks.fifths, ks.mode))

    # print(notes)

    # mp.from_representation(notes, "note").print()
    score = music_from_notes_prd(notes)
    # score.print()

    # mp.write_midi("essen.mid", music)
    mp.write_midi("essen_from_notes.mid", score)


fracts = [Fraction(dur, 24) for dur in durations]

# print(durations)
# print(fracts)
# print(resolutions)
# print(tempos)
# print(key_signatures)
