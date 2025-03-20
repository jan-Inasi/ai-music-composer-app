import music21 as m21


def to_m21_stream(
    pitches: list[int], durations: list[int], resolution=24
) -> m21.stream.Stream:
    stream = m21.stream.Stream()
    for pitch, duration in zip(pitches, durations):
        note = m21.note.Note()
        note.pitch = m21.pitch.Pitch(pitch)
        note.duration = m21.duration.Duration(duration / resolution)
        stream.append(note)
    return stream


pitches_str = "65 63 75 71 56 68 67 66 72 70 74 74 75 72 71 71 70 73"
durations_str = "4  8  8 40 36 12 12  9 16 16  9 12 12 24 12  9  8 16"

pitches_str = "57 59 70 65 68 70 65 70 68 73 74 62 73 69 66 72 71 69"
durations_str = "54 12 18 18 12 30 15 24 21 18 30 21 30 15 30  8 28 24"

pitches_str = "67 63 64 70 65 63 67 67 70 62 69 67 66 63 71 68 73 76"
durations_str = "12 24 16  6 42  2 15 12 16 18 21  9 36 21 15 16 16 30"

pitches = [int(x) for x in pitches_str.split(" ")]
durations = [int(x) for x in durations_str.split(" ") if x.isnumeric()]

stream = to_m21_stream(pitches, durations)
# player = m21.midi.realtime.StreamPlayer(stream)
# player.play()

for note in stream:
    print(note.duration.type)
# m21.duration.Duration().fullN
