import music21 as m21
import pathlib

from const import ORCHSET_QL

music = m21.converter.parse("data/orchset/midi/Beethoven-S3-I-ex3.mid")


def find_all_quater_lengths(path_to_dir_with_midi_files):
    paths = pathlib.Path(path_to_dir_with_midi_files).glob("*.mid")
    all_durations = set()
    for file_path in paths:
        music = m21.converter.parse(file_path)
        for measure in music[1]:
            for elem in measure:
                if isinstance(elem, m21.note.Note) or isinstance(elem, m21.note.Rest):
                    all_durations.add(elem.duration.quarterLength)
    return sorted(list(all_durations))


# stream = m21.stream.Stream()
# ql = set()
# for measure in music[1]:
#     print()
#     print(measure)
#     for elem in measure:
#         stream.append(elem)
#         if isinstance(elem, m21.note.Note):
#             print("NOTE", elem.name, elem.octave, elem.duration, elem.duration.type)
#             ql.add(elem.duration.quarterLength)
#         elif isinstance(elem, m21.note.Rest):
#             print("REST", elem.duration, elem.duration.type)
#             ql.add(elem.duration.quarterLength)
#         else:
#             print("ELSE", type(elem), elem)


# print(ql)


# music.show("text")
# stream.show("text")
# stream.write(fmt="musicxml", fp="stream.mxl")
music.write(fmt="musicxml", fp="music.mxl")
