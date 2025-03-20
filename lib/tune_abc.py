from __future__ import annotations
from typing import Optional, Iterator, Iterable, ClassVar
import re, sys, glob, json, string
import dataclasses as dc
import numpy as np
import muspy as mp
import music21 as m21

sys.path.append(f"{sys.path[0]}/..")
from lib.abc_tool import abctune


@dc.dataclass
class TuneABC:
    """
    A class to represent a tune in ABC notation from EsAC dataset.

    Class attribute names corresponds to field name from ABC standard.

    Attributes:
        T     (Optional[str])   : the tune title  (default None)
        O     (list[str])       : the list of geografical origins of a tune  (default [])
        M     (Optional[str])   : the tune meter  (default None)
        L     (str)             : the tune unit note length  (default "1/8")
        K     (str)             : the tune key  (default "C")
        R     (Optional[str])   : the tune rhythm  (default None)
        tune  (list[list[str]]) : the tune body  (default [])
    """

    MULTI_CHARS: ClassVar[set[str]] = set(("(3", "12", "16", "24", "32"))
    DEFAULT_L: ClassVar[str] = "1/8"
    DEFAULT_K: ClassVar[str] = "C"

    T: Optional[str] = None
    O: list[str] = dc.field(default_factory=list)
    M: Optional[str] = None
    L: str = DEFAULT_L
    K: str = DEFAULT_K
    R: Optional[str] = None
    tune: list[list[str]] = dc.field(default_factory=list)

    metrics: Metrics = dc.field(init=False)

    def __post_init__(self):
        """After initialisation creates metrics object for the tune."""
        self.metrics = Metrics(self)

    def __str__(self):
        """Converts TuneABC to str"""
        text = "X: 0\n"
        text += f"T: {self.T}\n" if self.T else ""
        text += f"O: {self.O}\n" if self.O else ""
        text += f"R: {self.R}\n" if self.R else ""
        text += f"M: {self.M}\n" if self.M else ""
        text += f"L: {self.L}\n" if self.L else ""
        text += f"K: {self.K}\n" if self.K else ""
        text += self.tune_str()
        return text

    def __hash__(self):
        """Defines hash function for TuneABC"""
        return hash("\n".join([" | ".join(line) for line in self.tune]))

    def tune_str(self) -> str:
        """Returns tune body as a str"""
        text = ""
        for bars in self.tune:
            text += f"{' | '.join(bars)}\n"
        return text

    def clear(self) -> TuneABC:
        """Clears all data from the TuneABC"""
        self.T = None
        self.O = []
        self.M = None
        self.L = TuneABC.DEFAULT_L
        self.K = TuneABC.DEFAULT_K
        self.R = None
        self.tune = []
        return self

    def remove_inline_fields(self):
        """Removes inline fileds from the tune,
        tune fragments between square brackets e.g. [abc,e]
        """
        new_tune = []
        regex = re.compile("\[.*?\]")
        for line in self.tune:
            new_line = []
            for bar in line:
                new_line.append(regex.sub("", bar))
            new_tune.append(new_line)
        self.tune = new_tune

    def remove_comments(self):
        """Removes comments from the tune,
        that is lines beggining from "%%"
        """
        new_tune = []
        for line in self.tune:
            if any(True for bar in line if re.match("%%", bar)):
                continue
            new_tune.append(line)
        self.tune = new_tune

    def remove_empty_lines(self):
        """Removes empty lies from the tune body."""
        self.tune = [line for line in self.tune if len(line) > 0]

    def oneline_if_barless(self):
        """If every line of the tune body is not divided in bars,
        move all line to the first one as separate bars.
        """
        if max(len(line) for line in self.tune) == 1:
            self.tune = [[line[0] for line in self.tune]]

    def convert_to_music21(self) -> m21.stream.Part:
        """Converts tune to music21 Part"""
        abcfile = m21.abcFormat.ABCFile()
        return m21.abcFormat.translate.abcToStreamPart(abcfile.readstr(str(self)))

    def save_as_sheet_music(self, path: str, part: Optional[m21.stream.Part] = None):
        """Saves a sheet music representation of the tune to a PNG file.

        Parameters:
            path (str): Path defining where to save a file.
            part (Optional[m21.stream.Part]):
                if given renders music representation from this part instead
        """
        if part is None:
            part = self.convert_to_music21()
        part.write("musicxml.png", fp=path)

    def synthesize(self, rate=44100, part: Optional[m21.stream.Part] = None):
        """Synthesize the tune to a wave form.

        Parameters:
            rate (int): Sampling rate in samples per second.
            part (Optional[m21.stream.Part]):
                if given syntesizes from the part instead of the tune.
        """
        if part is None:
            part = self.convert_to_music21()
        tracks = mp.from_music21(part)
        music = mp.Music(tracks=tracks)
        return (rate, mp.synthesize(music, rate=rate))

    def play(self):
        """Plays synthesized tune."""
        part = self.convert_to_music21()
        player = m21.midi.realtime.StreamPlayer(part)
        player.play()

    def transpose(self, interval: int) -> TuneABC:
        """Transposes the tune for a given interval.

        Parameters:
            interval (int): Interval of the transposition in semitones.

        Returns:
            (TuneABC): New transposed tune.
        """
        tune = abctune(str(self))
        tune.transpose(interval)
        return parse_abc_tune(tune.getAbc().splitlines(), self)

    def transpose_to_key(self, key: str) -> TuneABC:
        """Transposes the tune to the given key.

        Parameters:
            key: (str): Name of the key to which the tune should be transposed to.

        Returns:
            (TuneABC): New transposed tune.
        """
        interval = m21.interval.Interval(m21.note.Note(self.K), m21.note.Note(key))
        return self.transpose(interval.semitones)

    def bars(self) -> Iterator[str]:
        """
        Returns:
            (Iterator[str]): iterator to the bars of the tune.
        """
        for line in self.tune:
            for bar in line:
                yield bar

    def tokenize(self, bar: Optional[str] = None) -> list[str]:
        """Converts the tune to a list of tokens.

        Parameters:
            bar (Optional[str]): if given tokenizes the bar instead

        Returns:
            (list[str]): List of tokens of the tune (or a bar if given).
        """
        if bar is None:
            tokens = []
            for i, line in enumerate(self.tune):
                for j, bar in enumerate(line):
                    tokens.extend(self.tokenize(bar))
                    if j + 1 != len(line):
                        tokens.append("|")
                if i + 1 != len(self.tune):
                    tokens.append("\n")
            return tokens
        else:
            return split_multicharly(bar, TuneABC.MULTI_CHARS)


def parse_abc_tune(
    abc_text_lines: list[str], tune: Optional[TuneABC] = None
) -> TuneABC:
    """Parses a string of text as a TuneABC object.

    This methods does NOT validate a tune given as an input.

    Parameters:
        abc_text_lines (list[str]): list of lines defining a tune in ABC notation.
        tune (Optional[TuneABC]):
            If given instead of creating a new TuneABC object,
            a tune is parsed to the existing TuneABC object.

    Returns:
        tune (TuneABC) An object with parsed tune.
    """
    tune = tune.clear() if tune else TuneABC()
    is_done_with_header = False
    for line in abc_text_lines:
        line = remove_nonprintable(line)
        mark = line[:2]
        if is_done_with_header:
            tune.tune.append(parse_abc_tune_line(line))
        elif mark in {"N:", "X:", "S:", "%%"}:
            continue
        elif mark == "T:":
            tune.T = line[2:].strip()
        elif mark == "O:":
            tune.O = line[2:].strip()
        elif mark == "M:":
            tune.M = line[2:].strip()
        elif mark == "L:":
            tune.L = line[2:].strip()
        elif mark == "K:":
            tune.K = line[2:].strip()
        elif mark == "R:":
            tune.R = line[2:].strip()
        else:
            is_done_with_header = True
            tune.tune.append(parse_abc_tune_line(line))
    return tune


def remove_nonprintable(text: str) -> str:
    """Creates a new str with removed non-printable characters.

    Parameters:
        text (str): A string from which to remove characters.

    Returns:
        text (str): A string without non-printable characters.
    """
    return "".join([char for char in text if char in string.printable])


def parse_abc_tune_line(tune_line: str) -> list[str]:
    """Removes spaces from a line of ABC notation tune and splits bars.

    Parameters:
        tune_line (str): A line of a tune in ABC notation.

    Returns:
        (list[str]): List of bars with deleted spaces.
    """
    list_of_bars = tune_line.rstrip().replace(" ", "").split("|")
    return [bar for bar in list_of_bars if bar and bar != "]"]


def browse_dataset(path_specification: str) -> Iterable[str]:
    """Iterates over lines of the specified files.

    Parameters:
        path_specification (str):
            A path defining which files to read.
            Function accepts wildcards * and ?.

    Returns:
        (Iterable[str]): Sequenial lines of given files
    """
    abc_file_paths = glob.glob(path_specification)

    for path in abc_file_paths:
        with open(path, "r") as file:
            for line in file:
                yield line


def split_abc_tunes(abc_file_lines: Iterable[str]) -> Iterator[list[str]]:
    """Splits lines of different tunes in a single ABC file.

    Parameters:
        abc_file_lines (Iterable[str]): Iterator over lines of an ABC file.

    Returns:
        (Iterator[list[str]]): Iterator over list of lines of the whole tunes.
    """
    tune_lines = []
    for line in abc_file_lines:
        if line == "\n":
            if len(tune_lines) > 0:
                yield tune_lines
                tune_lines = []
        elif line[:2] == "X:" and len(tune_lines) > 0:
            yield tune_lines
            tune_lines = []
        else:
            tune_lines.append(line)


class DatasetTuneABC:
    MULTI_CHARS: ClassVar[set[str]] = set(("(3", "12", "16", "24", "32"))

    def __init__(self, abc_tunes: list[TuneABC] | None = None):
        if abc_tunes is None:
            self.tune_codec = None
            self.key_codec = None
            self.bit_codec = None
            self.tune_data = None
            self.key_data = None
            self.bit_data = None
        else:
            self.tune_codec = Codec(np.array(self.gather_tune_alphabet(abc_tunes)))
            self.key_codec = Codec(np.array(self.gather_key_alphabet(abc_tunes)))
            self.bit_codec = Codec(np.array(self.gather_bit_alphabet(abc_tunes)))

            self.tune_data: list[np.ndarray[int]] = self.convert_to_tune_data(abc_tunes)
            self.key_data: np.ndarray[int] = self.convert_to_key_data(abc_tunes)
            self.bit_data: np.ndarray[int] = self.convert_to_bit_data(abc_tunes)

    def abc_tune(self, idx: int) -> TuneABC:
        tune = self.tune_codec.decode(self.tune_data[idx])
        key = str(self.key_codec.decode(self.key_data[idx]))
        bit = str(self.bit_codec.decode(self.bit_data[idx]))

        tune = [line.split("|") for line in "".join(tune).split("\n")]
        return TuneABC(T=str(idx), tune=tune, K=key, L=bit)

    def convert_to_tune_data(self, abc_tunes: list[TuneABC]) -> list[list[int]]:
        return [self.get_data_from_tune(tune) for tune in abc_tunes]

    def convert_to_key_data(self, abc_tunes: list[TuneABC]) -> np.ndarray[int]:
        return np.array([self.key_codec.encode(tune.K) for tune in abc_tunes])

    def convert_to_bit_data(self, abc_tunes: list[TuneABC]) -> np.ndarray[int]:
        return np.array([self.bit_codec.encode(tune.L) for tune in abc_tunes])

    def gather_tune_alphabet(self, abc_tunes: list[TuneABC]) -> list[str]:
        token_set = set()
        for tune in abc_tunes:
            for line in self.tune_tokenized(tune):
                for bar in line:
                    token_set |= set(bar)
        return ["\n", "|"] + sorted(list(token_set))

    def gather_key_alphabet(self, abc_tunes: list[TuneABC]) -> list[str]:
        return sorted(list(set(tune.K for tune in abc_tunes)))

    def gather_bit_alphabet(self, abc_tunes: list[TuneABC]) -> list[str]:
        return sorted(list(set(tune.L for tune in abc_tunes)))

    def tune_tokenized(self, tune: TuneABC) -> list[list[list[str]]]:
        tokenized_tune = []
        for line in tune.tune:
            tokenized_line = []
            for bar in line:
                tokenized_line.append(
                    split_multicharly(bar, DatasetTuneABC.MULTI_CHARS)
                )
            tokenized_tune.append(tokenized_line)
        return tokenized_tune

    def squeeze_tokenized_tune(self, ttune: list[list[list[str]]]) -> list[str]:
        tune_data = []
        for line in ttune:
            for bar in line:
                tune_data.extend(bar)
                tune_data.append("|")
            tune_data.pop()
            tune_data.append("\n")
        tune_data.pop()
        return tune_data

    def get_data_from_tune(self, tune: TuneABC) -> list[int]:
        return self.tune_codec.encode(
            self.squeeze_tokenized_tune(self.tune_tokenized(tune))
        )

    def save(self, path: str):
        with open(path, "w") as file:
            json.dump(
                [
                    self.tune_codec.code2raw.tolist(),
                    self.key_codec.code2raw.tolist(),
                    self.bit_codec.code2raw.tolist(),
                    [x.tolist() for x in self.tune_data],
                    self.key_data.tolist(),
                    self.bit_data.tolist(),
                ],
                file,
            )

    def load(self, path: str) -> DatasetTuneABC:
        with open(path, "r") as file:
            data = json.load(file)

        self.tune_codec = Codec(np.array(data[0]))
        self.key_codec = Codec(np.array(data[1]))
        self.bit_codec = Codec(np.array(data[2]))

        self.tune_data = [np.array(x) for x in data[3]]
        self.key_data = np.array(data[4])
        self.bit_data = np.array(data[5])
        return self


def load_and_validate_dataset(path: str, verbose=False) -> list[TuneABC]:
    valid_tunes = []
    for i, str_tune in enumerate(split_abc_tunes(browse_dataset(f"{path}/*.abc"))):
        tune = parse_abc_tune(str_tune)
        try:
            abcfile = m21.abcFormat.ABCFile()
            m21.abcFormat.translate.abcToStreamPart(abcfile.readstr(str(tune)))
        except Exception as ex:
            if verbose:
                print("tune nr:", i, "error:", ex)
        else:
            valid_tunes.append(tune)
    return valid_tunes


def contains_invalid_duration(tune: TuneABC) -> bool:
    for line in tune.tune:
        for bar in line:
            if re.findall("22|42|62|66|222|424", bar):
                return True
            if ":" in bar:
                return True
    return False


def split_multicharly(text: str, multi_chars: set[str]) -> list[str]:
    chars = []
    max_len = max(len(char) for char in multi_chars)
    multi_char = ""
    for char in text:
        multi_char = multi_char + char
        if multi_char in multi_chars:
            chars.append(multi_char)
            multi_char = ""
        elif len(multi_char) > max_len:
            chars.extend(multi_char)
            multi_char = ""
    chars.extend(multi_char)
    return chars


def bit_counter(bar: str) -> int:
    note_chars = set("abcdefgABCDEFGz")

    count = 0
    for token in TuneABC().tokenize(bar):
        if token in note_chars:
            count += 2
        elif token.isnumeric():
            count += 2 * (int(token) - 1)
        elif token == "/":
            count -= 1
    return count


class Metrics:
    def __init__(self, tune: TuneABC):
        self.tune = tune
        self._music: mp.Music | None = None
        self._music21: m21.stream.Part | None = None

    @property
    def music(self):
        if not self._music:
            self._music = mp.read_abc_string(str(self.tune))
        return self._music

    @property
    def music21(self):
        if not self._music21:
            abc_file = m21.abcFormat.ABCFile()
            abc_handler = abc_file.readstr(str(self.tune))
            self._music21 = m21.abcFormat.translate.abcToStreamPart(abc_handler)
            if not isinstance(self._music21[0], m21.stream.Stream):
                self._music21 = [self._music21]
        return self._music21

    def pitch_class_entropy(self) -> float | None:
        try:
            return mp.pitch_class_entropy(self.music)
        except Exception as ex:
            return None

    def scale_consistency(self) -> float | None:
        try:
            if len(self.music.key_signatures) > 0:
                key = self.music.key_signatures[0]
                return mp.pitch_in_scale_rate(self.music, key.root, key.mode)
            return mp.scale_consistency(self.music)
        except Exception as ex:
            return None

    def bar_lengths(self) -> Iterator[int]:
        yield from (bit_counter(bar) for bar in self.tune.bars())

    def n_bar_lengths(self) -> int | None:
        try:
            lens = list(self.tune.metrics.bar_lengths())
            return len(np.unique(lens))
        except Exception as ex:
            return None


class Codec:
    def __init__(self, raw: np.ndarray):
        self.code2raw = raw
        self.raw2code = {elem: i for i, elem in enumerate(raw)}

        self._encode = np.vectorize(lambda x: self.raw2code[x])
        self._decode = np.vectorize(lambda x: self.code2raw[x])

    def encode(self, raw: np.ndarray) -> np.ndarray:
        return self._encode(raw)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        return self._decode(encoded)


if __name__ == "__main__":
    ds = DatasetTuneABC().load("./data/dataset_abctunes.json")

    print(len(ds.tune_data))

    print(len(ds.tune_codec.code2raw))
    print(len(ds.bit_codec.code2raw))
    print(len(ds.key_codec.code2raw))

    tune = ds.abc_tune(900)
    print(tune)

    music: mp.Music = mp.read_abc_string(str(tune))
    stream = music.to_music21()
    print(stream.duration.quarterLength)

    abc_file = m21.abcFormat.ABCFile()
    abc_handler = abc_file.readstr(str(tune))
    score = m21.abcFormat.translate.abcToStreamPart(abc_handler)
