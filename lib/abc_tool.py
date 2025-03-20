# source: https://github.com/a773music/abctool
import sys, string, re, os, tempfile

external_programs = {}
external_functionality = {}

external_programs["abc2ps"] = "abcm2ps"
external_functionality["abc2ps"] = "conversion to postscript"

external_programs["abc2abc"] = "abc2abc"
external_functionality["abc2abc"] = "external transposition"

external_programs["abc2xml"] = "abc2xml"
external_functionality["abc2xml"] = "conversion to musicXML"

external_programs["abc2midi"] = "abc2midi"
external_functionality["abc2midi"] = "conversion to midi"

external_programs["gv"] = "gv"
external_functionality["gv"] = "displaying postscript"

external_programs["gs"] = ["gs", "gswin32c"]
external_functionality["gs"] = "converting to PDF"

external_programs["lpr"] = "lpr"
external_functionality["lpr"] = "printing"

external_programs["midiplayer"] = "timidity"
external_functionality["midiplayer"] = "playing back abc"

external_programs["editor"] = "emacs"
external_functionality["editor"] = "editing"

external_programs["file"] = "file"
external_functionality["file"] = "determining file type"

external_programs["iconv"] = "iconv"
external_functionality["iconv"] = "converting incoding"

external_programs["pdftk"] = "pdftk"
external_functionality["pdftk"] = "adding correct title to PDF files"


class abctune:
    abc = ""

    voices = []

    root_splitter = re.compile("[a-gA-G]")

    key_aliases = {"": "maj", "none": ""}

    modes = {
        "ionian": 0,
        "ion": 0,
        "dorian": 1,
        "dor": 1,
        "phrygian": 2,
        "phr": 2,
        "lydian": 3,
        "lyd": 3,
        "mixolydian": 4,
        "mix": 4,
        "aeolian": 5,
        "aeo": 5,
        "locrian": 6,
        "loc": 6,
        "maj": 0,
        "m": 5,
        "minor": 5,
        "min": 5,
    }

    white_keys = ["C", "D", "E", "F", "G", "A", "B"]

    scales = {
        "": ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        "m": ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
    }
    scales["major"] = scales[""]

    accidentals = {
        "C": [],
        "Db": ["Db", "Eb", "Gb", "Ab", "Bb"],
        "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
        "D": ["F#", "C#"],
        "Eb": ["Eb", "Ab", "Bb"],
        "E": ["F#", "G#", "C#", "D#"],
        "Fb": ["Fb", "Gb", "Ab", "Bbb", "Cb", "Db", "Eb"],
        "F": ["Bb"],
        "F#": ["F#", "G#", "A#", "C#", "D#", "E#"],
        "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb"],
        "G": ["F#"],
        "Ab": ["Ab", "Bb", "Db", "Eb"],
        "G#": ["G#", "A#", "B#", "C#", "D#", "E#", "F##"],
        "A": ["C#", "F#", "G#"],
        "Bbb": ["Bbb", "Cb", "Db", "Ebb", "Fb", "Gb", "Ab"],
        "Bb": ["Bb", "Eb"],
        "B": ["C#", "D#", "F#", "G#", "A#"],
        "Cb": ["Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb"],
    }

    def __init__(self, string):
        self.abc = string
        self.setVoices()

    def writeAbcToFile(self, file):
        self.writeFile(self.abc, file)

    def writeFile(self, string, file):
        f = open(file, "w")
        f.write(string)
        f.close()

    """
    def readFile(self, file):
        f = open(file,'r')
        filecontent = f.read()
        f.close()
        tunes = filecontent.split("\n\n")
        for tune in tunes:
            if  self.isTune(tune):
                self.abcs.append(abctune(tune))
    """

    def getX(self):
        X = -1
        for line in self.getLines():
            if line.strip()[:2] == "X:":
                X = line.strip()[2:]
        return X

    def getAbc(self):
        return self.abc

    def abc2lines(self, string):
        return string.split("\n")

    def lines2abc(self, lines):
        return "\n".join(lines)

    def getLines(self):
        return self.abc2lines(self.abc)

    def parseMusicLine(self, line):
        parsedLine = []
        note = re.compile("[\^_]*[a-gA-G]")
        while len(line) > 0:
            node = None
            if note.match(line):
                note.match(line).group()
                node = {"type": "note", "abc": note.match(line).group()}
                line = note.sub("", line, 1)
            else:
                node = {"type": "misc", "abc": line[0]}
                parsedLine.append(node)
                line = line[1:]
            if node:
                parsedLine.append(node)
        print(line)
        print(parsedLine)
        sys.exit()

    def generateMidi(self, midiFile):
        abcTmpFile = tempfile.mktemp()
        self.writeAbcToFile(abcTmpFile)
        os.system(
            external_programs["abc2midi"] + " " + abcTmpFile + " -quiet -o " + midiFile
        )
        os.remove(abcTmpFile)

    # def generateXml(file, filename):
    #     remove_words(file)
    #     remove_extra_words(file)
    #     os.system(external_programs["abc2xml"] + " -o " + filename + " " + file)

    def playMidi(self):
        tmpfile = tempfile.mktemp()
        self.generateMidi(tmpfile)
        os.system(external_programs["midiplayer"] + " " + tmpfile)
        os.remove(tmpfile)

    def getLineType(self, line):
        line = line.strip()
        if len(line) > 1 and line[0].isupper() and line[1] == ":":
            return line[:2]
        elif line[:3] == "[V:":
            return "[V:"
        elif line == "":
            return "blank"
        elif line[:2] == "w:":
            return "w:"
        elif line[:8] == "%%staves":
            return "%%staves"
        elif line.split(" ")[0] in ["%%begintext", "%%endtext"]:
            return "pseudocomment"
        elif line[:1] == "%":
            return "comment"
        else:
            return "music"

    def removeExtraLyrics(self):
        lines = self.getLines()
        self.abc = ""
        for line in lines:
            if line[:2] != "W:":
                self.abc += "\n" + line

    def getTitle(self, nb=1):
        i = 1
        # HERHER
        title = []
        for line in self.getLines():
            line = line.strip()
            if line[:2] == "T:":
                if i > nb:
                    break
                i += 1
                title.append(line[2:].strip())
        return ", ".join(title)

    def addTitle(self, title):
        lines = self.getLines()
        self.abc = ""
        inside_title = False
        was_title = False
        for line in lines:
            if line[:2] == "T:":
                inside_title = True
                was_title = True
            else:
                inside_title = False
            if (not inside_title) and was_title:
                self.abc += "\nT:" + title + "\n" + line
                was_title = False
            else:
                self.abc += "\n" + line

    def replaceStrings(self, replaceStrings):
        lines = self.getLines()
        replaceStrings = replaceStrings.split(",")
        while len(replaceStrings) > 1:
            i = 0
            for line in lines:
                lines[i] = line.replace(replaceStrings[0], replaceStrings[1])
                i = i + 1
            replaceStrings = replaceStrings[2:]
        self.abc = "\n".join(lines)

    def explicitAccidentals(self):
        lines = self.getLines()
        modifiedLines = []
        pastHeader = False
        for line in lines:
            if self.getLineType(line) == "K:":
                pastHeader = True
            elif pastHeader and self.getLineType(line) == "music":
                parsedLine = self.parseMusicLine(line)
            modifiedLines.append(line)
        sys.exit()

    def removeLyrics(self):
        lines = self.getLines()
        self.abc = ""
        for line in lines:
            if line[:2] != "w:":
                self.abc += "\n" + line

    def extractLyrics(self):
        slash_save = re.compile("-$")
        slash_replace = re.compile(" *- *")
        lyrics = ""
        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + "\n"
            i = i + 1
        file = ""
        for line in lines:
            line = line.strip()
            if line[:2] == "w:" or line[:2] == "W:":
                current_line = line[2:]
                current_line = current_line.replace("*", "")
                current_line = slash_save.sub("_____ATTEHACK_____", current_line)
                current_line = slash_replace.sub("", current_line)
                current_line = current_line.replace("_____ATTEHACK_____", "-")
                current_line = current_line.strip()
                if current_line != "":
                    lyrics += current_line + "\n"
            elif line[:2] == "P:" and lyrics != "":
                lyrics += "\n"
        print(lyrics[:-1])

    def stavesKeepVoices(self, line, voicesToKeep):
        cleanup1 = re.compile("\( *\)")
        cleanup2 = re.compile("\{ *\}")
        for voice in self.voices:
            if voice not in voicesToKeep:
                line = line.replace(voice, "")
        line = cleanup1.sub("", line)
        line = cleanup2.sub("", line)
        return line

    def keepVoices(self, voicesToKeep):
        voiceRE = re.compile("\[V\:([^\]]*)\]")
        voiceCleanAfterRE = re.compile("[\] ].*")
        voicesToKeep = voicesToKeep.split(",")
        lines = self.getLines()
        newLines = []
        past_header = False
        voiceName = ""
        removeThisLine = False
        protect = False
        for line in lines:
            split = voiceRE.split(line)
            if len(split) == 1:
                newLines.append(split[0])
            elif line.strip()[:3] == "[V:":
                line = split[1:]
                while len(line) > 0:
                    if len(line) > 2:
                        newLines.append("[V:" + line[0] + "]" + line[1] + "\\")
                    else:
                        newLines.append("[V:" + line[0] + "]" + line[1])
                    line = line[2:]
            else:
                line = split[1:]
                newLines.append(split[0] + "\\")
                while len(line) > 0:
                    if len(line) > 2:
                        newLines.append("[V:" + line[0] + "]" + line[1] + "\\")
                    else:
                        newLines.append("[V:" + line[0] + "]" + line[1])
                    line = line[2:]

        lines = newLines
        newLines = []
        for line in lines:
            lineType = self.getLineType(line)
            removeThisLine = False
            if lineType == "K:":
                past_header = True
            elif lineType == "%%staves":
                line = self.stavesKeepVoices(line, voicesToKeep)
            elif past_header:
                if lineType == "V:":
                    voice = line.replace("V:", "").strip()
                    voice = voice.split(" ")
                    if len(voice) > 0:
                        voiceName = voice[0]
                elif lineType == "[V:":
                    voiceName = voiceCleanAfterRE.split(
                        line.replace("[V:", "").strip()
                    )[0]
                elif line.strip().split(" ")[0] in ["%%begintext"]:
                    protect = True
                elif line.strip().split(" ")[0] in ["%%endtext"]:
                    protect = True

                if (
                    voiceName not in voicesToKeep
                    and voiceName != ""
                    and lineType in ["music", "V:", "[V:", "W:", "w:"]
                    and not protect
                ):
                    removeThisLine = True
                else:
                    removeThisLine = False
            if not removeThisLine:
                newLines.append(line)
        self.abc = "\n".join(newLines)

    #        sys.exit()

    def setVoices(self):
        voiceRE = re.compile("\[V\:([^\]]*)\]")
        voices = []
        lines = self.getLines()
        for line in lines:
            if line.strip()[:2] == "V:":
                voice = line.replace("V:", "").strip()
                voice = voice.split(" ")
                if len(voice) > 0:
                    voiceName = voice[0]
                    if voiceName not in voices:
                        voices.append(voiceName)
            else:
                match = voiceRE.match(line)
                if match != None:
                    voiceName = match.group(1).split(" ")[0]
                    if voiceName not in voices:
                        voices.append(voiceName)

        self.voices = voices

    def listVoices(self):
        for voice in self.voices:
            print(voice)

    def listTitles(self, i):
        lines = self.abc.split("\n")
        for line in lines:
            line = line.strip()
            if line[:2] == "T:":
                print(str(i) + ":" + line[2:].strip())
                break

    def removeChordsFromLine(self, line):
        split = line.split('"')
        if len(split) > 1:
            line = ""
            i = 0
            for cell in split:
                if (i % 2) == 0:
                    line = line + cell
                elif len(cell) > 0 and cell[0] == "@":
                    line = line + '"' + cell + '"'
                i = i + 1
        return line

    def removeChords(self):
        lines = self.getLines()
        past_header = None
        p = re.compile('"[^@][^"]*"')
        lineNb = 0
        for line in lines:
            if line[:2] == "K:":
                past_header = 1
            if past_header:
                # lines[lineNb] = p.sub('',line)
                lines[lineNb] = self.removeChordsFromLine(line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def removeAnnotations(self):
        lines = self.getLines()
        past_header = None
        p = re.compile('"[@][^"]*"')
        lineNb = 0
        for line in lines:
            if line[:2] == "K:":
                past_header = 1
            if past_header:
                lines[lineNb] = p.sub("", line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def danishChords(self):
        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + "\n"
            i = i + 1

        header_done = False
        abc = ""
        for line in lines:
            strip = line.strip()
            if header_done:
                line = self.danishChordsLine(line)
            elif len(strip) == 0 or strip[0] == "K":
                header_done = True
            abc += line

        self.abc = abc

    def danishChordsLine(self, abc):
        i = 0
        result = ""
        abc = abc.split('"')
        while i < len(abc):
            if i % 2 == 1:
                if abc[i][:1] == "@":
                    subst = abc[i]
                else:
                    # subst = string.replace(abc[i],'Bb','bbbbbbb')
                    subst = abc[i].replace("Bb", "bbbbbbb")
                    # subst = string.replace(subst,'B','H')
                    subst = subst.replace("B", "H")
                    # subst = string.replace(subst,'bbbbbbb','Bb')
                    subst = subst.replace("bbbbbbb", "Bb")
                result += '"' + subst + '"'
            else:
                result += abc[i]
            i = i + 1
        return result

    def removeFingerings(self):
        lines = self.getLines()
        past_header = None
        p = re.compile("![12345]!")
        lineNb = 0
        for line in lines:
            if line[:2] == "K:":
                past_header = 1
            if past_header and line.strip()[:2] != "w:" and line.strip()[:2] != "W:":
                lines[lineNb] = p.sub("", line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def removeDecorations(self):
        lines = self.getLines()
        past_header = None
        d1 = re.compile("![^!]*!")
        d2 = re.compile("\+[^\+]*\+")
        lineNb = 0
        for line in lines:
            if line[:2] == "K:":
                past_header = 1
            if past_header and line.strip()[:2] != "w:" and line.strip()[:2] != "W:":
                lines[lineNb] = d2.sub("", d1.sub("", line))
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def jazzChords(self):
        lines = self.getLines()
        file = ""
        past_header = None
        lineNb = 0
        for line in lines:
            if line[:2] == "K:":
                past_header = 1
            if past_header:
                split = line.split('"')
                for i in range(1, len(split), 2):
                    new = split[i]
                    new = new.replace("dim7", "°")
                    new = new.replace("m7b5", "ø")
                    new = new.replace("maj", "Δ")
                    split[i] = new
                lines[lineNb] = '"'.join(split)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def externalTranspose(self, halfsteps):
        tmpfile1 = tempfile.mktemp()
        tmpfile2 = tempfile.mktemp()
        self.writeAbcToFile(tmpfile1)
        os.system(
            external_programs["abc2abc"]
            + " "
            + tmpfile1
            + " -e -t "
            + str(halfsteps)
            + " > "
            + tmpfile2
        )
        f = open(tmpfile2, "r")
        filecontent = f.read()
        f.close()
        os.remove(tmpfile1)
        os.remove(tmpfile2)
        self.abc = filecontent

    def transpose(self, transpose):
        transpose = int(transpose)
        key = "none"
        result = []
        org_applied_accidentals = {}
        applied_accidentals = {}
        in_chord = False
        not_music = ["V", "K", "P", "%", "w", "W", "T"]
        note = re.compile("^[_^=]*[a-gA-G][,']*")
        chord_root = re.compile("^[A-G][b#]*")
        annotation = re.compile(
            '^"[><_^@hijcklmnopqrstuvxxyzæøåHIJKLMNOPQRSTUVWXYZÆØÅ].*?"'
        )
        white_key_root = re.compile("[a-gA-G]")
        k = re.compile("^K: *")
        hp_key = re.compile("^K: *H[pP]")
        tmp_save_key = re.compile("[^ ]*")
        ik = re.compile("^\[ *K: *")
        ik_end = re.compile("^.*?\]")
        deco = re.compile("^\!.*?\!")
        new_deco = re.compile("^\+.*?\+")
        sqr = re.compile("^\[[KMIV].*?\]")

        skip = False

        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + "\n"
            i = i + 1
        file = ""

        # easy solution for K:Hp and K:HP suggested by Hudson Lacerda
        new_lines = []
        for line in lines:
            strip = line.strip()
            if hp_key.match(strip) != None:
                new_lines.append("%" + line)
                new_lines.append("K:Amix\n")
            else:
                new_lines.append(line)
        lines = new_lines

        voice = ""
        voice_entry_key = {}
        previous_key = ""
        header_done = False
        for line in lines:
            strip = line.strip()
            if len(strip) == 0:
                header_done = False
                key = "none"
                result.append(line)
            elif strip[:11] == "%%begintext":
                skip = True
                result.append(line)
            elif strip[:9] == "%%endtext":
                skip = False
                result.append(line)
            elif skip:
                result.append(line)
            elif strip[0] == "K" and not header_done:
                header_done = True
                key_res = k.match(strip)
                key = strip[key_res.span()[1] :]
                # this stips off everything after first space
                # FIXME; should handle modes and exp, UPDATE: it does handle modes
                # tmp_saved_key = tmp_save_key.match(key)
                # key = key[:tmp_saved_key.span()[1]]

                # print(key)
                # sys.exit()
                key_setup = self.setup_key(key, transpose)
                key = key_setup["key"]
                new_key = key_setup["new_key"]
                key_root = key_setup["key_root"]
                key_type = key_setup["key_type"]
                source_scale = key_setup["source_scale"]
                target_scale = key_setup["target_scale"]
                source_chord_scale = key_setup["source_chord_scale"]
                target_chord_scale = key_setup["target_chord_scale"]
                result.append("K:" + new_key + "\n")
            elif strip[0] == "P":
                org_applied_accidentals = {}
                applied_accidentals = {}
                # voice_entry_key = {}
                result.append(line)
            elif strip[0] == "V" and header_done:
                voice = re.sub(" .*", "", strip[2:].strip())
                org_applied_accidentals = {}
                applied_accidentals = {}
                if voice in voice_entry_key.keys() and voice_entry_key[voice] != "":
                    key = voice_entry_key[voice]
                    key_setup = self.setup_key(key, transpose)
                    key = key_setup["key"]
                    new_key = key_setup["new_key"]
                    key_root = key_setup["key_root"]
                    key_type = key_setup["key_type"]
                    source_scale = key_setup["source_scale"]
                    target_scale = key_setup["target_scale"]
                    source_chord_scale = key_setup["source_chord_scale"]
                    target_chord_scale = key_setup["target_chord_scale"]
                else:
                    voice_entry_key[voice] = key
                result.append(line)
            elif not header_done or strip[0] in not_music:
                result.append(line)
            else:
                while len(line) > 0:
                    next = None
                    note_res = note.match(line)
                    inkey_res = ik.match(line)
                    deco_res = deco.match(line)
                    new_deco_res = new_deco.match(line)
                    sqr_res = sqr.match(line)
                    annotation_res = annotation.match(line)
                    if line[0] == "|":
                        org_applied_accidentals = {}
                        applied_accidentals = {}
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif annotation_res != None and not in_chord:
                        annotation_length = annotation_res.span()[1]
                        next = line[:annotation_length]
                        result.append(next)
                        line = line[annotation_length:]
                    elif line[0] == '"':
                        in_chord = not in_chord
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif in_chord:
                        chord_res = chord_root.match(line)
                        if chord_res != None:
                            chord_length = chord_res.span()[1]
                            next = line[:chord_length]
                            next = self.transpose_in_scales(
                                next, source_chord_scale, target_chord_scale
                            )
                            result.append(next)
                            line = line[chord_length:]
                        else:
                            next = line[0]
                            result.append(next)
                            line = line[1:]

                    elif inkey_res != None:
                        k_start = line[: inkey_res.span()[1]]
                        result.append(k_start)
                        line = line[inkey_res.span()[1] :]
                        inkey_end_res = ik_end.match(line)
                        if (
                            voice in voice_entry_key.keys()
                            and voice_entry_key[voice] == ""
                        ):
                            voice_entry_key[voice] = key
                        key = line[: inkey_end_res.span()[1] - 1]
                        voice_entry_key[voice] = key

                        # this stips off everything after first space
                        # FIXME; should handle modes and exp
                        tmp_saved_key = tmp_save_key.match(key)
                        key = key[: tmp_saved_key.span()[1]]

                        key_setup = self.setup_key(key, transpose)
                        key = key_setup["key"]
                        new_key = key_setup["new_key"]
                        key_root = key_setup["key_root"]
                        key_type = key_setup["key_type"]
                        source_scale = key_setup["source_scale"]
                        target_scale = key_setup["target_scale"]
                        source_chord_scale = key_setup["source_chord_scale"]
                        target_chord_scale = key_setup["target_chord_scale"]

                        result.append(new_key)

                        line = line[inkey_end_res.span()[1] - 1 :]
                    elif deco_res != None:
                        next = line[: deco_res.span()[1]]
                        result.append(next)
                        line = line[deco_res.span()[1] :]
                    elif new_deco_res != None:
                        next = line[: new_deco_res.span()[1]]
                        result.append(next)
                        line = line[new_deco_res.span()[1] :]
                    elif sqr_res != None:
                        next = line[: sqr_res.span()[1]]
                        result.append(next)
                        line = line[sqr_res.span()[1] :]
                    elif note_res != None:
                        note_length = note_res.span()[1]
                        next = line[:note_length]
                        next = self.normalize_octave(next)
                        if key_type == "none":
                            next_res = white_key_root.search(next)
                            next_root = next[next_res.span()[0] :]
                            next_accidental = next[: next_res.span()[0]]
                            if next_accidental != "":
                                org_applied_accidentals[next_root] = next_accidental
                            else:
                                if next_root not in org_applied_accidentals.keys():
                                    org_applied_accidentals[next_root] = "="
                            next = org_applied_accidentals[next_root] + next_root
                            next = self.transpose_in_scales(
                                next, source_scale, target_scale
                            )
                            next_res = white_key_root.search(next)
                            next_root = next[next_res.span()[0] :]
                            next_accidental = next[: next_res.span()[0]]
                            if (
                                next_accidental == "="
                                and next_root not in applied_accidentals.keys()
                            ):
                                applied_accidentals[next_root] = "="

                            if (
                                next_root in applied_accidentals.keys()
                                and applied_accidentals[next_root] == next_accidental
                            ):
                                next = next_root
                            applied_accidentals[next_root] = next_accidental
                        else:
                            next = self.transpose_in_scales(
                                next, source_scale, target_scale
                            )
                        result.append(next)
                        line = line[note_length:]
                    else:
                        next = line[0]
                        result.append(next)
                        line = line[1:]

        result = "".join(result)

        self.abc = result.strip("\n")

    def setup_key(self, key, transpose):
        # remove %comment after key
        comment = re.compile("%.*")
        key = comment.sub("", key).strip()

        root = re.compile("^[A-G][b#]?")
        none = re.compile("none")
        # strip = string.lower(key).strip()
        strip = key.lower().strip()
        if strip in ["none", ""]:
            key = "Cnone"  # Hmmm, what's up with this, changed to...
            # key = 'C'
        else:
            # look for alternate ways to write keys
            root_res = root.match(key)
            # print(root_res)
            # sys.exit()
            key_root = key[: root_res.span()[1]]
            key_type = key[root_res.span()[1] :]
            # key_type = string.lower(key_type).strip()
            key_type = key_type.lower().strip()
            if key_type != "" and key_type in self.modes:
                for real_key_type in self.key_aliases:
                    if key_type == self.key_aliases[real_key_type]:
                        key_type = real_key_type
                        key = key_root + key_type
                        break
            else:
                if key_type == "":
                    key = key_root
                else:
                    exit

            # make F# -> Gb
            key = self.non_enharmonic_key(key)
        # recalculate key_root, in case it was changed in non_enharmonic_key()
        root_res = root.match(key)
        key_root = key[: root_res.span()[1]]
        key_type = key[root_res.span()[1] :]

        result = {}

        source_scale = self.find_scale(key)
        source_chord_scale = self.find_chord_scale(key)
        new_key = self.find_key(key, transpose)
        target_scale = self.find_scale(new_key)
        scale_offset = self.get_midi_key(source_scale[0]) - self.get_midi_key(
            target_scale[0]
        )
        if (scale_offset + transpose) % 12 == 0:
            while scale_offset + transpose != 0:
                if scale_offset + transpose > 0:
                    target_scale = target_scale[42:]
                else:
                    source_scale = source_scale[42:]

                scale_offset = self.get_midi_key(source_scale[0]) - self.get_midi_key(
                    target_scale[0]
                )
        else:
            pass

        target_chord_scale = self.find_chord_scale(new_key)

        none_res = none.search(new_key)
        if none_res != None:
            new_key = "none"

        result["key"] = key
        result["new_key"] = new_key
        result["source_scale"] = source_scale
        result["target_scale"] = target_scale
        result["source_chord_scale"] = source_chord_scale
        result["target_chord_scale"] = target_chord_scale
        result["key_root"] = key_root
        result["key_type"] = key_type

        return result

    def non_enharmonic_key(self, key):
        if key == "F#":
            result = "Gb"
        elif key == "D#m":
            result = "Ebm"
        elif key == "D#minor":
            result = "Ebminor"
        else:
            result = key

        return result

    def find_scale(self, key):
        none = False
        root = re.compile("^[A-G][b#]?")
        root_res = root.match(key)
        scale_root = key[: root_res.span()[1]]
        scale_type = key[root_res.span()[1] :]
        actual_scale = self.map_key_to_accidentals(key)
        actual_scale = actual_scale.split(" ")[0]

        if scale_type == "major":
            key = scale_root
        if scale_type == "none":
            key = scale_root
            none = True
        result = []
        octaves = [
            ",,,,,,",
            ",,,,,",
            ",,,,",
            ",,,",
            ",,",
            ",",
            "",
            "lower",
            "'",
            "''",
            "'''",
            "''''",
            "'''''",
        ]
        offset = self.white_keys.index(key[0])
        i = offset
        lower = False
        for modifier in octaves:
            if modifier == "lower":
                lower = True
                modifier = ""
            while i < 7:
                white_key = self.white_keys[i]
                real_key = self.apply_accidentals(
                    white_key, self.accidentals[actual_scale]
                )
                if lower:
                    # white_key = string.lower(white_key)
                    white_key = white_key.lower()
                white_key = white_key + modifier
                if len(real_key) == 1:
                    if none:
                        result.append("__" + white_key)
                        result.append("_" + white_key)
                        result.append("=" + white_key)
                        result.append("^" + white_key)
                        result.append("^^" + white_key)
                        result.append("")
                    else:
                        result.append("__" + white_key)
                        result.append("_" + white_key)
                        result.append(white_key)
                        result.append("=" + white_key)
                        result.append("^" + white_key)
                        result.append("^^" + white_key)
                elif real_key[1] == "b":
                    if none:
                        result.append("___" + white_key)
                        result.append("__" + white_key)
                        result.append("_" + white_key)
                        result.append("=" + white_key)
                        result.append("^" + white_key)
                        result.append("")
                    else:
                        result.append("___" + white_key)
                        result.append("__" + white_key)
                        result.append(white_key)
                        result.append("_" + white_key)
                        result.append("=" + white_key)
                        result.append("^" + white_key)
                elif real_key[1] == "#":
                    if none:
                        result.append("_" + white_key)
                        result.append("=" + white_key)
                        result.append("^" + white_key)
                        result.append("^^" + white_key)
                        result.append("^^^" + white_key)
                        result.append("")
                    else:
                        result.append("_" + white_key)
                        result.append("=" + white_key)
                        result.append(white_key)
                        result.append("^" + white_key)
                        result.append("^^" + white_key)
                        result.append("^^^" + white_key)
                else:
                    pass
                i = i + 1
            i = 0
        return result

    def map_key_to_accidentals(self, key):
        root = re.compile("^[A-G][b#]?")
        root_res = root.match(key)
        scale_root = key[: root_res.span()[1]]
        scale_type = key[root_res.span()[1] :]
        # scale_type = string.lower(scale_type).strip()
        scale_type = scale_type.lower().strip()

        if scale_type == "none":
            return scale_root

        degree = None
        for i in self.modes:
            if scale_type == i:
                degree = self.modes[i]
                break

        if degree == None:
            return key

        for potential_key in self.accidentals:
            potential_scale_root = potential_key[0]
            start_on = self.white_keys.index(potential_scale_root)
            if (
                self.apply_accidentals(
                    self.white_keys[(degree + start_on) % 7],
                    self.accidentals[potential_key],
                )
                == scale_root
            ):
                return potential_key

        return key

    def apply_accidentals(self, white_key, accidentals):
        result = white_key
        for i in accidentals:
            if white_key == i[0]:
                result = i
                break
        return result

    def find_chord_scale(self, key):
        none = False
        root = re.compile("^[A-G][b#]?")
        root_res = root.match(key)
        scale_root = key[: root_res.span()[1]]
        scale_type = key[root_res.span()[1] :]
        if scale_type == "major":
            key = scale_root
        if scale_type == "none":
            key = scale_root
            none = True

        result = []
        offset = self.white_keys.index(key[0])
        degree = offset
        for i in range(7):
            white_key = self.white_keys[degree % 7]
            actual_scale = self.map_key_to_accidentals(key)
            actual_scale = actual_scale.split(" ")[0]
            real_key = self.apply_accidentals(white_key, self.accidentals[actual_scale])
            if len(real_key) == 1:
                result.append(white_key + "bb")
                result.append(white_key + "b")
                result.append(white_key)
                result.append(white_key + "#")
                result.append(white_key + "##")
            elif real_key[1] == "b":
                result.append(white_key + "bbb")
                result.append(white_key + "bb")
                result.append(white_key + "b")
                result.append(white_key)
                result.append(white_key + "#")
            elif real_key[1] == "#":
                result.append(white_key + "b")
                result.append(white_key)
                result.append(white_key + "#")
                result.append(white_key + "##")
                result.append(white_key + "###")
            else:
                pass
            degree = degree + 1
        return result

    def find_key(self, source_key, transpose):
        none = False
        root = re.compile("^[A-G][b#]?")
        root_res = root.match(source_key)
        scale_root = source_key[: root_res.span()[1]]
        scale_type = source_key[root_res.span()[1] :]
        if scale_type == "none":
            none = True
            scale_type = ""
        new_root = self.transpose_root(scale_root, scale_type, transpose)
        new_key = new_root + scale_type
        if none:
            new_key = new_key + "none"
        return new_key

    def transpose_root(self, scale_root, scale_type, transpose):
        # scale_type = string.lower(scale_type).strip()
        scale_type = scale_type.lower().strip()
        if scale_type not in self.modes.keys() and scale_type not in self.scales.keys():
            scale_type = "major"

        if scale_type in self.modes.keys():
            actual_scale = self.map_key_to_accidentals(scale_root + scale_type)
            new_actual_scale = self.find_key(actual_scale, transpose)
            new_root = self.white_keys[
                (self.white_keys.index(new_actual_scale[0]) + self.modes[scale_type])
                % 7
            ]
            new_root = self.apply_accidentals(
                new_root, self.accidentals[new_actual_scale]
            )

        else:
            # herher
            # print(scale_type)
            # print(self.scales)
            # print(self.scales[scale_type])
            # sys.exit()
            index = self.scales[scale_type].index(scale_root)
            new_root = self.scales[scale_type][(index + transpose) % 12]
        return new_root

    def get_midi_key(self, note):
        # doesn't work with any abc note, since C could mean C# in A major, etc
        # but since it's only used for the lowest note in a scale where explicit
        # flats are always present, that's ok...
        midi_notes = {
            "C": 60,
            "D": 62,
            "E": 64,
            "F": 65,
            "G": 67,
            "A": 69,
            "B": 71,
            "c": 72,
            "d": 74,
            "e": 76,
            "f": 77,
            "g": 79,
            "a": 81,
            "b": 83,
        }

        root = re.compile("[a-gA-G]")
        root_res = root.search(note)
        root = note[root_res.span()[0] : root_res.span()[1]]
        midi_key = midi_notes[root]

        # midi_key = midi_key - string.count(note,'_')
        # midi_key = midi_key - string.count(note,'^')
        # midi_key = midi_key - (string.count(note,',') * 12)
        # midi_key = midi_key + (string.count(note,"'") * 12)

        midi_key = midi_key - note.count("_")
        midi_key = midi_key - note.count("^")
        midi_key = midi_key - (note.count(",") * 12)
        midi_key = midi_key + (note.count("'") * 12)

        return midi_key

    def transpose_in_scales(self, note, source_scale, target_scale):
        index = source_scale.index(note)
        result = target_scale[index]
        return result

    def normalize_octave(self, note):
        root_res = self.root_splitter.search(note)
        span = root_res.span()
        accidentals = note[: span[0]]
        root = note[span[0] : span[1]]
        octave = note[span[1] :]

        octave_count = octave.count("'") - octave.count(",")
        if octave_count == 0:
            pass
        elif octave_count < 0 and root.islower():
            root = string.upper(root)
            octave_count = octave_count + 1
        elif octave_count > 0 and root.isupper():
            root = string.lower(root)
            octave_count = octave_count - 1

        if octave_count == 0:
            octave = ""
        elif octave_count > 0:
            octave = "'" * octave_count
        else:
            octave = "," * (-octave_count)

        return accidentals + root + octave

    def whiteChordRoot(self, root):
        result = root
        if root == "Cbbb":
            result = "A"
        elif root == "Cbb":
            result = "Bb"
        elif root == "Cb":
            result = "B"
        elif root == "C##":
            result = "D"

        elif root == "Dbb":
            result = "C"
        elif root == "D##":
            result = "E"

        elif root == "Ebb":
            result = "D"
        elif root == "E#":
            result = "F"
        elif root == "E##":
            result = "F#"
        elif root == "E###":
            result = "G"

        elif root == "Fbbb":
            result = "D"
        elif root == "Fbb":
            result = "Eb"
        elif root == "Fb":
            result = "E"
        elif root == "F##":
            result = "G"

        elif root == "Gbb":
            result = "F"
        elif root == "G##":
            result = "A"

        elif root == "Abb":
            result = "G"
        elif root == "A##":
            result = "B"

        elif root == "Bbb":
            result = "A"
        elif root == "B#":
            result = "C"
        elif root == "B##":
            result = "C#"
        elif root == "B###":
            result = "D"
        return result

    def whiteChordRootsTune(self):
        result = []
        in_chord = False
        not_music = ["V", "K", "P", "%", "w", "W"]
        chord = re.compile("^[A-G][b#]*")
        annotation = re.compile('^"[><_^@].*?"')

        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + "\n"
            i = i + 1

        voice_entry_key = ""
        previous_key = ""
        header_done = False
        for line in lines:
            strip = line.strip()
            if len(strip) == 0:
                header_done = False
                result.append(line)
            elif strip[0] == "K" and not header_done:
                header_done = True
                result.append(line)
            elif not header_done or strip[0] in not_music:
                result.append(line)
            else:
                while len(line) > 0:
                    next = None
                    annotation_res = annotation.match(line)
                    if line[0] == "|":
                        org_applied_accidentals = {}
                        applied_accidentals = {}
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif annotation_res != None and not in_chord:
                        annotation_length = annotation_res.span()[1]
                        next = line[:annotation_length]
                        result.append(next)
                        line = line[annotation_length:]
                    elif line[0] == '"':
                        in_chord = not in_chord
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif in_chord:
                        chord_res = chord.match(line)
                        if chord_res != None:
                            chord_length = chord_res.span()[1]
                            next = line[:chord_length]
                            next = self.whiteChordRoot(next)
                            result.append(next)
                            line = line[chord_length:]
                        else:
                            next = line[0]
                            result.append(next)
                            line = line[1:]
                    else:
                        next = line[0]
                        result.append(next)
                        line = line[1:]
        result = "".join(result)

        self.abc = result
