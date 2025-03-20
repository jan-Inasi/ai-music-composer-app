import numpy as np
import muspy as mp


"""
event representation code
  0-127 - note on
128-255 - note off
256-355 - time shift
356-387 - velocity (default off)
"""
TOP_NOTE_ON = 127
TOP_NOTE_OFF = 255
NUM_OF_NOTES = TOP_NOTE_OFF - TOP_NOTE_ON
NUM_OF_DURATIONS = 100

MAX_PD_EVENT = TOP_NOTE_ON + NUM_OF_DURATIONS


def event_representation_without_off_notes(
    event_representation: np.ndarray,
) -> np.ndarray:
    new_events = []
    for event in event_representation:
        if event <= TOP_NOTE_ON:
            new_events.append(event)
        # skipping notes off <=> (127 < event < 256)
        elif event > TOP_NOTE_OFF:
            new_events.append(event - NUM_OF_NOTES)
    return np.array(new_events)


def to_pitch_and_duration_event_representation(music: mp.Music) -> np.ndarray:
    # event representation without off notes
    event_repr = mp.to_event_representation(music)
    return event_representation_without_off_notes(event_repr)


def pitch_and_duration_to_event_representation(events: np.ndarray) -> np.ndarray:
    """example
    from: on_E5, on_A6, dur_12, on_E5, dur_6
      to: on_E5, on_A6, dur_12, off_E5, off_A6, on_E5, dur_6, off_E5
    """

    new_events = []
    notes_on_since_last_duration = []
    for event in events:
        if event <= TOP_NOTE_ON:
            new_events.append(event)
            notes_on_since_last_duration.append(event)
        else:
            new_events.append(event + NUM_OF_NOTES)  # new non-pitch event
            if event <= TOP_NOTE_ON + NUM_OF_DURATIONS:  # is duration event
                # notes off events
                for note_on in notes_on_since_last_duration:
                    new_events.append(note_on + NUM_OF_NOTES)
    return np.array(new_events)


def from_pitch_and_duration_event_representation(events: np.ndarray) -> mp.Music:
    event_representation = pitch_and_duration_to_event_representation(events)
    return mp.from_event_representation(event_representation)
