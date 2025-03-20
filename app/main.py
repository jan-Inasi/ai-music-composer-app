import gradio as gr
import muspy as mp
from generation import (
    tune_extension,
    tune_generation,
    parse_tune,
    TuneABC,
    ModelHandler,
)
import os


def extend_tune(tune_body: str, key: str, bit_len: str):
    tune = tune_extension(tune_body, key, bit_len)
    return display_tune(tune)


def generate_tune():
    tune = tune_generation()
    return display_tune(tune)


def resynthesize_tune(tune_text):
    tune = parse_tune(tune_text)
    return display_tune(tune)[1:]


def display_tune(tune: TuneABC):
    part = tune.convert_to_music21()
    wav = tune.synthesize(part=part)
    save_sheet_music(tune, "data_out/tmp/sheet_music.png", part)
    return (str(tune), wav, "data_out/tmp/sheet_music-1.png")


def save_sheet_music(tune: TuneABC, path: str, part):
    if not os.path.exists(path):
        os.makedirs(path)
    tune.save_as_sheet_music(path, part)


def tune_body_input():
    return gr.Textbox(
        label="melodia do przedłużenia",
        placeholder="ABCD | DCBA ",
        show_label=True,
        interactive=True,
        autofocus=True,
        lines=10,
        max_lines=100,
    )


def copy_to_prompt(text: str):
    tune = parse_tune(text)
    return tune.tune_str(), tune.K, tune.L


def key_choice():
    return gr.Dropdown(
        # fmt: off
        choices=[ "A", "Ab", "B", "Bb", "C", "Cb", "Cm", "D", "Db", "Dm", "E", "Eb", "Em", "F", "F#", "G", "Gm", ],
        # fmt: on
        value="C",
        multiselect=False,
        label="tonacja",
    )


def bit_len_choice():
    return gr.Dropdown(
        choices=["1/1", "1/2", "1/4", "1/8", "1/16", "1/32"],
        value="1/8",
        multiselect=False,
        label="jednostkowa długość bitu",
    )


with gr.Blocks(title="Generator melodii ludowych") as interface:
    inputs = []
    outputs = []
    with gr.Row():
        btn_prompt = gr.Button("wygeneruj kontynuację melodii")
        btn_gen = gr.Button("wygeneruj nową melodię")
        btn_resythesize = gr.Button("zsyntetyzuj melodię")
        btn_as_a_prompt = gr.Button("przenieś wygenerowaną melodię na wejście")
    with gr.Row():
        with gr.Column():
            inputs.append(tune_body_input())
            with gr.Row():
                inputs.append(key_choice())
                inputs.append(bit_len_choice())
        with gr.Column():
            outputs.append(
                gr.Textbox(
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                    label="wygenerowana melodia",
                )
            )
            outputs.append(gr.Audio(label="zsyntezowana melodia"))

    sheet_music = gr.Image(interactive=False, label="notacja nutowa")
    outputs.append(sheet_music)

    btn_prompt.click(fn=extend_tune, inputs=inputs, outputs=outputs)
    btn_gen.click(fn=generate_tune, outputs=outputs)
    btn_resythesize.click(
        fn=resynthesize_tune, inputs=[outputs[0]], outputs=[outputs[1], outputs[2]]
    )
    btn_as_a_prompt.click(fn=copy_to_prompt, inputs=[outputs[0]], outputs=inputs)


def main():
    mp.download_musescore_soundfont(overwrite=False)
    interface.launch()
    ModelHandler.get()


if __name__ == "__main__":
    main()
