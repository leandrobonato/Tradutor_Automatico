"""Demo interativa (Gradio): tradução de texto EN->PT com o modelo treinado.

Uso:
    python app/app.py
"""
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.translate import load_model, translate

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CHECKPOINT = config.MODELS_DIR / "seq2seq_best.pt"

_model = None
_sp_en = None
_sp_pt = None


def _ensure_model_loaded():
    global _model, _sp_en, _sp_pt
    if _model is None:
        if not _CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado em {_CHECKPOINT}. "
                "Rode `python -m src.train` primeiro."
            )
        _model, _sp_en, _sp_pt = load_model(_CHECKPOINT, _DEVICE)


def translate_text(text: str, beam_width: int) -> str:
    if not text or not text.strip():
        return ""
    _ensure_model_loaded()
    return translate(text.strip(), _model, _sp_en, _sp_pt, _DEVICE, beam_width=int(beam_width))


EXAMPLES = [
    ["How are you today?"],
    ["The company needs to translate documents quickly."],
    ["I would like to schedule a meeting for tomorrow morning."],
    ["Thank you very much for your help."],
    ["This report contains important financial information."],
]

with gr.Blocks(title="Tradutor Automático EN → PT") as demo:
    gr.Markdown(
        """
        # 🌐 Tradutor Automático (Inglês → Português)
        Modelo Seq2Seq com Atenção de Bahdanau (Encoder BiLSTM + Decoder LSTM),
        treinado do zero em pares de frases EN→PT do corpus OPUS.
        Inferência via **Beam Search**.
        """
    )
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Texto em inglês", placeholder="Digite um texto em inglês...", lines=4)
            beam_width = gr.Slider(minimum=1, maximum=5, value=config.BEAM_WIDTH, step=1, label="Beam width")
            translate_btn = gr.Button("Traduzir", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Tradução em português", lines=4, interactive=False)

    translate_btn.click(fn=translate_text, inputs=[input_text, beam_width], outputs=output_text)
    input_text.submit(fn=translate_text, inputs=[input_text, beam_width], outputs=output_text)

    gr.Examples(examples=EXAMPLES, inputs=[input_text])

if __name__ == "__main__":
    demo.launch()
