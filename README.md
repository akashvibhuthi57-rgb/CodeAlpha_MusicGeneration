# AI Music Generation with an LSTM (CodeAlpha Task 3)

A deep learning model that learns the melodic patterns of J.S. Bach's chorales and composes new melodies,
exported as MIDI and WAV.

**Pipeline:** MIDI/MusicXML files -> `music21` note tokens -> LSTM -> sampled melody -> `.mid` + `.wav`

## 1. Setup
```bash
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or normal torch for GPU
pip install music21 numpy matplotlib
```

## 2. Data
- Default: music21's built-in Bach corpus (410 pieces), no download needed: `--corpus bach`
- Your own MIDI (classical, jazz, ...): put `.mid` files in `data/midi/` (e.g. MAESTRO, Lakh MIDI, Classical Piano MIDI). Use one style per run.

## 3. Run
```bash
python music_ai.py preprocess --corpus bach --transpose     # parse + tokenise (a few minutes)
python music_ai.py train --epochs 150                        # early stopping ends it automatically
python music_ai.py generate --length 300 --temperature 0.8 --num-pieces 5
```
Output goes to `output/` (`generated_N.mid`, `generated_N.wav`); the loss plot is `artifacts/loss_curve.png`.
For better-sounding audio: install FluidSynth and pass `--soundfont file.sf2` to `generate`.

## 4. How it works
| Step | Details |
|------|---------|
| Preprocess | `music21` parses each piece, merges voices with `chordify()`, keeps the top voice (melody "skyline"), transposes to C major / A minor, and turns every note into a token such as `E4_0.5` (pitch + duration) |
| Model | Embedding -> 2-layer LSTM (128 units) -> Linear; predicts the next token at every position (264k parameters) |
| Training | AdamW, label smoothing, dropout 0.5, gradient clipping, LR scheduler, validation split, early stopping, best checkpoint saved |
| Generation | Seed with a real phrase, then sample token by token with temperature + top-k |
| Export | `music21` writes MIDI; a built-in synthesiser writes WAV |

## 5. Results
- Data: 410 Bach pieces -> 37,230 note events, vocabulary of 170 tokens
- Best validation loss: **1.71** (random guessing = ln 170 = 5.14), i.e. the model is choosing between about 5-6 plausible notes at each step
- Early stopping at epoch 108 of 150
- Sample outputs and the loss curve are in `examples/`

### Lessons learned (what went wrong first)
| Attempt | Problem | Fix |
|---------|---------|-----|
| Chord tokens (all voices) | 2,103 tokens for only 10k events -> severe overfitting (val loss 8.6 -> 10.8) | Melody-only tokens: vocabulary 170 |
| Overlapping training windows | Each epoch saw every note ~64 times, so the model memorised in epoch 1 | Window stride + label smoothing + embedding dropout + smaller model |

## 6. Tuning tips
- `--temperature`: 0.7 = safe and repetitive, 1.1 = experimental
- Larger model once regularised: `--hidden 256`
- Repeating loops: raise temperature or `--top-k`

## 7. Possible extensions
- Polyphonic output (event-based tokens: note-on / time-shift)
- Transformer instead of LSTM; GAN/VAE variant
- Condition on composer or genre
