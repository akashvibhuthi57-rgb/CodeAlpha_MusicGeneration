"""
AI Music Generation with an LSTM  (CodeAlpha Task 3)
====================================================
Pipeline:  MIDI files -> music21 tokens -> LSTM -> sampled tokens -> MIDI + WAV

Usage
-----
  python music_ai.py preprocess --midi-dir data/midi            # your own MIDI files
  python music_ai.py preprocess --corpus bach --max-files 100   # or music21's built-in corpus
  python music_ai.py train --epochs 30
  python music_ai.py generate --length 300 --temperature 0.9
  python music_ai.py all --corpus bach --epochs 30              # everything in one go
"""
import argparse, glob, os, pickle, random, shutil, subprocess, wave
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

OUT_DIR = "output"
ART_DIR = "artifacts"
DATA_FILE = os.path.join(ART_DIR, "data.pkl")
MODEL_FILE = os.path.join(ART_DIR, "model.pt")

# Allowed note lengths (in quarter notes). Real durations get snapped to the closest one,
# which keeps the vocabulary small enough for the model to learn.
DURATIONS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


# ----------------------------------------------------------------------------
# 1. PREPROCESSING  (MIDI -> token sequence)
# ----------------------------------------------------------------------------
def snap_duration(d):
    return float(DURATIONS[np.abs(DURATIONS - float(d)).argmin()])


def midi_to_tokens(path, transpose=False, mode="melody"):
    """Parse one MIDI file into tokens like 'C4_0.5', '60.64.67_1.0' (chord) or 'rest_0.5'."""
    from music21 import converter, interval, pitch, note, chord

    score = converter.parse(path)
    if transpose:  # normalise every piece to C major / A minor
        key = score.analyze("key")
        target = "C" if key.mode == "major" else "A"
        score = score.transpose(interval.Interval(key.tonic, pitch.Pitch(target)))

    # chordify merges all voices into one stream of "what sounds at each onset"
    events = score.chordify().flatten().notesAndRests
    tokens = []
    for el in events:
        dur = snap_duration(el.duration.quarterLength)
        if isinstance(el, note.Rest):
            tokens.append(f"rest_{dur}")
        elif isinstance(el, chord.Chord):
            ps = sorted(p.midi for p in el.pitches)
            if mode == "melody" or len(ps) == 1:
                # "skyline": keep only the highest note -> tiny vocabulary, much easier to learn
                top = max(el.pitches, key=lambda p: p.midi)
                tokens.append(f"{top.nameWithOctave}_{dur}")
            else:
                tokens.append(".".join(str(p) for p in ps) + f"_{dur}")
        elif isinstance(el, note.Note):
            tokens.append(f"{el.pitch.nameWithOctave}_{dur}")
    return tokens


def collect_files(args):
    if args.corpus:
        from music21 import corpus
        paths = [str(p) for p in corpus.getComposer(args.corpus)
                 if str(p).lower().endswith((".mxl", ".xml", ".musicxml", ".mid", ".midi"))]
    else:
        paths = []
        for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
            paths += glob.glob(os.path.join(args.midi_dir, "**", ext), recursive=True)
    random.shuffle(paths)
    return paths[: args.max_files] if args.max_files else paths


def preprocess(args):
    paths = collect_files(args)
    if not paths:
        raise SystemExit(f"No MIDI files found. Put .mid files in '{args.midi_dir}' or use --corpus bach")
    print(f"Parsing {len(paths)} files ...")

    all_tokens = []
    for i, p in enumerate(paths, 1):
        try:
            toks = midi_to_tokens(p, args.transpose, args.mode)
            all_tokens += toks
            print(f"  [{i}/{len(paths)}] {os.path.basename(p)}: {len(toks)} events")
        except Exception as e:  # corrupt / unsupported files are skipped
            print(f"  [{i}/{len(paths)}] skipped {os.path.basename(p)} ({type(e).__name__})")

    # Drop very rare tokens so the softmax isn't dominated by one-off events
    counts = Counter(all_tokens)
    all_tokens = [t for t in all_tokens if counts[t] >= args.min_count]
    vocab = sorted(set(all_tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    seq = np.array([stoi[t] for t in all_tokens], dtype=np.int64)

    os.makedirs(ART_DIR, exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"seq": seq, "vocab": vocab}, f)
    print(f"Saved {len(seq)} events, vocabulary size {len(vocab)} -> {DATA_FILE}")
    if len(vocab) > len(seq) / 20:
        print("WARNING: vocabulary is large compared to the data - the model will overfit. "
              "Use --mode melody, more files, or a higher --min-count.")


# ----------------------------------------------------------------------------
# 2. MODEL  (Embedding -> stacked LSTM -> Linear)
# ----------------------------------------------------------------------------
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        out, state = self.lstm(self.drop(self.embed(x)), state)
        return self.fc(self.drop(out)), state


class WindowDataset(Dataset):
    """Input = events[i : i+L], target = the same window shifted by one (next-event prediction)."""
    def __init__(self, seq, length, stride=1):
        self.seq, self.length, self.stride = torch.from_numpy(seq), length, stride

    def __len__(self):
        return (len(self.seq) - self.length - 1) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        return self.seq[i : i + self.length], self.seq[i + 1 : i + self.length + 1]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------------------------
def train(args):
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    seq, vocab = data["seq"], data["vocab"]
    if len(seq) < args.seq_len * 3:
        raise SystemExit("Not enough data - add more MIDI files.")

    split = int(len(seq) * 0.9)  # contiguous split avoids leaking neighbouring windows
    train_dl = DataLoader(WindowDataset(seq[:split], args.seq_len, args.stride), args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(WindowDataset(seq[split:], args.seq_len, args.seq_len // 2), args.batch_size)

    device = get_device()
    model = MusicLSTM(len(vocab), args.embed, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()                                   # validation: plain loss
    train_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # training: smoothed
    print(f"Training on {device} | {len(seq)} events | vocab {len(vocab)} | "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    best, bad, history = float("inf"), 0, {"train": [], "val": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = train_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # keeps LSTM gradients stable
            opt.step()
            tot += loss.item()
        tr = tot / max(1, len(train_dl))

        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model(x.to(device))[0].reshape(-1, len(vocab)),
                                  y.to(device).reshape(-1)).item() for x, y in val_dl])
        sched.step(vl)
        history["train"].append(tr); history["val"].append(vl)
        flag = ""
        if vl < best:
            best, bad = vl, 0
            torch.save({"state": model.state_dict(), "vocab": vocab,
                        "cfg": dict(embed=args.embed, hidden=args.hidden,
                                    layers=args.layers, dropout=args.dropout,
                                    seq_len=args.seq_len)}, MODEL_FILE)
            flag = "  <- saved"
        else:
            bad += 1
        print(f"Epoch {epoch:3d}/{args.epochs}  train {tr:.3f}  val {vl:.3f}{flag}")
        if bad >= args.patience:
            print(f"Early stopping: validation loss has not improved for {args.patience} epochs.")
            break

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(history["train"], label="train"); plt.plot(history["val"], label="validation")
        plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.legend()
        plt.savefig(os.path.join(ART_DIR, "loss_curve.png"), dpi=120)
        print("Loss curve saved to artifacts/loss_curve.png")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 4. GENERATION  (sample tokens -> MIDI -> WAV)
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample_tokens(model, vocab, seed_ids, length, temperature, top_k, device):
    model.eval()
    x = torch.tensor([seed_ids], device=device)
    logits, state = model(x)                      # warm up the LSTM on the seed
    out, last = [], logits[0, -1]
    for _ in range(length):
        logits = last / max(temperature, 1e-3)    # low T = safe/repetitive, high T = adventurous
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        out.append(vocab[nxt])
        logits, state = model(torch.tensor([[nxt]], device=device), state)
        last = logits[0, -1]
    return out


def token_to_parts(tok):
    """'C4_0.5' -> ([midi numbers], 0.5)   |   'rest_1.0' -> ([], 1.0)"""
    from music21 import pitch
    p, d = tok.rsplit("_", 1)
    if p == "rest":
        return [], float(d)
    if p.replace(".", "").isdigit():
        return [int(n) for n in p.split(".")], float(d)
    return [pitch.Pitch(p).midi], float(d)


def tokens_to_midi(tokens, path, bpm):
    from music21 import stream, note, chord, tempo, instrument
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=bpm))
    s.insert(0, instrument.Piano())
    offset = 0.0
    for tok in tokens:
        midis, dur = token_to_parts(tok)
        if len(midis) == 1:
            el = note.Note(midis[0])
        elif len(midis) > 1:
            el = chord.Chord(midis)
        else:
            el = note.Rest()
        el.quarterLength = dur
        s.insert(offset, el)
        offset += dur
    s.write("midi", fp=path)


def tokens_to_wav(tokens, path, bpm, sr=22050):
    """Dependency-free synth: sine + one overtone with a plucked-piano style envelope."""
    spb = 60.0 / bpm
    events, t = [], 0.0
    for tok in tokens:
        midis, dur = token_to_parts(tok)
        events.append((t, dur * spb, midis))
        t += dur * spb
    buf = np.zeros(int((t + 2.0) * sr), dtype=np.float32)
    for start, length, midis in events:
        if not midis:
            continue
        n = int((length + 0.4) * sr)
        tt = np.arange(n) / sr
        env = np.minimum(tt / 0.01, 1.0) * np.exp(-3.0 * tt / max(length, 0.2))
        for m in midis:
            f = 440.0 * 2 ** ((m - 69) / 12)
            wave_ = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt)
            i = int(start * sr)
            buf[i : i + n] += (env * wave_ / len(midis)).astype(np.float32)[: len(buf) - i]
    buf /= max(1e-6, np.abs(buf).max())
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((buf * 32000).astype(np.int16).tobytes())


def generate(args):
    ckpt = torch.load(MODEL_FILE, map_location="cpu")
    vocab, cfg = ckpt["vocab"], ckpt["cfg"]
    device = get_device()
    model = MusicLSTM(len(vocab), cfg["embed"], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state"])

    with open(DATA_FILE, "rb") as f:
        seq = pickle.load(f)["seq"]
    start = random.randint(0, len(seq) - cfg["seq_len"] - 1)
    seed_ids = seq[start : start + cfg["seq_len"]].tolist()  # a real phrase to "prime" the model

    os.makedirs(OUT_DIR, exist_ok=True)
    for k in range(1, args.num_pieces + 1):
        toks = sample_tokens(model, vocab, seed_ids, args.length, args.temperature, args.top_k, device)
        midi_path = os.path.join(OUT_DIR, f"generated_{k}.mid")
        tokens_to_midi(toks, midi_path, args.bpm)
        print(f"Saved {midi_path}")

        wav_path = os.path.join(OUT_DIR, f"generated_{k}.wav")
        if args.soundfont and shutil.which("fluidsynth"):  # high quality audio if available
            subprocess.run(["fluidsynth", "-ni", args.soundfont, midi_path, "-F", wav_path, "-r", "44100"],
                           check=False, stdout=subprocess.DEVNULL)
        else:
            tokens_to_wav(toks, wav_path, args.bpm)
        print(f"Saved {wav_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_pre(p):
        p.add_argument("--midi-dir", default="data/midi")
        p.add_argument("--corpus", help="use a composer from music21's built-in corpus, e.g. bach, mozart, beethoven")
        p.add_argument("--max-files", type=int, default=0)
        p.add_argument("--min-count", type=int, default=2)
        p.add_argument("--transpose", action="store_true", help="normalise all pieces to C major / A minor")
        p.add_argument("--mode", choices=["melody", "chords"], default="melody",
                       help="melody = top voice only (small vocab, recommended); chords = all voices (huge vocab)")

    def add_train(p):
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--seq-len", type=int, default=64)
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--embed", type=int, default=64)
        p.add_argument("--hidden", type=int, default=128)
        p.add_argument("--layers", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.5)
        p.add_argument("--weight-decay", type=float, default=1e-2)
        p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--stride", type=int, default=8, help="step between training windows (bigger = shorter epochs, less memorising)")
        p.add_argument("--label-smoothing", type=float, default=0.1)

    def add_gen(p):
        p.add_argument("--length", type=int, default=300, help="number of note/chord events to generate")
        p.add_argument("--temperature", type=float, default=0.9)
        p.add_argument("--top-k", type=int, default=20)
        p.add_argument("--bpm", type=int, default=110)
        p.add_argument("--num-pieces", type=int, default=3)
        p.add_argument("--soundfont", help="optional .sf2 file (needs fluidsynth installed) for realistic audio")

    add_pre(sub.add_parser("preprocess")); add_train(sub.add_parser("train")); add_gen(sub.add_parser("generate"))
    a = sub.add_parser("all"); add_pre(a); add_train(a); add_gen(a)

    args = ap.parse_args()
    if args.cmd in ("preprocess", "all"): preprocess(args)
    if args.cmd in ("train", "all"): train(args)
    if args.cmd in ("generate", "all"): generate(args)


if __name__ == "__main__":
    main()
