# XTTS-v2 Complete Fine-Tuning & Multi-GPU Guide  
(Everything you need to fine-tune one or many voices on 2× RTX 4090)

---

## 0.  Quick Reference

| Step | What you do | What you get |
|---|---|---|
| 1  | Prepare one folder per speaker (`wavs` + `metadata.csv`) | Clean, speaker-specific dataset |
| 2  | Run one `train_xtts.py` process **per speaker** | One checkpoint **per speaker** (~1.8 GB each) |
| 3  | Export the best checkpoint to a self-contained folder | Ready-to-load custom voice |
| 4  | Load with `TTS(model_path=…)` | Native-quality Spanish (or any other language) clone |

> You **cannot** continuously fine-tune the same weights on different speakers—the model will forget the previous voice.  
> Instead, keep the **base XTTS-v2 checkpoint untouched** and spawn **independent fine-tunes**.

---

## 1.  One-Time Environment Setup

```bash
# 1.  Coqui-TTS source (pip wheels are too old)
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e ".[all]"

# 2.  CUDA 11.8 wheels for 2× RTX 4090
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118