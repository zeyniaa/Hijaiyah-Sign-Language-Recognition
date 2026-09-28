# Dataset Information & Access

## 📖 Background & Acquisition
The dataset used in this research is an exclusive primary dataset recorded at the studio of **Lajnah Pentashihan Mushaf Al-Qur'an (LPMQ), Kementerian Agama RI**, under direct supervision of LPMQ sign language experts.

### Key Specifications:
- **Total Classes**: 112 unique classes (28 Hijaiyah letters $\times$ 4 harakat variations: *Fathah, Kasrah, Dhammah, Sukun*).
- **Total Raw Frames**: 44,697 video frames.
- **Total Video Sequences**: 2,235 sequences ($N = 2,235$).
- **Acquisition Hardware**: Canon EOS 70D with 18-55mm lens.
- **Lighting Setup**: Godox SL60W LED with Softbox diffuser (121 – 130 Lux).

---

## 🗂️ Preprocessed Tensor Format
All extracted keypoints are saved in NumPy `.npy` format:
- `X_padded.npy`: Shape `(2235, 20, 75)` — Float32 tensor with 20 timesteps and 75 normalized keypoints per frame.
- `y_encoded.npy`: Shape `(2235, 112)` — One-hot encoded categorical labels.
- `X_train.npy` & `y_train.npy`: Training split.
- `X_test.npy` & `y_test.npy`: Unseen test evaluation split.

---

## 📥 Dataset Download (Google Drive)
Due to file size constraints on GitHub, the complete dataset and pre-trained weights are hosted on Google Drive:

🔗 **Google Drive Dataset & Weights Link**:
[https://colab.research.google.com/drive/1EqHWbovCn-lnD4LRET4GSBMm_w4u4ch4](https://colab.research.google.com/drive/1EqHWbovCn-lnD4LRET4GSBMm_w4u4ch4)

Place downloaded files into this folder:
```text
data/
├── sample/
│   ├── X_padded.npy
│   └── y_encoded.npy
```
*(If no files are placed in `data/sample/`, the training pipeline automatically initializes synthetic demonstration tensors so the codebase remains runnable immediately).*
