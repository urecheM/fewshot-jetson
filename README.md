# fewshot-jetson

Few-shot object identification on NVIDIA Jetson Nano-class boards using DINOv2

(a) Identify objects from a few labelled examples
(b) Run entirely on a Jetson Nano
(c) Performance data per run
(d) Can substitute DINOv2 with similar models

## Operation

1. **Embed.** A pretrained vision model (DINOv2 ViT-S/14 by default) converts each image into a vector. Images of similar objects get similar vectors.
2. **Enrol.** The vectors of the labelled examples are stored. Each class' vectors are averaged into one reference vector
3. **Identify.** A new image is embedded and compared with each class ref. Closest class wins and anything not close enough to any class is reported as unknown.

## Resources
- **DINOv2** weights and architecture from Meta AI (Apache-2.0), loaded thru [`timm`](https://github.com/huggingface/pytorch-image-models) or meta's [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) torch.hub entry.
- **Nearest-class-mean/prototype classification** (Prototypical Networks, Snell et al. 2017)



On original Jetson nano board the model is run from an exported `.onnx` file. 1 time file conversion -> afterwards enrolment,
identification and benchmarking all execute on the Nano.



## Repo Layout

```
fewshot-jetson/
├── fewshot/                 library
│   ├── backbones.py         model loading
│   ├── data.py              image discovery and preprocessing
│   ├── classifier.py        prototype and k-NN few-shot classifiers
│   ├── profiler.py          latency, throughput, memory, Jetson measurements
│   ├── report.py            writes metrics summary ...
│   └── cli.py               
├── scripts/
│   ├── check_env.py         checks board: is nano original or nano orin
│   ├── enroll.py            learn objects from labelled examples
│   ├── predict.py           identify objects in files, folders or usb camera
│   ├── benchmark.py         full measured test run
│   ├── compare_runs.py      run comparisons charts
│   ├── export_onnx.py       convert model to ONNX (for nano original)
│   └── make_demo_data.py    placeholders
├── data/
│   ├── support/<class>/     your few labelled examples go here
│   └── query/<class>/       your test images go here
├── results/                 one folder per benchmark run
├── tests/test_smoke.py      automated checks
└── docs/                    temp
```

---

## 1. hardware setup

## 2. Add photos

Put images in folders named after the object. The folder name is the label

```
data/support/mug/            5 or more photos of mugs
data/support/water_bottle/   5 or more photos of bottles
data/support/pen/            5 or more photos of pens
data/query/mug/              other mug photos, NOT in support
data/query/water_bottle/
data/query/pen/
```
Note:
- Vary the examples (angles/background/distance from camera)
- Keep query images separate from support images or accuracy is inflated.
- Approx. 10 query images per class

## 3. use

```bash
# Learn objects (re-run whenever examples change)
python3 scripts/enroll.py --support data/support --shots 5 --output enrollment.npz

# Identify
python3 scripts/predict.py --enrollment enrollment.npz --image some_photo.jpg
python3 scripts/predict.py --enrollment enrollment.npz --image data/query/ # a folder
python3 scripts/predict.py --enrollment enrollment.npz --camera 0 --show # USB camera
```

Output per image: predicted label, score, margin and latency.

Options:
a. --shots N | labelled examples per class (0= all in the folder)
b. --method prototype\|knn | classification rule, --k sets neighbours for knn
c. --unknown-threshold 0.4 | answer unknown when the best score is below this
d. --fp16 | half precision on the GPU (less memory, usually faster (PyTorch route))
e. --img-size 224 | input resolution
f. --backbone NAME | model to use

## 4. Measured test runs

```bash
python3 scripts/benchmark.py --support data/support --query data/query --shots 5
```

Each run creates `results/<timestamp>_<model>_<shots>shot/` with:

| File | Contents |
|---|---|
| `summary.md` | summary report: headline numbers, latency table, throughput, memory, accuracy per class, confusion matrix, glossary |
| `metrics.json` | every key carries its unit (`_ms`, `_s`, `_mb`, `_percent`, `_per_s`) |
| `per_image.csv` | one row per processed image: true label, prediction, score, and the time of each stage |

## Swapping models
```bash
python3 scripts/benchmark.py --backbone dinov2_vits14        # default
python3 scripts/benchmark.py --backbone dinov2_vits14_reg    # DINOv2 w registers
python3 scripts/benchmark.py --backbone dino_vits16          # DINO v1
python3 scripts/benchmark.py --backbone dinov3_vits16        # DINOv3
python3 scripts/benchmark.py --backbone clip_vitb32          # CLIP image encoder
python3 scripts/benchmark.py --backbone deit_small           # supervised ViT, same size
python3 scripts/benchmark.py --backbone resnet50             # CNN baseline
python3 scripts/benchmark.py --backbone mobilenetv3_large    # lightweight CNN

python3 scripts/compare_runs.py     # -> results/comparison.csv, .md and .png
```

`python3 scripts/benchmark.py --list-backbones` prints all presets. Any of these work:

- `--backbone timm:<any timm model name>`
- `--backbone hub:<github user>/<repo>:<entry point>` (e.g. `hub:facebookresearch/dinov2:dinov2_vits14`)
- `--backbone onnx:<path to exported .onnx>`

## Deployment
**Original Jetson Nano (ONNX)**:
Unfortunately nano original does not support Jetson PyTorch wheel.
```bash
# one time setup
python3 scripts/export_onnx.py --backbone dinov2_vits14 --output models/dinov2_vits14.onnx
# copy models/dinov2_vits14.onnx and models/dinov2_vits14.onnx.json to the Nano, then:
python3 scripts/enroll.py    --backbone onnx:models/dinov2_vits14.onnx --support data/support
python3 scripts/predict.py   --backbone onnx:models/dinov2_vits14.onnx --image photo.jpg
python3 scripts/benchmark.py --backbone onnx:models/dinov2_vits14.onnx
```

## Testing

```bash
python3 -m pytest -q
```

Classifiers checked on synthetic clusters, unknown rejection, save/load, latency stats, preset resolution, and a benchmark that confirms all report files are produced w/ expected fields. They use random weights so they need no downloads and no GPU.

## Licences and credits

- This repository's code: MIT 
- DINOv2 code and weights: Meta AI, Apache-2.0. Oquab et al., *DINOv2: Learning
  Robust Visual Features without Supervision*, 2023.
- `timm` (PyTorch Image Models): Ross Wightman / Hugging Face, Apache-2.0.
- DINOv3 weights are under Meta's separate DINOv3 License
