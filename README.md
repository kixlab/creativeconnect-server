# CreativeConnect Server
## Setup

1. Install SAM
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

2. Install BLIP
```bash
pip install transformers
pip install numpy==1.23.0
```

3. Install OVSeg

- Clone OVSeg
```bash
git clone https://github.com/facebookresearch/ov-seg
mv ov-seg ovseg
```
- Install dependencies
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
cd ovseg
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
cd third_party/CLIP
python -m pip install -Ue .
```

4. Download checkpoints from [here](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view) and put them in `ovseg/checkpoints`
5. Make 'uploaded' folder in root directory