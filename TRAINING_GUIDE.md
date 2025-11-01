# Handwritten OCR Training Guide

## Quick Start

Train the handwritten OCR model with a single command:

```bash
python train_handwritten_ocr.py
```

## Training Options

### Basic Training
Train with default settings (50 epochs, batch size 8):
```bash
python train_handwritten_ocr.py
```

### Custom Epochs
Train for more epochs to improve accuracy:
```bash
python train_handwritten_ocr.py --epochs 100
```

### Larger Batch Size
Use larger batch size if you have GPU memory (faster training):
```bash
python train_handwritten_ocr.py --batch-size 16
```

### All Options
```bash
python train_handwritten_ocr.py --epochs 100 --batch-size 16 --dataset-path dataset/train_data --model-save-path models
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 8 | Batch size for training (increase if you have GPU) |
| `--dataset-path` | dataset/train_data | Path to your dataset directory |
| `--model-save-path` | models | Where to save trained models |

## Dataset Requirements

Your dataset should have:
- PNG/JPG images of grocery lists
- Corresponding `.txt` files with ground truth text
- Same base name for image and text files

Example:
```
dataset/train_data/
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
...
```

## Training Output

After training, you'll find:
- `models/best_model.pth` - Best model based on validation loss
- `models/checkpoint_epoch_X.pth` - Checkpoints every 10 epochs
- `models/training_curves.png` - Training visualization

## GPU Requirements

- **Recommended**: CUDA-capable GPU with at least 4GB VRAM
- **Minimum**: CPU (will be slower)
- The script automatically uses GPU if available

## Monitoring Training

Watch for:
1. **Training Loss**: Should decrease over epochs
2. **Validation Loss**: Should decrease (indicates learning)
3. **Similarity**: Test accuracy on sample images

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
python train_handwritten_ocr.py --batch-size 4
```

### Slow Training
- Enable GPU (if available)
- Increase batch size
- Use fewer workers (already set to 0 on Windows)

### Low Accuracy
- Train for more epochs
- Check your dataset quality
- Ensure ground truth text is correct

## Advanced Usage

### Resume Training
The model saves checkpoints - you can modify the code to resume from a checkpoint.

### Custom Architecture
Edit `models/handwritten_ocr_model.py` to modify the CNN/RNN architecture.

## Examples

### Fast Training (Quick Test)
```bash
python train_handwritten_ocr.py --epochs 10 --batch-size 16
```

### Production Training (Best Results)
```bash
python train_handwritten_ocr.py --epochs 100 --batch-size 8
```

### Large Dataset (Maximum Throughput)
```bash
python train_handwritten_ocr.py --epochs 50 --batch-size 32
```

## Tips

1. Start with default settings to ensure everything works
2. Monitor GPU usage: `nvidia-smi` (Linux) or Task Manager (Windows)
3. Training curves help identify overfitting
4. Best model is automatically saved based on validation loss
5. Sample predictions show model performance

## Training Time Estimates

| Dataset Size | Batch Size | Epochs | Est. Time (GPU) | Est. Time (CPU) |
|--------------|------------|--------|-----------------|-----------------|
| 100 images   | 8          | 50     | 5-10 min        | 30-60 min       |
| 500 images   | 8          | 50     | 30-60 min       | 2-4 hours       |
| 1000 images  | 8          | 50     | 1-2 hours       | 4-8 hours       |

*Times are approximate and depend on GPU/CPU performance*

