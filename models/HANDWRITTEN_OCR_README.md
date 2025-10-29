# Handwritten OCR Model for Grocery Lists

A deep learning-based OCR system specifically designed for recognizing handwritten grocery lists. This system uses a CRNN (Convolutional Recurrent Neural Network) architecture with CTC (Connectionist Temporal Classification) loss for sequence-to-sequence learning.

## Features

- **CRNN Architecture**: Combines CNN for feature extraction and RNN for sequence modeling
- **CTC Loss**: Enables training without perfect alignment between input and output sequences
- **Data Augmentation**: Improves model generalization with various image transformations
- **Custom Preprocessing**: Optimized for handwritten text recognition
- **Comprehensive Evaluation**: Detailed metrics and error analysis
- **Easy-to-use Interface**: Simple training and inference scripts

## Model Architecture

```
Input Image (1x64x512) 
    ↓
CNN Backbone (7 conv blocks)
    ↓
Feature Maps (256x1x128)
    ↓
Reshape to (128x256)
    ↓
Bidirectional LSTM (2 layers)
    ↓
Output Projection
    ↓
Character Probabilities (128xnum_classes)
```

## Dataset Structure

The model expects a dataset with the following structure:
```
dataset/train_data/
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
├── ...
└── 200.png
    └── 200.txt
```

- **Images**: Handwritten grocery lists in PNG/JPG format
- **Text files**: Ground truth text with grocery items and quantities

Example text file content:
```
Milk - 1 litre
Ghee - 1/2 kg
Sugar - 1 kg
Wheat flour - 5 kg
Basmati rice - 2 kg
```

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements_ocr.txt
```

2. **Verify PyTorch installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### 1. Training the Model

Train the model on your dataset:

```bash
python train_handwritten_ocr.py --dataset_path dataset/train_data --epochs 50
```

**Training Parameters**:
- `--dataset_path`: Path to training dataset (default: `dataset/train_data`)
- `--model_save_path`: Path to save trained models (default: `models`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate (default: 0.001)

### 2. Inference

Predict text from a single image:

```bash
python inference_handwritten_ocr.py --model_path models/best_model.pth --image_path path/to/image.png
```

Predict text from multiple images:

```bash
python inference_handwritten_ocr.py --model_path models/best_model.pth --image_dir path/to/images/ --output_file predictions.txt
```

### 3. Evaluation

Evaluate model performance on the dataset:

```bash
python evaluate_handwritten_ocr.py --model_path models/best_model.pth --dataset_path dataset/train_data
```

## Model Files

After training, the following files will be created in the `models/` directory:

- `best_model.pth`: Best model based on validation loss
- `checkpoint_epoch_X.pth`: Model checkpoints every 10 epochs
- `training_curves.png`: Training and validation loss curves

## Evaluation Results

The evaluation script generates comprehensive results in the `evaluation_results/` directory:

- `detailed_results.csv`: Detailed results for each sample
- `metrics.json`: Overall performance metrics
- `error_analysis.json`: Error pattern analysis
- `summary_report.txt`: Human-readable summary
- `*.png`: Various performance plots

### Key Metrics

- **Character Accuracy**: Character-level similarity between predicted and ground truth text
- **Word Precision/Recall/F1**: Word-level matching metrics
- **Exact Match Rate**: Percentage of perfectly predicted samples
- **Success Rate**: Percentage of samples processed without errors

## Model Performance

The model is designed to achieve high accuracy on handwritten grocery lists. Typical performance metrics:

- **Character Accuracy**: 85-95%
- **Word F1 Score**: 80-90%
- **Exact Match Rate**: 60-80%

Performance may vary based on:
- Handwriting quality and style
- Image quality and resolution
- Amount of training data
- Training duration

## Customization

### Adding New Characters

To add support for new characters, modify the vocabulary building in `handwritten_ocr_model.py`:

```python
def _build_vocabulary(self):
    # Add your custom characters here
    custom_chars = ['₹', '°', '½', '¼', '¾']
    # ... rest of the function
```

### Adjusting Model Architecture

Modify the model architecture in `HandwrittenOCRModel` class:

```python
class HandwrittenOCRModel(nn.Module):
    def __init__(self, num_classes, img_height=64, img_width=512, 
                 hidden_size=256, rnn_layers=2):
        # Adjust these parameters as needed
```

### Data Augmentation

Customize data augmentation in the `GroceryOCRDataset` class:

```python
self.augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # Add more augmentations as needed
])
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 4`
   - Reduce image size in model configuration

2. **Poor Accuracy**:
   - Increase training epochs
   - Add more training data
   - Adjust learning rate
   - Check image preprocessing

3. **Model Not Loading**:
   - Ensure model path is correct
   - Check if model was trained successfully
   - Verify PyTorch version compatibility

### Debug Mode

Enable detailed logging by modifying the training script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Training Loop

For advanced users, you can customize the training process by modifying the `HandwrittenOCRTrainer` class:

```python
# Custom learning rate schedule
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Custom loss function
criterion = CTCLoss(blank_idx=char_to_idx['<BLANK>'], reduction='sum')
```

### Model Ensemble

Combine multiple models for better accuracy:

```python
# Load multiple models
models = [HandwrittenOCRInference(f'models/model_{i}.pth') for i in range(3)]

# Average predictions
predictions = [model.predict(image_path) for model in models]
final_prediction = max(set(predictions), key=predictions.count)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV for image processing
- Albumentations for data augmentation
- The research community for CRNN and CTC loss implementations
