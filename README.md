# Vision Aid - Histopathological Image Super-Resolution

A deep learning project that implements knowledge distillation for super-resolution enhancement of lung and colon cancer histopathological images using PyTorch.

## Overview

This project uses a teacher-student knowledge distillation framework to enhance the resolution of medical histopathological images. The system is designed to improve image quality for better medical diagnosis and analysis of lung and colon cancer tissue samples.

## Features

- **Knowledge Distillation**: Implements teacher-student architecture for efficient model training
- **Medical Image Processing**: Specialized for histopathological images of lung and colon cancer
- **Super-Resolution Enhancement**: Improves image resolution while preserving medical details
- **SSIM Evaluation**: Uses Structural Similarity Index for quality assessment
- **Multi-Loss Training**: Combines L1 loss, SSIM loss, and distillation loss

## Dataset

The project uses the [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) dataset from Kaggle, which contains:

- **Lung adenocarcinoma (lung_aca)**
- **Lung benign tissue (lung_n)**
- **Colon adenocarcinoma (colon_aca)**
- **Colon benign tissue (colon_n)**

## Architecture

### Teacher Network (TeacherSharpNet)
- 6 ResNet blocks with 128 channels
- Encoder-decoder architecture
- Higher capacity for better feature extraction

### Student Network (StudentSharpNet)
- 2 ResNet blocks with 64 channels
- Lightweight design for efficient inference
- Learns from teacher network through knowledge distillation

## Requirements

```
torch
torchvision
pytorch-msssim
kagglehub
PIL (Pillow)
numpy
matplotlib
glob
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BSRohit20/Vision-aid.git
cd Vision-aid
```

2. Install required packages:
```bash
pip install torch torchvision pytorch-msssim kagglehub pillow numpy matplotlib
```

3. Download the dataset (handled automatically in the notebook):
```python
import kagglehub
dataset_path = kagglehub.dataset_download('andrewmvd/lung-and-colon-cancer-histopathological-images')
```

## Usage

1. **Data Loading**: The `HistologyImageDataset` class handles loading and preprocessing of histopathological images
2. **Model Training**: Train the student network using knowledge distillation from the teacher network
3. **Evaluation**: Assess model performance using SSIM scores
4. **Visualization**: Compare low-resolution input, ground truth, and enhanced output

### Running the Notebook

Open `Intel_Unatti.ipynb` and run the cells sequentially:

1. **Cell 1**: Download dataset using Kaggle API
2. **Cell 2**: Install pytorch-msssim dependency
3. **Cell 3-4**: Import libraries and define dataset class
4. **Cell 5**: Define neural network architectures
5. **Cell 6**: Define loss functions
6. **Cell 7**: Create dataset and data loader
7. **Cell 8**: Train the model
8. **Cell 9-11**: Evaluate and visualize results

## Model Performance

The model is evaluated using:
- **SSIM (Structural Similarity Index)**: Measures structural similarity between enhanced and ground truth images
- **L1 Loss**: Pixel-wise absolute difference
- **Knowledge Distillation Loss**: Ensures student learns from teacher

## Key Components

### Loss Function
The training uses a weighted combination of three losses:
```python
loss = (α × L1_loss + β × SSIM_loss + γ × Distillation_loss)
```
Where α=0.4, β=0.3, γ=0.3

### Data Augmentation
- Random Gaussian blur application
- Resize transformations for super-resolution task
- Tensor normalization

## File Structure

```
Vision-aid/
├── Intel_Unatti.ipynb    # Main notebook with implementation
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Results

The model produces enhanced histopathological images with improved resolution while maintaining medical image characteristics essential for diagnostic purposes.

## Future Improvements

- Implement additional super-resolution architectures (ESRGAN, SRGAN)
- Add more sophisticated evaluation metrics
- Extend to other medical imaging modalities
- Implement real-time inference capabilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019](https://arxiv.org/abs/1912.12142)
- PyTorch team for the deep learning framework
- Kaggle for hosting the dataset

## Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Note**: This project is for research and educational purposes. Always consult with medical professionals for clinical applications.
