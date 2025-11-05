# Smart Bakery POS System

A smart Point of Sale (POS) system for bakeries that uses AI to automatically identify bakery items through camera input. The system streamlines the checkout process by detecting and recognizing bakery items placed on a tray.

## Features

- 🎥 Real-time bakery item detection using computer vision
- 🧠 Product recognition using CNN
- 💻 Web-based POS interface
- 📊 Support for multiple bakery items (10 types)
- 🔄 Flexible tray detection system
- 💳 Integrated payment processing

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML/Computer Vision**: 
  - TensorFlow/Keras
  - OpenCV
  - NumPy
- **Model**: Custom CNN trained on bakery items

## Project Structure

```
├── app.py             # Flask application server
├── data/              # Training data
│   ├── train/         # Training dataset
│   ├── test/          # Testing dataset
│   └── valid/         # Validation dataset
├── models/            # Trained models
│   ├── bakery_cnn.h5  # Main CNN model
│   └── labels.txt     # Label mappings
├── scripts/           # Training scripts
└── web/              # Web interface
    ├── index.html
    ├── script.js
    └── style.css
```

## Setup and Installation

1. Create a conda environment using the provided environment.yml:
   ```bash
   conda env create -f environment.yml
   conda activate base
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Access the web interface at `http://localhost:5000`

## Supported Bakery Items

The system can currently recognize the following items:
- Bánh chuối nướng
- Bánh cua bơ
- Bánh da lợn
- Bánh mì dưa lưới
- Chà bông cây
- Cookies dừa
- Croissant
- Egg tart
- Muffin việt quất
- Patechaud

## Usage

1. Start the application and ensure your camera is connected
2. Place bakery items on the tray
3. Position the tray in front of the camera
4. The system will automatically detect and identify items
5. Review the items and prices
6. Complete the transaction

## Training

To train the model with new data:
1. Add images to the respective folders in `data/train/`
2. Update `labels.txt` if adding new categories
3. Run the training script:
   ```bash
   python scripts/train.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
