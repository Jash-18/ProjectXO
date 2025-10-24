# 🍁 Plant Disease Detection & Treatment Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/AceMaster018/plant-disease-detector)

AI-powered tool that detects plant diseases from leaf images and provides actionable treatment recommendations. Try the hosted demo with a clean Gradio interface.

## 🚀 Live Demo

**🌐 [Try the Interactive Demo](https://huggingface.co/spaces/AceMaster018/plant-disease-detector)**

Upload a plant leaf image and get instant AI-powered disease diagnosis with treatment recommendations!

---

## ✨ Features

- 🔍 **Disease Detection**: AI analysis of plant leaf images
- 📋 **Treatment Recommendations**: Practical steps for disease management  
- 🌱 **Prevention Tips**: Guidance for maintaining healthy plants
- 💻 **Multiple Interfaces**: Gradio (Space) and Flask (local) versions
- 🚀 **Easy Deployment**: Ready for Hugging Face Spaces, Vercel, or local hosting

---

## 🌿 Supported Plant Types

**⚠️ Important**: This model is trained and intended to work **only** for the following plant types:

| Plant Category | Supported Varieties |
|----------------|-------------------|
| **Tree Fruits** | Apple, Cherry, Orange, Peach |
| **Berries** | Blueberry, Raspberry, Strawberry |
| **Vegetables** | Corn, Pepper (Bell), Potato, Soybean, Squash, Tomato |
| **Other** | Grape |
| **Special Classes** | Background detection, Healthy variants |

**Total Classes**: 38 PlantVillage classes (including disease variants and healthy classifications)

> 📝 **Note**: Using images outside these categories may produce unreliable predictions. For best results, upload clear, well-lit images of individual plant leaves.

---

## 🧠 Model Architecture & Training Details

The core of this project is a **custom Convolutional Neural Network (CNN)** designed and optimized specifically for plant disease classification using the PlantVillage dataset.

### **Architecture Highlights:**
- **Custom CNN Design**: Multiple convolutional layers with ReLU activation, batch normalization, and pooling blocks
- **Regularization**: Dropout layers to prevent overfitting and improve generalization  
- **Classification Head**: Fully connected layers for final 38-class classification
- **Lightweight**: Optimized for fast inference (~0.5 seconds on CPU) with minimal memory footprint
- **Input Processing**: 224×224 RGB images with normalization

### **Training Methodology:**
- **Dataset**: [PlantVillage Kaggle Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) - 54,000+ labeled images
- **Data Augmentation**: Rotation, flipping, brightness/contrast adjustments for better generalization
- **Optimization**: Adam optimizer with cross-entropy loss function
- **Validation Strategy**: Train/validation split with early stopping to prevent overfitting
- **Performance**: Achieves >97% accuracy on test data
- **Batch Processing**: Optimized batch sizes for stable training convergence

### **Model Specifications:**
- **File**: `model/ProjectXOAdv.pt` (PyTorch state dictionary)
- **Framework**: PyTorch (CPU-optimized for deployment)
- **Input**: 224×224×3 RGB leaf images
- **Output**: 38-class probability distribution
- **Size**: Compact model suitable for web deployment

### **Why Custom CNN?**
1. **Deployment Efficiency**: Much smaller than pre-trained models (VGG16/ResNet)
2. **Speed**: Fast inference suitable for real-time web applications
3. **Specialized**: Architecture tailored for plant leaf features and disease patterns
4. **Performance**: Outperforms transfer learning approaches on this specific dataset
5. **Flexibility**: Easy to modify and extend for additional plant types

---

## 🏃‍♂️ Quick Start

### Option 1: Use Live Demo (Recommended)
Simply visit: **[https://huggingface.co/spaces/AceMaster018/plant-disease-detector](https://huggingface.co/spaces/AceMaster018/plant-disease-detector)**

### Option 2: Run Locally (Gradio Interface)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jash-18/ProjectXO.git
   cd ProjectXO
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model file**:
   ```
   model/ProjectXOAdv.pt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser**: `http://localhost:7860`

### Option 3: Run Flask Version (Original Interface)

- Ensure `templates/` and `static/` directories are present
- Place CSV data files in `data/` directory (if using CSV-backed UI)
- Place model at `model/ProjectXOAdv.pt`
- Run the Flask entry point as configured

---

## 📁 Project Structure

```
ProjectXO/
├── 📄 README.md              # This file
├── 🐍 app.py                 # Gradio interface (Spaces-ready)
├── 🧠 CNN.py                 # Model architecture definition
├── 📋 requirements.txt       # Python dependencies  
├── ⚙️ runtime.txt            # Python version (for Spaces)
├── 📊 data/                  # Dataset and info files
│   ├── disease_info.csv
│   └── supplement_info.csv
├── 🤖 model/                 # Model weights
│   └── ProjectXOAdv.pt
├── 🎨 templates/             # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   ├── home.html
│   ├── contact-us.html
│   └── submit.html
├── 📁 static/                # Static assets (CSS, JS, images)
└── 🔧 config.py              # Configuration utilities
```

---

## 🛠️ Model Storage (Git LFS)

If your model file exceeds 100MB, use Git LFS for storage:

```bash
# Enable Git LFS
git lfs install

# Track the model file
git lfs track "model/ProjectXOAdv.pt"

# Commit LFS configuration
git add .gitattributes model/ProjectXOAdv.pt
git commit -m "Add model via LFS"
git push
```

---

## 🚀 Deployment Options

### Hugging Face Spaces (Recommended)
- **Automatic deployment** from repository
- **Free hosting** with CPU compute
- **Built-in Gradio interface**
- **Easy sharing** with public URL

### Vercel
- **Serverless deployment** 
- **Automatic scaling**
- **Custom domains** available
- **GitHub integration**

### Local Hosting
- **Full control** over environment
- **Custom modifications** supported
- **No external dependencies**

---

## 🔧 Troubleshooting

### Common Issues

**🚫 404 or Image Load Errors**
- Use local/static images instead of external URLs
- Check file paths and permissions

**📊 CSV Parse Errors** 
- Ensure CSVs have consistent comma-separated format
- Use built-in data variant in `app.py` as fallback

**🤖 Model Not Loading**
- Verify model path: `model/ProjectXOAdv.pt`
- Check file size (should be >10KB, not LFS pointer)
- Ensure proper PyTorch model format

**🐍 Import Errors**
- Install all requirements: `pip install -r requirements.txt`
- Check Python version compatibility (3.11 recommended)

---

## 📱 Usage Instructions

1. **📷 Upload Image**: Select a clear photo of a plant leaf
2. **🔍 Analyze**: Click the "Analyze Plant Disease" button  
3. **📋 Review Results**: Read the AI analysis and recommendations
4. **🌱 Take Action**: Follow the suggested treatment steps

### 📸 Photo Guidelines

- ✅ **Good**: Well-lit, focused leaf images
- ✅ **Good**: Single leaf filling most of the frame
- ✅ **Good**: Clear disease symptoms visible
- ❌ **Avoid**: Blurry or dark photos
- ❌ **Avoid**: Multiple plants in one image
- ❌ **Avoid**: Extreme close-ups or distant shots

---

## 🔬 Model Training & Development

### **Retraining the Model:**
To retrain or extend the model:

1. **Modify Architecture**: Edit `CNN.py` to adjust layers, filters, or add new components
2. **Data Preparation**: Prepare your dataset in PlantVillage format (224×224 images)
3. **Training Script**: Use PyTorch training loop with data augmentation
4. **Validation**: Implement proper train/validation split and early stopping
5. **Export**: Save model state dict as `ProjectXOAdv.pt`

### **Performance Metrics:**
- **Training Accuracy**: >98%
- **Validation Accuracy**: >97%
- **Inference Time**: ~0.5 seconds per image (CPU)
- **Model Size**: Compact for web deployment
- **Memory Usage**: Optimized for resource-constrained environments

---

## ⚠️ Disclaimer

This tool provides AI-generated suggestions for **educational purposes only**. 

**Always consult with:**
- 🌾 Agricultural professionals
- 🔬 Plant pathologists  
- 🏪 Local extension services
- 📚 Certified plant care experts

For serious plant health issues or commercial agriculture decisions.

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

- 🐛 Report bugs and issues
- 💡 Suggest new features  
- 📖 Improve documentation
- 🧪 Add test cases
- 🌱 Extend plant type support
- 🔧 Improve model architecture

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Gradio Team** for the easy-to-use interface library
- **Hugging Face** for free model hosting and Spaces platform
- **PlantVillage Dataset** creators for the comprehensive plant disease dataset
- **Plant Pathology Community** for disease classification research

---

## 📞 Support

- 🌐 **Live Demo**: [https://huggingface.co/spaces/AceMaster018/plant-disease-detector](https://huggingface.co/spaces/AceMaster018/plant-disease-detector)
- 📧 **Issues**: [GitHub Issues](https://github.com/Jash-18/ProjectXO/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Jash-18/ProjectXO/discussions)

---

<div align="center">

**🌱 Happy Gardening! 🌱**

*Made with ❤️ for plant lovers and gardeners worldwide*

[![Hugging Face](https://img.shields.io/badge/🤗%20Try%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/AceMaster018/plant-disease-detector)

</div>