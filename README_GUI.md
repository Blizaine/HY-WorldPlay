# HY-WorldPlay 1.5 Gradio Interface

A user-friendly web interface for the HY-WorldPlay 1.5 real-time interactive world model.

## Features

- **🎨 Intuitive Web Interface**: Clean, tabbed interface for easy navigation
- **🎥 Multiple Model Support**: Choose between Bidirectional, Autoregressive, and Distilled models
- **📹 Camera Control Presets**: Built-in camera movements (Static, Forward, Orbit, Pan, Zoom)
- **⚙️ Full Configuration**: Easily set model paths and parameters
- **📊 Real-time Progress**: Track generation progress with live updates
- **💾 Persistent Settings**: Configuration saved between sessions
- **🎯 Example Library**: Pre-configured examples to get started quickly

## Quick Start

### Windows
```bash
# Simply double-click or run:
launch_gui.bat
```

### Linux/Mac
```bash
chmod +x launch_gui.sh
./launch_gui.sh
```

### Manual Launch
```bash
# Activate your environment
conda activate worldplay

# Install Gradio if needed
pip install gradio>=4.0.0

# Launch the interface
python app_gradio.py
```

## First-Time Setup

1. **Download Models**:
   ```bash
   # Download base HunyuanVideo-1.5 model (required)
   huggingface-cli download tencent/HunyuanVideo-1.5

   # Download HY-WorldPlay models
   huggingface-cli download tencent/HY-WorldPlay
   ```

2. **Configure Model Paths**:
   - Launch the GUI
   - Go to the "Configuration" tab
   - Set paths to your downloaded models:
     - Base HunyuanVideo-1.5 model path
     - At least one action model (AR, Bidirectional, or Distilled)
   - Click "Save Configuration"

3. **Generate Your First Video**:
   - Go to "Generate" tab
   - Enter a text prompt or upload an image
   - Select model type and settings
   - Choose a camera preset
   - Click "Generate Video"

## Interface Tabs

### 🎬 Generate Tab
Main generation interface with:
- Text prompt and optional image input
- Model selection (quality vs speed tradeoff)
- Resolution and aspect ratio settings
- Frame count control (17-125 frames)
- Camera movement presets
- Advanced options (super resolution, prompt rewriting)

### ⚙️ Configuration Tab
Set up model paths and system settings:
- Base model path configuration
- Action model paths (AR, Bidirectional, Distilled)
- Output directory settings
- Multi-GPU configuration
- Persistent configuration saving

### 💡 Examples Tab
Pre-configured examples with:
- Sample prompts for different scenarios
- Recommended camera movements
- Best practices guide
- Tips for optimal results

### ℹ️ About Tab
Information about the project:
- Key features overview
- Links to documentation
- Citation information
- Community resources

## Camera Movement Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| **Static** | No camera movement | Observing detailed scenes |
| **Forward** | Move forward smoothly | Walking/flying through scenes |
| **Orbit Left/Right** | Circle around subject | Showcasing objects from all angles |
| **Pan Left/Right** | Horizontal rotation | Revealing panoramic views |
| **Zoom In/Out** | Change focal distance | Focus on details or reveal context |
| **Custom** | Upload JSON trajectory | Complex camera paths |

## Model Selection Guide

### Bidirectional Model
- **Quality**: Highest
- **Speed**: Slowest
- **Use Case**: Offline generation, final renders
- **Memory**: ~20GB+

### Autoregressive Model
- **Quality**: Good
- **Speed**: Medium
- **Use Case**: Interactive exploration
- **Memory**: ~16GB

### Distilled Model
- **Quality**: Good
- **Speed**: Fast (4 steps)
- **Use Case**: Real-time demos, quick iterations
- **Memory**: ~14GB

## Tips for Best Results

1. **Detailed Prompts**: Be specific about lighting, style, and composition
2. **Start Small**: Begin with 65 frames to test prompts quickly
3. **Seed Control**: Use the same seed for consistent results
4. **GPU Memory**: Enable model offloading if you have <24GB VRAM
5. **Camera Movement**: Start with static camera, then experiment with movement

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
- Reduce frame count
- Use distilled model
- Enable model offloading in configuration
- Close other GPU applications

### Models Not Found
- Verify model paths in Configuration tab
- Ensure models are fully downloaded
- Check file permissions

### Generation Fails
- Check console output for detailed error messages
- Verify all dependencies installed: `pip install -r requirements.txt`
- Ensure sufficient disk space for outputs

## Advanced Usage

### Custom Camera Trajectories
Create a JSON file with camera positions and actions:
```json
{
  "trajectory": [
    {
      "frame": 0,
      "position": [0, 0, 0],
      "rotation": [0, 0, 0],
      "action": [0, 0, 0, 0]
    },
    ...
  ]
}
```

### Batch Processing
Use the command-line interface for batch processing:
```python
import app_gradio
interface = app_gradio.WorldPlayInterface()
# Configure and run multiple generations
```

## System Requirements

- **GPU**: NVIDIA with CUDA support
- **Memory**: 14GB+ VRAM (minimum)
- **Storage**: 50GB+ for models and outputs
- **Python**: 3.10+
- **OS**: Windows/Linux/Mac (with CUDA)

## Support

- **Discord**: [Join Community](https://discord.gg/dNBrdrGGMa)
- **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Documentation**: [Technical Report](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf)