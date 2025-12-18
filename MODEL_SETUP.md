# Model Setup Guide for HY-WorldPlay

## Model Download Locations

When you use `huggingface-cli download`, models are downloaded to your HuggingFace cache directory:

**Windows:** `C:\Users\bliza\.cache\huggingface\hub\`
**Linux/Mac:** `~/.cache/huggingface/hub/`

## Your Downloaded Models

I found your HY-WorldPlay models at:
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\
```

This directory contains:
- ✅ `ar_distilled_action_model/` - Distilled autoregressive model (fastest)
- ✅ `ar_model/` - Autoregressive model (balanced)
- ✅ `bidirectional_model/` - Bidirectional model (highest quality)

## ⚠️ Missing: HunyuanVideo-1.5 Base Model

You still need to download the base HunyuanVideo-1.5 model. Run:
```bash
huggingface-cli download tencent/HunyuanVideo-1.5
```

This will download to:
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HunyuanVideo-1.5\
```

## Setting Up Model Paths in GUI

When you launch the Gradio interface, go to the **Configuration** tab and set:

### 1. Base Model Path
After downloading HunyuanVideo-1.5, the path will be:
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HunyuanVideo-1.5\snapshots\[hash]\transformer\480p_i2v\
```

### 2. Action Model Paths

**Autoregressive Model:**
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\ar_model
```

**Bidirectional Model:**
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\bidirectional_model
```

**Distilled Model:**
```
C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\ar_distilled_action_model
```

## Alternative: Create Symlinks (Optional)

To make paths simpler, you can create symbolic links in your project directory:

### Windows (Run as Administrator):
```cmd
cd C:\Users\bliza\conda\HY-WorldPlay-Blizaine\HY-WorldPlay
mkdir models

mklink /D models\ar_model "C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\ar_model"

mklink /D models\bidirectional_model "C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\bidirectional_model"

mklink /D models\ar_distilled_action_model "C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\d093ba09183b304c476e3bac3d1e906f1f37af12\ar_distilled_action_model"
```

After creating symlinks, you can use simpler paths in the GUI:
- AR Model: `./models/ar_model`
- Bidirectional: `./models/bidirectional_model`
- Distilled: `./models/ar_distilled_action_model`

## Quick Verification

To verify your models are set up correctly:

1. Launch the GUI: `launch_gui.bat`
2. Go to Configuration tab
3. Enter the model paths
4. Click "Save Configuration"
5. Try generating a video with a simple prompt

## Model Sizes

Approximate sizes for planning storage:
- HunyuanVideo-1.5 base: ~20-30 GB
- Each WorldPlay model: ~5-10 GB
- Total needed: ~50-60 GB

## Troubleshooting

### "Model not found" error
- Check that the snapshot hash in your path matches what's in your cache directory
- The hash (like `d093ba09183b304c476e3bac3d1e906f1f37af12`) may be different for you

### Finding the correct snapshot
```bash
# List all snapshots for a model
ls C:\Users\bliza\.cache\huggingface\hub\models--tencent--HY-WorldPlay\snapshots\
```

### GPU Memory Issues
- Start with the distilled model (uses least memory)
- Reduce frame count to 17-33
- Close other GPU applications