"""
HY-WorldPlay 1.5 Gradio Interface
Interactive GUI for real-time world model generation
"""

import gradio as gr
import torch
import os
import json
import subprocess
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from typing import Optional, Tuple, List, Dict, Any
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_CONFIG = {
    "model_path": "",
    "ar_action_model_path": "",
    "bi_action_model_path": "",
    "ar_distill_action_model_path": "",
    "output_base_path": "./outputs/",
    "n_inference_gpu": 1,
    "default_seed": 42,
}

# Camera movement presets
CAMERA_PRESETS = {
    "Static": {"description": "No camera movement", "file": "static.json"},
    "Forward": {"description": "Move forward smoothly", "file": "forward.json"},
    "Orbit Left": {"description": "Orbit around subject to the left", "file": "orbit_left.json"},
    "Orbit Right": {"description": "Orbit around subject to the right", "file": "orbit_right.json"},
    "Zoom In": {"description": "Zoom in slowly", "file": "zoom_in.json"},
    "Zoom Out": {"description": "Zoom out slowly", "file": "zoom_out.json"},
    "Pan Left": {"description": "Pan camera to the left", "file": "pan_left.json"},
    "Pan Right": {"description": "Pan camera to the right", "file": "pan_right.json"},
    "Custom": {"description": "Upload your own trajectory JSON", "file": None},
}

class WorldPlayInterface:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
        self.current_process = None

    def load_config(self):
        """Load configuration from file if exists"""
        config_path = Path("gui_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                self.config.update(saved_config)

    def save_config(self):
        """Save current configuration"""
        with open("gui_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

    def validate_paths(self) -> Tuple[bool, str]:
        """Validate required model paths"""
        errors = []

        if not self.config["model_path"]:
            errors.append("Base HunyuanVideo-1.5 model path not set")
        elif not Path(self.config["model_path"]).exists():
            errors.append(f"Base model path does not exist: {self.config['model_path']}")

        # Check if at least one action model is configured
        has_action_model = any([
            self.config.get("ar_action_model_path"),
            self.config.get("bi_action_model_path"),
            self.config.get("ar_distill_action_model_path")
        ])

        if not has_action_model:
            errors.append("At least one action model path must be configured")

        if errors:
            return False, "\n".join(errors)
        return True, "All paths validated"

    def generate_trajectory_json(self, preset: str, custom_file: Optional[str],
                                num_frames: int) -> str:
        """Generate or load camera trajectory JSON"""

        if preset == "Custom" and custom_file:
            return custom_file

        # Generate trajectory based on preset
        trajectory_path = Path("temp_trajectory.json")

        if preset == "Static":
            # Generate static camera trajectory
            trajectory = self.create_static_trajectory(num_frames)
        elif preset == "Forward":
            trajectory = self.create_forward_trajectory(num_frames)
        elif preset == "Orbit Left":
            trajectory = self.create_orbit_trajectory(num_frames, direction="left")
        elif preset == "Orbit Right":
            trajectory = self.create_orbit_trajectory(num_frames, direction="right")
        elif preset == "Zoom In":
            trajectory = self.create_zoom_trajectory(num_frames, zoom_in=True)
        elif preset == "Zoom Out":
            trajectory = self.create_zoom_trajectory(num_frames, zoom_in=False)
        elif preset == "Pan Left":
            trajectory = self.create_pan_trajectory(num_frames, direction="left")
        elif preset == "Pan Right":
            trajectory = self.create_pan_trajectory(num_frames, direction="right")
        else:
            # Default to static
            trajectory = self.create_static_trajectory(num_frames)

        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)

        return str(trajectory_path)

    def create_static_trajectory(self, num_frames: int) -> Dict:
        """Create static camera trajectory"""
        frames = []
        for i in range(num_frames):
            frames.append({
                "frame": i,
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "action": [0, 0, 0, 0]  # No action
            })
        return {"trajectory": frames}

    def create_forward_trajectory(self, num_frames: int) -> Dict:
        """Create forward movement trajectory"""
        frames = []
        for i in range(num_frames):
            frames.append({
                "frame": i,
                "position": [0, 0, i * 0.1],  # Move forward
                "rotation": [0, 0, 0],
                "action": [1, 0, 0, 0]  # Forward action
            })
        return {"trajectory": frames}

    def create_orbit_trajectory(self, num_frames: int, direction: str = "left") -> Dict:
        """Create orbital camera trajectory"""
        frames = []
        angle_step = (2 * np.pi / num_frames) * (1 if direction == "left" else -1)
        radius = 5.0

        for i in range(num_frames):
            angle = i * angle_step
            frames.append({
                "frame": i,
                "position": [radius * np.cos(angle), 0, radius * np.sin(angle)],
                "rotation": [0, angle, 0],  # Rotate to face center
                "action": [0, 0, 1 if direction == "left" else 0, 0 if direction == "left" else 1]
            })
        return {"trajectory": frames}

    def create_zoom_trajectory(self, num_frames: int, zoom_in: bool = True) -> Dict:
        """Create zoom trajectory"""
        frames = []
        zoom_factor = 0.05 if zoom_in else -0.05

        for i in range(num_frames):
            z_pos = -5 + (i * zoom_factor)  # Start at -5, move based on zoom
            frames.append({
                "frame": i,
                "position": [0, 0, z_pos],
                "rotation": [0, 0, 0],
                "action": [1, 0, 0, 0] if zoom_in else [0, 1, 0, 0]
            })
        return {"trajectory": frames}

    def create_pan_trajectory(self, num_frames: int, direction: str = "left") -> Dict:
        """Create panning trajectory"""
        frames = []
        pan_speed = 0.02 * (1 if direction == "left" else -1)

        for i in range(num_frames):
            frames.append({
                "frame": i,
                "position": [0, 0, 0],
                "rotation": [0, i * pan_speed, 0],
                "action": [0, 0, 1 if direction == "left" else 0, 0 if direction == "left" else 1]
            })
        return {"trajectory": frames}

    def run_generation(
        self,
        prompt: str,
        image_path: Optional[str],
        model_type: str,
        resolution: str,
        aspect_ratio: str,
        num_frames: int,
        seed: int,
        camera_preset: str,
        custom_trajectory: Optional[str],
        enable_sr: bool,
        rewrite_prompt: bool,
        num_inference_steps: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """Run the video generation process"""

        # Validate inputs
        valid, error_msg = self.validate_paths()
        if not valid:
            return None, error_msg, "Error: " + error_msg

        if not prompt and not image_path:
            return None, "Please provide either a text prompt or an image", "Error: No input provided"

        # Determine which model checkpoint to use
        if model_type == "bidirectional":
            action_ckpt = self.config["bi_action_model_path"]
            few_step = "false"
        elif model_type == "autoregressive":
            action_ckpt = self.config["ar_action_model_path"]
            few_step = "false"
        elif model_type == "autoregressive_distilled":
            action_ckpt = self.config["ar_distill_action_model_path"]
            few_step = "true"
        else:
            return None, f"Unknown model type: {model_type}", "Error"

        if not action_ckpt or not Path(action_ckpt).exists():
            return None, f"Model checkpoint not found for {model_type}", "Error: Model not found"

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config["output_base_path"]) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate or get trajectory JSON
        trajectory_path = self.generate_trajectory_json(camera_preset, custom_trajectory, num_frames)

        # Build command
        cmd = [
            "python", "generate.py",
            "--prompt", prompt,
            "--resolution", resolution,
            "--aspect_ratio", aspect_ratio,
            "--video_length", str(num_frames),
            "--seed", str(seed),
            "--rewrite", str(rewrite_prompt).lower(),
            "--sr", str(enable_sr).lower(),
            "--save_pre_sr_video",
            "--pose_json_path", trajectory_path,
            "--output_path", str(output_dir),
            "--model_path", self.config["model_path"],
            "--action_ckpt", action_ckpt,
            "--few_step", few_step,
            "--model_type", "bi" if model_type == "bidirectional" else "ar"
        ]

        if image_path:
            cmd.extend(["--image_path", image_path])

        if model_type == "autoregressive_distilled":
            cmd.extend(["--num_inference_steps", str(num_inference_steps)])

        # Update progress
        progress(0.1, desc="Starting generation...")

        try:
            # Run the generation command
            logger.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            output_log = []
            for line in process.stdout:
                output_log.append(line)
                logger.info(line.strip())

                # Update progress based on output
                if "Generating" in line:
                    progress(0.5, desc="Generating video...")
                elif "Super resolution" in line:
                    progress(0.8, desc="Applying super resolution...")

            process.wait()

            if process.returncode != 0:
                error_msg = "".join(output_log[-20:])  # Last 20 lines
                return None, f"Generation failed:\n{error_msg}", "Error"

            progress(0.9, desc="Finalizing output...")

            # Find the output video
            video_files = list(output_dir.glob("*.mp4"))
            if not video_files:
                return None, "No video file generated", "Error: No output"

            video_path = str(video_files[0])
            log_output = "".join(output_log)

            progress(1.0, desc="Complete!")

            return video_path, f"Video saved to: {video_path}", log_output

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return None, f"Generation failed: {str(e)}", str(e)

    def update_config(
        self,
        model_path: str,
        ar_path: str,
        bi_path: str,
        distill_path: str,
        output_path: str,
        gpu_count: int
    ) -> str:
        """Update configuration settings"""

        self.config.update({
            "model_path": model_path,
            "ar_action_model_path": ar_path,
            "bi_action_model_path": bi_path,
            "ar_distill_action_model_path": distill_path,
            "output_base_path": output_path,
            "n_inference_gpu": gpu_count
        })

        self.save_config()

        return "Configuration saved successfully!"

def create_interface():
    """Create the Gradio interface"""

    interface = WorldPlayInterface()

    with gr.Blocks(title="HY-WorldPlay 1.5", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎮 HY-WorldPlay 1.5 Interface
        ### Real-time Interactive World Model Generation

        Generate immersive, geometrically consistent world videos with camera control.
        """)

        with gr.Tabs():
            # Generation Tab
            with gr.TabItem("🎬 Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input Section
                        gr.Markdown("### 📝 Input")
                        prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Describe the world you want to create...",
                            lines=3
                        )

                        image_input = gr.Image(
                            label="Reference Image (Optional)",
                            type="filepath"
                        )

                        # Model Settings
                        gr.Markdown("### ⚙️ Model Settings")
                        model_type = gr.Radio(
                            label="Model Type",
                            choices=["bidirectional", "autoregressive", "autoregressive_distilled"],
                            value="autoregressive",
                            info="Bidirectional: High quality | AR: Balanced | Distilled: Fast"
                        )

                        with gr.Row():
                            resolution = gr.Dropdown(
                                label="Resolution",
                                choices=["480p", "720p"],
                                value="480p"
                            )

                            aspect_ratio = gr.Dropdown(
                                label="Aspect Ratio",
                                choices=["16:9", "9:16", "1:1", "4:3"],
                                value="16:9"
                            )

                        with gr.Row():
                            num_frames = gr.Slider(
                                label="Number of Frames",
                                minimum=17,
                                maximum=125,
                                value=65,
                                step=16,
                                info="Multiples of 16 work best"
                            )

                            seed = gr.Number(
                                label="Seed",
                                value=42,
                                precision=0
                            )

                        # Camera Control
                        gr.Markdown("### 📹 Camera Control")
                        camera_preset = gr.Dropdown(
                            label="Camera Movement",
                            choices=list(CAMERA_PRESETS.keys()),
                            value="Static",
                            info="Select preset camera movement"
                        )

                        custom_trajectory = gr.File(
                            label="Custom Trajectory JSON (when 'Custom' selected)",
                            file_types=[".json"],
                            visible=False
                        )

                        # Advanced Options
                        with gr.Accordion("Advanced Options", open=False):
                            enable_sr = gr.Checkbox(
                                label="Enable Super Resolution",
                                value=False,
                                info="Upscale output (requires more GPU memory)"
                            )

                            rewrite_prompt = gr.Checkbox(
                                label="Rewrite Prompt",
                                value=False,
                                info="Use AI to enhance prompt (requires vLLM server)"
                            )

                            num_inference_steps = gr.Slider(
                                label="Inference Steps (Distilled model only)",
                                minimum=1,
                                maximum=50,
                                value=4,
                                step=1
                            )

                        generate_btn = gr.Button(
                            "🚀 Generate Video",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        # Output Section
                        gr.Markdown("### 🎥 Output")
                        video_output = gr.Video(
                            label="Generated Video",
                            autoplay=True
                        )

                        status_output = gr.Textbox(
                            label="Status",
                            lines=2,
                            interactive=False
                        )

                        with gr.Accordion("Generation Log", open=False):
                            log_output = gr.Textbox(
                                label="Detailed Log",
                                lines=10,
                                interactive=False,
                                elem_classes=["monospace"]
                            )

                # Event handlers
                def toggle_custom_trajectory(preset):
                    return gr.update(visible=preset == "Custom")

                camera_preset.change(
                    toggle_custom_trajectory,
                    inputs=[camera_preset],
                    outputs=[custom_trajectory]
                )

                generate_btn.click(
                    interface.run_generation,
                    inputs=[
                        prompt, image_input, model_type, resolution,
                        aspect_ratio, num_frames, seed, camera_preset,
                        custom_trajectory, enable_sr, rewrite_prompt,
                        num_inference_steps
                    ],
                    outputs=[video_output, status_output, log_output]
                )

            # Configuration Tab
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown("""
                ### 📁 Model Paths Configuration
                Configure the paths to your downloaded models.
                """)

                with gr.Column():
                    base_model_path = gr.Textbox(
                        label="HunyuanVideo-1.5 Base Model Path",
                        value=interface.config.get("model_path", ""),
                        placeholder="/path/to/hunyuanvideo-1.5/",
                        info="Path to the base HunyuanVideo-1.5 480p-i2v model"
                    )

                    gr.Markdown("### Action Model Checkpoints")

                    ar_model_path = gr.Textbox(
                        label="Autoregressive Model Path",
                        value=interface.config.get("ar_action_model_path", ""),
                        placeholder="/path/to/ar_model/",
                        info="Path to HY-World1.5-Autoregressive-480P-I2V"
                    )

                    bi_model_path = gr.Textbox(
                        label="Bidirectional Model Path",
                        value=interface.config.get("bi_action_model_path", ""),
                        placeholder="/path/to/bidirectional_model/",
                        info="Path to HY-World1.5-Bidirectional-480P-I2V"
                    )

                    distill_model_path = gr.Textbox(
                        label="Distilled Model Path",
                        value=interface.config.get("ar_distill_action_model_path", ""),
                        placeholder="/path/to/ar_distilled_action_model/",
                        info="Path to HY-World1.5-Autoregressive-480P-I2V-distill"
                    )

                    gr.Markdown("### Output Settings")

                    output_base_path = gr.Textbox(
                        label="Output Directory",
                        value=interface.config.get("output_base_path", "./outputs/"),
                        info="Base directory for saving generated videos"
                    )

                    gpu_count = gr.Number(
                        label="Number of GPUs for Parallel Inference",
                        value=interface.config.get("n_inference_gpu", 1),
                        precision=0,
                        minimum=1,
                        maximum=8,
                        info="Use multiple GPUs for faster inference (max recommended: 4)"
                    )

                    save_config_btn = gr.Button(
                        "💾 Save Configuration",
                        variant="primary"
                    )

                    config_status = gr.Textbox(
                        label="Configuration Status",
                        interactive=False
                    )

                    save_config_btn.click(
                        interface.update_config,
                        inputs=[
                            base_model_path, ar_model_path, bi_model_path,
                            distill_model_path, output_base_path, gpu_count
                        ],
                        outputs=[config_status]
                    )

                    gr.Markdown("""
                    ### 📋 Requirements
                    - **GPU**: NVIDIA GPU with CUDA support
                    - **Minimum GPU Memory**: 14 GB (with model offloading)
                    - **Models**: Download from [HuggingFace](https://huggingface.co/tencent/HY-WorldPlay)

                    ### 🔗 Quick Download Commands
                    ```bash
                    # Download base HunyuanVideo-1.5 model
                    huggingface-cli download tencent/HunyuanVideo-1.5

                    # Download HY-WorldPlay models
                    huggingface-cli download tencent/HY-WorldPlay
                    ```
                    """)

            # Examples Tab
            with gr.TabItem("💡 Examples"):
                gr.Markdown("""
                ### Example Prompts

                Try these prompts to get started:
                """)

                gr.Examples(
                    examples=[
                        [
                            "A serene Japanese garden with a stone bridge over a koi pond, cherry blossoms in bloom",
                            "Static", "480p", "16:9", 65
                        ],
                        [
                            "First-person view walking through a futuristic neon-lit cyberpunk city at night",
                            "Forward", "480p", "16:9", 65
                        ],
                        [
                            "Ancient ruins in a mystical forest, sunlight streaming through the canopy",
                            "Orbit Left", "480p", "16:9", 65
                        ],
                        [
                            "Underwater coral reef teeming with colorful fish and marine life",
                            "Pan Right", "480p", "16:9", 65
                        ],
                        [
                            "Snow-covered mountain peak with clouds rolling through valleys below",
                            "Zoom Out", "480p", "16:9", 65
                        ],
                    ],
                    inputs=[prompt, camera_preset, resolution, aspect_ratio, num_frames],
                    label="Click to load example"
                )

                gr.Markdown("""
                ### 🎮 Camera Movement Tips

                - **Static**: Best for observing detailed scenes without distraction
                - **Forward/Backward**: Simulates walking or flying through the scene
                - **Orbit**: Great for showcasing objects or environments from all angles
                - **Pan**: Reveals panoramic views gradually
                - **Zoom**: Focus on details or reveal wider context
                - **Custom**: Upload your own JSON for complex camera paths

                ### 🎯 Best Practices

                1. **Start Simple**: Begin with static camera and short videos (65 frames)
                2. **Detailed Prompts**: More descriptive prompts yield better results
                3. **Model Selection**:
                   - Use **Bidirectional** for highest quality (slower)
                   - Use **Autoregressive** for good balance
                   - Use **Distilled** for real-time demos (fastest)
                4. **GPU Memory**: Enable model offloading if you have <24GB VRAM
                5. **Seeds**: Use the same seed for consistent results
                """)

            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## HY-WorldPlay 1.5

                A systematic framework for interactive world modeling with real-time latency and geometric consistency.

                ### Key Features
                - 🎮 **Real-time Generation**: Up to 24 FPS streaming
                - 🎯 **Action Control**: Keyboard/mouse input simulation
                - 🔄 **Long-term Consistency**: Geometric coherence over extended sequences
                - 🚀 **Multiple Models**: Bidirectional, Autoregressive, and Distilled variants
                - 📹 **Camera Control**: Various preset movements and custom trajectories

                ### Resources
                - [📄 Technical Report](https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf)
                - [🤗 HuggingFace Models](https://huggingface.co/tencent/HY-WorldPlay)
                - [🌐 Project Page](https://3d-models.hunyuan.tencent.com/world/)
                - [💬 Discord Community](https://discord.gg/dNBrdrGGMa)

                ### Citation
                ```bibtex
                @article{hyworld2025,
                  title={HY-World 1.5: A Systematic Framework for Interactive World Modeling},
                  author={Team HunyuanWorld},
                  journal={arXiv preprint},
                  year={2025}
                }
                ```

                ---
                *Built with Gradio | Version 1.0*
                """)

    return app

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ Warning: CUDA not available. GPU acceleration will not work.")
    except ImportError:
        print("❌ PyTorch not installed. Please install requirements first.")
        sys.exit(1)

    # Create and launch the app
    app = create_interface()

    print("\n" + "="*50)
    print("🎮 HY-WorldPlay 1.5 Gradio Interface")
    print("="*50)
    print("\n📋 Before running, ensure you have:")
    print("1. Downloaded the HunyuanVideo-1.5 base model")
    print("2. Downloaded HY-WorldPlay model checkpoints")
    print("3. Configured model paths in the Configuration tab")
    print("\nStarting server...\n")

    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True for public URL
        inbrowser=True  # Auto-open in browser
    )