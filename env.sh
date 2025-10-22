#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME="cs224n"

echo "Setting up environment for Blackwell GPU..."

# Complete clean slate
conda remove -n $ENV_NAME --all -y 2>/dev/null
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# Install PyTorch nightly (all together, no version specs)
echo "Installing PyTorch nightly..."
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install other packages
echo "Installing ML packages..."
pip install transformers accelerate peft
pip install pandas pyarrow numpy matplotlib seaborn scikit-learn tqdm datasets


# Set up environment variables
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo ""
echo "Testing..."
python << 'PYTHON_EOF'
import torch
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("="*60)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute:", torch.cuda.get_device_capability(0))
    
    try:
        x = torch.randn(100, 100).cuda()
        y = x + x
        z = torch.tensor([1, 2, 3]).cuda()
        mask = z.ne(1)
        
        print("\n✓ CUDA operations work!")
        print("✓ Environment ready!")
        
    except Exception as e:
        print(f"\n✗ CUDA failed: {e}")
else:
    print("✗ CUDA not available")
    
print("="*60)
PYTHON_EOF

echo ""
echo "Setup complete!"
echo ""
echo "Run experiments:"
echo "  conda activate cs224n"
echo "  ./experiments.sh --only-fp32"
echo "  ./experiments.sh --only-fp16"
echo "  ./experiments.sh --only-lora"