#!/bin/bash
# Remote GPU training script for RANK Enhanced V3

set -e

echo "============================================================"
echo "RANK Enhanced V3 Training on Remote GPU"
echo "============================================================"
echo "Server: jx@172.29.2.51"
echo "GPU: RTX 4090"
echo "Start time: $(date)"
echo ""

# Step 1: Extract files
echo "1. Extracting training files..."
cd ~
tar -xzf rank_v3_training.tar.gz
echo "   ✓ Files extracted"

# Step 2: Check Python environment
echo ""
echo "2. Checking Python environment..."
python3 --version
pip3 list | grep -E "(torch|lightgbm|transformers|scipy)" || echo "   ⚠️  Some packages may need installation"

# Step 3: Install missing dependencies if needed
echo ""
echo "3. Ensuring dependencies..."
pip3 install scipy --quiet || echo "   scipy already installed"

# Step 4: Generate synthetic data if not present
echo ""
echo "4. Checking synthetic retrieval data..."
if [ ! -f "data/synthetic_retrieval/retrieval_history.jsonl" ]; then
    echo "   Generating synthetic data..."
    python3 scripts/generate_synthetic_retrieval_history.py 10000 1.5
else
    echo "   ✓ Synthetic data exists"
fi

# Step 5: Start training
echo ""
echo "5. Starting RANK V3 training..."
echo "   This will take 30-60 minutes (E5 encoding + LightGBM training)"
echo "   Log: ~/logs/rank_v3_training.log"
echo ""

mkdir -p ~/logs
nohup python3 scripts/train_rank_enhanced_v3.py > ~/logs/rank_v3_training.log 2>&1 &
TRAIN_PID=$!

echo "   ✓ Training started (PID: $TRAIN_PID)"
echo ""
echo "============================================================"
echo "Monitoring Commands"
echo "============================================================"
echo "# Check training progress:"
echo "  ssh jx@172.29.2.51 'tail -f ~/logs/rank_v3_training.log'"
echo ""
echo "# Check process status:"
echo "  ssh jx@172.29.2.51 'ps aux | grep train_rank_enhanced_v3'"
echo ""
echo "# Kill training if needed:"
echo "  ssh jx@172.29.2.51 'kill $TRAIN_PID'"
echo "============================================================"
