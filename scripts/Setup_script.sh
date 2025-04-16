# Step 1: Clone the GitHub repository
echo "Cloning GitHub repo..."
git clone https://github.com/Mariodraghidicomodo/Custom-Time-LLM.git || exit

# Step 2: Navigate into the repo
#cd Custom-Time-LLM || exit
#Kaggle repo
cd kaggle/working/Custom/Custom-Time-LLM || exit

# Step 3: Set up Python virtual environment (optional)
# python3 -m venv .venv
# source .venv/bin/activate

# Step 4: Install requirements
echo "Installing requirements"
#pip install --upgrade pip
pip install -r requirements2.txt

# Step 5: Install LLama 7B model (huggyllama)
echo "⬇️ Downloading LLaMA 7B model (huggyllama)..."
python3 - <<EOF
from transformers import LlamaModel, LlamaConfig
print("Loading LLaMA config and model")
model = LlamaModel.from_pretrained(
    "huggyllama/llama-7b",
    trust_remote_code=True,
    local_files_only=False,
    config=LlamaConfig.from_pretrained("huggyllama/llama-7b"),
    load_in_4bit=True
)
print("LLaMA 7B model downloaded successfully.")
EOF

echo "Setup complete!"
echo "Now if you want to run the main execute Script_15_min_poi.sh"
