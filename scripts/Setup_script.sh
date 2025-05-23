#echo 'Install git'
#sudo apt install git

# Step 1: Clone the GitHub repository
echo "Cloning GitHub repo"
git clone https://github.com/Mariodraghidicomodo/Custom-Time-LLM.git || exit

# Step 2: Navigate into the repo
#cd Custom-Time-LLM || exit

# Step 3: Set up Python virtual environment (optional)
echo "Set up Python virtual environment"
#python -m venv .venv
python3 -m venv new_venv
source new_venv/bin/activate #Linux
#source .venv/Scripts/activate #Windows

# Step 4: Install requirements
echo "Installing requirements"
#pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements2.txt || exit
#pip install deepspeed
#pip install deepspeed --prefer-binary

# Step 5: Install LLama 7B model (huggyllama)
echo "Downloading LLaMA 7B model (huggyllama)"
python - <<EOF
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
print("Loading LLaMA config and model")
model = LlamaModel.from_pretrained(
    "huggyllama/llama-7b",
    trust_remote_code=True,
    local_files_only=False,
    config=LlamaConfig.from_pretrained("huggyllama/llama-7b"),
    load_in_4bit=True
)
print("Download tokenizer files")
tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
print("LLaMA 7B model downloaded successfully.")
EOF

echo "Setup complete!"
echo "Now, if you want to run the main, execute Script_15_min_poi.sh"
