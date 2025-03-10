# FedTextGrad: Federated Textual Gradient

[![Conference](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://iclr.cc/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Arxiv](https://img.shields.io/badge/ArXiv-2502.19980-b31b1b.svg)](https://arxiv.org/abs/2502.19980)

📌 **Official repository for our ICLR 2025 paper:**
  
> **Can Textual Gradient Work in Federated Learning?**  
> *[Minghui Chen](https://chenminghui.com), [Ruinan Jin](https://nanboy-ronan.github.io/Personal-Web/), [Wenlong Deng](https://vengdeng.github.io/), Yuanyuan Chen, [Zhi Huang](https://www.zhihuang.ai/), [Han Yu](https://trustful.federated-learning.org/), [Xiaoxiao Li](https://tea.ece.ubc.ca/)*  
> 📄 [[Arxiv](https://arxiv.org/abs/2502.19980)]  

![FedTextGradFramework](/resources/FedTextGrad_Framework.png)

---


## 📌 Abstract

Recent studies highlight the promise of LLM-based prompt optimization, especially with TextGrad, which automates differentiation via texts and backpropagates textual feedback. This approach facilitates training in various real-world applications that do not support numerical gradient propagation or loss calculation. In this paper, we systematically explore the potential and challenges of incorporating textual gradient into Federated Learning (FL). Our contributions are fourfold. Firstly, we introduce a novel FL paradigm, Federated Textual Gradient (FedTextGrad), that allows clients to upload locally optimized prompts derived from textual gradients, while the server aggregates the received prompts. Unlike traditional FL frameworks, which are designed for numerical aggregation, FedTextGrad is specifically tailored for handling textual data, expanding the applicability of FL to a broader range of problems that lack well-defined numerical loss functions. Secondly, building on this design, we conduct extensive experiments to explore the feasibility of FedTextGrad. Our findings highlight the importance of properly tuning key factors (e.g., local steps) in FL training. Thirdly, we highlight a major challenge in FedTextGrad aggregation: retaining essential information from distributed prompt updates. Last but not least, in response to this issue, we improve the vanilla variant of FedTextGrad by providing actionable guidance to the LLM when summarizing client prompts by leveraging the Uniform Information Density principle. Through this principled study, we enable the adoption of textual gradients in FL for optimizing LLMs, identify important issues, and pinpoint future directions, thereby opening up a new research area that warrants further investigation.

## 📂 Project Structure

```
📁 repo/
│-- 📜 README.md                    # Project documentation
│-- 📜 .gitignore                   # Ignored files
│-- 📜 requirements.txt             # Required packages
│-- 📜 main.py                      # Main Python file
│-- 📜 train_centralized.py         # Centralized training
│-- 📜 train_homo_fed.py            # Homogeneous Federated Learning
│-- 📜 train_hetero_fed.py          # Heterogeneous Federated Learning
│-- 📂 scripts/                     # Scripts for running Python files
│   │-- 📜 run_centralized.sh       # Main script for centralized training
│   │-- 📜 run_homo_fed.sh          # Homogeneous FL script
│   │-- 📜 run_hetero_fed.sh        # Heterogeneous FL script
│   │-- 📜 vllm_serve.sh            # VLLM serve
│   │-- 📜 sglang_serve.sh          # SG serve
│-- 📂 textgrad/                    # TextGrad package
│-- 📂 utils/                       # Utility functions for training
│-- 📂 logs/                        # Results and logs
```

## 🚀 Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/ubc-tea/FedTextGrad
cd FedTextGrad

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Tips: Installation
- Use our local package textgrad (no local install needed).  


## 🏗️ Usage

### 1️⃣ Data Preparation

#### Downloading Datasets
- BBH Word Counting (automatic downloading)
- BBH Word Sorting (automatic downloading)
- GSM8k (requires `pip install datasets` from Hugging Face)

### 2️⃣ Setting Up LLM APIs

#### Using OpenAI API
- Specify `OPENAI_API_KEY`.

#### Using Other APIs
- Specify `BASE_URL` for third-party providers.

#### Using Local LLM API: Ollama
- Automatic installation of Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If download with ollama pre-built binaries from the [Ollama GitHub release](https://github.com/ollama/ollama/releases).
Start the Ollama Serving in the backend
```bash
tar -xzvf ollamaxxx.tgz &
chmod +x ./bin/ollama &
cd ./bin/ollama &
./ollama serve
```

##### 🔧 Tips

- **[Improve Ollama Speed](https://anakin.ai/blog/how-to-make-ollama-faster/)** – Techniques to optimize Ollama’s performance.
- **[Specify Ollama GPU](https://gist.github.com/pykeras/0b1e32b92b87cdce1f7195ea3409105c)** – Guide on selecting a specific GPU for Ollama.
- **[Set Ollama Serve URL](https://github.com/langchain-ai/langchain/issues/15365)** – Configure a custom serve URL for Ollama.

#### Using Local LLM API: vLLM

- Install `vllm`:

```bash
pip install vllm
```

- Start vLLM as an API server:

```bash
sh scripts/vllm_serve.py
```

##### 🔧 Tips
- **[vLLM Tutorial](https://ploomber.io/blog/vllm-deploy/)** – Guide to deploying vLLM.  
- **Use Compute Capability 8.0+** (e.g., **A100**) for better performance.  
- ️**Serving on Tesla V100-SXM2** – Use `--dtype=half --gpu-memory-utilization=0.9 --max-model-len=101728`.  
- **Offline Mode** – Download models manually, set `TRANSFORMERS_OFFLINE=1` (or `HF_HUB_OFFLINE=1`).  
- **Serving with vLLM** – Use the **Instruct version** of the LLM.  

#### Using Local LLM API: SGLang

- Install `SGLang`:

```bash
pip install --upgrade pip
pip install "sglang[all]"
```

- Install FlashInfer CUDA kernels:

```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

- Start SGLang Server:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

##### 🔧 Tips: Model Selection  
- If occasional format misalignment occurs, try a more advanced or larger LLM.  


### 3️⃣ Running Experiments

- Start local LLM server:

```bash
sh <LLM_API_tool>_serve.sh
```

- Run the corresponding module:

```bash
sh scripts/run_<module_name>.sh
```

**Examples:**

Using Ollama API:
```bash
OLLAMA_BASE_URL='http://localhost:11434/v1' OLLAMA_API_KEY='xxxxxxxxxxxxxxxx' python main.py --evaluation_engine ollama-llama3.1 --test_engine ollama-llama3.1 --task BBH_object_counting --module train_centralized
```

##### 🔧 Tips: API URL & Key  
- When integrating third-party LLM API services like OpenRouter or Togther AI, please configure the URL and key as specified in the Ollama documentation, and ensure that the model name is prefixed with “ollama-”. 


## 📊 Results

Results can be found in the `logs` directory. Additionally, you can configure [comet_ml](https://www.comet.com/site/) for logging.

## 📜 Citation

If you find our work useful and relevant, please cite:

```bibtex
@inproceedings{chencan,
  title={Can Textual Gradient Work in Federated Learning?},
  author={Chen, Minghui and Jin, Ruinan and Deng, Wenlong and Chen, Yuanyuan and Huang, Zhi and Yu, Han and Li, Xiaoxiao},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
