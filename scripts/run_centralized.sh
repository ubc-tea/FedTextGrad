# Note that third-party API should also use OLLAMA_BASE_URL, e.g., openrouter
OLLAMA_BASE_URL='YOUR_URL' OLLAMA_API_KEY='YOUR_API_KEY' python main.py --evaluation_engine ollama-meta-llama/llama-3.2-11b-vision-instruct --test_engine ollama-meta-llama/llama-3.2-11b-vision-instruct --task BBH_object_counting --module train_centralized --proximal_update | tee -a ./logs/testing.log
