# ShopRAG

An e-commerce RAG for experimenting with chunking, reranking, and query strategies on product retrieval performance.

# 1. clone
git clone <your-repo-url> shoprag
cd shoprag

# 2. environment setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .         
