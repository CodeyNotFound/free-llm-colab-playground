# Getting started

## Prerequisites

- A Google account that can start a Colab runtime
- A public Hugging Face GGUF repository (gated models may require a Hugging Face token)
- Patience for compilation and multi-gigabyte downloads

Free Colab resources and GPU types vary dynamically and runtimes time out. Google describes these
constraints in its [official FAQ](https://research.google.com/colaboratory/faq.html).

## Run

1. Open `colab/Free_LLM_Colab_Playground.ipynb` in Colab.
2. Choose **Runtime → Change runtime type → GPU**.
3. Set `REPOSITORY_URL` in the install cell, then run every cell in order.
4. Wait for the CUDA llama.cpp build, note the random UI login, and open the Gradio share link.
5. Search for a model name. Results are restricted to repositories tagged GGUF.
6. Select a repository, discover files, and choose a quantization.
7. Start with Q4_K_M and 4K or 8K context unless the estimate strongly supports more.
8. Review the approximate GPU/RAM plan, download, and start the model.
9. Chat in the browser. Performance and raw logs are under Nerd Mode.

The Hugging Face client caches existing downloads; see its
[download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

## External apps

The public API is optional. Start the model first, open **API**, read the warning, and click
**Create temporary Cloudflare tunnel**. Copy the Base URL, key, and model ID, then test the API.
Stop the tunnel after use.

## Runtime resets

A reset removes the cloned repository, llama.cpp build, downloaded weights, API key, public URL,
conversation, and logs. Mounting Drive for persistence is a future option and is intentionally not
automatic because it grants notebook code access to Drive.
