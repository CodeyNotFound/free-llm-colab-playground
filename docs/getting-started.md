# Getting started

## Prerequisites

- A Google account that can start a Colab runtime
- A public Hugging Face GGUF repository (gated models may require a Hugging Face token)
- Enough time to download model weights (often several gigabytes)

Free Colab resources and GPU types vary dynamically and runtimes time out. Google describes these
constraints in its [official FAQ](https://research.google.com/colaboratory/faq.html).

## Run

1. Open `colab/Free_LLM_Colab_Playground.ipynb` in Colab.
2. Choose **Runtime → Change runtime type → GPU**.
3. Set `REPOSITORY_URL` in the install cell, then run every cell in order.
4. Wait for runtime preparation (precompiled T4 download when compatible; source build otherwise).
   Note the printed username and random password, then open the app and log in.
5. Open **Setup → First time? Read this 60-second guide**.
6. In **01 · Choose a model**, search, click a result, and choose **Inspect this model's files**.
   You can also paste an `author/model-name` repository ID directly.
7. In **02 · Check memory & download**, choose one GGUF. Q4_K_M is preselected when available.
   Keep conversation memory at **4096** for the first run and review the approximate memory check.
8. Click **Download selected file**. After it completes, click **Start model** in step 3.
   Advanced loading settings are optional and collapsed by default.
9. Wait for **Ready to chat**, then click **Open Chat →**. Downloading alone does not start the model.

**Chat** contains optional reply settings and document attachments. Upload documents, click
**Prepare files**, then ask a question; remove attachments when no longer needed.
**Monitor** contains performance, server logs, and **Stop model & API tunnel**.
**Help** explains terms such as GGUF, quantization, context, and VRAM.

Download and Start buttons stay disabled until their prerequisites are met. Changing the selected
file requires downloading that file before starting. Memory estimates update when context or
memory settings change; an insufficient-memory estimate blocks downloading and starting.

The Hugging Face client caches existing downloads; see its
[download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

## External apps

The public API is optional. Start the model first, open **Connect apps**, read the warning, and click
**Create temporary Cloudflare tunnel**. Copy the Base URL, key, and model ID, then test the API.
Stop the tunnel after use.

## Runtime resets

A reset removes the cloned repository, llama.cpp build, downloaded weights, API key, public URL,
conversation, and logs. Mounting Drive for persistence is a future option and is intentionally not
automatic because it grants notebook code access to Drive.
