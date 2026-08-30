# OpenAI-compatible API

`llama-server` provides the API; the playground does not invent a proxy protocol. Relevant routes:

- `GET /v1/models`
- `POST /v1/chat/completions`

The official [llama-server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
documents its OpenAI compatibility and streaming behavior.

## Authentication

Every application process generates a random `colab-…` key and passes it through `--api-key`.
`llama-server` binds to `127.0.0.1`; only an explicit tunnel action makes it public. The launch log
redacts the key.

```bash
curl https://YOUR_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR_MODEL","messages":[{"role":"user","content":"Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://YOUR_URL/v1", api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="YOUR_MODEL",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

The OpenAI Python package is required only for this client example, not for the playground.

## Tunnel limitation

Cloudflare calls Quick Tunnels a testing/development feature with no SLA. Its current
[official documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
says Quick Tunnels do not support Server-Sent Events. Therefore built-in localhost chat streams, but
external clients using the free Quick Tunnel should disable streaming. Non-streaming Chat Completions
and the API test work through the tunnel. The URL disappears when the process/runtime ends.

