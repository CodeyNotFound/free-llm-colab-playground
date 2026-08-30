# OpenCode

OpenCode supports custom OpenAI-compatible providers. Because its configuration evolves, follow the
current [official provider documentation](https://opencode.ai/docs/providers/) and use the values
displayed in the playground.

1. Run `/connect`, select **Other**, choose a memorable provider ID, and enter the temporary key.
2. Add a custom provider to `opencode.json` using `@ai-sdk/openai-compatible`.
3. Set `options.baseURL` to the playground URL including `/v1`.
4. Add the displayed model ID under `models`.
5. Run `/models` and select it.

Illustrative shape (verify against the linked current docs):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "colab": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Colab LLM",
      "options": {"baseURL": "https://YOUR_URL/v1"},
      "models": {"YOUR_MODEL": {"name": "Colab model"}}
    }
  }
}
```

Do not commit the API key. Quick Tunnel does not support SSE, so use non-streaming behavior where the
client exposes that choice.

