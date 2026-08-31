# Security

## Threat model

Creating a tunnel exposes an inference endpoint from a temporary Colab VM to the public internet.
Possession of both the URL and key is sufficient to consume model capacity. Quick Tunnel provides no
production access policy or SLA.

## Implemented controls

- A cryptographically random per-process API key; no default/shared secret
- A separate random username/password gate on public Gradio share links
- `llama-server` listens on loopback and enforces its API key
- Public tunnel creation is explicit and accompanied by a warning
- Tunnel and server have separate stop actions
- API keys are replaced with `<API_KEY>` in captured launch commands
- Keys and GGUF weights are ignored by Git
- Uploaded filenames are normalized, files are never executed, unsupported extensions are rejected
- Upload size is limited to 8 MiB and extracted content to 80,000 characters
- Document text is wrapped as untrusted reference material, not system instructions
- Raw server logs are available in Monitor while normal failures get actionable messages

## User responsibilities

- Keep the URL and key private; rotate by restarting the app if either leaks.
- Stop the tunnel immediately after use.
- Do not upload secrets or untrusted sensitive documents.
- Review model licenses and model-generated content.
- Do not treat model output as trusted code or security advice.

## Reporting

Report vulnerabilities privately using the process in the repository `SECURITY.md`. Do not include
working API URLs, credentials, personal documents, or model weights in reports.
