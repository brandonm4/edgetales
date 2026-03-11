## Handoff Note

Repository to continue from:
- `/Volumes/Data/Development/projects/edgetales`

What was done:
- Added a provider abstraction in [provider.py](/Volumes/Data/Development/projects/edgetales/provider.py).
- Updated [app.py](/Volumes/Data/Development/projects/edgetales/app.py) to construct model clients through `get_model_gateway(...)` instead of directly instantiating Anthropic/OpenAI SDK clients.
- Updated [engine.py](/Volumes/Data/Development/projects/edgetales/engine.py) so model-facing functions accept `ModelGateway` and use a unified responses-based provider path.
- Added a local OpenAI-compatible proxy server in [proxy_server.py](/Volumes/Data/Development/projects/edgetales/proxy_server.py) with:
  - `GET /health`
  - `POST /v1/responses`
  - `mock` backend for local tests
  - `openai` passthrough backend for upstream-compatible usage
- Added proxy integration tests in [tests/test_proxy.py](/Volumes/Data/Development/projects/edgetales/tests/test_proxy.py).
- Extended [config.example.json](/Volumes/Data/Development/projects/edgetales/config.example.json) with:
  - `provider_mode`
  - `provider_base_url`

Verification already run in this repo:
- `python3 -m py_compile /Volumes/Data/Development/projects/edgetales/app.py /Volumes/Data/Development/projects/edgetales/engine.py /Volumes/Data/Development/projects/edgetales/provider.py /Volumes/Data/Development/projects/edgetales/proxy_server.py /Volumes/Data/Development/projects/edgetales/tests/test_proxy.py`
- `PYTHONPATH=/Volumes/Data/Development/projects/edgetales python3 -m unittest /Volumes/Data/Development/projects/edgetales/tests/test_proxy.py`
- Result: tests passed (`3/3`)

Important current state:
- All new work should happen in `/Volumes/Data/Development/projects/edgetales`, not `/Users/brandon2/Downloads/edgetales-main`.
- The target environment did not have the `openai` package installed during testing, so [provider.py](/Volumes/Data/Development/projects/edgetales/provider.py) includes an HTTP fallback for proxy mode. Direct OpenAI mode still expects the SDK once the app auto-installs it at runtime.
- The target repo had newer work than the scratch repo. Only the provider/proxy integration was merged; existing newer gameplay/accessibility changes in the target repo were intentionally preserved.

Recommended next resume point:
1. Switch working directory to `/Volumes/Data/Development/projects/edgetales`.
2. Decide the next backend target for [proxy_server.py](/Volumes/Data/Development/projects/edgetales/proxy_server.py):
   - keep `openai` + `mock` only, or
   - add a `codex` / ChatGPT-plan-backed adapter.
3. If adding a real non-OpenAI backend, implement it behind the proxy boundary rather than changing game logic again.
4. If needed, add a small README section documenting:
   - direct OpenAI mode
   - local proxy mode
   - how to run `proxy_server.py --backend mock` for local testing
