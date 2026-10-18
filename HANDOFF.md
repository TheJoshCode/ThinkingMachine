# Thinking Machine — Neural Visualizer Handoff

## Project Overview
Building a visualizer particle sphere that shows the inner workings of a VLM/LLM in llama.cpp — attention heads, neurons, embeddings, vision layers, MoE experts — rendered as a globe with continents scaled by memory usage. A custom llama.cpp build exposes internal data via a `/neural` endpoint in `llama-server`.

## File Paths
- **Project root:** `C:/Users/joshr/Downloads/ThinkingMachine/`
- **Custom llama.cpp:** `C:/Users/joshr/Downloads/ThinkingMachine/llama.cpp/`
- **Testing binary:** `C:/Users/joshr/Downloads/ThinkingMachine/llama.cpp/build/bin/Release/llama-server.exe`
- **Model files:** `F:/aimodels/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/` (Qwen3.6-35B-A3B-MXFP4_MOE + mmproj-F32.gguf)
- **Visualizer:** `C:/Users/joshr/Downloads/ThinkingMachine/index.html`
- **Backend:** `C:/Users/joshr/Downloads/ThinkingMachine/llama.cpp/tools/server/server-neural.h` (and `.cpp`)

## Current Architecture

### Backend (server-neural.h)
- Hooks into ggml eval callback to capture per-layer activations
- Captures attention (`a`), feed-forward (`f`), residual (`r`) tensors per layer
- Captures MoE expert gating (`ffn_gate_exps`), expert outputs (`exps.*`), routing vectors
- Streams data via SSE at `/neural` (layer frames, vision frames, meta)
- Exposes JSON at `/neural/meta` (model metadata, layer bytes, components, MoE info)
- Frame types: `t:meta`, `t:L` (layer), `t:V` (vision)

### Frontend (index.html)
- Three.js particle sphere rendered as a globe
- **Continents on a globe** — each tensor component is a continent scaled by byte weight
- Vertical span ∝ memory; within layer continents, attn/ffn/residual arranged horizontally by byte ratio
- Continent types: embed (orange), attn (cyan), ffn (green), res (blue), moe (red), out (magenta), vis (violet)
- Subtle continent borders (opacity 0.25), no grid lines or discrete rings
- Activation array: `L + 3` (one per layer + 3 special slots for embed/out/vis)
- MoE heat tracked separately in `viz._moeHeat` (per-layer gate activation)
- SSE live stream connects to `/neural` endpoint, polls `/neural/meta` first
- Simulation fallback when no server connected

## What's Done
1. Backend extended with MoE capture (K_MOE_GATE, K_MOE_EXPS, K_MOE_ROUTED)
2. Backend meta payload includes n_experts, top_k_experts, is_moe[], kv_size, kv_used
3. Frontend migrated from band-based rings to continent-based globe layout
4. All old constants removed (BAND_COLOR, B_ATTN, B_FFN, B_RES, B_SPEC, B_MOE, MOE_COLOR)
5. Activation array simplified to L + 3 structure
6. Legend, HUD, applyLayerFrame, applyVisionFrame, stepSimulation, stepEngine all updated
7. `/neural/meta` confirmed working on port 8081 — returns full model metadata

## What's NOT Working (Current Issues)

### ISSUE 1: Visualizer not connecting to live server
- The visualizer is still running off simulated data instead of real model data from the `/neural` endpoint
- The `/neural/meta` endpoint IS responding correctly on port 8081 (verified with curl)
- The SSE stream at `/neural` should be working but isn't feeding data to the sphere
- **Possible causes:**
  - Default endpoint in `index.html` line 86 is `http://localhost:8081` — verify this matches what you're running
  - The SSE connection might be failing silently (onerror handler sets status to "stream lost — retrying")
  - The `applyMeta` function may not be receiving the meta frame
  - CORS issues if the server isn't sending proper CORS headers for SSE
- **Debug steps:**
  1. Check browser console for any errors (network tab → SSE connection status)
  2. Check HUD — "source" should say "LIVE X f/s" when connected, not "SIM"
  3. Check HUD — "model" field should show the model name from meta, not "model"
  4. Verify the server is actually streaming layer frames (check `/neural` SSE manually)

### ISSUE 2: Vision frame handling
- `applyVisionFrame` exists but only handles `f.r` (residual) data
- Vision frames come with `t:V` type from the SSE stream
- The function writes to `actFlat[viz.L + SPECIAL_SLOT.vis]` — this is correct for the new structure

### ISSUE 3: Layer frame data format
- The meta response shows layer bytes with keys `a`, `f`, `o` (attention, feed-forward, output/residual)
- Layer frames from `/neural` should contain `a`, `f`, `r` arrays (attention activations, feed-forward activations, residual activations)
- `applyLayerFrame` expects `f.a`, `f.f`, `f.r` — verify the backend sends these exact keys
- MoE data in frames: `f.moe` (gate scores array), `f.routed` (routing indices array)

### ISSUE 4: No MoE expert visualization yet
- The backend captures MoE gate scores and routing vectors
- `viz._moeHeat` tracks per-layer MoE activation
- But there's no visual representation of MoE experts on the globe yet
- MoE layers are identified by `viz.is_moe_layer[]` array from meta
- The MoE heat data exists but isn't being rendered as a visual element

## Server Launch
- Binary: `llama-server.exe` in `C:/Users/joshr/Downloads/ThinkingMachine/llama.cpp/build/bin/Release/`
- Custom build exposes `/neural` endpoint
- Tested on port 8081 (port 8080 is the user's production brain — DO NOT KILL)
- Launch script: `C:/Users/joshr/Downloads/ThinkingMachine/runstuff.bat`

## Model Info
- Model: Qwen3.6-35B-A3B-MXFP4_MOE (MoE architecture)
- Vision adapter: mmproj-F32.gguf
- From meta response on port 8081:
  - L=32 layers, S=16, heads=16, hkv=4, nff=9216, vocab=248320
  - Embedding: ~1.35 GiB, Attention: ~1.1 GiB, Feed-forward: ~2.4 GiB
  - Some layers have `a:39006208` (MoE layers — smaller attn), others `a:44880640`
  - recr=[1,1,1,0,...] — SSM layers at indices 3,7,11,15,19,23,27,31

## Key Functions to Know
- `layoutLayers()` — creates continent zones from model bytes
- `layoutMemory()` — creates memory view zones (treemap layout)
- `buildBrain()` — builds the Three.js sphere from zones
- `applyMeta(f)` — processes meta frame, updates viz state, rebuilds sphere
- `applyLayerFrame(f)` — processes layer frame, writes heat to actFlat
- `applyVisionFrame(f)` — processes vision frame
- `stepSimulation(dt, now)` — generates fake data when no server connected
- `stepEngine(dt, now)` — applies decay, updates uniforms, drives animation
- `connectLive()` — polls /neural/meta, opens SSE stream to /neural

## What Needs To Be Done (Priority Order)
1. **Fix live connection** — get the visualizer actually streaming from the `/neural` endpoint instead of simulation
2. **Verify layer frame format** — confirm backend sends `a`, `f`, `r` arrays in layer frames matching what `applyLayerFrame` expects
3. **Add MoE expert visualization** — render MoE expert activations on the globe (maybe as intensity within MoE layer continents, or as a separate visual element)
4. **Add vision visualization** — render vision encoder activations when processing images
5. **Test with MoE model** — load Qwen3.6-35B-A3B-MXFP4_MOE and verify all data streams correctly
6. **Test with vision model** — load mmproj-F32.gguf and verify vision frames work
7. **Performance tuning** — ensure smooth animation with real model data at full layer count

## Constraints
- NEVER interact with llama-server on port 8080 (it's YOUR BRAIN)
- Port 8081 is safe to test against

- run the test llama-server like this: 
PS C:\Users\joshr\Downloads\ThinkingMachine> .\llamacppstuff\llama-server.exe -m .\unsloth\Qwen3.5-4B-MTP-GGUF\Qwen3.5-4B-Q8_0.gguf --host 127.0.0.1 --port 8081 --n-gpu-layers -1 -c 2048 -np 1 -fa on --jinja -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --kv-unified --ctx-checkpoints 64 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0


- All paths use forward slashes for shell commands, or raw strings for Windows paths
