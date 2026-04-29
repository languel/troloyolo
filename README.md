---
title: troloyolo
emoji: 🏆
colorFrom: red
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: A YOLO26 browser demo for interactive art experiments
---

# troloyolo

## Local demo

troloyolo is a local static replica and remix point for the YOLO26 WebGPU Space:
https://huggingface.co/spaces/webml-community/YOLO26-WebGPU

Run it from the project folder:

```bash
npm run dev
```

Then open http://localhost:5173.

The browser downloads Transformers.js and onnxruntime-web from jsDelivr, then loads the selected YOLO26 models from Hugging Face. Detection and pose use the `onnx-community/yolo26*-ONNX` Transformers.js exports. Segmentation uses AXERA's YOLO26 segmentation ONNX files through onnxruntime-web.

Use a WebGPU-capable browser for best performance, or choose the WASM backend when WebGPU is unavailable. Auto mode tries WebGPU first and falls back to WASM when the browser only exposes WASM.

Allow camera access when prompted, or choose `Movie / GIF file` to run the same model layers over a local video or animated GIF. `Direct media URL` works for MP4, WebM, and GIF URLs that the browser can load and read from canvas; remote servers may need permissive CORS headers. YouTube watch links are not supported directly by the static browser demo because YouTube frames are cross-origin and cannot be read for model input.

## OSC output

`npm run dev` starts a static web server with a local OSC bridge. The browser posts each completed inference frame to the bridge, and the bridge sends UDP OSC to `127.0.0.1:8000` by default. The OSC panel in the app can start/stop output, change the destination host/port, send a test packet, and open the in-app spec overlay.

Change the OSC destination with environment variables:

```bash
OSC_HOST=127.0.0.1 OSC_PORT=9000 npm run dev
```

Initial bridge output can be disabled with:

```bash
OSC_ENABLED=0 npm run dev
```

Runtime bridge API:

- `GET /osc/config` returns `{ enabled, host, port, packetsSent }`.
- `POST /osc/config` accepts `{ enabled, host, port }` and returns the updated config.
- `POST /osc/test` sends `/troloyolo/test` to the current destination.

OSC messages:

| Address | Arguments |
| --- | --- |
| `/troloyolo/frame` | `frameId:int count:int width:int height:int timestamp:float source:string` |
| `/troloyolo/object` | `frameId:int id:int classId:int label:string type:string score:float nx:float ny:float nw:float nh:float ncx:float ncy:float x:float y:float w:float h:float` |
| `/troloyolo/test` | `timestamp:float host:string port:int` |

Object field meanings:

| Field | Meaning |
| --- | --- |
| `frameId` | Incrementing inference frame counter, reset when the source or model stack restarts. |
| `id` | Stable tracked object ID shown in the overlay as `#ID`. |
| `classId`, `label` | COCO class index and resolved label. |
| `type` | Detection source, currently `object` or `segment`. |
| `score` | Model confidence from `0..1`. |
| `nx`, `ny`, `nw`, `nh` | Normalized top-left box and size in source coordinates. |
| `ncx`, `ncy` | Normalized box center in source coordinates. |
| `x`, `y`, `w`, `h` | Pixel top-left box and size in the camera/movie/GIF source coordinate space. |

The normalized coordinates are `0..1`; the final `x y w h` values are source pixels. Use `npm run static` for the old no-bridge static server.

## Layers

The controls expose the selected visual source and model outputs as independent layers:

- `Camera` hides or shows the camera, movie, or GIF source without stopping processing.
- `Detection` draws object boxes from the selected YOLO26 detect model and tags each tracked object with a stable `#ID`.
- `Pose` draws skeleton/keypoint overlays from the matching YOLO26 pose model.
- `Segmentation` loads the matching AXERA YOLO26 segmentation ONNX file and draws mask overlays.
- `OBB` and `Classify` are present in the UI for future model sources. They automatically mark themselves unavailable when a browser-ready model is not found.

Changing model size, backend, or model layers reloads only the required model layers. The overlay keeps the last completed inference frame while the next frame is processing, so slower layers should not flicker between inference passes. Track IDs reset when the source or model stack restarts.

## Stream-friendly UI

The app fills the whole browser window so it can be captured into other apps. Controls live in a translucent overlay on the right.

- Press `Alt+U` to toggle clean mode.
- Click or hover the right-edge chevron to bring the controls back.
- Open `http://localhost:5173/?clean=1` to start with only the camera/tracking output visible.

## Remote testing over Tailscale

Browsers only allow camera access from secure contexts. `http://localhost:5173` is allowed, but a remote URL like `http://machine-name:5173` or `http://100.x.y.z:5173` is not. The browser can deny camera access before showing a prompt.

Use Tailscale Serve to expose the local dev server with a valid HTTPS URL inside the tailnet:

```bash
npm run dev
```

In another terminal:

```bash
tailscale serve --bg --set-path /troloyolo 5173
```

Tailscale prints a URL like:

```text
https://your-machine.your-tailnet.ts.net
```

Open that HTTPS URL from the remote device. If Tailscale asks you to enable HTTPS certificates for the tailnet, follow its consent flow first.

On this macOS machine, the GUI app's CLI is the reliable one to use:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve status
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve reset
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg --set-path /troloyolo 5173
```

The expected config is:

```text
https://office.solarflare-stonecat.ts.net (tailnet only)
|-- /troloyolo proxy http://127.0.0.1:5173
```

Test it from the host:

```bash
curl -I https://office.solarflare-stonecat.ts.net/troloyolo/
```

You should see `HTTP/2 200`. If you see `ERR_SSL_PROTOCOL_ERROR`, reset Serve and re-run the `serve --bg --set-path /troloyolo 5173` command above.

For a more reliable class/demo URL that also works from browsers that are not correctly routed through the tailnet, use Tailscale Funnel:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale funnel --bg --set-path /troloyolo 5173
```

Expected config:

```text
# Funnel on:
#     - https://office.solarflare-stonecat.ts.net

https://office.solarflare-stonecat.ts.net (Funnel on)
|-- /troloyolo proxy http://127.0.0.1:5173
```

Turn it off when the demo is done:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale funnel --https=443 off
```
