// Shared host_gpu_* bridge factory used by both the main-thread
// playground (`wlift.js::MainWlift`) and the worker
// (`worker.js`). All WebGPU APIs the bridge calls
// (`navigator.gpu`, `OffscreenCanvas.getContext("webgpu")`,
// `device.createBuffer`, etc.) work identically in window and
// worker scope; the only thread-specific bit was the canvas
// element, and `OffscreenCanvas` slots in for `HTMLCanvasElement`
// at the same `.getContext("webgpu")` site.
//
// Caller contract:
//   * `wasm`       — the namespace returned by `init()` (raw
//                    `wlift_wasm.wasm` exports). Used for slot
//                    reads (`wrenGetSlotString`, `wrenGetSlotBytes`,
//                    `wrenGetListCount`, etc.).
//   * `canvases`   — Map<canvas_handle, { element, ... }>. Both
//                    threads populate this from their own DOM
//                    bridge layer; the GPU bridge only reads
//                    `c.element` (`HTMLCanvasElement` on main,
//                    `OffscreenCanvas` on worker) at attach time.
//
// Returns:
//   * `bridge`           — the `host_gpu_*` imports object the
//                          plugin wasm consumes via `env`.
//   * `ensureGpuDevice`  — async fn that lazily resolves
//                          `navigator.gpu`. Called once at plugin
//                          install time so subsequent sync
//                          host_gpu_* calls have a device ready.

export function createGpuBridge({ wasm, canvases }) {
  // GPU device — single per page/worker, lazy-resolved. Browsers
  // require an async `requestAdapter` -> `requestDevice` dance
  // before any GPU work, but Wren-side foreign methods are sync.
  // The plugin loader runs `ensureGpuDevice` at install time so
  // every subsequent `host_gpu_*` call has the device ready.
  let gpuDevicePromise = null;
  let gpuDevice = null;
  let gpuPreferredFormat = null;

  async function ensureGpuDevice() {
    if (gpuDevice) return gpuDevice;
    if (!gpuDevicePromise) {
      gpuDevicePromise = (async () => {
        const gpu = (typeof navigator !== "undefined") ? navigator.gpu : null;
        if (!gpu) return null;
        try {
          const adapter = await gpu.requestAdapter();
          if (!adapter) return null;
          const device = await adapter.requestDevice();
          gpuDevice = device;
          gpuPreferredFormat = gpu.getPreferredCanvasFormat();
          return device;
        } catch (err) {
          console.warn("[wlift gpu] requestAdapter/requestDevice failed:", err);
          return null;
        }
      })();
    }
    return gpuDevicePromise;
  }

  // GPU resource registry — surfaces, frames (texture view +
  // command encoder), buffers, textures, samplers, shaders,
  // pipelines, bind groups, render passes share one numeric
  // handle space. JS holds the refs; releasing on `endFrame` /
  // `destroy` deletes from this map.
  const gpuObjects = new Map();
  let nextGpuHandle = 1;

  // Variable-length data-carrying bridges (shader src, buffer
  // writes, descriptor JSON) take a `vm` + slot index and read
  // directly from the host-side Wren slot. The slot read happens
  // entirely inside the host's wasm context — same memory in
  // both threads, no plugin-memory copy dance.
  const readSlotString = (vm, slot) => {
    const ptr = wasm.wrenGetSlotString(vm, slot);
    if (!ptr) return null;
    const view = new Uint8Array(wasm.memory.buffer);
    let len = 0;
    while (view[ptr + len] !== 0) len++;
    return new TextDecoder().decode(view.subarray(ptr, ptr + len));
  };

  const readSlotBytes = (vm, slot) => {
    const len_ptr = wasm.wlift_host_alloc(4);
    const ptr = wasm.wrenGetSlotBytes(vm, slot, len_ptr);
    const len = new Int32Array(wasm.memory.buffer, len_ptr, 1)[0];
    wasm.wlift_host_free(len_ptr, 4);
    if (!ptr || len <= 0) return null;
    return new Uint8Array(wasm.memory.buffer, ptr, len);
  };

  const decodeSlotJson = (vm, slot) => {
    const text = readSlotString(vm, slot);
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (err) {
      console.warn("[wlift gpu] descriptor JSON parse failed:", err);
      return null;
    }
  };

  // GPUShaderStage flags as numeric: VERTEX=1, FRAGMENT=2, COMPUTE=4.
  const parseVisibility = (v) => {
    if (typeof v === "number") return v;
    if (typeof v !== "string") return 0;
    let bits = 0;
    for (const part of v.split("|").map((s) => s.trim().toLowerCase())) {
      if (part === "vertex") bits |= 1;
      else if (part === "fragment") bits |= 2;
      else if (part === "compute") bits |= 4;
    }
    return bits;
  };

  const resolveBindResource = (r) => {
    if (!r) return undefined;
    if (r.kind === "buffer") {
      const obj = gpuObjects.get(r.buffer);
      if (!obj || obj.kind !== "buffer") return undefined;
      return {
        buffer: obj.buffer,
        ...(r.offset !== undefined ? { offset: r.offset } : {}),
        ...(r.size   !== undefined ? { size:   r.size   } : {}),
      };
    }
    if (r.kind === "textureView") {
      const obj = gpuObjects.get(r.view);
      if (!obj || obj.kind !== "texture_view") return undefined;
      return obj.view;
    }
    if (r.kind === "sampler") {
      const obj = gpuObjects.get(r.sampler);
      if (!obj || obj.kind !== "sampler") return undefined;
      return obj.sampler;
    }
    return undefined;
  };

  const bridge = {
    host_gpu_ready: () => (gpuDevice ? 1 : 0),

    host_gpu_attach_canvas: (canvas_handle) => {
      if (!gpuDevice) return 0;
      const c = canvases.get(canvas_handle);
      if (!c) return 0;
      // Already configured? Reuse the existing surface handle.
      for (const [h, obj] of gpuObjects) {
        if (obj && obj.kind === "surface" && obj.canvasHandle === canvas_handle) {
          return h;
        }
      }
      let context;
      try {
        context = c.element.getContext("webgpu");
      } catch (_) {
        return 0;
      }
      if (!context) return 0;
      context.configure({
        device: gpuDevice,
        format: gpuPreferredFormat,
        alphaMode: "premultiplied",
      });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, {
        kind: "surface",
        canvasHandle: canvas_handle,
        context,
      });
      return handle;
    },

    host_gpu_begin_frame: (surface_handle) => {
      if (!gpuDevice) return 0;
      const surface = gpuObjects.get(surface_handle);
      if (!surface || surface.kind !== "surface") return 0;
      let texture;
      try {
        texture = surface.context.getCurrentTexture();
      } catch (err) {
        console.warn("[wlift gpu] getCurrentTexture failed:", err);
        return 0;
      }
      const view = texture.createView();
      const encoder = gpuDevice.createCommandEncoder();
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "frame", surface, texture, view, encoder });
      return handle;
    },

    host_gpu_clear: (frame_handle, r, g, b, a) => {
      const frame = gpuObjects.get(frame_handle);
      if (!frame || frame.kind !== "frame") return;
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: [
          {
            view: frame.view,
            clearValue: { r, g, b, a },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.end();
    },

    host_gpu_end_frame: (frame_handle) => {
      const frame = gpuObjects.get(frame_handle);
      if (!frame || frame.kind !== "frame") return;
      const buffer = frame.encoder.finish();
      gpuDevice.queue.submit([buffer]);
      gpuObjects.delete(frame_handle);
    },

    // ----- Buffers -----

    host_gpu_create_buffer: (size, usage_bits) => {
      if (!gpuDevice) return 0;
      const buf = gpuDevice.createBuffer({ size, usage: usage_bits });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "buffer", buffer: buf, size });
      return handle;
    },

    host_gpu_buffer_write: (vm, buffer_handle, offset, slot) => {
      if (!gpuDevice) return;
      const obj = gpuObjects.get(buffer_handle);
      if (!obj || obj.kind !== "buffer") return;
      const bytes = readSlotBytes(vm, slot);
      if (!bytes) return;
      gpuDevice.queue.writeBuffer(obj.buffer, offset, bytes.slice());
    },

    host_gpu_destroy_buffer: (handle) => {
      const obj = gpuObjects.get(handle);
      if (!obj || obj.kind !== "buffer") return;
      obj.buffer.destroy?.();
      gpuObjects.delete(handle);
    },

    host_gpu_create_buffer_from_f32: (vm, list_slot, usage_bits) => {
      if (!gpuDevice) return 0;
      const count = wasm.wrenGetListCount(vm, list_slot);
      if (count <= 0) return 0;
      const floats = new Float32Array(count);
      wasm.wrenEnsureSlots(vm, list_slot + 2);
      const dest = list_slot + 1;
      for (let i = 0; i < count; i++) {
        wasm.wrenGetListElement(vm, list_slot, i, dest);
        floats[i] = wasm.wrenGetSlotDouble(vm, dest);
      }
      const buf = gpuDevice.createBuffer({
        size: floats.byteLength,
        usage: usage_bits,
      });
      gpuDevice.queue.writeBuffer(buf, 0, floats);
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "buffer", buffer: buf, size: floats.byteLength });
      return handle;
    },

    host_gpu_buffer_write_f32: (vm, buffer_handle, offset, list_slot) => {
      if (!gpuDevice) return;
      const obj = gpuObjects.get(buffer_handle);
      if (!obj || obj.kind !== "buffer") return;
      const count = wasm.wrenGetListCount(vm, list_slot);
      if (count <= 0) return;
      const floats = new Float32Array(count);
      wasm.wrenEnsureSlots(vm, list_slot + 2);
      const dest = list_slot + 1;
      for (let i = 0; i < count; i++) {
        wasm.wrenGetListElement(vm, list_slot, i, dest);
        floats[i] = wasm.wrenGetSlotDouble(vm, dest);
      }
      gpuDevice.queue.writeBuffer(obj.buffer, offset, floats);
    },

    host_gpu_buffer_write_u32: (vm, buffer_handle, offset, list_slot) => {
      if (!gpuDevice) return;
      const obj = gpuObjects.get(buffer_handle);
      if (!obj || obj.kind !== "buffer") return;
      const count = wasm.wrenGetListCount(vm, list_slot);
      if (count <= 0) return;
      const uints = new Uint32Array(count);
      wasm.wrenEnsureSlots(vm, list_slot + 2);
      const dest = list_slot + 1;
      for (let i = 0; i < count; i++) {
        wasm.wrenGetListElement(vm, list_slot, i, dest);
        uints[i] = wasm.wrenGetSlotDouble(vm, dest);
      }
      gpuDevice.queue.writeBuffer(obj.buffer, offset, uints);
    },

    host_gpu_create_shader: (vm, slot) => {
      if (!gpuDevice) return 0;
      const code = readSlotString(vm, slot);
      if (!code) return 0;
      const module = gpuDevice.createShaderModule({ code });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "shader", module });
      return handle;
    },

    host_gpu_create_bind_group_layout: (vm, slot) => {
      if (!gpuDevice) return 0;
      const desc = decodeSlotJson(vm, slot);
      if (!desc) return 0;
      const entries = (desc.entries || []).map((e) => ({
        binding: e.binding,
        visibility: parseVisibility(e.visibility),
        ...(e.buffer ? { buffer: e.buffer } : {}),
        ...(e.sampler ? { sampler: e.sampler } : {}),
        ...(e.texture ? { texture: e.texture } : {}),
        ...(e.storageTexture ? { storageTexture: e.storageTexture } : {}),
      }));
      const layout = gpuDevice.createBindGroupLayout({ entries });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "bind_group_layout", layout });
      return handle;
    },

    host_gpu_create_bind_group: (vm, slot) => {
      if (!gpuDevice) return 0;
      const desc = decodeSlotJson(vm, slot);
      if (!desc) return 0;
      const layoutObj = gpuObjects.get(desc.layout);
      if (!layoutObj || layoutObj.kind !== "bind_group_layout") return 0;
      const entries = (desc.entries || []).map((e) => ({
        binding: e.binding,
        resource: resolveBindResource(e.resource),
      }));
      const bindGroup = gpuDevice.createBindGroup({
        layout: layoutObj.layout,
        entries,
      });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "bind_group", bindGroup });
      return handle;
    },

    host_gpu_create_texture: (vm, slot) => {
      if (!gpuDevice) return 0;
      const desc = decodeSlotJson(vm, slot);
      if (!desc) return 0;
      const size = desc.depth
        ? { width: desc.width, height: desc.height, depthOrArrayLayers: desc.depth }
        : [desc.width, desc.height];
      const tex = gpuDevice.createTexture({
        size,
        format: desc.format || "rgba8unorm",
        usage: desc.usage || 0,
        ...(desc.dimension ? { dimension: desc.dimension } : {}),
        ...(desc.mipLevelCount !== undefined ? { mipLevelCount: desc.mipLevelCount } : {}),
        ...(desc.sampleCount  !== undefined ? { sampleCount:  desc.sampleCount  } : {}),
      });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, {
        kind: "texture", texture: tex,
        width: desc.width, height: desc.height,
        format: desc.format || "rgba8unorm",
      });
      return handle;
    },

    host_gpu_create_texture_view: (vm, texture_handle, slot) => {
      if (!gpuDevice) return 0;
      const tex = gpuObjects.get(texture_handle);
      if (!tex || tex.kind !== "texture") return 0;
      const desc = decodeSlotJson(vm, slot);
      const view = desc ? tex.texture.createView(desc) : tex.texture.createView();
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "texture_view", view, texture: texture_handle });
      return handle;
    },

    host_gpu_queue_write_texture: (vm, texture_handle, bytes_slot, desc_slot) => {
      if (!gpuDevice) return 0;
      const tex = gpuObjects.get(texture_handle);
      if (!tex || tex.kind !== "texture") return 0;
      const bytes = readSlotBytes(vm, bytes_slot);
      if (!bytes) return 0;
      const desc = decodeSlotJson(vm, desc_slot) || {};
      const bytesPerRow = desc.bytesPerRow ?? (tex.width * 4);
      const writeSize = {
        width:  desc.width  ?? tex.width,
        height: desc.height ?? tex.height,
        depthOrArrayLayers: desc.depth ?? 1,
      };
      gpuDevice.queue.writeTexture(
        {
          texture: tex.texture,
          ...(desc.mipLevel !== undefined ? { mipLevel: desc.mipLevel } : {}),
          ...(desc.origin  ? { origin: desc.origin } : {}),
          ...(desc.aspect  ? { aspect: desc.aspect } : {}),
        },
        bytes.slice(),
        {
          bytesPerRow,
          ...(desc.rowsPerImage !== undefined ? { rowsPerImage: desc.rowsPerImage } : {}),
        },
        writeSize,
      );
      return 1;
    },

    host_gpu_create_sampler: (vm, slot) => {
      if (!gpuDevice) return 0;
      const desc = decodeSlotJson(vm, slot) || {};
      const sampler = gpuDevice.createSampler({
        ...(desc.magFilter    ? { magFilter:    desc.magFilter    } : {}),
        ...(desc.minFilter    ? { minFilter:    desc.minFilter    } : {}),
        ...(desc.mipmapFilter ? { mipmapFilter: desc.mipmapFilter } : {}),
        ...(desc.addressModeU ? { addressModeU: desc.addressModeU } : {}),
        ...(desc.addressModeV ? { addressModeV: desc.addressModeV } : {}),
        ...(desc.addressModeW ? { addressModeW: desc.addressModeW } : {}),
        ...(desc.lodMinClamp !== undefined ? { lodMinClamp: desc.lodMinClamp } : {}),
        ...(desc.lodMaxClamp !== undefined ? { lodMaxClamp: desc.lodMaxClamp } : {}),
        ...(desc.compare ? { compare: desc.compare } : {}),
        ...(desc.maxAnisotropy !== undefined ? { maxAnisotropy: desc.maxAnisotropy } : {}),
      });
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "sampler", sampler });
      return handle;
    },

    host_gpu_create_pipeline: (vm, slot) => {
      if (!gpuDevice) return 0;
      const desc = decodeSlotJson(vm, slot);
      if (!desc) return 0;
      const vertShader = gpuObjects.get(desc.vertex.shader);
      if (!vertShader || vertShader.kind !== "shader") return 0;
      const fragShader =
        desc.fragment && gpuObjects.get(desc.fragment.shader);
      if (desc.fragment && (!fragShader || fragShader.kind !== "shader")) return 0;
      let layout = "auto";
      if (Array.isArray(desc.layouts)) {
        const bgls = desc.layouts.map((h) => {
          const obj = gpuObjects.get(h);
          return obj && obj.kind === "bind_group_layout" ? obj.layout : null;
        }).filter(Boolean);
        layout = gpuDevice.createPipelineLayout({ bindGroupLayouts: bgls });
      }
      const targets = (desc.fragment?.targets || [{}]).map((t) => ({
        format: t.format === "preferred" || !t.format ? gpuPreferredFormat : t.format,
        ...(t.blend ? { blend: t.blend } : {}),
        ...(t.writeMask !== undefined ? { writeMask: t.writeMask } : {}),
      }));
      const pipelineDesc = {
        layout,
        vertex: {
          module: vertShader.module,
          entryPoint: desc.vertex.entry || "vs_main",
          buffers: desc.vertex.buffers || [],
        },
        ...(fragShader ? {
          fragment: {
            module: fragShader.module,
            entryPoint: desc.fragment.entry || "fs_main",
            targets,
          },
        } : {}),
        primitive: desc.primitive || { topology: "triangle-list" },
        ...(desc.depthStencil ? { depthStencil: desc.depthStencil } : {}),
      };
      const pipeline = gpuDevice.createRenderPipeline(pipelineDesc);
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "pipeline", pipeline });
      return handle;
    },

    host_gpu_render_pass_begin: (vm, frame_handle, slot) => {
      const frame = gpuObjects.get(frame_handle);
      if (!frame || frame.kind !== "frame") return 0;
      const desc = decodeSlotJson(vm, slot);
      const cas = (desc?.colorAttachments || [{}]).map((c) => {
        let view = frame.view;
        if (c.view !== undefined) {
          const v = gpuObjects.get(c.view);
          if (v && v.kind === "texture_view") view = v.view;
        }
        return {
          view,
          clearValue: c.clearValue || { r: 0, g: 0, b: 0, a: 1 },
          loadOp:  c.loadOp  || "clear",
          storeOp: c.storeOp || "store",
        };
      });
      const passDesc = { colorAttachments: cas };
      if (desc?.depthStencilAttachment) {
        const ds = desc.depthStencilAttachment;
        const v = gpuObjects.get(ds.view);
        if (v && v.kind === "texture_view") {
          passDesc.depthStencilAttachment = {
            view: v.view,
            ...(ds.depthClearValue !== undefined
              ? { depthClearValue: ds.depthClearValue } : { depthClearValue: 1.0 }),
            depthLoadOp:  ds.depthLoadOp  || "clear",
            depthStoreOp: ds.depthStoreOp || "store",
            ...(ds.stencilClearValue !== undefined ? { stencilClearValue: ds.stencilClearValue } : {}),
            ...(ds.stencilLoadOp  ? { stencilLoadOp:  ds.stencilLoadOp  } : {}),
            ...(ds.stencilStoreOp ? { stencilStoreOp: ds.stencilStoreOp } : {}),
          };
        }
      }
      const pass = frame.encoder.beginRenderPass(passDesc);
      const handle = nextGpuHandle++;
      gpuObjects.set(handle, { kind: "render_pass", pass, frame: frame_handle });
      return handle;
    },

    host_gpu_render_pass_set_pipeline: (pass_handle, pipeline_handle) => {
      const p = gpuObjects.get(pass_handle);
      const pl = gpuObjects.get(pipeline_handle);
      if (!p || p.kind !== "render_pass") return;
      if (!pl || pl.kind !== "pipeline") return;
      p.pass.setPipeline(pl.pipeline);
    },

    host_gpu_render_pass_set_bind_group: (pass_handle, group_index, bg_handle) => {
      const p = gpuObjects.get(pass_handle);
      const bg = gpuObjects.get(bg_handle);
      if (!p || p.kind !== "render_pass") return;
      if (!bg || bg.kind !== "bind_group") return;
      p.pass.setBindGroup(group_index, bg.bindGroup);
    },

    host_gpu_render_pass_set_vertex_buffer: (pass_handle, slot, buffer_handle) => {
      const p = gpuObjects.get(pass_handle);
      const buf = gpuObjects.get(buffer_handle);
      if (!p || p.kind !== "render_pass") return;
      if (!buf || buf.kind !== "buffer") return;
      p.pass.setVertexBuffer(slot, buf.buffer);
    },

    host_gpu_render_pass_set_index_buffer: (pass_handle, buffer_handle, format32) => {
      const p = gpuObjects.get(pass_handle);
      const buf = gpuObjects.get(buffer_handle);
      if (!p || p.kind !== "render_pass") return;
      if (!buf || buf.kind !== "buffer") return;
      p.pass.setIndexBuffer(buf.buffer, format32 ? "uint32" : "uint16");
    },

    host_gpu_render_pass_draw: (pass_handle, vertex_count, instance_count, first_vertex, first_instance) => {
      const p = gpuObjects.get(pass_handle);
      if (!p || p.kind !== "render_pass") return;
      p.pass.draw(vertex_count, instance_count || 1, first_vertex || 0, first_instance || 0);
    },

    host_gpu_render_pass_draw_indexed: (pass_handle, index_count, instance_count, first_index, base_vertex, first_instance) => {
      const p = gpuObjects.get(pass_handle);
      if (!p || p.kind !== "render_pass") return;
      p.pass.drawIndexed(index_count, instance_count || 1, first_index || 0, base_vertex || 0, first_instance || 0);
    },

    host_gpu_render_pass_end: (pass_handle) => {
      const p = gpuObjects.get(pass_handle);
      if (!p || p.kind !== "render_pass") return;
      p.pass.end();
      gpuObjects.delete(pass_handle);
    },

    host_gpu_destroy: (handle) => {
      const obj = gpuObjects.get(handle);
      if (!obj) return;
      if (obj.kind === "buffer")  obj.buffer.destroy?.();
      if (obj.kind === "texture") obj.texture.destroy?.();
      gpuObjects.delete(handle);
    },
  };

  return { bridge, ensureGpuDevice };
}
