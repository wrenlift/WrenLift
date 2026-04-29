// Solid-color test shader. Loaded at runtime via @hatch:assets's
// web backend (Browser.fetch + manifest); the page's smoke test
// hands this WGSL to Gpu.createShaderModule and renders a quad.

@vertex
fn vs_main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
  return vec4f(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
  return vec4f(0.36, 0.78, 0.50, 1.0);
}
