//! Noise plugin for WrenLift. Exposes Simplex, Perlin, and Value
//! noise in 2D + 3D plus a fractal Brownian motion (fBm) helper
//! that composes octaves of a base function. Built as a cdylib
//! and bundled into @hatch:noise.
//!
//! All host interaction goes through `wlift_abi`. No `wren_lift`
//! Rust dep — VM/Gc layout shifts can't reach plugin code.

#![allow(clippy::missing_safety_doc)]

use noise::core::worley::distance_functions::euclidean;
use noise::{NoiseFn, OpenSimplex, Perlin, Value, Worley};
use wlift_abi::{runtime_error, set_return, slot, Value as WlValue, WrenVm};

/// Plugin ABI handshake. Called by the host immediately after
/// `dlopen`; mismatching values refuse to bind any other symbols.
#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
}

// Pull a finite f64 from `slot(idx)`. Surfaces a useful error
// message rather than silently defaulting on non-numeric / NaN /
// infinite arguments — every noise sample propagates the badness
// otherwise.
unsafe fn req_num(vm: *mut WrenVm, idx: u32, label: &str, what: &str) -> Option<f64> {
    match slot(vm, idx).as_num() {
        Some(n) if n.is_finite() => Some(n),
        _ => {
            runtime_error(
                vm,
                &format!("{}: argument `{}` must be a finite number.", label, what),
            );
            None
        }
    }
}

unsafe fn req_u32_arg(vm: *mut WrenVm, idx: u32, label: &str, what: &str) -> Option<u32> {
    match slot(vm, idx).as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64 => {
            Some(n as u32)
        }
        _ => {
            runtime_error(
                vm,
                &format!(
                    "{}: argument `{}` must be a non-negative 32-bit integer.",
                    label, what
                ),
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// 2D + 3D scalar samplers
// ---------------------------------------------------------------------------

/// `Noise.simplex2(x, y, seed)` — 2D OpenSimplex noise, range ≈
/// [-1, 1]. Smooth, isotropic, patent-free; the default choice
/// for terrain heightmaps and 2D scalar fields.
///
/// Slot layout: `(_, x, y, seed)`.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_simplex2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.simplex2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.simplex2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.simplex2", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    set_return(vm, WlValue::num(n.get([x, y])));
}

/// `Noise.simplex3(x, y, z, seed)` — 3D OpenSimplex noise. Same
/// range as the 2D variant; use for volumetric / animated noise.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_simplex3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.simplex3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.simplex3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.simplex3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.simplex3", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    set_return(vm, WlValue::num(n.get([x, y, z])));
}

/// `Noise.perlin2(x, y, seed)` — 2D Perlin (gradient) noise, the
/// classic textbook implementation. Range ≈ [-1, 1]. Slightly
/// cheaper than Simplex; shows mild axis-aligned artefacts at
/// high octaves.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_perlin2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.perlin2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.perlin2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.perlin2", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Perlin::new(seed);
    set_return(vm, WlValue::num(n.get([x, y])));
}

/// `Noise.perlin3(x, y, z, seed)` — 3D Perlin noise.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_perlin3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.perlin3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.perlin3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.perlin3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.perlin3", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Perlin::new(seed);
    set_return(vm, WlValue::num(n.get([x, y, z])));
}

/// `Noise.value2(x, y, seed)` — 2D Value noise, range ≈ [-1, 1].
/// Cheapest of the three; lattice-based with smooth interpolation,
/// gives a blockier look than gradient noise. Good for placeholder
/// terrain and CPU-bound use cases.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_value2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.value2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.value2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.value2", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Value::new(seed);
    set_return(vm, WlValue::num(n.get([x, y])));
}

/// `Noise.value3(x, y, z, seed)` — 3D Value noise.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_value3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.value3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.value3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.value3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.value3", "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Value::new(seed);
    set_return(vm, WlValue::num(n.get([x, y, z])));
}

// ---------------------------------------------------------------------------
// Fractal Brownian motion — octave-sum of a base function
// ---------------------------------------------------------------------------

/// `Noise.fbm2(x, y, seed, octaves, lacunarity, persistence)` —
/// fractal Brownian motion over OpenSimplex 2D. Sums `octaves`
/// samples, doubling (or `lacunarity`-folding) frequency and
/// scaling amplitude by `persistence` per step. The output is
/// normalised so the range still sits roughly in [-1, 1] regardless
/// of octave count.
///
/// Typical settings:
///   - `lacunarity` 2.0
///   - `persistence` 0.5
///   - `octaves` 4–8 (terrain) / 1–3 (cheap variation)
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fbm2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.fbm2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.fbm2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.fbm2", "seed") {
        Some(n) => n,
        None => return,
    };
    let octaves = match req_u32_arg(vm, 4, "Noise.fbm2", "octaves") {
        Some(n) if n > 0 && n <= 16 => n,
        Some(_) => {
            runtime_error(vm, "Noise.fbm2: octaves must be in 1..=16.");
            return;
        }
        None => return,
    };
    let lacunarity = match req_num(vm, 5, "Noise.fbm2", "lacunarity") {
        Some(n) => n,
        None => return,
    };
    let persistence = match req_num(vm, 6, "Noise.fbm2", "persistence") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    let mut acc = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        acc += amp * n.get([x * freq, y * freq]);
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    set_return(vm, WlValue::num(if norm > 0.0 { acc / norm } else { 0.0 }));
}

/// `Noise.fbm3(x, y, z, seed, octaves, lacunarity, persistence)` —
/// 3D variant of `fbm2`.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fbm3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.fbm3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.fbm3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.fbm3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.fbm3", "seed") {
        Some(n) => n,
        None => return,
    };
    let octaves = match req_u32_arg(vm, 5, "Noise.fbm3", "octaves") {
        Some(n) if n > 0 && n <= 16 => n,
        Some(_) => {
            runtime_error(vm, "Noise.fbm3: octaves must be in 1..=16.");
            return;
        }
        None => return,
    };
    let lacunarity = match req_num(vm, 6, "Noise.fbm3", "lacunarity") {
        Some(n) => n,
        None => return,
    };
    let persistence = match req_num(vm, 7, "Noise.fbm3", "persistence") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    let mut acc = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        acc += amp * n.get([x * freq, y * freq, z * freq]);
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    set_return(vm, WlValue::num(if norm > 0.0 { acc / norm } else { 0.0 }));
}

// ---------------------------------------------------------------------------
// Batched samplers — fill a Float32Array in one call
// ---------------------------------------------------------------------------

/// `Noise.fillSimplex2(out, originX, originY, stepX, stepY, width, height, seed)` —
/// sample the simplex2 function at every (row, col) of a width ×
/// height grid starting at `(originX, originY)` and stepping by
/// `(stepX, stepY)` per cell. Writes `width * height` floats into
/// `out` (row-major) in one call so the Wren-side loop overhead
/// drops away for heightmap generation.
///
/// `out` must be a `Float32Array` with at least `width * height`
/// slots. Mismatched length aborts cleanly.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fill_simplex2(vm: *mut WrenVm) {
    let label = "Noise.fillSimplex2";
    let bytes = match wlift_abi::typed_array_bytes_mut(slot(vm, 1)) {
        Some(b) => b,
        None => {
            runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
            return;
        }
    };
    let kind = wlift_abi::typed_array_kind(slot(vm, 1));
    if kind != Some(wlift_abi::TypedArrayKind::F32) {
        runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
        return;
    }
    let origin_x = match req_num(vm, 2, label, "originX") {
        Some(n) => n,
        None => return,
    };
    let origin_y = match req_num(vm, 3, label, "originY") {
        Some(n) => n,
        None => return,
    };
    let step_x = match req_num(vm, 4, label, "stepX") {
        Some(n) => n,
        None => return,
    };
    let step_y = match req_num(vm, 5, label, "stepY") {
        Some(n) => n,
        None => return,
    };
    let width = match req_u32_arg(vm, 6, label, "width") {
        Some(n) => n,
        None => return,
    };
    let height = match req_u32_arg(vm, 7, label, "height") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 8, label, "seed") {
        Some(n) => n,
        None => return,
    };
    let needed = (width as usize) * (height as usize) * 4;
    if bytes.len() < needed {
        runtime_error(
            vm,
            &format!(
                "{}: `out` holds {} floats, need {} for {}×{}.",
                label,
                bytes.len() / 4,
                width as usize * height as usize,
                width,
                height
            ),
        );
        return;
    }
    let n = OpenSimplex::new(seed);
    // Reinterpret as &mut [f32] for direct stores; valid because
    // typed_array_bytes_mut hands us the F32 array's backing
    // buffer with matching alignment guarantees.
    let dst = std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, needed / 4);
    let mut idx = 0;
    for row in 0..height {
        let y = origin_y + (row as f64) * step_y;
        for col in 0..width {
            let x = origin_x + (col as f64) * step_x;
            dst[idx] = n.get([x, y]) as f32;
            idx += 1;
        }
    }
    set_return(vm, WlValue::NULL);
}

// ---------------------------------------------------------------------------
// Worley (cellular) noise — F1 distance to nearest seed point.
// ---------------------------------------------------------------------------
//
// Worley generates jittered cell-grid seed points; the value at
// `(x, y)` is the Euclidean distance to the nearest seed. The
// underlying crate returns roughly [-1, 1]; we forward as-is so
// callers can compose with the other noise outputs.

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_worley2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.worley2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.worley2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.worley2", "seed") {
        Some(n) => n,
        None => return,
    };
    let w = Worley::new(seed).set_distance_function(euclidean);
    set_return(vm, WlValue::num(w.get([x, y])));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_worley3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.worley3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.worley3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.worley3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.worley3", "seed") {
        Some(n) => n,
        None => return,
    };
    let w = Worley::new(seed).set_distance_function(euclidean);
    set_return(vm, WlValue::num(w.get([x, y, z])));
}

// ---------------------------------------------------------------------------
// Ridged-multi fBM — sharp valleys / ridges, the classic
// mountain-range generator. Per-octave: take 1 - |noise|, square,
// accumulate. Returns ≈ [0, 1].
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_ridged_fbm2(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.ridgedFbm2", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.ridgedFbm2", "y") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 3, "Noise.ridgedFbm2", "seed") {
        Some(n) => n,
        None => return,
    };
    let octaves = match req_u32_arg(vm, 4, "Noise.ridgedFbm2", "octaves") {
        Some(n) if n > 0 && n <= 16 => n,
        Some(_) => {
            runtime_error(vm, "Noise.ridgedFbm2: octaves must be in 1..=16.");
            return;
        }
        None => return,
    };
    let lacunarity = match req_num(vm, 5, "Noise.ridgedFbm2", "lacunarity") {
        Some(n) => n,
        None => return,
    };
    let persistence = match req_num(vm, 6, "Noise.ridgedFbm2", "persistence") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    let mut acc = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        let s = n.get([x * freq, y * freq]);
        let r = 1.0 - s.abs();
        acc += amp * r * r;
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    set_return(vm, WlValue::num(if norm > 0.0 { acc / norm } else { 0.0 }));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_ridged_fbm3(vm: *mut WrenVm) {
    let x = match req_num(vm, 1, "Noise.ridgedFbm3", "x") {
        Some(n) => n,
        None => return,
    };
    let y = match req_num(vm, 2, "Noise.ridgedFbm3", "y") {
        Some(n) => n,
        None => return,
    };
    let z = match req_num(vm, 3, "Noise.ridgedFbm3", "z") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 4, "Noise.ridgedFbm3", "seed") {
        Some(n) => n,
        None => return,
    };
    let octaves = match req_u32_arg(vm, 5, "Noise.ridgedFbm3", "octaves") {
        Some(n) if n > 0 && n <= 16 => n,
        Some(_) => {
            runtime_error(vm, "Noise.ridgedFbm3: octaves must be in 1..=16.");
            return;
        }
        None => return,
    };
    let lacunarity = match req_num(vm, 6, "Noise.ridgedFbm3", "lacunarity") {
        Some(n) => n,
        None => return,
    };
    let persistence = match req_num(vm, 7, "Noise.ridgedFbm3", "persistence") {
        Some(n) => n,
        None => return,
    };
    let n = OpenSimplex::new(seed);
    let mut acc = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        let s = n.get([x * freq, y * freq, z * freq]);
        let r = 1.0 - s.abs();
        acc += amp * r * r;
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    set_return(vm, WlValue::num(if norm > 0.0 { acc / norm } else { 0.0 }));
}

// ---------------------------------------------------------------------------
// 3D + alternate-flavour bulk fills.
// ---------------------------------------------------------------------------
//
// Shape matches `fill_simplex2`: validate `out` typed-array kind +
// length, decode the grid spec, write `width * height * depth`
// (or `* height` for the 2D variants) f32 samples in row-major
// (z outermost, then y, then x for 3D).

unsafe fn require_f32_out(vm: *mut WrenVm, label: &str) -> Option<&'static mut [u8]> {
    let bytes = wlift_abi::typed_array_bytes_mut(slot(vm, 1))?;
    if wlift_abi::typed_array_kind(slot(vm, 1)) != Some(wlift_abi::TypedArrayKind::F32) {
        runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
        return None;
    }
    Some(bytes)
}

unsafe fn fill_2d_inner<F: Fn(f64, f64) -> f64>(vm: *mut WrenVm, label: &str, sample: F) {
    let bytes = match require_f32_out(vm, label) {
        Some(b) => b,
        None => {
            runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
            return;
        }
    };
    let origin_x = match req_num(vm, 2, label, "originX") {
        Some(n) => n,
        None => return,
    };
    let origin_y = match req_num(vm, 3, label, "originY") {
        Some(n) => n,
        None => return,
    };
    let step_x = match req_num(vm, 4, label, "stepX") {
        Some(n) => n,
        None => return,
    };
    let step_y = match req_num(vm, 5, label, "stepY") {
        Some(n) => n,
        None => return,
    };
    let width = match req_u32_arg(vm, 6, label, "width") {
        Some(n) => n,
        None => return,
    };
    let height = match req_u32_arg(vm, 7, label, "height") {
        Some(n) => n,
        None => return,
    };
    let needed = (width as usize) * (height as usize) * 4;
    if bytes.len() < needed {
        runtime_error(
            vm,
            &format!(
                "{}: `out` holds {} floats, need {} for {}×{}.",
                label,
                bytes.len() / 4,
                width as usize * height as usize,
                width,
                height
            ),
        );
        return;
    }
    let dst = std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, needed / 4);
    let mut idx = 0;
    for row in 0..height {
        let y = origin_y + (row as f64) * step_y;
        for col in 0..width {
            let x = origin_x + (col as f64) * step_x;
            dst[idx] = sample(x, y) as f32;
            idx += 1;
        }
    }
    set_return(vm, WlValue::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fill_perlin2(vm: *mut WrenVm) {
    let label = "Noise.fillPerlin2";
    let seed = match req_u32_arg(vm, 8, label, "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Perlin::new(seed);
    fill_2d_inner(vm, label, |x, y| n.get([x, y]));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fill_value2(vm: *mut WrenVm) {
    let label = "Noise.fillValue2";
    let seed = match req_u32_arg(vm, 8, label, "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Value::new(seed);
    fill_2d_inner(vm, label, |x, y| n.get([x, y]));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fill_worley2(vm: *mut WrenVm) {
    let label = "Noise.fillWorley2";
    let seed = match req_u32_arg(vm, 8, label, "seed") {
        Some(n) => n,
        None => return,
    };
    let n = Worley::new(seed).set_distance_function(euclidean);
    fill_2d_inner(vm, label, |x, y| n.get([x, y]));
}

/// 3D Simplex bulk fill. Samples `width * height * depth` values
/// laid out row-major with z outermost, then y, then x. Used by
/// procedural volumes, voxel terrain, and 3D weather effects that
/// need a deterministic noise field generated in one foreign call.
#[no_mangle]
pub unsafe extern "C" fn wlift_noise_fill_simplex3(vm: *mut WrenVm) {
    let label = "Noise.fillSimplex3";
    let bytes = match wlift_abi::typed_array_bytes_mut(slot(vm, 1)) {
        Some(b) => b,
        None => {
            runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
            return;
        }
    };
    if wlift_abi::typed_array_kind(slot(vm, 1)) != Some(wlift_abi::TypedArrayKind::F32) {
        runtime_error(vm, &format!("{}: `out` must be a Float32Array.", label));
        return;
    }
    let origin_x = match req_num(vm, 2, label, "originX") {
        Some(n) => n,
        None => return,
    };
    let origin_y = match req_num(vm, 3, label, "originY") {
        Some(n) => n,
        None => return,
    };
    let origin_z = match req_num(vm, 4, label, "originZ") {
        Some(n) => n,
        None => return,
    };
    let step_x = match req_num(vm, 5, label, "stepX") {
        Some(n) => n,
        None => return,
    };
    let step_y = match req_num(vm, 6, label, "stepY") {
        Some(n) => n,
        None => return,
    };
    let step_z = match req_num(vm, 7, label, "stepZ") {
        Some(n) => n,
        None => return,
    };
    let width = match req_u32_arg(vm, 8, label, "width") {
        Some(n) => n,
        None => return,
    };
    let height = match req_u32_arg(vm, 9, label, "height") {
        Some(n) => n,
        None => return,
    };
    let depth = match req_u32_arg(vm, 10, label, "depth") {
        Some(n) => n,
        None => return,
    };
    let seed = match req_u32_arg(vm, 11, label, "seed") {
        Some(n) => n,
        None => return,
    };
    let needed = (width as usize) * (height as usize) * (depth as usize) * 4;
    if bytes.len() < needed {
        runtime_error(
            vm,
            &format!(
                "{}: `out` holds {} floats, need {} for {}×{}×{}.",
                label,
                bytes.len() / 4,
                width as usize * height as usize * depth as usize,
                width,
                height,
                depth
            ),
        );
        return;
    }
    let n = OpenSimplex::new(seed);
    let dst = std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, needed / 4);
    let mut idx = 0;
    for slice in 0..depth {
        let z = origin_z + (slice as f64) * step_z;
        for row in 0..height {
            let y = origin_y + (row as f64) * step_y;
            for col in 0..width {
                let x = origin_x + (col as f64) * step_x;
                dst[idx] = n.get([x, y, z]) as f32;
                idx += 1;
            }
        }
    }
    set_return(vm, WlValue::NULL);
}
