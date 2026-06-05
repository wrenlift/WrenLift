//! Particle-system math plugin for WrenLift.
//!
//! Exposes two foreign functions that the Wren-side `ParticleSystem3D`
//! delegates its per-frame hot loops to:
//!
//! - `wlift_particles_integrate` — advances `liveCount` particles in
//!   the caller's sim Float32Array by `dt`, applying gravity + linear
//!   drag + lifetime decay (and the optional kill-plane crossing
//!   recorded in a side `deaths` Float32Array). Returns the new live
//!   count (compaction by swap-with-last).
//!
//! - `wlift_particles_pack` — packs the per-instance billboard data
//!   the GPU draw expects (16 f32 per slot — position + size + uv-rect
//!   + tint + rotation + lod) for each live slot, with optional
//!   distance-scaled width and atmospheric alpha falloff.
//!
//! The hot work happens in tight `for off in 0..count*8 step 8` loops
//! against the raw `&mut [f32]` view of the host Float32Array. The
//! Wren-side caller keeps the simulation state (positions, velocities,
//! age, inverse-lifetime), spawn logic, lifecycle, and renderer
//! dispatch — only the per-particle update + per-particle pack run
//! in native code.
//!
//! ## Sim slot layout (8 f32 per slot)
//!
//! ```text
//!   off + 0 = px
//!   off + 1 = py
//!   off + 2 = pz
//!   off + 3 = vx
//!   off + 4 = vy
//!   off + 5 = vz
//!   off + 6 = age
//!   off + 7 = inv_lifetime  (= 1.0 / lifetime; written at spawn)
//! ```
//!
//! ## Instance slot layout (16 f32 per slot — matches `drawBillboardN`)
//!
//! ```text
//!   off + 0..2  = origin xyz
//!   off + 3..4  = size x/y
//!   off + 5..8  = uv-rect (u0, v0, u1, v1)
//!   off + 9..12 = tint rgba
//!   off + 13    = rotation (radians around camera-forward axis)
//!   off + 14    = lod index (unused for billboards; zero)
//!   off + 15    = pad
//! ```
//!
//! All host interaction routes through `wlift_abi`. No `wren_lift`
//! Rust dep — VM layout shifts can't reach plugin code.

#![allow(clippy::missing_safety_doc)]

use wlift_abi::{runtime_error, set_return, slot, typed_array_bytes_mut, typed_array_kind, Value, WrenVm};

/// Plugin ABI handshake — see wlift_gpu / wlift_physics for the
/// rationale. Native-only; the wasm static-link path can't drift.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn req_f32_slice<'a>(
    vm: *mut WrenVm,
    idx: u32,
    label: &str,
    what: &str,
) -> Option<&'a mut [f32]> {
    let bytes = typed_array_bytes_mut(slot(vm, idx))?;
    if typed_array_kind(slot(vm, idx)) != Some(wlift_abi::TypedArrayKind::F32) {
        runtime_error(
            vm,
            &format!("{}: `{}` must be a Float32Array.", label, what),
        );
        return None;
    }
    let len = bytes.len() / 4;
    // SAFETY: typed_array_bytes_mut returned an F32 array's backing
    // buffer (verified by the kind check just above); the byte ptr is
    // 4-byte aligned and `len` reflects the f32 element count.
    Some(unsafe {
        core::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, len)
    })
}

unsafe fn req_u32(vm: *mut WrenVm, idx: u32, label: &str, what: &str) -> Option<u32> {
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
// integrate
// ---------------------------------------------------------------------------

/// `ParticleSim3DCore.integrate(sim, liveCount, params, deaths)` — runs
/// one update step over `liveCount` particles in `sim`.
///
/// Sim slot layout: 8 f32 per slot (see module doc).
///
/// `params` is a Float32Array of 8 floats:
///
/// ```text
///   [0] = dt
///   [1] = gravityX
///   [2] = gravityY
///   [3] = gravityZ
///   [4] = drag
///   [5] = killPlaneOn (0.0 = off, 1.0 = on)
///   [6] = killPlaneY
///   [7] = reserved (currently unused; reserved for future params)
/// ```
///
/// `deaths` is a Float32Array of at least `liveCount * 3 + 2` floats.
/// On return:
///
/// - `deaths[0]` = death count this frame (number of particles that
///   crossed the kill plane OR hit their lifetime ceiling)
/// - `deaths[1]` = reserved (padding so the kill-position stride
///   starts cleanly at index 2)
/// - `deaths[2..2 + deathCount*3]` = death positions (x, y, z) triples
///
/// Returns the new live count (the surviving slot count) via
/// `set_return`. Dead slots are swap-removed (the last live slot fills
/// the gap), matching the legacy Wren `killSlot_` semantics so
/// downstream code (renderer, observers) sees a packed `[0, newLive)`
/// range.
#[no_mangle]
pub unsafe extern "C" fn wlift_particles_integrate(vm: *mut WrenVm) {
    let label = "ParticleSim3DCore.integrate";

    let Some(sim) = req_f32_slice(vm, 1, label, "sim") else {
        return;
    };
    let Some(live_count_arg) = req_u32(vm, 2, label, "liveCount") else {
        return;
    };
    let Some(params) = req_f32_slice(vm, 3, label, "params") else {
        return;
    };
    let Some(deaths) = req_f32_slice(vm, 4, label, "deaths") else {
        return;
    };
    if params.len() < 8 {
        runtime_error(
            vm,
            &format!("{}: `params` must have at least 8 floats.", label),
        );
        return;
    }
    if deaths.len() < 2 {
        runtime_error(
            vm,
            &format!("{}: `deaths` must have at least 2 floats for the sentinel header.", label),
        );
        return;
    }

    let dt          = params[0];
    let gx          = params[1];
    let gy          = params[2];
    let gz          = params[3];
    let drag        = params[4];
    let kp_on       = params[5] != 0.0;
    let kp_y        = params[6];

    // Pre-multiply gravity and drag by dt once — saves four
    // multiplications per particle vs the textbook
    // `v += g*dt - v*drag*dt` form.
    let gxdt   = gx * dt;
    let gydt   = gy * dt;
    let gzdt   = gz * dt;
    let damp_dt = 1.0 - drag * dt;

    let mut live: usize = live_count_arg as usize;
    if live * 8 > sim.len() {
        runtime_error(
            vm,
            &format!(
                "{}: liveCount ({}) exceeds sim capacity ({}).",
                label,
                live,
                sim.len() / 8
            ),
        );
        return;
    }

    let mut death_count: usize = 0;
    let deaths_cap = (deaths.len().saturating_sub(2)) / 3;

    let mut i: usize = 0;
    while i < live {
        let off = i * 8;
        let age = sim[off + 6] + dt;
        let inv_life = sim[off + 7];

        // Lifetime death: age * invLife >= 1.0  iff  age >= lifetime.
        if age * inv_life >= 1.0 {
            if death_count < deaths_cap {
                let d_off = 2 + death_count * 3;
                deaths[d_off]     = sim[off];
                deaths[d_off + 1] = sim[off + 1];
                deaths[d_off + 2] = sim[off + 2];
                death_count += 1;
            }
            // Swap-with-last to keep [0, live) packed.
            live -= 1;
            if i != live {
                let src_off = live * 8;
                for k in 0..8 {
                    sim[off + k] = sim[src_off + k];
                }
            }
            continue;
        }

        // Integrate: v_new = v * (1 - drag*dt) + g*dt; p_new = p + v_new*dt.
        let vx = sim[off + 3];
        let vy = sim[off + 4];
        let vz = sim[off + 5];
        let nvx = vx * damp_dt + gxdt;
        let nvy = vy * damp_dt + gydt;
        let nvz = vz * damp_dt + gzdt;
        let px = sim[off];
        let py = sim[off + 1];
        let pz = sim[off + 2];
        let nx = px + nvx * dt;
        let ny = py + nvy * dt;
        let nz = pz + nvz * dt;

        // Kill-plane crossing — particle crosses y = kp_y this tick.
        // Linearly interpolate the crossing position so death events
        // sit exactly on the plane.
        if kp_on && ny <= kp_y {
            let mut u = 1.0;
            let span = py - ny;
            if span > 1e-5 {
                u = (py - kp_y) / span;
            }
            if u < 0.0 {
                u = 0.0;
            }
            if u > 1.0 {
                u = 1.0;
            }
            let hx = px + (nx - px) * u;
            let hz = pz + (nz - pz) * u;
            if death_count < deaths_cap {
                let d_off = 2 + death_count * 3;
                deaths[d_off]     = hx;
                deaths[d_off + 1] = kp_y;
                deaths[d_off + 2] = hz;
                death_count += 1;
            }
            live -= 1;
            if i != live {
                let src_off = live * 8;
                for k in 0..8 {
                    sim[off + k] = sim[src_off + k];
                }
            }
            continue;
        }

        // Commit the integration.
        sim[off + 3] = nvx;
        sim[off + 4] = nvy;
        sim[off + 5] = nvz;
        sim[off]     = nx;
        sim[off + 1] = ny;
        sim[off + 2] = nz;
        sim[off + 6] = age;
        i += 1;
    }

    deaths[0] = death_count as f32;
    deaths[1] = 0.0;
    set_return(vm, Value::num(live as f64));
}

// ---------------------------------------------------------------------------
// pack
// ---------------------------------------------------------------------------

/// `ParticleSim3DCore.pack(sim, inst, liveCount, params)` — packs
/// `liveCount` live particles from `sim` into `inst` for the next
/// `Renderer3D.drawBillboardN` call.
///
/// `params` is a Float32Array of 16 floats:
///
/// ```text
///   [0..4]   = colorStart rgba
///   [4..8]   = colorDelta rgba (= colorEnd - colorStart, pre-computed)
///   [8]      = sxBase   (configured X width)
///   [9]      = sy       (configured Y height)
///   [10]     = rotation (radians around camera-forward axis)
///   [11]     = widthScaleOn (0.0 = off, 1.0 = on)
///   [12]     = refDist  (reference distance for screen-space width)
///   [13..16] = cameraEye xyz (used when widthScaleOn = 1)
/// ```
///
/// Instance slot constants (UV-rect 0/0/1/1, lodIndex 0, pad 0) are
/// assumed to have been pre-filled at construct time. The hot loop
/// only writes the changing fields (origin, size, rgba, rotation).
#[no_mangle]
pub unsafe extern "C" fn wlift_particles_pack(vm: *mut WrenVm) {
    let label = "ParticleSim3DCore.pack";

    let Some(sim) = req_f32_slice(vm, 1, label, "sim") else {
        return;
    };
    let Some(inst) = req_f32_slice(vm, 2, label, "inst") else {
        return;
    };
    let Some(live_count_arg) = req_u32(vm, 3, label, "liveCount") else {
        return;
    };
    let Some(params) = req_f32_slice(vm, 4, label, "params") else {
        return;
    };
    if params.len() < 16 {
        runtime_error(
            vm,
            &format!("{}: `params` must have at least 16 floats.", label),
        );
        return;
    }

    let live: usize = live_count_arg as usize;
    if live * 8 > sim.len() {
        runtime_error(
            vm,
            &format!(
                "{}: liveCount ({}) exceeds sim capacity ({}).",
                label,
                live,
                sim.len() / 8
            ),
        );
        return;
    }
    if live * 16 > inst.len() {
        runtime_error(
            vm,
            &format!(
                "{}: liveCount ({}) exceeds inst capacity ({}).",
                label,
                live,
                inst.len() / 16
            ),
        );
        return;
    }

    let cs0 = params[0];
    let cs1 = params[1];
    let cs2 = params[2];
    let cs3 = params[3];
    let cd0 = params[4];
    let cd1 = params[5];
    let cd2 = params[6];
    let cd3 = params[7];
    let sx_base = params[8];
    let sy      = params[9];
    let rot     = params[10];
    let width_scale_on = params[11] != 0.0;
    let ref_dist = params[12];
    let ex = params[13];
    let ey = params[14];
    let ez = params[15];

    let mut sim_off: usize = 0;
    let mut off:     usize = 0;
    let mut i: usize = 0;
    while i < live {
        let age = sim[sim_off + 6];
        let inv_life = sim[sim_off + 7];
        // t = age * invLife — multiplication avoids the per-particle
        // divide in `age / lifetime`. Clamped because update guarantees
        // the [0, 1) range under normal operation, but a freshly
        // spawned slot in the same frame may briefly have t very near
        // 0 with floating noise.
        let mut t = age * inv_life;
        if t < 0.0 { t = 0.0; }
        if t > 1.0 { t = 1.0; }

        let r = cs0 + cd0 * t;
        let g = cs1 + cd1 * t;
        let b = cs2 + cd2 * t;
        let mut a = cs3 + cd3 * t;

        let px = sim[sim_off];
        let py = sim[sim_off + 1];
        let pz = sim[sim_off + 2];

        let mut sx = sx_base;
        if width_scale_on {
            let dx = px - ex;
            let dy = py - ey;
            let dz = pz - ez;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            // Square-root falloff with 22% floor — see particles.wren
            // for the rationale (matches the legacy Wren behaviour
            // exactly so the visual output doesn't shift).
            let mut lin = dist / ref_dist;
            if lin > 1.0 { lin = 1.0; }
            let mut scale = lin.sqrt();
            if scale < 0.22 { scale = 0.22; }
            sx = sx_base * scale;
            // Atmospheric alpha fade: closer streaks at 65% of base,
            // reference-distance streaks at full alpha.
            a = a * (0.65 + 0.35 * lin);
        }

        inst[off]      = px;
        inst[off + 1]  = py;
        inst[off + 2]  = pz;
        inst[off + 3]  = sx;
        inst[off + 4]  = sy;
        // Slots 5..8 (UV-rect 0,0,1,1), 14 (lodIndex 0), 15 (pad 0)
        // are pre-filled at construct in the Wren wrapper.
        inst[off + 9]  = r;
        inst[off + 10] = g;
        inst[off + 11] = b;
        inst[off + 12] = a;
        inst[off + 13] = rot;

        sim_off += 8;
        off     += 16;
        i       += 1;
    }

    set_return(vm, Value::num(live as f64));
}
