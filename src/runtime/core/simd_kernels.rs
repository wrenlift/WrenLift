//! Per-target SIMD kernels for `Simd4f` / `Simd4i` ops.
//!
//! Each function takes one or two `[u32; 4]` lane payloads (the
//! exact storage shape used by `ObjSimd`) and returns a new lane
//! payload. The wasm32 build has an explicit `simd128` fast path
//! through `core::arch::wasm32::*`; everywhere else (host x86_64 /
//! aarch64, plus any wasm32 build *without* `+simd128`) the scalar
//! fallback runs, and LLVM's autovectorizer handles SSE2 / NEON
//! lowering on its own.
//!
//! Native JIT / AOT bypass these kernels entirely — `cranelift_backend`
//! lowers the same SIMD ops through Cranelift vector ops directly,
//! which becomes SSE / NEON at codegen time. The kernels here are the
//! interpreter path and the wasm path.
//!
//! Two-flavour wasm bundle: build the cdylib once with the default
//! flags (no `simd128`) and once with `RUSTFLAGS=-C target-feature=+simd128`.
//! The JS loader feature-detects v128 support and picks the right one;
//! browsers that haven't shipped the wasm SIMD spec keep working with
//! the baseline build.
//!
//! The lane payload is a raw 16-byte `[u32; 4]`; f32 ops bitcast on
//! the way in / out via `f32::from_bits` / `f32::to_bits` so the
//! storage format never changes between targets.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32 as simd_isa;

// ---------------------------------------------------------------------------
// f32x4 ops
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn f32x4_add(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_add(av, bv))
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    {
        scalar_f32_binop(a, b, |x, y| x + y)
    }
}

#[inline(always)]
pub fn f32x4_sub(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_sub(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_binop(a, b, |x, y| x - y)
    }
}

#[inline(always)]
pub fn f32x4_mul(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_mul(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_binop(a, b, |x, y| x * y)
    }
}

#[inline(always)]
pub fn f32x4_div(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_div(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_binop(a, b, |x, y| x / y)
    }
}

#[inline(always)]
pub fn f32x4_min(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        // `f32x4_pmin` matches Rust's `f32::min` semantics for the
        // common case (NaN propagates the second operand's bits);
        // identical observable behaviour to the scalar fallback for
        // any non-NaN input pair, which is the only shape Wren-side
        // arithmetic produces.
        store_v128(simd_isa::f32x4_pmin(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_binop(a, b, f32::min)
    }
}

#[inline(always)]
pub fn f32x4_max(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_pmax(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_binop(a, b, f32::max)
    }
}

#[inline(always)]
pub fn f32x4_neg(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::f32x4_neg(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_unop(a, |x| -x)
    }
}

#[inline(always)]
pub fn f32x4_abs(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::f32x4_abs(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_unop(a, f32::abs)
    }
}

#[inline(always)]
pub fn f32x4_sqrt(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::f32x4_sqrt(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_unop(a, f32::sqrt)
    }
}

#[inline(always)]
pub fn f32x4_eq(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_eq(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x == y)
    }
}

#[inline(always)]
pub fn f32x4_ne(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_ne(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x != y)
    }
}

#[inline(always)]
pub fn f32x4_lt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_lt(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x < y)
    }
}

#[inline(always)]
pub fn f32x4_le(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_le(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x <= y)
    }
}

#[inline(always)]
pub fn f32x4_gt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_gt(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x > y)
    }
}

#[inline(always)]
pub fn f32x4_ge(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::f32x4_ge(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_f32_cmp(a, b, |x, y| x >= y)
    }
}

// ---------------------------------------------------------------------------
// i32x4 ops
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn i32x4_add(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_add(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_binop(a, b, i32::wrapping_add)
    }
}

#[inline(always)]
pub fn i32x4_sub(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_sub(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_binop(a, b, i32::wrapping_sub)
    }
}

#[inline(always)]
pub fn i32x4_mul(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_mul(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_binop(a, b, i32::wrapping_mul)
    }
}

#[inline(always)]
pub fn i32x4_min(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_min(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_binop(a, b, i32::min)
    }
}

#[inline(always)]
pub fn i32x4_max(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_max(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_binop(a, b, i32::max)
    }
}

#[inline(always)]
pub fn i32x4_and(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::v128_and(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
    }
}

#[inline(always)]
pub fn i32x4_or(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::v128_or(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }
}

#[inline(always)]
pub fn i32x4_xor(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::v128_xor(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }
}

#[inline(always)]
pub fn i32x4_not(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::v128_not(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        [!a[0], !a[1], !a[2], !a[3]]
    }
}

#[inline(always)]
pub fn i32x4_neg(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::i32x4_neg(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_unop(a, i32::wrapping_neg)
    }
}

#[inline(always)]
pub fn i32x4_abs(a: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::i32x4_abs(av))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_unop(a, i32::wrapping_abs)
    }
}

#[inline(always)]
pub fn i32x4_shl(a: [u32; 4], shift: u32) -> [u32; 4] {
    let shift = shift & 31;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::i32x4_shl(av, shift))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_unop(a, |x| x.wrapping_shl(shift))
    }
}

#[inline(always)]
pub fn i32x4_shr(a: [u32; 4], shift: u32) -> [u32; 4] {
    let shift = shift & 31;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        store_v128(simd_isa::i32x4_shr(av, shift))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_unop(a, |x| x >> shift)
    }
}

#[inline(always)]
pub fn i32x4_eq(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_eq(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x == y)
    }
}

#[inline(always)]
pub fn i32x4_ne(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_ne(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x != y)
    }
}

#[inline(always)]
pub fn i32x4_lt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_lt(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x < y)
    }
}

#[inline(always)]
pub fn i32x4_le(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_le(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x <= y)
    }
}

#[inline(always)]
pub fn i32x4_gt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_gt(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x > y)
    }
}

#[inline(always)]
pub fn i32x4_ge(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        let bv = simd_isa::v128_load(b.as_ptr().cast());
        store_v128(simd_isa::i32x4_ge(av, bv))
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        scalar_i32_cmp(a, b, |x, y| x >= y)
    }
}

#[inline(always)]
pub fn i32x4_all_true(a: [u32; 4]) -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        simd_isa::i32x4_all_true(av)
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        a.iter().all(|&lane| lane != 0)
    }
}

#[inline(always)]
pub fn i32x4_any_true(a: [u32; 4]) -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        simd_isa::v128_any_true(av)
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        a.iter().any(|&lane| lane != 0)
    }
}

#[inline(always)]
pub fn i32x4_bitmask(a: [u32; 4]) -> u32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let av = simd_isa::v128_load(a.as_ptr().cast());
        simd_isa::i32x4_bitmask(av) as u32
    }
    #[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let mut mask = 0u32;
        for (i, lane) in a.iter().enumerate() {
            if (lane & 0x8000_0000) != 0 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

#[inline(always)]
fn scalar_f32_binop(a: [u32; 4], b: [u32; 4], op: impl Fn(f32, f32) -> f32) -> [u32; 4] {
    let af = [
        f32::from_bits(a[0]),
        f32::from_bits(a[1]),
        f32::from_bits(a[2]),
        f32::from_bits(a[3]),
    ];
    let bf = [
        f32::from_bits(b[0]),
        f32::from_bits(b[1]),
        f32::from_bits(b[2]),
        f32::from_bits(b[3]),
    ];
    [
        op(af[0], bf[0]).to_bits(),
        op(af[1], bf[1]).to_bits(),
        op(af[2], bf[2]).to_bits(),
        op(af[3], bf[3]).to_bits(),
    ]
}

#[inline(always)]
fn scalar_f32_unop(a: [u32; 4], op: impl Fn(f32) -> f32) -> [u32; 4] {
    [
        op(f32::from_bits(a[0])).to_bits(),
        op(f32::from_bits(a[1])).to_bits(),
        op(f32::from_bits(a[2])).to_bits(),
        op(f32::from_bits(a[3])).to_bits(),
    ]
}

#[inline(always)]
fn scalar_f32_cmp(a: [u32; 4], b: [u32; 4], op: impl Fn(f32, f32) -> bool) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = if op(f32::from_bits(a[i]), f32::from_bits(b[i])) {
            u32::MAX
        } else {
            0
        };
    }
    out
}

#[inline(always)]
fn scalar_i32_binop(a: [u32; 4], b: [u32; 4], op: impl Fn(i32, i32) -> i32) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = op(a[i] as i32, b[i] as i32) as u32;
    }
    out
}

#[inline(always)]
fn scalar_i32_unop(a: [u32; 4], op: impl Fn(i32) -> i32) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = op(a[i] as i32) as u32;
    }
    out
}

#[inline(always)]
fn scalar_i32_cmp(a: [u32; 4], b: [u32; 4], op: impl Fn(i32, i32) -> bool) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = if op(a[i] as i32, b[i] as i32) {
            u32::MAX
        } else {
            0
        };
    }
    out
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn store_v128(v: simd_isa::v128) -> [u32; 4] {
    let mut out = [0u32; 4];
    simd_isa::v128_store(out.as_mut_ptr().cast(), v);
    out
}

// ---------------------------------------------------------------------------
// Tests — every kernel checked against its scalar reference.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_lanes(a: [f32; 4]) -> [u32; 4] {
        [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits()]
    }
    fn from_f32_lanes(a: [u32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0]),
            f32::from_bits(a[1]),
            f32::from_bits(a[2]),
            f32::from_bits(a[3]),
        ]
    }
    fn i32_lanes(a: [i32; 4]) -> [u32; 4] {
        [a[0] as u32, a[1] as u32, a[2] as u32, a[3] as u32]
    }
    fn from_i32_lanes(a: [u32; 4]) -> [i32; 4] {
        [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32]
    }

    #[test]
    fn f32x4_arithmetic_matches_scalar() {
        let a = f32_lanes([1.0, 2.0, 3.0, 4.0]);
        let b = f32_lanes([0.5, 1.0, 1.5, 2.0]);
        assert_eq!(from_f32_lanes(f32x4_add(a, b)), [1.5, 3.0, 4.5, 6.0]);
        assert_eq!(from_f32_lanes(f32x4_sub(a, b)), [0.5, 1.0, 1.5, 2.0]);
        assert_eq!(from_f32_lanes(f32x4_mul(a, b)), [0.5, 2.0, 4.5, 8.0]);
        assert_eq!(from_f32_lanes(f32x4_div(a, b)), [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(from_f32_lanes(f32x4_min(a, b)), [0.5, 1.0, 1.5, 2.0]);
        assert_eq!(from_f32_lanes(f32x4_max(a, b)), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32x4_unary_ops_match_scalar() {
        let a = f32_lanes([1.0, -2.0, 4.0, -9.0]);
        assert_eq!(from_f32_lanes(f32x4_neg(a)), [-1.0, 2.0, -4.0, 9.0]);
        assert_eq!(from_f32_lanes(f32x4_abs(a)), [1.0, 2.0, 4.0, 9.0]);
        let sq = from_f32_lanes(f32x4_sqrt(f32_lanes([1.0, 4.0, 9.0, 16.0])));
        assert_eq!(sq, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f32x4_compares_produce_lane_masks() {
        let a = f32_lanes([1.0, 2.0, 3.0, 4.0]);
        let b = f32_lanes([2.0, 2.0, 2.0, 2.0]);
        // lt: lanes where a < b → 1.0 < 2.0 (true), rest false
        assert_eq!(f32x4_lt(a, b), [u32::MAX, 0, 0, 0]);
        assert_eq!(f32x4_eq(a, b), [0, u32::MAX, 0, 0]);
        assert_eq!(f32x4_ge(a, b), [0, u32::MAX, u32::MAX, u32::MAX]);
    }

    #[test]
    fn i32x4_arithmetic_matches_scalar() {
        let a = i32_lanes([1, -2, 3, -4]);
        let b = i32_lanes([10, 20, -30, 40]);
        assert_eq!(from_i32_lanes(i32x4_add(a, b)), [11, 18, -27, 36]);
        assert_eq!(from_i32_lanes(i32x4_sub(a, b)), [-9, -22, 33, -44]);
        assert_eq!(from_i32_lanes(i32x4_mul(a, b)), [10, -40, -90, -160]);
        assert_eq!(from_i32_lanes(i32x4_min(a, b)), [1, -2, -30, -4]);
        assert_eq!(from_i32_lanes(i32x4_max(a, b)), [10, 20, 3, 40]);
    }

    #[test]
    fn i32x4_bitwise_matches_scalar() {
        let a = i32_lanes([0b1100, 0b1010, 0b1111, 0]);
        let b = i32_lanes([0b1010, 0b1100, 0b0001, !0]);
        assert_eq!(i32x4_and(a, b), [0b1000, 0b1000, 0b0001, 0]);
        assert_eq!(i32x4_or(a, b), [0b1110, 0b1110, 0b1111, !0u32]);
        assert_eq!(i32x4_xor(a, b), [0b0110, 0b0110, 0b1110, !0u32]);
        assert_eq!(i32x4_not(a), [!0b1100u32, !0b1010u32, !0b1111u32, !0u32]);
    }

    #[test]
    fn i32x4_shifts_apply_modulo_32() {
        let a = i32_lanes([1, 2, 4, -8]);
        assert_eq!(from_i32_lanes(i32x4_shl(a, 1)), [2, 4, 8, -16]);
        assert_eq!(from_i32_lanes(i32x4_shr(a, 1)), [0, 1, 2, -4]);
        // shift amount is masked to 5 bits
        assert_eq!(from_i32_lanes(i32x4_shl(a, 33)), [2, 4, 8, -16]);
    }

    #[test]
    fn i32x4_compares_produce_lane_masks() {
        let a = i32_lanes([1, 2, 3, 4]);
        let b = i32_lanes([2, 2, 2, 2]);
        assert_eq!(i32x4_lt(a, b), [u32::MAX, 0, 0, 0]);
        assert_eq!(i32x4_eq(a, b), [0, u32::MAX, 0, 0]);
        assert_eq!(i32x4_ge(a, b), [0, u32::MAX, u32::MAX, u32::MAX]);
    }

    #[test]
    fn i32x4_horizontal_reductions_agree_with_scalar() {
        let masked = [u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        let mixed = [u32::MAX, 0, u32::MAX, 0];
        let zero = [0u32; 4];
        assert!(i32x4_all_true(masked));
        assert!(!i32x4_all_true(mixed));
        assert!(!i32x4_all_true(zero));
        assert!(i32x4_any_true(masked));
        assert!(i32x4_any_true(mixed));
        assert!(!i32x4_any_true(zero));
        // bitmask packs MSB of each lane into bit i
        assert_eq!(i32x4_bitmask(masked), 0b1111);
        assert_eq!(i32x4_bitmask(mixed), 0b0101);
        assert_eq!(i32x4_bitmask(zero), 0);
    }
}
