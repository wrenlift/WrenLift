/**
 * @enum {0 | 1 | 2}
 */
export const ErrorKind = Object.freeze({
    None: 0, "0": "None",
    CompileError: 1, "1": "CompileError",
    RuntimeError: 2, "2": "RuntimeError",
});

/**
 * @enum {0 | 1 | 2}
 */
export const FutureState = Object.freeze({
    Pending: 0, "0": "Pending",
    Resolved: 1, "1": "Resolved",
    Rejected: 2, "2": "Rejected",
});

/**
 * Result of a single `run` call, exported to JS as a structural
 * object via `wasm_bindgen`'s getter convention.
 */
export class RunResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(RunResult.prototype);
        obj.__wbg_ptr = ptr;
        RunResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RunResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_runresult_free(ptr, 0);
    }
    /**
     * @returns {ErrorKind}
     */
    get errorKind() {
        const ret = wasm.runresult_errorKind(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get ok() {
        const ret = wasm.runresult_ok(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {string}
     */
    get output() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.runresult_output(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) RunResult.prototype[Symbol.dispose] = RunResult.prototype.free;

/**
 * @enum {0 | 1 | 2}
 */
export const WebSocketState = Object.freeze({
    Connecting: 0, "0": "Connecting",
    Open: 1, "1": "Open",
    Closed: 2, "2": "Closed",
});

/**
 * One-time setup. The wasm-bindgen `start` attribute makes this
 * run on module instantiation, before any JS-side `run()` /
 * `version()` call.
 *
 * Two jobs:
 *
 *   * Install `console_error_panic_hook` so panics surface as a
 *     readable JS `console.error` instead of the bare
 *     "unreachable executed" trap.
 *
 *   * Publish every statically-linked plugin's foreign-method
 *     symbols to the runtime's foreign-method registry. This is
 *     the wasm replacement for the host's `libloading::dlsym`
 *     pass — Wren `foreign class` declarations annotated with
 *     `#!native = "wlift_image"` look up here at install time.
 */
export function _wasm_init() {
    wasm._wasm_init();
}

/**
 * JS-callable: read the current state of a handle. Used by
 * JS-side debugging / introspection tooling AND by the
 * `browser::browser_peek_state` foreign method that Wren's
 * `Future.await` polls in its yield loop.
 * @param {number} handle
 * @returns {FutureState}
 */
export function future_state(handle) {
    const ret = wasm.future_state(handle);
    return ret;
}

/**
 * Number of MIR functions successfully compiled to wasm.
 * @returns {number}
 */
export function jit_compile_count() {
    const ret = wasm.jit_compile_count();
    return ret >>> 0;
}

/**
 * Number of MIR functions rejected by the helper-set gate.
 * @returns {number}
 */
export function jit_compile_reject_count() {
    const ret = wasm.jit_compile_reject_count();
    return ret >>> 0;
}

/**
 * Reset all counters — handy for per-run measurements.
 * Doesn't touch the runtime's `dispatch_hook_hits` counter
 * (it's `pub fn` in tier_wasm.rs without a reset hook); read
 * the delta yourself if you need a per-run number.
 */
export function jit_counters_reset() {
    wasm.jit_counters_reset();
}

/**
 * Number of dispatches via `wren_call_1`'s short-circuit
 * (JIT'd code calling another JIT'd function directly,
 * without re-entering the BC interp). High value = recursion
 * stays in JIT; low value with high BC count = the alternating
 * JIT/BC pattern we're trying to avoid.
 * @returns {bigint}
 */
export function jit_dispatch_fast_path_count() {
    const ret = wasm.jit_dispatch_fast_path_count();
    return BigInt.asUintN(64, ret);
}

/**
 * Number of dispatches via the BC interp's hook (a normal
 * `Op::Call` in interpreted code that found a JIT'd slot).
 * @returns {bigint}
 */
export function jit_dispatch_from_bc_count() {
    const ret = wasm.jit_dispatch_from_bc_count();
    return BigInt.asUintN(64, ret);
}

/**
 * Total times the BC interpreter's wasm dispatch hook ran —
 * i.e. how many Wren closure-method calls reached
 * `dispatch_closure_bc_inner`'s wasm-only block. If this stays
 * 0 while a script runs, closure dispatch is going through a
 * path that bypasses the hook (rare but possible — e.g. if a
 * call is intercepted by an earlier match arm in `Op::Call`).
 * @returns {bigint}
 */
export function jit_dispatch_hook_hits() {
    const ret = wasm.jit_dispatch_hook_hits();
    return BigInt.asUintN(64, ret);
}

/**
 * Phase 1 — emit wasm bytes for the *first* MIR function
 * produced by compiling `source`. Lets the host smoke-test
 * codegen on real Wren code (e.g. a simple math function),
 * not just hand-built MIR.
 *
 * Returns an empty Vec on any compile / emit failure — the
 * host can `bytes.length === 0` to detect.
 * @param {string} source
 * @returns {Uint8Array}
 */
export function jit_emit_from_source(source) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jit_emit_from_source(ptr0, len0);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Phase 2b smoke — emit the const-42 module, hand bytes to the
 * JS instantiate shim, call the resulting function via the
 * JS call-0 shim, return the raw u64. Returns `0` on any
 * failure (cleanest signal for the JS test code).
 * @returns {bigint}
 */
export function jit_smoke_run_const() {
    const ret = wasm.jit_smoke_run_const();
    return BigInt.asUintN(64, ret);
}

/**
 * Phase 1 smoke test — emit a hand-built MIR function that
 * returns `42` as a NaN-boxed `Value` and hand back the wasm
 * bytes. JS-side test:
 *
 * ```js
 * const bytes = wlift_wasm.jit_test_emit();
 * const mod   = new WebAssembly.Module(bytes);
 * const inst  = new WebAssembly.Instance(mod, { wren: {} });
 * const r     = inst.exports.fn_0();
 * // r is a BigInt — 42.0's NaN-boxed bits = 0x4045000000000000
 * console.log(r === BigInt.asIntN(64, 0x4045000000000000n));
 * ```
 *
 * `fn_0` is the export name `emit_mir` uses for this MIR's
 * `name.index() == 0`.
 * @returns {Uint8Array}
 */
export function jit_test_emit() {
    const ret = wasm.jit_test_emit();
    var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v1;
}

/**
 * @param {string} text
 * @returns {any}
 */
export function parse_hatchfile_toml(text) {
    const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.parse_hatchfile_toml(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Inspect a `.hatch` byte stream and return its manifest as a
 * JS object — same shape as `parse_hatchfile_toml`. Used by
 * the JS-side dep walker to discover transitive `[dependencies]`
 * without installing the bundle.
 * @param {Uint8Array} bytes
 * @returns {any}
 */
export function peek_manifest(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.peek_manifest(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Number of lines `run()` prepends to the user source. Today
 * it's just the one-line `PRELUDE_IMPORT`. Exposed so callers
 * can shift error spans back to user-source line numbers — the
 * big `BROWSER_PRELUDE` is loaded as a separate module and no
 * longer inflates the user module's line count.
 * @returns {number}
 */
export function prelude_line_count() {
    const ret = wasm.prelude_line_count();
    return ret >>> 0;
}

/**
 * JS-callable: flip a handle to `Rejected` with an error message.
 * @param {number} handle
 * @param {string} error
 */
export function reject_future(handle, error) {
    const ptr0 = passStringToWasm0(error, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.reject_future(handle, ptr0, len0);
}

/**
 * JS-callable: flip a handle to `Resolved` and stash the value.
 * Idempotent — resolving an already-resolved/rejected handle is
 * a no-op (matches Promise semantics; the first settlement wins).
 * @param {number} handle
 * @param {string} value
 */
export function resolve_future(handle, value) {
    const ptr0 = passStringToWasm0(value, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.resolve_future(handle, ptr0, len0);
}

/**
 * @param {string} source
 * @returns {Promise<RunResult>}
 */
export function run(source) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run(ptr0, len0);
    return ret;
}

/**
 * Run a `.hatch` bundle in the playground. Same setup as `run`
 * (fresh VM, prelude install, scheduler loop) but feeds the
 * user-code phase a hatch byte stream instead of source. Modules
 * inside the hatch install in their declared order — so a hatch
 * produced via `wlift src/ --bundle out.hatch --bundle-target
 * wasm32-unknown-unknown` runs end-to-end here, including any
 * in-bundle `import` between modules (cross-module resolution
 * in wasm).
 *
 * `@hatch:*` external package resolution is *not* implemented
 * here — the wasm runtime can't synchronously fetch packages
 * during compilation. Use the host's `hatch` CLI (or a
 * JS-side prefetch + multiple `install_hatch` calls, when that
 * API lands) to assemble the closure of needed packages, then
 * pass the result here as a single bundle.
 * @param {Uint8Array} bytes
 * @returns {Promise<RunResult>}
 */
export function run_hatch(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run_hatch(ptr0, len0);
    return ret;
}

/**
 * Run user source after pre-installing a list of `@hatch:*`
 * dependency bundles. The JS host is responsible for fetching
 * each dep's bytes (typically by scanning `import "@hatch:..."`
 * patterns in the source and pulling matching `.hatch` files
 * from a CDN); this entry point installs them before the user
 * source's parser hits an `import "@hatch:foo"` line, so the
 * import resolves against an already-loaded module.
 *
 * `deps` is a JS `Array` of `Uint8Array`. Each element is one
 * `.hatch` byte stream. Order matters only for transitive
 * dependencies — a hatch that imports another hatch must come
 * after its dep in the array. The JS-side helper that does the
 * fetch is responsible for that ordering (topological sort over
 * the dep graph).
 * @param {string} source
 * @param {Array<any>} deps
 * @returns {Promise<RunResult>}
 */
export function run_with_hatches(source, deps) {
    const ptr0 = passStringToWasm0(source, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.run_with_hatches(ptr0, len0, deps);
    return ret;
}

/**
 * Build identifier — hard-coded for the moment so JS can sanity-
 * check the loaded wasm matches what its bundler thought it was
 * importing.
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * JS-callable: read the current state of a socket. Used by JS
 * debug tooling and by `browser::browser_ws_state` if a Wren-
 * side state probe ends up wanted.
 * @param {number} handle
 * @returns {WebSocketState}
 */
export function websocket_state(handle) {
    const ret = wasm.websocket_state(handle);
    return ret;
}

/**
 * Wasm-bindgen export. Decode a `.hatch` bundle and surface every
 * wasm-targeted NativeLib section as `{lib, bytes}` so JS can
 * instantiate each plugin module before installing the bundle's
 * Wlbc / Source sections (foreign-class declarations need the
 * plugin's symbols already registered when they bind).
 *
 * Returns an empty array on parse failure rather than throwing —
 * the playground falls back to the no-plugin path naturally.
 * @param {Uint8Array} bytes
 * @returns {Array<any>}
 */
export function wlift_extract_wasm_plugins(bytes) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wlift_extract_wasm_plugins(ptr0, len0);
    return ret;
}

/**
 * Allocate `len` bytes inside the host's linear memory and
 * return a pointer. Used by the JS-side bridge fns
 * (`wlift_set_slot_str` etc.) to land plugin-supplied bytes in
 * the host's address space before calling Wren-API functions
 * that read from there. The companion `wlift_host_free` returns
 * the bytes when the caller is done.
 * @param {number} len
 * @returns {number}
 */
export function wlift_host_alloc(len) {
    const ret = wasm.wlift_host_alloc(len);
    return ret >>> 0;
}

/**
 * Counterpart to [`wlift_host_alloc`]. Reconstructs the `Vec`
 * from the raw pointer + length so it drops normally.
 * @param {number} ptr
 * @param {number} len
 */
export function wlift_host_free(ptr, len) {
    wasm.wlift_host_free(ptr, len);
}

/**
 * Wasm-bindgen export. JS calls this once per foreign-method
 * export discovered in a plugin module. The returned `idx` is
 * what JS keys its `(plugin_instance, export_name)` map with;
 * the same idx ends up in `Method::ForeignCDynamic(idx)` and
 * rides back through `env::wlift_dispatch_dynamic_plugin` at
 * call time.
 * @param {string} library
 * @param {string} symbol
 * @returns {number}
 */
export function wlift_register_plugin_dynamic_export(library, symbol) {
    const ptr0 = passStringToWasm0(library, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(symbol, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.wlift_register_plugin_dynamic_export(ptr0, len0, ptr1, len1);
    return ret >>> 0;
}

/**
 * @param {bigint} receiver_bits
 * @param {bigint} method_id
 * @param {bigint} arg_bits
 * @returns {bigint}
 */
export function wren_call_1(receiver_bits, method_id, arg_bits) {
    const ret = wasm.wren_call_1(receiver_bits, method_id, arg_bits);
    return BigInt.asUintN(64, ret);
}

/**
 * `wren_call_1_slow` — fallback for `wren_jit_slot_plus_one`'s
 * `0` case. Same body as the original `wren_call_1`, just renamed
 * so emit_mir can pick between fast (call_indirect) and slow
 * (this) without name collision. JIT'd code with `wren_call_1`
 * imports still works — it's an alias for the slow path so
 * pre-Phase-5 modules continue dispatching correctly.
 * @param {bigint} receiver_bits
 * @param {bigint} method_id
 * @param {bigint} arg_bits
 * @returns {bigint}
 */
export function wren_call_1_slow(receiver_bits, method_id, arg_bits) {
    const ret = wasm.wren_call_1_slow(receiver_bits, method_id, arg_bits);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_eq(a, b) {
    const ret = wasm.wren_cmp_eq(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_ge(a, b) {
    const ret = wasm.wren_cmp_ge(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_gt(a, b) {
    const ret = wasm.wren_cmp_gt(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_le(a, b) {
    const ret = wasm.wren_cmp_le(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_lt(a, b) {
    const ret = wasm.wren_cmp_lt(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_cmp_ne(a, b) {
    const ret = wasm.wren_cmp_ne(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} slot_idx
 * @returns {bigint}
 */
export function wren_get_module_var(slot_idx) {
    const ret = wasm.wren_get_module_var(slot_idx);
    return BigInt.asUintN(64, ret);
}

/**
 * Truthiness probe — emit_mir uses this for `if`/`while` tests
 * on boxed values. Returns `1` for truthy, `0` for falsy. Note
 * the result type is i32 (a wasm bool) not i64 — the codegen
 * wires it directly into a `br_if` so an i64 boxed bool would
 * need an extra unboxing step.
 * @param {bigint} a
 * @returns {number}
 */
export function wren_is_truthy(a) {
    const ret = wasm.wren_is_truthy(a);
    return ret >>> 0;
}

/**
 * @param {bigint} bits
 */
export function wren_jit_root_push(bits) {
    wasm.wren_jit_root_push(bits);
}

/**
 * @param {number} len
 */
export function wren_jit_roots_restore_len(len) {
    wasm.wren_jit_roots_restore_len(len);
}

/**
 * @returns {number}
 */
export function wren_jit_roots_snapshot_len() {
    const ret = wasm.wren_jit_roots_snapshot_len();
    return ret >>> 0;
}

/**
 * Phase 5d combined helper — load the closure stored in
 * `module_vars["main"][idx]` *and* return its JIT slot+1 in a
 * single cross-instance call. The function-prologue lookup
 * emitted by `codegen::wasm::emit_function` was previously
 * two hops (`wren_get_module_var` then `wren_jit_slot_plus_one`);
 * merging them halves the prologue's cross-instance overhead,
 * which matters because the prologue runs once per outer call
 * (fib(20) → ~22k invocations).
 *
 * Returns 0 if the module / var / closure isn't JIT'd, mirroring
 * the `slot + 1` encoding of `wren_jit_slot_plus_one`.
 * @param {bigint} slot_idx
 * @returns {number}
 */
export function wren_jit_slot_for_module_var(slot_idx) {
    const ret = wasm.wren_jit_slot_for_module_var(slot_idx);
    return ret >>> 0;
}

/**
 * One-arg method call — `receiver.method(arg)`.
 *
 * Phase 4 step 2: emit_mir lowers `Call` instructions to
 * `wren_call_<argc>` imports. The 1-arg variant is enough to
 * unlock self-recursive numeric code (`fib.call(n)`,
 * `factorial.call(n)`, etc.). Higher arities follow the same
 * template; the runtime's `call_method_on` handles any arity.
 *
 * Method dispatch:
 *   * `method_id` is the `SymbolId` index emit_mir baked in
 *     when compiling — `mir.name.index() as i64`. Resolves
 *     against the live VM's interner (must be the same VM
 *     that compiled the MIR; we can't migrate slots across
 *     VMs).
 *   * `call_method_on` short-circuits to `call_closure_sync`
 *     when the receiver is a Closure and the method name
 *     starts with `call`. That's the hot path for
 *     `fib.call(n)`-style recursion.
 *
 * **GC SAFETY (Phase 4 step 4 territory):** the JIT'd caller
 * holds NaN-boxed `Value`s in wasm locals while this runs.
 * The wlift GC has no visibility into wasm locals — if a
 * callee allocates and triggers a GC pass, those locals can
 * dangle. fib + similar pure-arithmetic recursive code
 * doesn't allocate and is safe. **Do not tier up code that
 * allocates inside its hot loop until step 4 lands.** The
 * MIR reject list catches most allocating instructions
 * (MakeList / MakeMap / StringConcat / ToString); the gap is
 * callees reached via `Call` that themselves allocate.
 * `wren_jit_slot_plus_one(receiver) -> i32` — Phase 5 helper.
 *
 * Look up the JIT slot for a closure receiver, returning
 * `slot + 1` (so `0` means "no JIT, take the slow path"). Used
 * by emit_mir's `Call` lowering to decide between
 * `call_indirect` (fast) and `wren_call_1_slow` (slow). Single
 * wasm-to-wasm cross-module call per Call site — no JS hop.
 *
 * The `+ 1` encoding lets the caller emit a single
 * `i32.eqz`-based branch rather than a sentinel comparison
 * against `-1` or similar.
 * @param {bigint} receiver_bits
 * @returns {number}
 */
export function wren_jit_slot_plus_one(receiver_bits) {
    const ret = wasm.wren_jit_slot_plus_one(receiver_bits);
    return ret >>> 0;
}

/**
 * @param {bigint} a
 * @returns {bigint}
 */
export function wren_not(a) {
    const ret = wasm.wren_not(a);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_num_add(a, b) {
    const ret = wasm.wren_num_add(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_num_div(a, b) {
    const ret = wasm.wren_num_div(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_num_mod(a, b) {
    const ret = wasm.wren_num_mod(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_num_mul(a, b) {
    const ret = wasm.wren_num_mul(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @returns {bigint}
 */
export function wren_num_neg(a) {
    const ret = wasm.wren_num_neg(a);
    return BigInt.asUintN(64, ret);
}

/**
 * @param {bigint} a
 * @param {bigint} b
 * @returns {bigint}
 */
export function wren_num_sub(a, b) {
    const ret = wasm.wren_num_sub(a, b);
    return BigInt.asUintN(64, ret);
}

/**
 * JS-callable: flip a socket to `Closed` and reject any parked
 * `recv()` futures. Idempotent — closing twice is a no-op.
 * @param {number} handle
 */
export function ws_close(handle) {
    wasm.ws_close(handle);
}

/**
 * JS-callable: deliver an incoming message. Wakes the head
 * `recv()` future if one is parked; otherwise buffers the
 * message for a later caller. Drops the message silently if the
 * handle is unknown — late arrivals on a closed/torn-down socket
 * shouldn't crash the page.
 * @param {number} handle
 * @param {string} msg
 */
export function ws_message(handle, msg) {
    const ptr0 = passStringToWasm0(msg, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.ws_message(handle, ptr0, len0);
}

/**
 * JS-callable: flip a socket to `Open`. Browser hosts call this
 * from the `WebSocket`'s `open` listener so subsequent `send`
 * calls aren't queued in JS-land for nothing.
 * @param {number} handle
 */
export function ws_open(handle) {
    wasm.ws_open(handle);
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_83742b46f01ce22d: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_3c846841762788c1: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_string_7ef6b97b02428fae: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_52709e72fb9f179c: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_6b5b6b8576d35cb1: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg__wlift_dom_add_class_cff7ce5749bda8b2: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_dom_add_class(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_dom_get_attribute_a60d5afe6268a6e8: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_dom_get_attribute(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_dom_query_all_974332d80ef3febe: function(arg0, arg1, arg2) {
            globalThis._wlift_dom_query_all(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_dom_remove_class_2d9ca4a1199e7cfd: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_dom_remove_class(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_dom_set_attribute_f37999638ed06f7e: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            globalThis._wlift_dom_set_attribute(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4), getStringFromWasm0(arg5, arg6));
        },
        __wbg__wlift_dom_set_text_a659bbf8d9719299: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_dom_set_text(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_dom_text_6d119a35df8b8b48: function(arg0, arg1, arg2) {
            globalThis._wlift_dom_text(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_fetch_ea0f744df833158f: function(arg0, arg1, arg2) {
            globalThis._wlift_fetch(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_jit_call_0_3c4a5763627598cf: function(arg0) {
            const ret = globalThis._wlift_jit_call_0(arg0 >>> 0);
            return ret;
        },
        __wbg__wlift_jit_call_1_909b55db88f8ace7: function(arg0, arg1) {
            const ret = globalThis._wlift_jit_call_1(arg0 >>> 0, BigInt.asUintN(64, arg1));
            return ret;
        },
        __wbg__wlift_jit_call_2_4f430a796b9fc356: function(arg0, arg1, arg2) {
            const ret = globalThis._wlift_jit_call_2(arg0 >>> 0, BigInt.asUintN(64, arg1), BigInt.asUintN(64, arg2));
            return ret;
        },
        __wbg__wlift_jit_instantiate_1a1facf843addf1f: function(arg0, arg1) {
            const ret = globalThis._wlift_jit_instantiate(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg__wlift_jit_reset_8583ff9d77c43989: function() {
            globalThis._wlift_jit_reset();
        },
        __wbg__wlift_perf_log_7b0935b5b8ba7abc: function(arg0, arg1, arg2) {
            globalThis._wlift_perf_log(getStringFromWasm0(arg0, arg1), arg2);
        },
        __wbg__wlift_set_timeout_8c02d3b8b01dd0ef: function(arg0, arg1) {
            globalThis._wlift_set_timeout(arg0 >>> 0, arg1);
        },
        __wbg__wlift_storage_clear_34759c102bff75a0: function(arg0, arg1, arg2) {
            globalThis._wlift_storage_clear(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_storage_get_9eab12e270e0ad2c: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_storage_get(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_storage_remove_bf39ed76a1faef7c: function(arg0, arg1, arg2, arg3, arg4) {
            globalThis._wlift_storage_remove(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        },
        __wbg__wlift_storage_set_3b6a14135ad38ea5: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            globalThis._wlift_storage_set(arg0 >>> 0, getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4), getStringFromWasm0(arg5, arg6));
        },
        __wbg__wlift_ws_close_a31a7765ea554764: function(arg0) {
            globalThis._wlift_ws_close(arg0 >>> 0);
        },
        __wbg__wlift_ws_open_e2ed087e890ec926: function(arg0, arg1, arg2) {
            globalThis._wlift_ws_open(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_ws_send_7863a03c6ea13828: function(arg0, arg1, arg2) {
            globalThis._wlift_ws_send(arg0 >>> 0, getStringFromWasm0(arg1, arg2));
        },
        __wbg__wlift_yield_to_event_loop_038733db998cc2d7: function() {
            const ret = globalThis._wlift_yield_to_event_loop();
            return ret;
        },
        __wbg_call_2d781c1f4d5c0ef8: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_call_e133b57c9155d22c: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_a1cf2e70b003a59d: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_get_unchecked_329cfe50afab7352: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_instanceof_Uint8Array_740438561a5b956d: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_b3416cf66a5452c8: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_ea16607d7b61445b: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_49d5571bd3f0c4d4: function() {
            const ret = new Map();
            return ret;
        },
        __wbg_new_a70fbab9066b301f: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_ab79df5bd7c26067: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_d098e265629cd10f: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = state0.b = 0;
            }
        },
        __wbg_new_from_slice_22da9388ac046e50: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_aaaeaf29cf802876: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = state0.b = 0;
            }
        },
        __wbg_now_16f0c993d5dd6c27: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_now_ad1121946ba97ea0: function() { return handleError(function () {
            const ret = Date.now();
            return ret;
        }, arguments); },
        __wbg_now_c16a1d2e10f66992: function() {
            const ret = performance.now();
            return ret;
        },
        __wbg_now_e7c6795a7f81e10f: function(arg0) {
            const ret = arg0.now();
            return ret;
        },
        __wbg_of_d6376e3774c51f89: function(arg0, arg1) {
            const ret = Array.of(arg0, arg1);
            return ret;
        },
        __wbg_performance_3fcf6e32a7e1ed0a: function(arg0) {
            const ret = arg0.performance;
            return ret;
        },
        __wbg_prototypesetcall_d62e5099504357e6: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_e87b0e732085a946: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_0c399741342fb10f: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_a082d78ce798393e: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_race_6ded4e7ff0d4d898: function(arg0) {
            const ret = Promise.race(arg0);
            return ret;
        },
        __wbg_resolve_ae8d83246e5bcc12: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_runresult_new: function(arg0) {
            const ret = RunResult.__wrap(arg0);
            return ret;
        },
        __wbg_set_282384002438957f: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_7eaa4f96924fd6b3: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_bf7251625df30a02: function(arg0, arg1, arg2) {
            const ret = arg0.set(arg1, arg2);
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_8adb955bd33fac2f: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_ad356e0db91c7913: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_f207c857566db248: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_bb9f1ba69d61b386: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_then_098abe61755d12f6: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_then_9e335f6dd892bc11: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_wliftDynamicPluginDispatch_5eb4e9291437b815: function(arg0, arg1) {
            globalThis.wliftDynamicPluginDispatch(arg0 >>> 0, arg1 >>> 0);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { dtor_idx: 107, function: Function { arguments: [Externref], shim_idx: 108, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm.wasm_bindgen_20bf61ce484b8279___closure__destroy___dyn_core_2b72ad5d24e5930c___ops__function__FnMut__wasm_bindgen_20bf61ce484b8279___JsValue____Output___core_2b72ad5d24e5930c___result__Result_____wasm_bindgen_20bf61ce484b8279___JsError___, wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___wasm_bindgen_20bf61ce484b8279___JsValue__core_2b72ad5d24e5930c___result__Result_____wasm_bindgen_20bf61ce484b8279___JsError___true_);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./wlift_wasm_bg.js": import0,
    };
}

function wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___wasm_bindgen_20bf61ce484b8279___JsValue__core_2b72ad5d24e5930c___result__Result_____wasm_bindgen_20bf61ce484b8279___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___wasm_bindgen_20bf61ce484b8279___JsValue__core_2b72ad5d24e5930c___result__Result_____wasm_bindgen_20bf61ce484b8279___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_20bf61ce484b8279___convert__closures_____invoke___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined___js_sys_d983f9b75f30b74___Function_fn_wasm_bindgen_20bf61ce484b8279___JsValue_____wasm_bindgen_20bf61ce484b8279___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}

const RunResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_runresult_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => state.dtor(state.a, state.b));

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            state.dtor(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('wlift_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
