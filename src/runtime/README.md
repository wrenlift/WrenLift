# Runtime

The runtime module implements the core value representation, heap-allocated object system, and garbage collector for the Wren language.

## NaN-Boxed Value Representation

All Wren values are encoded as a single 64-bit NaN-boxed `Value`, declared as `#[repr(transparent)]` over `u64`. This exploits IEEE 754 quiet NaN payloads to pack non-number types into unused bit patterns:

```text
Normal f64:  any bit pattern that is NOT a quiet NaN with our tag
Null:        0x7FFC_0000_0000_0000  (QNAN base)
False:       0x7FFC_0000_0000_0001  (QNAN | 1)
True:        0x7FFC_0000_0000_0002  (QNAN | 2)
Undefined:   0x7FFC_0000_0000_0003  (QNAN | 3, internal sentinel)
Object ptr:  0xFFFC_0000_0000_0000 | (ptr & 0x0000_FFFF_FFFF_FFFF)
```

Object pointers are tagged with sign bit + QNAN. The lower 48 bits store the pointer, which is sufficient for all userspace addresses on x86_64 and aarch64. Extraction sign-extends bit 47 to reconstruct canonical pointer form.

`Value` is `Copy` and exactly 8 bytes. Equality follows IEEE 754 for numbers (NaN != NaN, 0.0 == -0.0) and bitwise comparison for everything else (pointer identity for objects).

Wren truthiness: only `false` and `null` are falsy. `0`, empty string, and all objects are truthy.

## Heap Object Types

Every heap-allocated object begins with an `ObjHeader` containing:

- `obj_type: ObjType` -- `#[repr(u8)]` discriminant for runtime type dispatch
- `gc_mark: u8` -- tri-color marking (0=white, 1=gray, 2=black)
- `flags: u8` -- per-object flag bits (`FLAG_*` in `object.rs`)
- `class: *mut ObjClass` -- class pointer for method dispatch

All `Obj*` structs are `#[repr(C)]` with `ObjHeader` as the first field, enabling safe casting between `*mut ObjHeader` and concrete object types via `downcast_ref`/`downcast_mut`.

### Object types

| Type | Description |
|------|-------------|
| `ObjString` | Immutable string with precomputed FNV-1a hash for O(1) map lookups and deduplication. |
| `ObjList` | Growable array of `Value` elements. |
| `ObjMap` | HashMap from `Value` to `Value`. Keys are wrapped in `MapKey` which implements `Hash`/`Eq` using raw u64 bits. |
| `ObjRange` | Numeric range (`from..to`) with inclusive/exclusive flag and integer step iterator. |
| `ObjFn` | Compiled function metadata: name (`SymbolId`), arity, upvalue count, function table index. |
| `ObjClosure` | Function + captured upvalues. Wraps an `ObjFn` pointer and a `Vec<*mut ObjUpvalue>`. |
| `ObjUpvalue` | Captured variable. `location` initially points to a stack slot; on close, value is copied to an internal `closed` field and `location` is redirected there. Maintains an intrusive list sorted by stack slot (descending). |
| `ObjFiber` | Lightweight coroutine with its own value stack, call frame stack, execution state (New/Running/Suspended/Done/Error), caller chain, and error value. |
| `ObjClass` | Class with a `HashMap<SymbolId, Method>` method table. Methods are either `Closure(*mut ObjClosure)` or `Native(NativeFn)`. Inheritance copies the superclass method table at class creation. `num_fields` and `is_foreign` flag. |
| `ObjInstance` | Instance with a fixed-size `Vec<Value>` of fields indexed by slot number. |
| `ObjForeign` | Opaque host data stored as `Vec<u8>`. |
| `ObjModule` | Compilation unit with module-level variables and their names (parallel `Vec`s) for import resolution. |

Method dispatch is O(1) via `HashMap<SymbolId, Method>` lookup, where `SymbolId` is an interned integer symbol handle.

## Garbage Collector

The collector (`gc_immix.rs`) is a
non-moving mark-sweep whose memory comes through the runtime seam
(`rt.rs`): a versioned `#[repr(C)]` table of function-pointer slots for
allocation, address resolution, the per-cycle liveness claim, the stack
scan, the trigger and the sweep. A host installs its own memory with
`wlift_rt_install` before the first Immix VM exists; otherwise every slot
is wren_lift's block allocator (`gc_immix_heap.rs`), Immix geometry:
32 KiB blocks of 128-byte lines in demand-mapped 32 MiB chunks. Objects
up to one line bump-allocate and never straddle a line; larger objects
own whole lines. Object starts and span sizes live in side tables, so
any address inside an allocation resolves to its start. Heap tracing is
precise through `for_each_child`, whichever memory is underneath; native
stacks (the running one from a callee-saved register spill to the
thread's stack top, every suspended krio fiber from its saved sp) are
scanned conservatively, so compiled code needs no stack maps and Rust
helpers need no root pushes. Any allocation helper is a safepoint in
every tier; natives are never one. Instance fields are allocated inline
behind the header. Free blocks beyond a small float are returned to the
OS after a quiet collection. Knobs: `WLIFT_GC_HEAP_MB`,
`WLIFT_GC_TRIGGER_MB`, `WLIFT_GC_GROWTH`, `WLIFT_GC_STRESS`.

The collector is the only one; `--gc-stats` prints its counters and the
pause split (stop / mark / sweep). String interning deduplicates through
a hash table keyed by FNV-1a hash; interned strings are collected when
unreachable.

## Fiber Runtime

Placeholder for fiber/coroutine execution runtime (Phase 12). The fiber data structure (`ObjFiber`) is defined in `object.rs` with full stack, call frame, and state management. This file will contain the fiber scheduler and coroutine transfer logic.
