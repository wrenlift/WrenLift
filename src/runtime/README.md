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
- `generation: u8` -- 0=young (nursery), 1=old
- `next: *mut ObjHeader` -- intrusive linked list for GC sweep
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

## Generational Garbage Collector

Two-generation collector with bump-allocated nursery and mark-sweep old generation.

### Nursery (young generation)

A contiguous `Vec<u8>` arena with bump-pointer allocation. Allocation is O(1): align the bump pointer, write the object, advance. Default size is 256 KB. When the nursery is full, new allocations overflow directly to the old generation.

### Old generation

Box-allocated objects linked via the `ObjHeader::next` intrusive list. Mark-sweep collection: unreachable objects are freed by reconstructing the `Box` and dropping it.

### Minor GC

1. Mark from roots using a gray stack (tri-color marking).
2. Mark from the remembered set (old-to-young references).
3. Process the gray stack (mark gray objects black, trace their references).
4. Walk nursery objects: promote live (marked) objects to old gen via `ptr::read` into a new `Box`; drop dead objects in place.
5. Build a forwarding table (old nursery address to new old-gen address) and update all roots, old-gen internal pointers, and the intern table.
6. Reset the nursery bump pointer to 0.
7. Reset marks on old-gen objects.

Special handling for `ObjUpvalue`: when a closed upvalue is promoted, its self-referential `location` pointer (pointing to its own `closed` field in the nursery) is fixed up to point to the new location.

### Major GC

Same as minor GC, followed by a sweep of dead old-gen objects. Triggered every N minor collections (default 8, configurable via `major_gc_interval`).

### Write barrier

When an old-gen object receives a reference to a young-gen object, the old-gen object is added to the remembered set. This ensures minor GC can find young objects reachable only through old-gen references. The barrier is a no-op for young-to-young or old-to-old writes.

### String interning

Hash-based deduplication via `HashMap<u64, Vec<*mut ObjString>>` keyed by FNV-1a hash. Interned strings are collected when unreachable (no pinning). The intern table is updated during pointer forwarding after promotion.

### Adaptive threshold

After each collection, the old-gen threshold is recalculated as `max(live_count * heap_grow_factor, initial_threshold)`. Default grow factor is 2.0, default initial threshold is 256 objects. Collection is triggered when the nursery exceeds 75% usage or old-gen object count exceeds the threshold.

### Configuration (`GcConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `nursery_size` | 256 KB | Arena size in bytes |
| `initial_threshold` | 256 | Minimum old-gen object count before GC triggers |
| `heap_grow_factor` | 2.0 | Multiplier for adaptive threshold |
| `major_gc_interval` | 8 | Minor collections between each major collection |

## Fiber Runtime

Placeholder for fiber/coroutine execution runtime (Phase 12). The fiber data structure (`ObjFiber`) is defined in `object.rs` with full stack, call frame, and state management. This file will contain the fiber scheduler and coroutine transfer logic.
