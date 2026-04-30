# Comment format spec — short version

For authors. The full design rationale lives in
[hatch-docs-generator-plan.md](hatch-docs-generator-plan.md).

## Rules

1. **`///`** above a declaration → that declaration's docs.
2. **`//!`** at the top of a file → the module's docs.
3. **`//`** is a code comment. Tooling ignores it.

## Body is Markdown

CommonMark, with code fences. Wren's interpolation syntax doesn't
clash with Markdown emphasis, so write naturally.

```wren
/// Slices a fruit along the swipe segment.
///
/// Splits the fruit into two halves; each carries its own
/// rotational velocity and a follow-through velocity along
/// the swipe direction.
///
/// ## Parameters
///
/// - `x1, y1` — segment start (last frame's mouse position).
/// - `x2, y2` — segment end (this frame's mouse position).
///
/// ## Example
///
/// ```wren
/// game.checkSlice(10, 10, 200, 50)
/// ```
checkSlice(x1, y1, x2, y2) { ... }
```

## Cross-references

Bracketed identifiers auto-resolve:

| Form                     | Links to                           |
|--------------------------|------------------------------------|
| `[ClassName]`            | the class's page                   |
| `[Class.method]`         | the method anchor on that page     |
| `[#section-id]`          | a heading in the current doc       |
| `[@hatch:pkg]`           | the package's index                |
| `[@hatch:pkg ClassName]` | a class inside the package         |

## Headings with meaning

- `## Example` / `## Examples` → promoted into the example slot.
- `## Parameters` / `## Returns` / `## Throws` → sidebar metadata.

Any other heading renders as a normal section.

## One-sentence summary

The first paragraph is the symbol's summary. It shows up in the
parent class's method list, search results, and LSP completion
previews. Keep it under one line if you can.

## Hidden symbols

`/// @private` on a line of its own marks the symbol as
generator-only (not surfaced in the public API). Use sparingly;
prefer naming with a trailing underscore (`foo_`) which is the
existing convention for internal methods.
