# Publish holdouts

Packages that have a CHANGELOG.md prepared and a version bump
staged in their hatchfile, but are intentionally held back from
the current publish wave because their source is known broken
on this checkout. The CHANGELOG entry stays in tree so the doc
is ready to ship the moment the gating bug lands fixed; only
the publish tag is withheld.

Update this list whenever a new wave goes out — entries should
be removed once the underlying bug ships a fix and the package
publishes through `publish-pkg.yml` / `publish-plugin.yml`.

## Held back this wave (2026-06-04)

### @hatch:game — 0.3.28 → 0.3.29 (held)

- **Gating bug**: pre-existing module-load-order panic. The
  `game-example` corpus reproduces it: subclasses of `Game`
  compile before the `Game` module loads, so the resolver
  hits an unresolved class reference and the MIR builder
  panics rather than emitting a diagnostic.
- **Why held**: republishing `@hatch:game` while the gating
  bug is live would push a snapshot that downstream
  consumers (procedural-world, fruit-slicer, animation-
  showcase) cannot build against. The catalog row stays on
  0.3.28 — the last known-good version — until the load-
  order fix lands in `wren_lift`.
- **CHANGELOG.md**: written for 0.3.29 anyway so the doc is
  ready when the gating bug is fixed.
- **Unblocks**: a `wren_lift` fix that ensures parent
  modules of inheriting classes load before the inheriting
  module's compile pass runs. Drop this entry, tag
  `publish/hatch-game@0.3.29`, and the wave catches up.
