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

None — the wave is unblocked.

## Notes for the 2026-06-04 wave

- **`@hatch:game` resolver-bug holdout was lifted** by the
  `build_recursive` patch that extends the layout-harvest pass
  to version- and git-pinned deps. The CLI previously only
  harvested class field layouts from `Dependency::Path` deps,
  so a consumer with `version = "..."` on `@hatch:game` saw an
  empty `Known classes: []` map at MIR-build time and panicked
  on `class FruitSlicer is Game`. With the harvest extended,
  `hatch build` on `fruit-slicer` (which pins the published
  `@hatch:game@0.3.5`) succeeds: the cached `.hatch` is loaded,
  its `Source` sections compile-replay through a throwaway VM,
  and the field layouts drain into `state.class_field_layouts`
  before the consumer's modules compile.

- **Publish order matters.** The bumped versions (`@hatch:game
  0.3.29`, `@hatch:gpu 0.3.14`, etc.) don't yet exist on the
  registry. Examples whose hatchfiles were rewritten to those
  pins by the publish-prep workflow will fail with
  `dependency '@hatch:X@Y.Z' resolution failed: 404` until the
  publish workflows run. Publish in topological order — leaf
  libs first (`hatch-math`, `hatch-color`, `hatch-buffers`,
  `hatch-fsm`, `hatch-json`), then mid-stack
  (`hatch-ecs`, `hatch-spatial`, `hatch-noise`, `hatch-image`,
  `hatch-audio`, `hatch-window`, `hatch-gpu`), then leaves
  (`hatch-assets`, `hatch-hud`, `hatch-postfx`, `hatch-gltf`,
  `hatch-game`, plus `hatch-physics`). `hatch-web` + `hatch-http`
  shipped in their own prior wave.

- **Local-dev gap for unpublished pins.** Until the publish
  wave lands, building a workspace whose hatchfile pins a
  bumped version (e.g. `hatch/examples/game/ecs-cubes`)
  fails at fetch time. Either revert the consumer pin to a
  workspace path until publish, or run the publish CI to seed
  the registry. A future workspace-fallback resolver tweak
  could close this gap automatically; not planned for this
  wave.
