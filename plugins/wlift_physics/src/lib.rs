//! Physics plugin for WrenLift. Built as a cdylib; bundled into
//! @hatch:physics. One plugin holds both 2D and 3D worlds —
//! the dimensional split is at the API level (`World2D` /
//! `World3D` Wren classes), not the package level.
//!
//! v0 surface (per dimension):
//! - World creation / step / drop
//! - spawnDynamic / spawnStatic / spawnKinematic / despawn
//! - position read
//! - linearVelocity read + set
//! - applyForce / applyImpulse
//!
//! Out of scope for v0: raycast, joints, sensors, CCD options,
//! contact event streams, rotation read/write, kinematic-target
//! based motion. The wrapper surface is deliberately small;
//! rapier supports far more, accessible via direct entry-point
//! additions when needed.

#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use wlift_abi::{
    alloc_list, alloc_map, alloc_string, list_add, list_count, list_get, map_iter, map_set,
    obj_type, runtime_error, set_return, slot, string_str, ObjType, Value, WrenVm,
};

/// Plugin ABI handshake — see wlift_gpu::wlift_plugin_abi_version.
///
/// Native-only. Used by the host's `libloading::dlsym` path to
/// reject a dylib built against a different `wren_lift` than the
/// host. Statically-linked wasm plugins can't drift versions
/// (everything's compiled from the same tree), and re-exporting
/// the symbol from every linked rlib produces wasm-linker
/// duplicate-symbol errors.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn string_of(v: Value) -> Option<String> {
    string_str(v).map(|s| s.to_string())
}

fn map_get(v: Value, key: &str) -> Option<Value> {
    if obj_type(v) != Some(ObjType::Map) {
        return None;
    }
    for (k, val) in map_iter(v) {
        if let Some(s) = string_str(k) {
            if s == key {
                return Some(val);
            }
        }
    }
    None
}

fn read_num_at(list_v: Value, i: u32, default: f64) -> f64 {
    if i >= list_count(list_v) {
        return default;
    }
    list_get(list_v, i).as_num().unwrap_or(default)
}

fn read_2d(list_v: Value) -> Option<(f32, f32)> {
    if obj_type(list_v) != Some(ObjType::List) || list_count(list_v) < 2 {
        return None;
    }
    Some((
        read_num_at(list_v, 0, 0.0) as f32,
        read_num_at(list_v, 1, 0.0) as f32,
    ))
}

fn read_3d(list_v: Value) -> Option<(f32, f32, f32)> {
    if obj_type(list_v) != Some(ObjType::List) || list_count(list_v) < 3 {
        return None;
    }
    Some((
        read_num_at(list_v, 0, 0.0) as f32,
        read_num_at(list_v, 1, 0.0) as f32,
        read_num_at(list_v, 2, 0.0) as f32,
    ))
}

/// Allocate a list, push it as the return value (rooting it via
/// `api_stack[0]`), then append each Value via `list_add` — the
/// host helper re-reads the list pointer from the GC-tracked
/// root before each append, so a forwarded list under a moving
/// nursery stays consistent.
unsafe fn return_list(vm: *mut WrenVm, values: &[Value]) {
    let list = alloc_list(vm, values.len() as u32);
    set_return(vm, list);
    for v in values {
        let cur = slot(vm, 0);
        list_add(vm, cur, *v);
    }
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// 2D world
// ---------------------------------------------------------------------------

mod d2 {
    use super::*;
    use rapier2d::prelude::*;

    pub struct World {
        pub gravity: Vector,
        pub integration_parameters: IntegrationParameters,
        pub physics_pipeline: PhysicsPipeline,
        pub island_manager: IslandManager,
        pub broad_phase: BroadPhaseBvh,
        pub narrow_phase: NarrowPhase,
        pub bodies: RigidBodySet,
        pub colliders: ColliderSet,
        pub impulse_joints: ImpulseJointSet,
        pub multibody_joints: MultibodyJointSet,
        pub ccd_solver: CCDSolver,
        pub bodies_by_id: HashMap<u64, RigidBodyHandle>,
        pub ids_by_handle: HashMap<RigidBodyHandle, u64>,
        pub joints_by_id: HashMap<u64, ImpulseJointHandle>,
        /// Same shape as 3D's `contact_events` — see that struct's
        /// docstring for the lifecycle.
        pub contact_events: Vec<(u64, u64, bool)>,
    }

    impl World {
        pub fn new(gx: f32, gy: f32) -> Self {
            Self {
                gravity: Vector::new(gx, gy),
                integration_parameters: IntegrationParameters::default(),
                physics_pipeline: PhysicsPipeline::new(),
                island_manager: IslandManager::new(),
                broad_phase: BroadPhaseBvh::new(),
                narrow_phase: NarrowPhase::new(),
                bodies: RigidBodySet::new(),
                colliders: ColliderSet::new(),
                impulse_joints: ImpulseJointSet::new(),
                multibody_joints: MultibodyJointSet::new(),
                ccd_solver: CCDSolver::new(),
                bodies_by_id: HashMap::new(),
                ids_by_handle: HashMap::new(),
                joints_by_id: HashMap::new(),
                contact_events: Vec::new(),
            }
        }

        pub fn step(&mut self, dt: f32) {
            // Channel type has to be spelled explicitly: the
            // compiler can't infer `CollisionEvent` from
            // `ChannelEventCollector::new` alone, and inside the
            // 2D scope `CollisionEvent` resolves through `use
            // rapier2d::prelude::*` to the 2D variant.
            // rapier's `ChannelEventCollector` takes
            // `std::sync::mpsc::Sender`s — explicitly typed so
            // the compiler resolves to the right `CollisionEvent`
            // (the 2D and 3D rapier crates each export a distinct
            // one, brought in via the enclosing `use … prelude::*`).
            let (collision_tx, collision_rx) =
                std::sync::mpsc::channel::<CollisionEvent>();
            let (contact_force_tx, _contact_force_rx) =
                std::sync::mpsc::channel::<ContactForceEvent>();
            let event_handler = ChannelEventCollector::new(
                collision_tx,
                contact_force_tx,
            );

            self.integration_parameters.dt = dt;
            self.physics_pipeline.step(
                self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd_solver,
                &(),
                &event_handler,
            );

            // Mirror the 3D drain — see d3::World::step for the
            // full rationale. Identical body-ID resolution path.
            self.contact_events.clear();
            while let Ok(ev) = collision_rx.try_recv() {
                let (ch_a, ch_b, started) = match ev {
                    CollisionEvent::Started(a, b, _) => (a, b, true),
                    CollisionEvent::Stopped(a, b, _) => (a, b, false),
                };
                let body_a = self.colliders.get(ch_a).and_then(|c| c.parent());
                let body_b = self.colliders.get(ch_b).and_then(|c| c.parent());
                if let (Some(ba), Some(bb)) = (body_a, body_b) {
                    if let (Some(&ida), Some(&idb)) = (
                        self.ids_by_handle.get(&ba),
                        self.ids_by_handle.get(&bb),
                    ) {
                        self.contact_events.push((ida, idb, started));
                    }
                }
            }
        }
    }

    pub fn worlds() -> &'static Mutex<HashMap<u64, World>> {
        static REG: OnceLock<Mutex<HashMap<u64, World>>> = OnceLock::new();
        REG.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub unsafe fn collider_from_desc(desc: Value) -> Option<Collider> {
        let kind = map_get(desc, "kind").and_then(string_of)?;
        let mut builder = match kind.as_str() {
            "ball" => {
                let r = map_get(desc, "radius")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::ball(r)
            }
            "box" => {
                let hw = map_get(desc, "halfWidth")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                let hh = map_get(desc, "halfHeight")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::cuboid(hw, hh)
            }
            "capsule" => {
                let hh = map_get(desc, "halfHeight")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                let r = map_get(desc, "radius")
                    .and_then(Value::as_num)
                    .unwrap_or(0.25) as f32;
                ColliderBuilder::capsule_y(hh, r)
            }
            _ => return None,
        };
        let restitution = map_get(desc, "restitution")
            .and_then(Value::as_num)
            .unwrap_or(0.0) as f32;
        let friction = map_get(desc, "friction")
            .and_then(Value::as_num)
            .unwrap_or(0.5) as f32;
        // `sensor: true` flips the collider into trigger mode —
        // it still surfaces start / stop events through the
        // collision-event channel but the narrow phase skips
        // contact resolution. The game layer reads sensor pairs
        // back through `drainContactEvents` exactly like real
        // collisions.
        let is_sensor = map_get(desc, "sensor")
            .map(|v| v == Value::TRUE)
            .unwrap_or(false);
        builder = builder
            .restitution(restitution)
            .friction(friction)
            .sensor(is_sensor)
            // Opt every collider into rapier's collision-event
            // channel by default. The cost is negligible (rapier
            // only emits events for pairs whose narrow-phase
            // touched / untouched state actually changed), and
            // the alternative is requiring every user to call
            // an opt-in setter on every collider before contact
            // events do anything — a footgun the game-layer
            // `World3D.drainContactEvents` shouldn't carry.
            .active_events(ActiveEvents::COLLISION_EVENTS);
        Some(builder.build())
    }
}

// ---------------------------------------------------------------------------
// 3D world (mirrors d2)
// ---------------------------------------------------------------------------

mod d3 {
    use super::*;
    use rapier3d::prelude::*;

    pub struct World {
        pub gravity: Vector,
        pub integration_parameters: IntegrationParameters,
        pub physics_pipeline: PhysicsPipeline,
        pub island_manager: IslandManager,
        pub broad_phase: BroadPhaseBvh,
        pub narrow_phase: NarrowPhase,
        pub bodies: RigidBodySet,
        pub colliders: ColliderSet,
        pub impulse_joints: ImpulseJointSet,
        pub multibody_joints: MultibodyJointSet,
        pub ccd_solver: CCDSolver,
        pub bodies_by_id: HashMap<u64, RigidBodyHandle>,
        pub ids_by_handle: HashMap<RigidBodyHandle, u64>,
        pub joints_by_id: HashMap<u64, ImpulseJointHandle>,
        /// Contact events queued during the most recent `step` —
        /// drained by `wlift_physics_world3d_drain_contact_events`.
        /// Each entry is `(body_id_a, body_id_b, started)`; we
        /// translate rapier's `CollisionEvent` into rigid-body IDs
        /// up-front so the drain export doesn't need to walk
        /// handle maps under the runtime's lock.
        pub contact_events: Vec<(u64, u64, bool)>,
    }

    impl World {
        pub fn new(gx: f32, gy: f32, gz: f32) -> Self {
            Self {
                gravity: Vector::new(gx, gy, gz),
                integration_parameters: IntegrationParameters::default(),
                physics_pipeline: PhysicsPipeline::new(),
                island_manager: IslandManager::new(),
                broad_phase: BroadPhaseBvh::new(),
                narrow_phase: NarrowPhase::new(),
                bodies: RigidBodySet::new(),
                colliders: ColliderSet::new(),
                impulse_joints: ImpulseJointSet::new(),
                multibody_joints: MultibodyJointSet::new(),
                ccd_solver: CCDSolver::new(),
                bodies_by_id: HashMap::new(),
                ids_by_handle: HashMap::new(),
                joints_by_id: HashMap::new(),
                contact_events: Vec::new(),
            }
        }

        pub fn step(&mut self, dt: f32) {
            // Per-step channels for collision + contact-force
            // events. Constructed fresh each step so a stale
            // receiver from a previous step can't leak.
            // `ChannelEventCollector` is rapier's `EventHandler`
            // impl that just forwards events into the senders;
            // we drain the collision channel into `contact_events`
            // before this scope ends.
            // rapier's `ChannelEventCollector` takes
            // `std::sync::mpsc::Sender`s — explicitly typed so
            // the compiler resolves to the 3D `CollisionEvent`
            // brought in via the enclosing `use rapier3d::prelude::*`.
            let (collision_tx, collision_rx) =
                std::sync::mpsc::channel::<CollisionEvent>();
            let (contact_force_tx, _contact_force_rx) =
                std::sync::mpsc::channel::<ContactForceEvent>();
            let event_handler = ChannelEventCollector::new(
                collision_tx,
                contact_force_tx,
            );

            self.integration_parameters.dt = dt;
            self.physics_pipeline.step(
                self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd_solver,
                &(),
                &event_handler,
            );

            // Drain rapier's contact-event channel into our own
            // buffer, mapped from `ColliderHandle` → user-facing
            // body IDs. Skip pairs whose collider has no rigid-body
            // parent (sensor-only colliders fall through here;
            // once we expose user-spawnable sensors that pattern
            // grows a parallel resolver).
            self.contact_events.clear();
            while let Ok(ev) = collision_rx.try_recv() {
                let (ch_a, ch_b, started) = match ev {
                    CollisionEvent::Started(a, b, _) => (a, b, true),
                    CollisionEvent::Stopped(a, b, _) => (a, b, false),
                };
                let body_a = self.colliders.get(ch_a).and_then(|c| c.parent());
                let body_b = self.colliders.get(ch_b).and_then(|c| c.parent());
                if let (Some(ba), Some(bb)) = (body_a, body_b) {
                    if let (Some(&ida), Some(&idb)) = (
                        self.ids_by_handle.get(&ba),
                        self.ids_by_handle.get(&bb),
                    ) {
                        self.contact_events.push((ida, idb, started));
                    }
                }
            }
        }
    }

    pub fn worlds() -> &'static Mutex<HashMap<u64, World>> {
        static REG: OnceLock<Mutex<HashMap<u64, World>>> = OnceLock::new();
        REG.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub unsafe fn collider_from_desc(desc: Value) -> Option<Collider> {
        let kind = map_get(desc, "kind").and_then(string_of)?;
        let mut builder = match kind.as_str() {
            "ball" => {
                let r = map_get(desc, "radius")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::ball(r)
            }
            "box" => {
                let hx = map_get(desc, "halfX")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                let hy = map_get(desc, "halfY")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                let hz = map_get(desc, "halfZ")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::cuboid(hx, hy, hz)
            }
            "capsule" => {
                let hh = map_get(desc, "halfHeight")
                    .and_then(Value::as_num)
                    .unwrap_or(0.5) as f32;
                let r = map_get(desc, "radius")
                    .and_then(Value::as_num)
                    .unwrap_or(0.25) as f32;
                ColliderBuilder::capsule_y(hh, r)
            }
            _ => return None,
        };
        let restitution = map_get(desc, "restitution")
            .and_then(Value::as_num)
            .unwrap_or(0.0) as f32;
        let friction = map_get(desc, "friction")
            .and_then(Value::as_num)
            .unwrap_or(0.5) as f32;
        // `sensor: true` flips the collider into trigger mode —
        // it still surfaces start / stop events through the
        // collision-event channel but the narrow phase skips
        // contact resolution. The game layer reads sensor pairs
        // back through `drainContactEvents` exactly like real
        // collisions.
        let is_sensor = map_get(desc, "sensor")
            .map(|v| v == Value::TRUE)
            .unwrap_or(false);
        builder = builder
            .restitution(restitution)
            .friction(friction)
            .sensor(is_sensor)
            // Opt every collider into rapier's collision-event
            // channel by default. The cost is negligible (rapier
            // only emits events for pairs whose narrow-phase
            // touched / untouched state actually changed), and
            // the alternative is requiring every user to call
            // an opt-in setter on every collider before contact
            // events do anything — a footgun the game-layer
            // `World3D.drainContactEvents` shouldn't carry.
            .active_events(ActiveEvents::COLLISION_EVENTS);
        Some(builder.build())
    }
}

// ---------------------------------------------------------------------------
// 2D foreign entry points
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_create(vm: *mut WrenVm) {
    let desc = slot(vm, 1);
    let (gx, gy) = map_get(desc, "gravity")
        .and_then(read_2d)
        .unwrap_or((0.0, -9.81));
    let world = d2::World::new(gx, gy);
    let id = next_id();
    d2::worlds().lock().unwrap().insert(id, world);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_destroy(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    d2::worlds().lock().unwrap().remove(&id);
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_step(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let dt = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
    if let Some(w) = d2::worlds().lock().unwrap().get_mut(&id) {
        w.step(dt);
    }
    set_return(vm, Value::NULL);
}

unsafe fn read_world2d_id(vm: *mut WrenVm, slot_index: u32, label: &str) -> Option<u64> {
    match slot(vm, slot_index).as_num() {
        Some(n) if n >= 0.0 => Some(n as u64),
        _ => {
            runtime_error(
                vm,
                &format!("{}: world id must be a non-negative integer.", label),
            );
            None
        }
    }
}

unsafe fn spawn_2d(vm: *mut WrenVm, body_kind: &str) {
    use rapier2d::prelude::*;

    let world_id = match read_world2d_id(vm, 1, "World2D.spawn") {
        Some(i) => i,
        None => return,
    };
    let desc = slot(vm, 2);
    let (px, py) = map_get(desc, "position")
        .and_then(read_2d)
        .unwrap_or((0.0, 0.0));
    let (vx, vy) = map_get(desc, "linearVelocity")
        .and_then(read_2d)
        .unwrap_or((0.0, 0.0));
    let mass = map_get(desc, "mass").and_then(Value::as_num);

    let mut body_builder = match body_kind {
        "static" => RigidBodyBuilder::fixed(),
        "kinematic" => RigidBodyBuilder::kinematic_position_based(),
        _ => RigidBodyBuilder::dynamic(),
    };
    body_builder = body_builder
        .translation(Vector::new(px, py))
        .linvel(Vector::new(vx, vy));

    let collider = match map_get(desc, "shape") {
        Some(s) => match d2::collider_from_desc(s) {
            Some(c) => c,
            None => {
                runtime_error(
                    vm,
                    "World2D.spawn: descriptor `shape` is missing or has unknown `kind`.",
                );
                return;
            }
        },
        None => {
            runtime_error(vm, "World2D.spawn: descriptor must include a `shape` Map.");
            return;
        }
    };

    let mut reg = d2::worlds().lock().unwrap();
    let world = match reg.get_mut(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.spawn: unknown world id.");
            return;
        }
    };

    let mut body = body_builder.build();
    if let Some(m) = mass {
        body.set_additional_mass(m as f32, true);
    }
    let body_handle = world.bodies.insert(body);
    world
        .colliders
        .insert_with_parent(collider, body_handle, &mut world.bodies);

    let id = next_id();
    world.bodies_by_id.insert(id, body_handle);
    world.ids_by_handle.insert(body_handle, id);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_dynamic(vm: *mut WrenVm) {
    unsafe { spawn_2d(vm, "dynamic") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_static(vm: *mut WrenVm) {
    unsafe { spawn_2d(vm, "static") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_kinematic(vm: *mut WrenVm) {
    unsafe { spawn_2d(vm, "kinematic") }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_despawn(vm: *mut WrenVm) {
    let world_id = match read_world2d_id(vm, 1, "World2D.despawn") {
        Some(i) => i,
        None => return,
    };
    let body_id = match slot(vm, 2).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let mut reg = d2::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(handle) = world.bodies_by_id.remove(&body_id) {
            world.ids_by_handle.remove(&handle);
            world.bodies.remove(
                handle,
                &mut world.island_manager,
                &mut world.colliders,
                &mut world.impulse_joints,
                &mut world.multibody_joints,
                true,
            );
        }
    }
    set_return(vm, Value::NULL);
}

unsafe fn return_2d_pair(vm: *mut WrenVm, x: f32, y: f32) {
    return_list(vm, &[Value::num(x as f64), Value::num(y as f64)])
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_position(vm: *mut WrenVm) {
    let world_id = match read_world2d_id(vm, 1, "World2D.position") {
        Some(i) => i,
        None => return,
    };
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d2::worlds().lock().unwrap();
    let world = match reg.get(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.position: unknown world id.");
            return;
        }
    };
    let handle = match world.bodies_by_id.get(&body_id) {
        Some(h) => *h,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.position: unknown body id.");
            return;
        }
    };
    let body = &world.bodies[handle];
    let t = body.translation();
    let (x, y) = (t.x, t.y);
    drop(reg);
    return_2d_pair(vm, x, y);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_linear_velocity(vm: *mut WrenVm) {
    let world_id = match read_world2d_id(vm, 1, "World2D.linearVelocity") {
        Some(i) => i,
        None => return,
    };
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d2::worlds().lock().unwrap();
    let (vx, vy) = reg
        .get(&world_id)
        .and_then(|w| {
            w.bodies_by_id.get(&body_id).map(|h| {
                let v = w.bodies[*h].linvel();
                (v.x, v.y)
            })
        })
        .unwrap_or((0.0, 0.0));
    drop(reg);
    return_2d_pair(vm, vx, vy);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_set_linear_velocity(vm: *mut WrenVm) {
    use rapier2d::prelude::*;
    unsafe {
        let world_id = match read_world2d_id(vm, 1, "World2D.setLinearVelocity") {
            Some(i) => i,
            None => return,
        };
        let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
        let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
        let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
        let mut reg = d2::worlds().lock().unwrap();
        if let Some(world) = reg.get_mut(&world_id) {
            if let Some(&handle) = world.bodies_by_id.get(&body_id) {
                world.bodies[handle].set_linvel(Vector::new(x, y), true);
            }
        }
        set_return(vm, Value::NULL);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_apply_impulse(vm: *mut WrenVm) {
    use rapier2d::prelude::*;
    unsafe {
        let world_id = match read_world2d_id(vm, 1, "World2D.applyImpulse") {
            Some(i) => i,
            None => return,
        };
        let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
        let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
        let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
        let mut reg = d2::worlds().lock().unwrap();
        if let Some(world) = reg.get_mut(&world_id) {
            if let Some(&handle) = world.bodies_by_id.get(&body_id) {
                world.bodies[handle].apply_impulse(Vector::new(x, y), true);
            }
        }
        set_return(vm, Value::NULL);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_apply_force(vm: *mut WrenVm) {
    use rapier2d::prelude::*;
    unsafe {
        let world_id = match read_world2d_id(vm, 1, "World2D.applyForce") {
            Some(i) => i,
            None => return,
        };
        let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
        let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
        let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
        let mut reg = d2::worlds().lock().unwrap();
        if let Some(world) = reg.get_mut(&world_id) {
            if let Some(&handle) = world.bodies_by_id.get(&body_id) {
                world.bodies[handle].add_force(Vector::new(x, y), true);
            }
        }
        set_return(vm, Value::NULL);
    }
}

// ---------------------------------------------------------------------------
// 3D foreign entry points (mirror 2D — same Wren-side conventions).
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_create(vm: *mut WrenVm) {
    let desc = slot(vm, 1);
    let (gx, gy, gz) = map_get(desc, "gravity")
        .and_then(read_3d)
        .unwrap_or((0.0, -9.81, 0.0));
    let world = d3::World::new(gx, gy, gz);
    let id = next_id();
    d3::worlds().lock().unwrap().insert(id, world);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_destroy(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    d3::worlds().lock().unwrap().remove(&id);
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_step(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let dt = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
    if let Some(w) = d3::worlds().lock().unwrap().get_mut(&id) {
        w.step(dt);
    }
    set_return(vm, Value::NULL);
}

unsafe fn spawn_3d(vm: *mut WrenVm, body_kind: &str) {
    use rapier3d::prelude::*;

    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let desc = slot(vm, 2);
    let (px, py, pz) = map_get(desc, "position")
        .and_then(read_3d)
        .unwrap_or((0.0, 0.0, 0.0));
    let (vx, vy, vz) = map_get(desc, "linearVelocity")
        .and_then(read_3d)
        .unwrap_or((0.0, 0.0, 0.0));
    let mass = map_get(desc, "mass").and_then(Value::as_num);

    let mut body_builder = match body_kind {
        "static" => RigidBodyBuilder::fixed(),
        "kinematic" => RigidBodyBuilder::kinematic_position_based(),
        _ => RigidBodyBuilder::dynamic(),
    };
    body_builder = body_builder
        .translation(Vector::new(px, py, pz))
        .linvel(Vector::new(vx, vy, vz));

    let collider = match map_get(desc, "shape") {
        Some(s) => match d3::collider_from_desc(s) {
            Some(c) => c,
            None => {
                runtime_error(
                    vm,
                    "World3D.spawn: descriptor `shape` is missing or has unknown `kind`.",
                );
                return;
            }
        },
        None => {
            runtime_error(vm, "World3D.spawn: descriptor must include a `shape` Map.");
            return;
        }
    };

    let mut reg = d3::worlds().lock().unwrap();
    let world = match reg.get_mut(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            runtime_error(vm, "World3D.spawn: unknown world id.");
            return;
        }
    };
    let mut body = body_builder.build();
    if let Some(m) = mass {
        body.set_additional_mass(m as f32, true);
    }
    let body_handle = world.bodies.insert(body);
    world
        .colliders
        .insert_with_parent(collider, body_handle, &mut world.bodies);

    let id = next_id();
    world.bodies_by_id.insert(id, body_handle);
    world.ids_by_handle.insert(body_handle, id);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_dynamic(vm: *mut WrenVm) {
    unsafe { spawn_3d(vm, "dynamic") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_static(vm: *mut WrenVm) {
    unsafe { spawn_3d(vm, "static") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_kinematic(vm: *mut WrenVm) {
    unsafe { spawn_3d(vm, "kinematic") }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_position(vm: *mut WrenVm) {
    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d3::worlds().lock().unwrap();
    let (x, y, z) = reg
        .get(&world_id)
        .and_then(|w| {
            w.bodies_by_id.get(&body_id).map(|h| {
                let t = w.bodies[*h].translation();
                (t.x, t.y, t.z)
            })
        })
        .unwrap_or((0.0, 0.0, 0.0));
    drop(reg);
    return_list(
        vm,
        &[
            Value::num(x as f64),
            Value::num(y as f64),
            Value::num(z as f64),
        ],
    );
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_linear_velocity(vm: *mut WrenVm) {
    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d3::worlds().lock().unwrap();
    let (vx, vy, vz) = reg
        .get(&world_id)
        .and_then(|w| {
            w.bodies_by_id.get(&body_id).map(|h| {
                let v = w.bodies[*h].linvel();
                (v.x, v.y, v.z)
            })
        })
        .unwrap_or((0.0, 0.0, 0.0));
    drop(reg);
    return_list(
        vm,
        &[
            Value::num(vx as f64),
            Value::num(vy as f64),
            Value::num(vz as f64),
        ],
    );
}

/// Non-allocating position read: write the body's (x, y, z) into
/// a caller-provided `Float32Array` starting at `offset`. Returns
/// `null`. The bridge's hot path uses this instead of `position`
/// to dodge the per-body `List<Num>` allocation that pins
/// short-lived garbage faster than the GC can reclaim it.
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_position_into(vm: *mut WrenVm) {
    use wlift_abi::{typed_array_bytes_mut, typed_array_kind, TypedArrayKind};

    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let out_val = slot(vm, 3);
    let offset = slot(vm, 4).as_num().unwrap_or(0.0) as usize;

    if typed_array_kind(out_val) != Some(TypedArrayKind::F32) {
        runtime_error(vm, "World3D.positionInto: out must be a Float32Array.");
        return;
    }

    let reg = d3::worlds().lock().unwrap();
    let (x, y, z) = reg
        .get(&world_id)
        .and_then(|w| {
            w.bodies_by_id.get(&body_id).map(|h| {
                let t = w.bodies[*h].translation();
                (t.x, t.y, t.z)
            })
        })
        .unwrap_or((0.0, 0.0, 0.0));
    drop(reg);

    let Some(bytes) = typed_array_bytes_mut(out_val) else {
        runtime_error(vm, "World3D.positionInto: out has no backing buffer.");
        return;
    };
    let base = offset * 4;
    if base + 12 > bytes.len() {
        runtime_error(vm, "World3D.positionInto: offset overruns the array.");
        return;
    }
    bytes[base..base + 4].copy_from_slice(&x.to_le_bytes());
    bytes[base + 4..base + 8].copy_from_slice(&y.to_le_bytes());
    bytes[base + 8..base + 12].copy_from_slice(&z.to_le_bytes());
    set_return(vm, Value::NULL);
}

/// Non-allocating rotation read — same shape as `position_into`,
/// writes 4 f32s in scalar-first (w, x, y, z) order to match
/// `@hatch:math.Quat.new(w, x, y, z)`.
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_rotation_into(vm: *mut WrenVm) {
    use wlift_abi::{typed_array_bytes_mut, typed_array_kind, TypedArrayKind};

    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let out_val = slot(vm, 3);
    let offset = slot(vm, 4).as_num().unwrap_or(0.0) as usize;

    if typed_array_kind(out_val) != Some(TypedArrayKind::F32) {
        runtime_error(vm, "World3D.rotationInto: out must be a Float32Array.");
        return;
    }

    let reg = d3::worlds().lock().unwrap();
    let (w, x, y, z) = reg
        .get(&world_id)
        .and_then(|w_| {
            w_.bodies_by_id.get(&body_id).map(|h| {
                let q = w_.bodies[*h].rotation();
                (q.w, q.x, q.y, q.z)
            })
        })
        .unwrap_or((1.0, 0.0, 0.0, 0.0));
    drop(reg);

    let Some(bytes) = typed_array_bytes_mut(out_val) else {
        runtime_error(vm, "World3D.rotationInto: out has no backing buffer.");
        return;
    };
    let base = offset * 4;
    if base + 16 > bytes.len() {
        runtime_error(vm, "World3D.rotationInto: offset overruns the array.");
        return;
    }
    bytes[base..base + 4].copy_from_slice(&w.to_le_bytes());
    bytes[base + 4..base + 8].copy_from_slice(&x.to_le_bytes());
    bytes[base + 8..base + 12].copy_from_slice(&y.to_le_bytes());
    bytes[base + 12..base + 16].copy_from_slice(&z.to_le_bytes());
    set_return(vm, Value::NULL);
}

/// Return the body's orientation as a Wren `List<Num>` in
/// scalar-first layout: `[w, x, y, z]`. Matches `@hatch:math`'s
/// `Quat.new(w, x, y, z)` constructor so the bridge can build a
/// Quat from the list directly.
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_rotation(vm: *mut WrenVm) {
    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d3::worlds().lock().unwrap();
    let (w, x, y, z) = reg
        .get(&world_id)
        .and_then(|w_| {
            w_.bodies_by_id.get(&body_id).map(|h| {
                let q = w_.bodies[*h].rotation();
                // glam-style Quat: direct (x, y, z, w) field
                // access. Repack into scalar-first (w, x, y, z)
                // to match @hatch:math's `Quat.new(w, x, y, z)`.
                (q.w, q.x, q.y, q.z)
            })
        })
        .unwrap_or((1.0, 0.0, 0.0, 0.0));
    drop(reg);
    return_list(
        vm,
        &[
            Value::num(w as f64),
            Value::num(x as f64),
            Value::num(y as f64),
            Value::num(z as f64),
        ],
    );
}

/// 2D equivalent — 2D rotations are scalars (angle in radians).
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_rotation(vm: *mut WrenVm) {
    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let reg = d2::worlds().lock().unwrap();
    let angle = reg
        .get(&world_id)
        .and_then(|w_| {
            w_.bodies_by_id
                .get(&body_id)
                .map(|h| w_.bodies[*h].rotation().angle())
        })
        .unwrap_or(0.0);
    drop(reg);
    set_return(vm, Value::num(angle as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_set_linear_velocity(vm: *mut WrenVm) {
    use rapier3d::prelude::*;

    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let z = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let mut reg = d3::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(&handle) = world.bodies_by_id.get(&body_id) {
            world.bodies[handle].set_linvel(Vector::new(x, y, z), true);
        }
    }
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_apply_impulse(vm: *mut WrenVm) {
    use rapier3d::prelude::*;

    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let z = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let mut reg = d3::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(&handle) = world.bodies_by_id.get(&body_id) {
            world.bodies[handle].apply_impulse(Vector::new(x, y, z), true);
        }
    }
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_apply_force(vm: *mut WrenVm) {
    use rapier3d::prelude::*;

    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let x = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let y = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let z = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let mut reg = d3::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(&handle) = world.bodies_by_id.get(&body_id) {
            world.bodies[handle].add_force(Vector::new(x, y, z), true);
        }
    }
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_despawn(vm: *mut WrenVm) {
    let world_id = slot(vm, 1).as_num().map(|n| n as u64).unwrap_or(0);
    let body_id = slot(vm, 2).as_num().map(|n| n as u64).unwrap_or(0);
    let mut reg = d3::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(handle) = world.bodies_by_id.remove(&body_id) {
            world.ids_by_handle.remove(&handle);
            world.bodies.remove(
                handle,
                &mut world.island_manager,
                &mut world.colliders,
                &mut world.impulse_joints,
                &mut world.multibody_joints,
                true,
            );
        }
    }
    set_return(vm, Value::NULL);
}

// ---------------------------------------------------------------------------
// Raycasts + contact events
//
// Raycasts return a 6-key result Map ({ bodyId, point[x,y,z],
// normal[x,y,z], toi }) or null when nothing was hit. The
// Wren-side `World3D.castRay` / `World2D.castRay` wraps this
// shape with a small `RaycastHit3D` / `RaycastHit2D` class.
//
// `drain_contact_events` returns the buffer of `(a, b, started)`
// triplets captured during the most recent `step` as a List of
// 3-key Maps. The buffer is cleared on each `step`, so calling
// drain twice produces an empty result the second time.
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_cast_ray(vm: *mut WrenVm) {
    use rapier3d::prelude::*;
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let ox = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
    let oy = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let oz = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let dx = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let dy = slot(vm, 6).as_num().unwrap_or(0.0) as f32;
    let dz = slot(vm, 7).as_num().unwrap_or(0.0) as f32;
    let max_toi = slot(vm, 8).as_num().unwrap_or(f64::MAX) as f32;
    // `solid` defaults to true — most game uses (projectiles,
    // line-of-sight) want to hit the inside of a shape if the
    // ray starts there. Pass `false` for sensor-style sweeps.
    let solid = slot(vm, 9).as_bool().unwrap_or(true);

    let reg = d3::worlds().lock().unwrap();
    let world = match reg.get(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            set_return(vm, Value::NULL);
            return;
        }
    };
    let qp = world.broad_phase.as_query_pipeline(
        world.narrow_phase.query_dispatcher(),
        &world.bodies,
        &world.colliders,
        QueryFilter::default(),
    );
    // rapier 0.32 / parry 0.26: `Ray::origin` is a `Vector`, not
    // a `Point` (the change came when parry switched its 3D math
    // crate). Origin + dir are both Vectors; `point_at(toi)`
    // returns `origin + dir * toi` as a `Vector`.
    let ray = Ray::new(
        Vector::new(ox, oy, oz),
        Vector::new(dx, dy, dz),
    );
    let hit = match qp.cast_ray_and_get_normal(&ray, max_toi, solid) {
        Some((collider_handle, intersection)) => {
            let body_id = world
                .colliders
                .get(collider_handle)
                .and_then(|c| c.parent())
                .and_then(|bh| world.ids_by_handle.get(&bh).copied());
            body_id.map(|id| (id, ray, intersection))
        }
        None => None,
    };
    drop(reg);
    let (body_id, ray, intersection) = match hit {
        Some(h) => h,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let toi = intersection.time_of_impact;
    let hit_point = ray.origin + ray.dir * toi;
    let nx = intersection.normal.x;
    let ny = intersection.normal.y;
    let nz = intersection.normal.z;
    emit_raycast_hit_map(vm, body_id, hit_point.x, hit_point.y, hit_point.z, nx, ny, nz, toi as f64);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_cast_ray(vm: *mut WrenVm) {
    use rapier2d::prelude::*;
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let ox = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
    let oy = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let dx = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let dy = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let max_toi = slot(vm, 6).as_num().unwrap_or(f64::MAX) as f32;
    let solid = slot(vm, 7).as_bool().unwrap_or(true);

    let reg = d2::worlds().lock().unwrap();
    let world = match reg.get(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            set_return(vm, Value::NULL);
            return;
        }
    };
    let qp = world.broad_phase.as_query_pipeline(
        world.narrow_phase.query_dispatcher(),
        &world.bodies,
        &world.colliders,
        QueryFilter::default(),
    );
    // Same shape as the 3D path — `Ray::origin` is a `Vector`.
    let ray = Ray::new(
        Vector::new(ox, oy),
        Vector::new(dx, dy),
    );
    let hit = match qp.cast_ray_and_get_normal(&ray, max_toi, solid) {
        Some((collider_handle, intersection)) => {
            let body_id = world
                .colliders
                .get(collider_handle)
                .and_then(|c| c.parent())
                .and_then(|bh| world.ids_by_handle.get(&bh).copied());
            body_id.map(|id| (id, ray, intersection))
        }
        None => None,
    };
    drop(reg);
    let (body_id, ray, intersection) = match hit {
        Some(h) => h,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let toi = intersection.time_of_impact;
    let hit_point = ray.origin + ray.dir * toi;
    // 2D ray always reports `z = 0`; matches what `World2D` users
    // expect for the `point` field.
    emit_raycast_hit_map(vm, body_id, hit_point.x, hit_point.y, 0.0,
                         intersection.normal.x, intersection.normal.y, 0.0, toi as f64);
}

// Build a Map { bodyId, point: [x,y,z], normal: [x,y,z], toi }
// and return it as slot 0. Factored to share the GC-rooted Map
// build between the 2D and 3D paths.
unsafe fn emit_raycast_hit_map(
    vm: *mut WrenVm,
    body_id: u64,
    px: f32, py: f32, pz: f32,
    nx: f32, ny: f32, nz: f32,
    toi: f64,
) {
    let m = alloc_map(vm);
    set_return(vm, m);
    let k = alloc_string(vm, "bodyId");
    map_set(vm, m, k, Value::num(body_id as f64));

    let kp = alloc_string(vm, "point");
    let lp = alloc_list(vm, 3);
    list_add(vm, lp, Value::num(px as f64));
    list_add(vm, lp, Value::num(py as f64));
    list_add(vm, lp, Value::num(pz as f64));
    map_set(vm, m, kp, lp);

    let kn = alloc_string(vm, "normal");
    let ln = alloc_list(vm, 3);
    list_add(vm, ln, Value::num(nx as f64));
    list_add(vm, ln, Value::num(ny as f64));
    list_add(vm, ln, Value::num(nz as f64));
    map_set(vm, m, kn, ln);

    let kt = alloc_string(vm, "toi");
    map_set(vm, m, kt, Value::num(toi));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_drain_contact_events(vm: *mut WrenVm) {
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            let list = alloc_list(vm, 0);
            set_return(vm, list);
            return;
        }
    };
    let mut reg = d3::worlds().lock().unwrap();
    let events = match reg.get_mut(&world_id) {
        Some(w) => std::mem::take(&mut w.contact_events),
        None => Vec::new(),
    };
    drop(reg);
    emit_contact_events_list(vm, &events);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_drain_contact_events(vm: *mut WrenVm) {
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            let list = alloc_list(vm, 0);
            set_return(vm, list);
            return;
        }
    };
    let mut reg = d2::worlds().lock().unwrap();
    let events = match reg.get_mut(&world_id) {
        Some(w) => std::mem::take(&mut w.contact_events),
        None => Vec::new(),
    };
    drop(reg);
    emit_contact_events_list(vm, &events);
}

// Returns a List< Map { a, b, started } > as slot 0. Same GC-
// rooting pattern as `emit_raycast_hit_map`: list goes into slot
// 0 first so subsequent allocs can't tear the not-yet-rooted
// Maps; each Map's keys are interned strings the alloc-string
// path retains.
unsafe fn emit_contact_events_list(vm: *mut WrenVm, events: &[(u64, u64, bool)]) {
    let list = alloc_list(vm, events.len() as u32);
    set_return(vm, list);
    for &(a, b, started) in events {
        let m = alloc_map(vm);
        list_add(vm, list, m);
        let ka = alloc_string(vm, "a");
        map_set(vm, m, ka, Value::num(a as f64));
        let kb = alloc_string(vm, "b");
        map_set(vm, m, kb, Value::num(b as f64));
        let ks = alloc_string(vm, "started");
        map_set(vm, m, ks, if started { Value::TRUE } else { Value::FALSE });
    }
}

// ---------------------------------------------------------------------------
// Shape casts
//
// Sweep a shape (ball / box / capsule) through space along a
// direction vector. Returns the first hit as a Map with the same
// `bodyId / point / normal / toi` shape `cast_ray` uses, so
// callers don't have to learn two result shapes. Shape descriptor
// is the same Map `spawnDynamic({"shape": ...})` already uses.
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_cast_shape(vm: *mut WrenVm) {
    use rapier3d::prelude::*;
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let shape_desc = slot(vm, 2);
    let ox = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let oy = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let oz = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let dx = slot(vm, 6).as_num().unwrap_or(0.0) as f32;
    let dy = slot(vm, 7).as_num().unwrap_or(0.0) as f32;
    let dz = slot(vm, 8).as_num().unwrap_or(0.0) as f32;
    let max_toi = slot(vm, 9).as_num().unwrap_or(f64::MAX) as f32;

    // Build a one-shot Collider just to reach the shape — the
    // user-facing descriptor format is identical to
    // `spawnDynamic({"shape": ...})`, and `Collider::shared_shape`
    // hands us a clone of the actual `dyn Shape` we hand to
    // `cast_shape`.
    let collider = match d3::collider_from_desc(shape_desc) {
        Some(c) => c,
        None => {
            runtime_error(
                vm,
                "World3D.castShape: shape descriptor missing or has unknown `kind`.",
            );
            return;
        }
    };
    let shape = collider.shared_shape().clone();

    let reg = d3::worlds().lock().unwrap();
    let world = match reg.get(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            set_return(vm, Value::NULL);
            return;
        }
    };
    let qp = world.broad_phase.as_query_pipeline(
        world.narrow_phase.query_dispatcher(),
        &world.bodies,
        &world.colliders,
        QueryFilter::default(),
    );
    // rapier 0.32 / parry 0.26 swapped nalgebra for glamx —
    // `Pose` is the new translation+rotation type (constructed
    // via `Pose::translation(x, y, z)`); `Vector` is `Vec3`
    // (glamx, not nalgebra), so `ShapeCastHit::normal1` is a
    // bare `Vector` rather than nalgebra's `Unit<Vector>`.
    let pose = Pose::translation(ox, oy, oz);
    let vel = Vector::new(dx, dy, dz);
    let options = rapier3d::parry::query::ShapeCastOptions {
        max_time_of_impact: max_toi,
        stop_at_penetration: true,
        ..rapier3d::parry::query::ShapeCastOptions::default()
    };
    let hit = qp.cast_shape(&pose, vel, &*shape, options);
    drop(reg);

    let (collider_handle, sc_hit) = match hit {
        Some(h) => h,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let reg = d3::worlds().lock().unwrap();
    let world = reg.get(&world_id).unwrap();
    let body_id = world
        .colliders
        .get(collider_handle)
        .and_then(|c| c.parent())
        .and_then(|bh| world.ids_by_handle.get(&bh).copied());
    drop(reg);
    let body_id = match body_id {
        Some(id) => id,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let toi = sc_hit.time_of_impact;
    let p = sc_hit.witness1;
    let n = sc_hit.normal1;
    emit_raycast_hit_map(vm, body_id, p.x, p.y, p.z, n.x, n.y, n.z, toi as f64);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_cast_shape(vm: *mut WrenVm) {
    use rapier2d::prelude::*;
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let shape_desc = slot(vm, 2);
    let ox = slot(vm, 3).as_num().unwrap_or(0.0) as f32;
    let oy = slot(vm, 4).as_num().unwrap_or(0.0) as f32;
    let dx = slot(vm, 5).as_num().unwrap_or(0.0) as f32;
    let dy = slot(vm, 6).as_num().unwrap_or(0.0) as f32;
    let max_toi = slot(vm, 7).as_num().unwrap_or(f64::MAX) as f32;

    let collider = match d2::collider_from_desc(shape_desc) {
        Some(c) => c,
        None => {
            runtime_error(
                vm,
                "World2D.castShape: shape descriptor missing or has unknown `kind`.",
            );
            return;
        }
    };
    let shape = collider.shared_shape().clone();

    let reg = d2::worlds().lock().unwrap();
    let world = match reg.get(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            set_return(vm, Value::NULL);
            return;
        }
    };
    let qp = world.broad_phase.as_query_pipeline(
        world.narrow_phase.query_dispatcher(),
        &world.bodies,
        &world.colliders,
        QueryFilter::default(),
    );
    // Mirror the 3D shape: `Pose::translation` in glamx land,
    // `Vector` is `Vec2`, normal is bare.
    let pose = Pose::translation(ox, oy);
    let vel = Vector::new(dx, dy);
    let options = rapier2d::parry::query::ShapeCastOptions {
        max_time_of_impact: max_toi,
        stop_at_penetration: true,
        ..rapier2d::parry::query::ShapeCastOptions::default()
    };
    let hit = qp.cast_shape(&pose, vel, &*shape, options);
    drop(reg);

    let (collider_handle, sc_hit) = match hit {
        Some(h) => h,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let reg = d2::worlds().lock().unwrap();
    let world = reg.get(&world_id).unwrap();
    let body_id = world
        .colliders
        .get(collider_handle)
        .and_then(|c| c.parent())
        .and_then(|bh| world.ids_by_handle.get(&bh).copied());
    drop(reg);
    let body_id = match body_id {
        Some(id) => id,
        None => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let toi = sc_hit.time_of_impact;
    let p = sc_hit.witness1;
    let n = sc_hit.normal1;
    emit_raycast_hit_map(vm, body_id, p.x, p.y, 0.0, n.x, n.y, 0.0, toi as f64);
}

// ---------------------------------------------------------------------------
// Joints
//
// One foreign-method entry point per axis. Descriptor shape:
//
//   { "kind": "fixed" | "revolute" | "prismatic" | "spherical",
//     "anchor1": [x, y, z]?,    // local anchor on body A
//     "anchor2": [x, y, z]?,    // local anchor on body B
//     "axis":    [x, y, z]?     // revolute / prismatic only
//   }
//
// Returns a u64 joint id the caller hands to
// `wlift_physics_world3d_destroy_joint`.
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_create_joint(vm: *mut WrenVm) {
    use rapier3d::prelude::*;

    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let body_id_a = match slot(vm, 2).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let body_id_b = match slot(vm, 3).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let desc = slot(vm, 4);
    let kind = match map_get(desc, "kind").and_then(string_of) {
        Some(s) => s,
        None => {
            runtime_error(vm, "World3D.createJoint: descriptor missing `kind`.");
            return;
        }
    };
    // rapier 0.32 represents local anchors as `Vec3` (glamx, a
    // body-local *offset*), not the older nalgebra `Point3`.
    // `Vector::ZERO` / `Vector::Y` are glamx constants — not
    // nalgebra associated functions — so plain const access.
    let anchor1 = map_get(desc, "anchor1")
        .and_then(read_3d)
        .map(|(x, y, z)| Vector::new(x, y, z))
        .unwrap_or(Vector::ZERO);
    let anchor2 = map_get(desc, "anchor2")
        .and_then(read_3d)
        .map(|(x, y, z)| Vector::new(x, y, z))
        .unwrap_or(Vector::ZERO);
    let axis = map_get(desc, "axis")
        .and_then(read_3d)
        .map(|(x, y, z)| Vector::new(x, y, z))
        .unwrap_or(Vector::Y);

    let joint_data: GenericJoint = match kind.as_str() {
        "fixed" => FixedJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        "revolute" => RevoluteJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        "prismatic" => PrismaticJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        "spherical" => SphericalJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        other => {
            runtime_error(
                vm,
                &format!("World3D.createJoint: unknown kind '{}'.", other),
            );
            return;
        }
    };

    let mut reg = d3::worlds().lock().unwrap();
    let world = match reg.get_mut(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            runtime_error(vm, "World3D.createJoint: unknown world id.");
            return;
        }
    };
    let ha = match world.bodies_by_id.get(&body_id_a).copied() {
        Some(h) => h,
        None => {
            drop(reg);
            runtime_error(vm, "World3D.createJoint: unknown body A id.");
            return;
        }
    };
    let hb = match world.bodies_by_id.get(&body_id_b).copied() {
        Some(h) => h,
        None => {
            drop(reg);
            runtime_error(vm, "World3D.createJoint: unknown body B id.");
            return;
        }
    };
    let handle = world
        .impulse_joints
        .insert(ha, hb, joint_data, /* wake_up = */ true);
    let id = next_id();
    world.joints_by_id.insert(id, handle);
    drop(reg);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_destroy_joint(vm: *mut WrenVm) {
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let joint_id = match slot(vm, 2).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let mut reg = d3::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(handle) = world.joints_by_id.remove(&joint_id) {
            world.impulse_joints.remove(handle, /* wake_up = */ true);
        }
    }
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_create_joint(vm: *mut WrenVm) {
    use rapier2d::prelude::*;

    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let body_id_a = match slot(vm, 2).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let body_id_b = match slot(vm, 3).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            set_return(vm, Value::NULL);
            return;
        }
    };
    let desc = slot(vm, 4);
    let kind = match map_get(desc, "kind").and_then(string_of) {
        Some(s) => s,
        None => {
            runtime_error(vm, "World2D.createJoint: descriptor missing `kind`.");
            return;
        }
    };
    // rapier 0.32: anchors are `Vec2` offsets, not `Point2`s.
    let anchor1 = map_get(desc, "anchor1")
        .and_then(read_2d)
        .map(|(x, y)| Vector::new(x, y))
        .unwrap_or(Vector::ZERO);
    let anchor2 = map_get(desc, "anchor2")
        .and_then(read_2d)
        .map(|(x, y)| Vector::new(x, y))
        .unwrap_or(Vector::ZERO);
    let axis = map_get(desc, "axis")
        .and_then(read_2d)
        .map(|(x, y)| Vector::new(x, y))
        .unwrap_or(Vector::X);

    let joint_data: GenericJoint = match kind.as_str() {
        "fixed" => FixedJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        "revolute" => RevoluteJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        "prismatic" => PrismaticJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .build()
            .into(),
        other => {
            runtime_error(
                vm,
                &format!("World2D.createJoint: unknown kind '{}'.", other),
            );
            return;
        }
    };

    let mut reg = d2::worlds().lock().unwrap();
    let world = match reg.get_mut(&world_id) {
        Some(w) => w,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.createJoint: unknown world id.");
            return;
        }
    };
    let ha = match world.bodies_by_id.get(&body_id_a).copied() {
        Some(h) => h,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.createJoint: unknown body A id.");
            return;
        }
    };
    let hb = match world.bodies_by_id.get(&body_id_b).copied() {
        Some(h) => h,
        None => {
            drop(reg);
            runtime_error(vm, "World2D.createJoint: unknown body B id.");
            return;
        }
    };
    let handle = world
        .impulse_joints
        .insert(ha, hb, joint_data, /* wake_up = */ true);
    let id = next_id();
    world.joints_by_id.insert(id, handle);
    drop(reg);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_destroy_joint(vm: *mut WrenVm) {
    let world_id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let joint_id = match slot(vm, 2).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => return,
    };
    let mut reg = d2::worlds().lock().unwrap();
    if let Some(world) = reg.get_mut(&world_id) {
        if let Some(handle) = world.joints_by_id.remove(&joint_id) {
            world.impulse_joints.remove(handle, /* wake_up = */ true);
        }
    }
    set_return(vm, Value::NULL);
}

// ---------------------------------------------------------------------------
// Static-link symbol registry (wasm only)
//
// On `wasm32-*`, plugins ship as Rust crates statically linked into
// `wlift_wasm` rather than as separate `.wasm` cdylibs — the wasm
// runtime has no `dlsym`, so foreign-method exports register
// themselves into a static table at host-init. Hosts call this
// once after `wlift_wasm`'s panic hook is installed.
// ---------------------------------------------------------------------------

/// Register every `#[no_mangle] wlift_physics_*` foreign-method
/// export under the library name `"wlift_physics"`. Idempotent —
/// re-calling overwrites the same `(library, symbol)` keys.
///
/// Only exists on `wasm32-*` targets. Host builds resolve these
/// symbols via `dlsym` and never call this; the function isn't
/// compiled there at all.
#[cfg(target_arch = "wasm32")]
pub fn register_static_symbols() {
    // register_plugin_symbol_unsafe is now wlift_abi::register_symbol below.
    // SAFETY: each export is `unsafe extern "C" fn(*mut WrenVm)` with
    // the same ABI the host build dispatches via `dlsym`.
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_create",
            wlift_physics_world2d_create as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_destroy",
            wlift_physics_world2d_destroy as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_step",
            wlift_physics_world2d_step as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_spawn_dynamic",
            wlift_physics_world2d_spawn_dynamic as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_spawn_static",
            wlift_physics_world2d_spawn_static as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_spawn_kinematic",
            wlift_physics_world2d_spawn_kinematic as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_despawn",
            wlift_physics_world2d_despawn as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_position",
            wlift_physics_world2d_position as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_rotation",
            wlift_physics_world2d_rotation as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_linear_velocity",
            wlift_physics_world2d_linear_velocity as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_set_linear_velocity",
            wlift_physics_world2d_set_linear_velocity as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_apply_impulse",
            wlift_physics_world2d_apply_impulse as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_apply_force",
            wlift_physics_world2d_apply_force as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_create",
            wlift_physics_world3d_create as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_destroy",
            wlift_physics_world3d_destroy as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_step",
            wlift_physics_world3d_step as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_spawn_dynamic",
            wlift_physics_world3d_spawn_dynamic as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_spawn_static",
            wlift_physics_world3d_spawn_static as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_spawn_kinematic",
            wlift_physics_world3d_spawn_kinematic as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_despawn",
            wlift_physics_world3d_despawn as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_position",
            wlift_physics_world3d_position as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_position_into",
            wlift_physics_world3d_position_into as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_rotation",
            wlift_physics_world3d_rotation as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_rotation_into",
            wlift_physics_world3d_rotation_into as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_linear_velocity",
            wlift_physics_world3d_linear_velocity as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_set_linear_velocity",
            wlift_physics_world3d_set_linear_velocity as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_apply_impulse",
            wlift_physics_world3d_apply_impulse as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_apply_force",
            wlift_physics_world3d_apply_force as *const (),
        )
    };

    // Raycasts + contact events. Same wasm-side registration
    // pattern as every export above.
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_cast_ray",
            wlift_physics_world3d_cast_ray as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_cast_ray",
            wlift_physics_world2d_cast_ray as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_drain_contact_events",
            wlift_physics_world3d_drain_contact_events as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_drain_contact_events",
            wlift_physics_world2d_drain_contact_events as *const (),
        )
    };

    // Shape casts + joints. Same registration pattern as every
    // export above.
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_cast_shape",
            wlift_physics_world3d_cast_shape as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_cast_shape",
            wlift_physics_world2d_cast_shape as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_create_joint",
            wlift_physics_world3d_create_joint as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world3d_destroy_joint",
            wlift_physics_world3d_destroy_joint as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_create_joint",
            wlift_physics_world2d_create_joint as *const (),
        )
    };
    unsafe {
        wlift_abi::register_symbol(
            "wlift_physics",
            "wlift_physics_world2d_destroy_joint",
            wlift_physics_world2d_destroy_joint as *const (),
        )
    };
}
