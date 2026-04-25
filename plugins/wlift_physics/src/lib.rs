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

use wren_lift::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

// ---------------------------------------------------------------------------
// Slot helpers (mirror sibling plugins)
// ---------------------------------------------------------------------------

unsafe fn slot(vm: *mut VM, index: usize) -> Value {
    unsafe {
        let stack = &(*vm).api_stack;
        stack.get(index).copied().unwrap_or(Value::null())
    }
}

unsafe fn set_return(vm: *mut VM, v: Value) {
    unsafe {
        let stack = &mut (*vm).api_stack;
        if stack.is_empty() {
            stack.push(v);
        } else {
            stack[0] = v;
        }
    }
}

unsafe fn ctx<'a>(vm: *mut VM) -> &'a mut VM {
    unsafe { &mut *vm }
}

unsafe fn string_of(v: Value) -> Option<String> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::String {
        return None;
    }
    let s = ptr as *const ObjString;
    Some(unsafe { (*s).as_str().to_string() })
}

unsafe fn map_get(v: Value, key: &str) -> Option<Value> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Map {
        return None;
    }
    let map = ptr as *const ObjMap;
    unsafe {
        for (k, val) in (*map).entries.iter() {
            if let Some(s) = string_of(k.0) {
                if s == key {
                    return Some(*val);
                }
            }
        }
    }
    None
}

unsafe fn list_view<'a>(v: Value) -> Option<&'a ObjList> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        return None;
    }
    Some(unsafe { &*(ptr as *const ObjList) })
}

unsafe fn read_num_at(list: &ObjList, i: usize, default: f64) -> f64 {
    if i >= list.count as usize {
        return default;
    }
    unsafe { *list.elements.add(i) }.as_num().unwrap_or(default)
}

unsafe fn read_2d(list_v: Value) -> Option<(f32, f32)> {
    let list = unsafe { list_view(list_v)? };
    if list.count < 2 {
        return None;
    }
    Some((
        unsafe { read_num_at(list, 0, 0.0) } as f32,
        unsafe { read_num_at(list, 1, 0.0) } as f32,
    ))
}

unsafe fn read_3d(list_v: Value) -> Option<(f32, f32, f32)> {
    let list = unsafe { list_view(list_v)? };
    if list.count < 3 {
        return None;
    }
    Some((
        unsafe { read_num_at(list, 0, 0.0) } as f32,
        unsafe { read_num_at(list, 1, 0.0) } as f32,
        unsafe { read_num_at(list, 2, 0.0) } as f32,
    ))
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
            }
        }

        pub fn step(&mut self, dt: f32) {
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
                &(),
            );
        }
    }

    pub fn worlds() -> &'static Mutex<HashMap<u64, World>> {
        static REG: OnceLock<Mutex<HashMap<u64, World>>> = OnceLock::new();
        REG.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub unsafe fn collider_from_desc(desc: Value) -> Option<Collider> {
        let kind = unsafe { map_get(desc, "kind").and_then(|v| string_of(v)) }?;
        let mut builder = match kind.as_str() {
            "ball" => {
                let r = unsafe { map_get(desc, "radius") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::ball(r)
            }
            "box" => {
                let hw = unsafe { map_get(desc, "halfWidth") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                let hh = unsafe { map_get(desc, "halfHeight") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::cuboid(hw, hh)
            }
            "capsule" => {
                let hh = unsafe { map_get(desc, "halfHeight") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                let r = unsafe { map_get(desc, "radius") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.25) as f32;
                ColliderBuilder::capsule_y(hh, r)
            }
            _ => return None,
        };
        let restitution = unsafe { map_get(desc, "restitution") }
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as f32;
        let friction = unsafe { map_get(desc, "friction") }
            .and_then(|v| v.as_num())
            .unwrap_or(0.5) as f32;
        builder = builder.restitution(restitution).friction(friction);
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
            }
        }

        pub fn step(&mut self, dt: f32) {
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
                &(),
            );
        }
    }

    pub fn worlds() -> &'static Mutex<HashMap<u64, World>> {
        static REG: OnceLock<Mutex<HashMap<u64, World>>> = OnceLock::new();
        REG.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub unsafe fn collider_from_desc(desc: Value) -> Option<Collider> {
        let kind = unsafe { map_get(desc, "kind").and_then(|v| string_of(v)) }?;
        let mut builder = match kind.as_str() {
            "ball" => {
                let r = unsafe { map_get(desc, "radius") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::ball(r)
            }
            "box" => {
                let hx = unsafe { map_get(desc, "halfX") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                let hy = unsafe { map_get(desc, "halfY") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                let hz = unsafe { map_get(desc, "halfZ") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                ColliderBuilder::cuboid(hx, hy, hz)
            }
            "capsule" => {
                let hh = unsafe { map_get(desc, "halfHeight") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.5) as f32;
                let r = unsafe { map_get(desc, "radius") }
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.25) as f32;
                ColliderBuilder::capsule_y(hh, r)
            }
            _ => return None,
        };
        let restitution = unsafe { map_get(desc, "restitution") }
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as f32;
        let friction = unsafe { map_get(desc, "friction") }
            .and_then(|v| v.as_num())
            .unwrap_or(0.5) as f32;
        builder = builder.restitution(restitution).friction(friction);
        Some(builder.build())
    }
}

// ---------------------------------------------------------------------------
// 2D foreign entry points
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_create(vm: *mut VM) {
    unsafe {
        let desc = slot(vm, 1);
        let (gx, gy) = map_get(desc, "gravity")
            .and_then(|v| read_2d(v))
            .unwrap_or((0.0, -9.81));
        let world = d2::World::new(gx, gy);
        let id = next_id();
        d2::worlds().lock().unwrap().insert(id, world);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_destroy(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => return,
        };
        d2::worlds().lock().unwrap().remove(&id);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_step(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => return,
        };
        let dt = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
        if let Some(w) = d2::worlds().lock().unwrap().get_mut(&id) {
            w.step(dt);
        }
        set_return(vm, Value::null());
    }
}

unsafe fn read_world2d_id(vm: *mut VM, slot_index: usize, label: &str) -> Option<u64> {
    unsafe {
        match slot(vm, slot_index).as_num() {
            Some(n) if n >= 0.0 => Some(n as u64),
            _ => {
                ctx(vm).runtime_error(format!(
                    "{}: world id must be a non-negative integer.",
                    label
                ));
                None
            }
        }
    }
}

unsafe fn spawn_2d(vm: *mut VM, body_kind: &str) {
    use rapier2d::prelude::*;
    unsafe {
        let world_id = match read_world2d_id(vm, 1, "World2D.spawn") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);
        let (px, py) = map_get(desc, "position")
            .and_then(|v| read_2d(v))
            .unwrap_or((0.0, 0.0));
        let (vx, vy) = map_get(desc, "linearVelocity")
            .and_then(|v| read_2d(v))
            .unwrap_or((0.0, 0.0));
        let mass = map_get(desc, "mass").and_then(|v| v.as_num());

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
                    ctx(vm).runtime_error(
                        "World2D.spawn: descriptor `shape` is missing or has unknown `kind`."
                            .to_string(),
                    );
                    return;
                }
            },
            None => {
                ctx(vm).runtime_error(
                    "World2D.spawn: descriptor must include a `shape` Map.".to_string(),
                );
                return;
            }
        };

        let mut reg = d2::worlds().lock().unwrap();
        let world = match reg.get_mut(&world_id) {
            Some(w) => w,
            None => {
                drop(reg);
                ctx(vm).runtime_error("World2D.spawn: unknown world id.".to_string());
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
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_dynamic(vm: *mut VM) {
    unsafe { spawn_2d(vm, "dynamic") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_static(vm: *mut VM) {
    unsafe { spawn_2d(vm, "static") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_spawn_kinematic(vm: *mut VM) {
    unsafe { spawn_2d(vm, "kinematic") }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_despawn(vm: *mut VM) {
    unsafe {
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
        set_return(vm, Value::null());
    }
}

unsafe fn return_2d_pair(vm: *mut VM, x: f32, y: f32) {
    unsafe {
        let context = ctx(vm);
        let list = context.alloc_list(vec![Value::num(x as f64), Value::num(y as f64)]);
        set_return(vm, list);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_position(vm: *mut VM) {
    unsafe {
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
                ctx(vm).runtime_error("World2D.position: unknown world id.".to_string());
                return;
            }
        };
        let handle = match world.bodies_by_id.get(&body_id) {
            Some(h) => *h,
            None => {
                drop(reg);
                ctx(vm).runtime_error("World2D.position: unknown body id.".to_string());
                return;
            }
        };
        let body = &world.bodies[handle];
        let t = body.translation();
        let (x, y) = (t.x, t.y);
        drop(reg);
        return_2d_pair(vm, x, y);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_linear_velocity(vm: *mut VM) {
    unsafe {
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
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_set_linear_velocity(vm: *mut VM) {
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
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_apply_impulse(vm: *mut VM) {
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
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world2d_apply_force(vm: *mut VM) {
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
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// 3D foreign entry points (mirror 2D — same Wren-side conventions).
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_create(vm: *mut VM) {
    unsafe {
        let desc = slot(vm, 1);
        let (gx, gy, gz) = map_get(desc, "gravity")
            .and_then(|v| read_3d(v))
            .unwrap_or((0.0, -9.81, 0.0));
        let world = d3::World::new(gx, gy, gz);
        let id = next_id();
        d3::worlds().lock().unwrap().insert(id, world);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_destroy(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => return,
        };
        d3::worlds().lock().unwrap().remove(&id);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_step(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => return,
        };
        let dt = slot(vm, 2).as_num().unwrap_or(0.0) as f32;
        if let Some(w) = d3::worlds().lock().unwrap().get_mut(&id) {
            w.step(dt);
        }
        set_return(vm, Value::null());
    }
}

unsafe fn spawn_3d(vm: *mut VM, body_kind: &str) {
    use rapier3d::prelude::*;
    unsafe {
        let world_id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => return,
        };
        let desc = slot(vm, 2);
        let (px, py, pz) = map_get(desc, "position")
            .and_then(|v| read_3d(v))
            .unwrap_or((0.0, 0.0, 0.0));
        let (vx, vy, vz) = map_get(desc, "linearVelocity")
            .and_then(|v| read_3d(v))
            .unwrap_or((0.0, 0.0, 0.0));
        let mass = map_get(desc, "mass").and_then(|v| v.as_num());

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
                    ctx(vm).runtime_error(
                        "World3D.spawn: descriptor `shape` is missing or has unknown `kind`."
                            .to_string(),
                    );
                    return;
                }
            },
            None => {
                ctx(vm).runtime_error(
                    "World3D.spawn: descriptor must include a `shape` Map.".to_string(),
                );
                return;
            }
        };

        let mut reg = d3::worlds().lock().unwrap();
        let world = match reg.get_mut(&world_id) {
            Some(w) => w,
            None => {
                drop(reg);
                ctx(vm).runtime_error("World3D.spawn: unknown world id.".to_string());
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
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_dynamic(vm: *mut VM) {
    unsafe { spawn_3d(vm, "dynamic") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_static(vm: *mut VM) {
    unsafe { spawn_3d(vm, "static") }
}
#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_spawn_kinematic(vm: *mut VM) {
    unsafe { spawn_3d(vm, "kinematic") }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_position(vm: *mut VM) {
    unsafe {
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
        let context = ctx(vm);
        let list = context.alloc_list(vec![
            Value::num(x as f64),
            Value::num(y as f64),
            Value::num(z as f64),
        ]);
        set_return(vm, list);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_linear_velocity(vm: *mut VM) {
    unsafe {
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
        let context = ctx(vm);
        let list = context.alloc_list(vec![
            Value::num(vx as f64),
            Value::num(vy as f64),
            Value::num(vz as f64),
        ]);
        set_return(vm, list);
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_set_linear_velocity(vm: *mut VM) {
    use rapier3d::prelude::*;
    unsafe {
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
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_apply_impulse(vm: *mut VM) {
    use rapier3d::prelude::*;
    unsafe {
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
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_apply_force(vm: *mut VM) {
    use rapier3d::prelude::*;
    unsafe {
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
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_physics_world3d_despawn(vm: *mut VM) {
    unsafe {
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
        set_return(vm, Value::null());
    }
}
