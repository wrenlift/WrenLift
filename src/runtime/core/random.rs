/// Optional `random` module — provides the `Random` class (WELL512a PRNG).
///
/// Loaded on-demand when `import "random" for Random` is encountered.
use crate::runtime::object::{NativeContext, ObjClass, ObjForeign, ObjHeader, ObjList, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// ---------------------------------------------------------------------------
// WELL512a state (16 × u32 = 64 bytes + 4 byte index = 68 bytes)
// ---------------------------------------------------------------------------

const STATE_SIZE: usize = 16;

struct Well512 {
    state: [u32; STATE_SIZE],
    index: u32,
}

impl Well512 {
    fn from_seed(seed: u64) -> Self {
        let mut state = [0u32; STATE_SIZE];
        // Seed using a simple SplitMix-style expansion
        let mut s = seed;
        for slot in &mut state {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (s >> 33) as u32;
        }
        Self { state, index: 0 }
    }

    fn from_seed_list(seeds: &[f64]) -> Self {
        let mut state = [0u32; STATE_SIZE];
        for (i, &v) in seeds.iter().enumerate().take(STATE_SIZE) {
            state[i] = v.to_bits() as u32;
        }
        // If fewer than 16 seeds, cycle them
        if !seeds.is_empty() {
            let n = seeds.len().min(STATE_SIZE);
            for i in n..STATE_SIZE {
                state[i] = state[i % n];
            }
        }
        Self { state, index: 0 }
    }

    fn next_u32(&mut self) -> u32 {
        let i = self.index as usize;
        let mut a = self.state[i];
        let c = self.state[(i + 13) & 15];
        let b = a ^ c ^ (a << 16) ^ (c << 15);
        let c2 = self.state[(i + 9) & 15];
        let d = c2 ^ (c2 >> 11);
        self.state[i] = b ^ d;
        a = self.state[i];
        let e = a ^ ((a << 5) & 0xDA442D24);
        self.index = ((i + 15) & 15) as u32;
        let idx = self.index as usize;
        self.state[idx] ^= self.state[idx] ^ e;
        self.state[idx]
    }

    /// Returns a float in [0, 1)
    fn next_float(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64 + 1.0)
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(68);
        for &s in &self.state {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes.extend_from_slice(&self.index.to_le_bytes());
        bytes
    }

    fn from_bytes(data: &[u8]) -> Self {
        let mut state = [0u32; STATE_SIZE];
        for (i, chunk) in data[..64].chunks_exact(4).enumerate() {
            state[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let index = u32::from_le_bytes([data[64], data[65], data[66], data[67]]);
        Self { state, index }
    }
}

fn get_prng(args: &[Value]) -> Option<Well512> {
    if !args[0].is_object() {
        return None;
    }
    let ptr = args[0].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::Foreign {
            return None;
        }
        let foreign = &*(ptr as *const ObjForeign);
        if foreign.data.len() < 68 {
            return None;
        }
        Some(Well512::from_bytes(&foreign.data))
    }
}

fn save_prng(args: &[Value], prng: &Well512) {
    if !args[0].is_object() {
        return;
    }
    let ptr = args[0].as_object().unwrap();
    unsafe {
        let foreign = &mut *(ptr as *mut ObjForeign);
        foreign.data = prng.to_bytes();
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Random.new() — seed from current time
fn random_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let prng = Well512::from_seed(seed);
    let foreign = ctx.alloc_foreign(prng.to_bytes());
    // Set class to Random (args[0] is the class for static methods)
    if let Some(class_ptr) = args[0].as_object() {
        unsafe {
            (*foreign).header.class = class_ptr as *mut ObjClass;
        }
    }
    Value::object(foreign as *mut u8)
}

/// Random.new(seed) — seed from number or list
fn random_new_seed(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let seed_val = args[1];
    let prng = if seed_val.is_num() {
        Well512::from_seed(seed_val.as_num().unwrap().to_bits())
    } else if seed_val.is_object() {
        let ptr = seed_val.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        if unsafe { (*header).obj_type } == ObjType::List {
            let list = unsafe { &*(ptr as *const ObjList) };
            let seeds: Vec<f64> = (0..list.len())
                .filter_map(|i| list.get(i).and_then(|v| v.as_num()))
                .collect();
            Well512::from_seed_list(&seeds)
        } else {
            ctx.runtime_error("Seed must be a number or a list of numbers.".into());
            return Value::null();
        }
    } else {
        ctx.runtime_error("Seed must be a number or a list of numbers.".into());
        return Value::null();
    };
    let foreign = ctx.alloc_foreign(prng.to_bytes());
    // Set class to Random (args[0] is the class for static methods)
    if let Some(class_ptr) = args[0].as_object() {
        unsafe {
            (*foreign).header.class = class_ptr as *mut ObjClass;
        }
    }
    Value::object(foreign as *mut u8)
}

// ---------------------------------------------------------------------------
// Instance methods
// ---------------------------------------------------------------------------

/// random.float() → [0, 1)
fn random_float(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let result = prng.next_float();
    save_prng(args, &prng);
    Value::num(result)
}

/// random.float(end) → [0, end)
fn random_float_end(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(end) = super::validate_num(ctx, args[1], "End") else {
        return Value::null();
    };
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let result = prng.next_float() * end;
    save_prng(args, &prng);
    Value::num(result)
}

/// random.float(start, end) → [start, end)
fn random_float_range(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(start) = super::validate_num(ctx, args[1], "Start") else {
        return Value::null();
    };
    let Some(end) = super::validate_num(ctx, args[2], "End") else {
        return Value::null();
    };
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let result = start + prng.next_float() * (end - start);
    save_prng(args, &prng);
    Value::num(result)
}

/// random.int(end) → [0, end)
fn random_int_end(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(end) = super::validate_int(ctx, args[1], "End") else {
        return Value::null();
    };
    if end <= 0 {
        ctx.runtime_error("End must be positive.".into());
        return Value::null();
    }
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let result = (prng.next_float() * end as f64) as i64;
    save_prng(args, &prng);
    Value::num(result as f64)
}

/// random.int(start, end) → [start, end)
fn random_int_range(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(start) = super::validate_int(ctx, args[1], "Start") else {
        return Value::null();
    };
    let Some(end) = super::validate_int(ctx, args[2], "End") else {
        return Value::null();
    };
    if end <= start {
        ctx.runtime_error("Range cannot be empty.".into());
        return Value::null();
    }
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let range = (end - start) as f64;
    let result = start + (prng.next_float() * range) as i64;
    save_prng(args, &prng);
    Value::num(result as f64)
}

/// random.sample(list)
fn random_sample(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let ptr = args[1].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let list = unsafe { &*(ptr as *const ObjList) };
    let count = list.len();
    if count == 0 {
        ctx.runtime_error("List cannot be empty.".into());
        return Value::null();
    }
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    let idx = (prng.next_float() * count as f64) as usize;
    save_prng(args, &prng);
    list.get(idx).unwrap_or(Value::null())
}

/// random.sample(list, count) — without replacement
fn random_sample_count(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let ptr = args[1].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let list = unsafe { &*(ptr as *const ObjList) };
    let Some(n) = super::validate_int(ctx, args[2], "Count") else {
        return Value::null();
    };
    let n = n as usize;
    if n > list.len() {
        ctx.runtime_error("Not enough elements in the list.".into());
        return Value::null();
    }
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    // Fisher-Yates partial shuffle on a copy
    let mut items: Vec<Value> = (0..list.len())
        .map(|i| list.get(i).unwrap_or(Value::null()))
        .collect();
    for i in 0..n {
        let j = i + (prng.next_float() * (items.len() - i) as f64) as usize;
        items.swap(i, j);
    }
    save_prng(args, &prng);
    let result = ctx.new_list();
    let result_ptr = result.as_object().unwrap() as *mut ObjList;
    for &v in &items[..n] {
        unsafe { (*result_ptr).add(v) };
    }
    result
}

/// random.shuffle(list) — in-place Fisher-Yates
fn random_shuffle(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let ptr = args[1].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        ctx.runtime_error("Argument must be a list.".into());
        return Value::null();
    }
    let list = unsafe { &mut *(ptr as *mut ObjList) };
    let len = list.len();
    let Some(mut prng) = get_prng(args) else {
        ctx.runtime_error("Expected a Random instance.".into());
        return Value::null();
    };
    for i in (1..len).rev() {
        let j = (prng.next_float() * (i + 1) as f64) as usize;
        let a = list.get(i).unwrap_or(Value::null());
        let b = list.get(j).unwrap_or(Value::null());
        list.set(i, b);
        list.set(j, a);
    }
    save_prng(args, &prng);
    Value::null()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("Random", vm.object_class);

    // Constructors (static)
    vm.primitive_static(class, "new", random_new);
    vm.primitive_static(class, "new(_)", random_new_seed);

    // Instance methods
    vm.primitive(class, "float()", random_float);
    vm.primitive(class, "float(_)", random_float_end);
    vm.primitive(class, "float(_,_)", random_float_range);
    vm.primitive(class, "int(_)", random_int_end);
    vm.primitive(class, "int(_,_)", random_int_range);
    vm.primitive(class, "sample(_)", random_sample);
    vm.primitive(class, "sample(_,_)", random_sample_count);
    vm.primitive(class, "shuffle(_)", random_shuffle);

    class
}
