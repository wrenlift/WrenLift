//! Optional `toml` module — TOML parse + encode backed by the
//! Rust `toml` crate. Wren-side type mapping:
//!
//!   TOML value        Wren value
//!   ----------        ----------
//!   string            String
//!   integer           Num (i64 → f64)
//!   float             Num
//!   boolean           Bool
//!   datetime          String (RFC 3339 form)
//!   array             List
//!   table             Map (String keys)
//!
//! `@hatch:toml` layers idiomatic naming on top (Toml.parse /
//! Toml.encode).

use toml::Value as TomlValue;

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- TOML → Wren ------------------------------------------------

fn toml_to_wren(ctx: &mut dyn NativeContext, v: &TomlValue) -> Value {
    match v {
        TomlValue::String(s) => ctx.alloc_string(s.clone()),
        TomlValue::Integer(i) => Value::num(*i as f64),
        TomlValue::Float(f) => Value::num(*f),
        TomlValue::Boolean(b) => Value::bool(*b),
        // Datetimes get a lossy String representation — Wren's Time
        // class can parse the ISO form if callers care. Roundtrips
        // through `Toml.encode` will produce a string field unless
        // the caller opts back in to a proper datetime shape.
        TomlValue::Datetime(dt) => ctx.alloc_string(dt.to_string()),
        TomlValue::Array(items) => {
            let values: Vec<Value> = items.iter().map(|it| toml_to_wren(ctx, it)).collect();
            ctx.alloc_list(values)
        }
        TomlValue::Table(table) => {
            let out = ctx.alloc_map();
            let out_ptr = out.as_object().unwrap() as *mut ObjMap;
            // Deterministic iteration order matches the TOML file
            // order — toml::Table is a BTreeMap internally.
            for (k, val) in table.iter() {
                let key = ctx.alloc_string(k.clone());
                let v = toml_to_wren(ctx, val);
                unsafe { (*out_ptr).set(key, v) };
            }
            out
        }
    }
}

// --- Wren → TOML ------------------------------------------------

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

fn wren_to_toml(v: Value) -> Result<TomlValue, String> {
    if v.is_null() {
        return Err("null isn't a representable TOML value (TOML has no null).".to_string());
    }
    if let Some(b) = v.as_bool() {
        return Ok(TomlValue::Boolean(b));
    }
    if let Some(n) = v.as_num() {
        // Preserve integer vs float distinction: round-number Nums
        // serialise as Integer, everything else as Float. Matches
        // what @hatch:json does for the same ambiguity.
        if n.is_finite() && n == n.trunc() && (i64::MIN as f64..=i64::MAX as f64).contains(&n) {
            return Ok(TomlValue::Integer(n as i64));
        }
        if n.is_nan() {
            return Err("TOML can't represent NaN.".to_string());
        }
        if n.is_infinite() {
            return Err("TOML can't represent Infinity.".to_string());
        }
        return Ok(TomlValue::Float(n));
    }
    if !v.is_object() {
        return Err("unsupported Wren value type for TOML encode.".to_string());
    }
    let ptr = v.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    match unsafe { (*header).obj_type } {
        ObjType::String => {
            let s = ptr as *const ObjString;
            Ok(TomlValue::String(unsafe { (*s).as_str().to_string() }))
        }
        ObjType::List => {
            let lst = ptr as *const ObjList;
            let (count, data) = unsafe { ((*lst).count as usize, (*lst).elements) };
            let mut items: Vec<TomlValue> = Vec::with_capacity(count);
            for i in 0..count {
                let elem = unsafe { *data.add(i) };
                items.push(wren_to_toml(elem)?);
            }
            Ok(TomlValue::Array(items))
        }
        ObjType::Map => {
            let m = ptr as *const ObjMap;
            // Collect entries out of the inner HashMap so we can
            // drop the pointer borrow before recursing.
            let entries: Vec<(Value, Value)> =
                unsafe { (*m).entries.iter().map(|(k, v)| (k.0, *v)).collect() };
            let mut table = toml::map::Map::new();
            for (k, v) in entries {
                let key = match unsafe { string_of(k) } {
                    Some(s) => s,
                    None => {
                        return Err("TOML table keys must be strings.".to_string());
                    }
                };
                let value = wren_to_toml(v)?;
                table.insert(key, value);
            }
            Ok(TomlValue::Table(table))
        }
        _ => Err("unsupported Wren object type for TOML encode.".to_string()),
    }
}

// --- Native functions ------------------------------------------

fn toml_parse(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(text) = super::validate_string(ctx, args[1], "Toml.parse") else {
        return Value::null();
    };
    match text.parse::<TomlValue>() {
        Ok(v) => toml_to_wren(ctx, &v),
        Err(e) => {
            ctx.runtime_error(format!("Toml.parse: {}", e));
            Value::null()
        }
    }
}

fn toml_encode(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // Top-level must be a Map — TOML documents are always tables.
    // Accepting a raw array / scalar would just produce an error
    // from the crate; catching it here gives a clearer message.
    let value = match wren_to_toml(args[1]) {
        Ok(v) => v,
        Err(e) => {
            ctx.runtime_error(format!("Toml.encode: {}", e));
            return Value::null();
        }
    };
    let TomlValue::Table(table) = value else {
        ctx.runtime_error(
            "Toml.encode: top-level value must be a Map (TOML documents are tables).".to_string(),
        );
        return Value::null();
    };
    match toml::to_string(&table) {
        Ok(s) => ctx.alloc_string(s),
        Err(e) => {
            ctx.runtime_error(format!("Toml.encode: {}", e));
            Value::null()
        }
    }
}

fn toml_encode_pretty(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let value = match wren_to_toml(args[1]) {
        Ok(v) => v,
        Err(e) => {
            ctx.runtime_error(format!("Toml.encodePretty: {}", e));
            return Value::null();
        }
    };
    let TomlValue::Table(table) = value else {
        ctx.runtime_error("Toml.encodePretty: top-level value must be a Map.".to_string());
        return Value::null();
    };
    match toml::to_string_pretty(&table) {
        Ok(s) => ctx.alloc_string(s),
        Err(e) => {
            ctx.runtime_error(format!("Toml.encodePretty: {}", e));
            Value::null()
        }
    }
}

// --- Registration ----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("TomlCore", vm.object_class);
    vm.primitive_static(class, "parse(_)", toml_parse);
    vm.primitive_static(class, "encode(_)", toml_encode);
    vm.primitive_static(class, "encodePretty(_)", toml_encode_pretty);
    class
}
