//! Audio plugin for WrenLift. Built as a cdylib, bundled into
//! @hatch:audio. Output via cpal; WAV decode via hound.
//!
//! Host interaction goes through `wlift_abi`'s C-ABI shim — no
//! `wren_lift` Rust dep, so VM/Gc layout changes can't reach
//! plugin code.
//!
//! # Architecture
//!
//! One global `AudioContext` per process — opens cpal's default
//! output stream, runs a tiny mixer in cpal's audio thread that
//! reads from a `Mutex<Vec<Voice>>`. Voices reference cached PCM
//! samples by id; play() pushes a Voice into the mix list, the
//! audio thread advances each voice every callback.
//!
//! Format: f32 stereo at the device's preferred sample rate. WAV
//! sources are resampled with a naive nearest-neighbour pass
//! during decode (no rate-conversion library on the dependency
//! list yet); pitch shift on rate mismatch is on the v0 list of
//! known limitations.

#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use wlift_abi::{
    list_count, list_get, map_iter, obj_type, runtime_error, set_return, slot, string_str,
    typed_array_bytes, typed_array_kind, ObjType, TypedArrayKind, Value, WrenVm,
};

/// Plugin ABI handshake — see wlift_gpu::wlift_plugin_abi_version.
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_get(map: Value, key: &str) -> Option<Value> {
    if obj_type(map) != Some(ObjType::Map) {
        return None;
    }
    for (k, val) in map_iter(map) {
        if let Some(s) = string_str(k) {
            if s == key {
                return Some(val);
            }
        }
    }
    None
}

unsafe fn read_byte_buffer(vm: *mut WrenVm, v: Value, label: &str) -> Option<Vec<u8>> {
    if !v.is_object() {
        runtime_error(
            vm,
            &format!("{}: expected a List<Num> or ByteArray.", label),
        );
        return None;
    }
    match obj_type(v) {
        Some(ObjType::List) => {
            let n = list_count(v);
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let entry = list_get(v, i);
                let byte = match entry.as_num() {
                    Some(v) if (0.0..=255.0).contains(&v) && v.fract() == 0.0 => v as u8,
                    _ => {
                        runtime_error(vm, &format!("{}: byte {} not in 0..=255.", label, i));
                        return None;
                    }
                };
                out.push(byte);
            }
            Some(out)
        }
        Some(ObjType::TypedArray) => {
            if typed_array_kind(v) == Some(TypedArrayKind::U8) {
                Some(typed_array_bytes(v).map(|b| b.to_vec()).unwrap_or_default())
            } else {
                runtime_error(
                    vm,
                    &format!("{}: typed array must be ByteArray (u8).", label),
                );
                None
            }
        }
        _ => {
            runtime_error(
                vm,
                &format!("{}: expected a List<Num> or ByteArray.", label),
            );
            None
        }
    }
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Mixer state — shared between the Wren thread and cpal's audio thread
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MixerState {
    /// Decoded PCM samples per Sound id. Stereo, f32, interleaved.
    /// Empty Vec means "loaded but silent" — never None.
    sounds: HashMap<u64, Arc<Vec<f32>>>,
    /// Live voices.
    voices: Vec<Voice>,
}

struct Voice {
    /// Kept for diagnostics + future "stop voices for sound id"
    /// queries. The mixer doesn't read it on the hot path.
    #[allow(dead_code)]
    sound_id: u64,
    samples: Arc<Vec<f32>>,
    cursor: usize, // index into samples (already in stereo frames * 2)
    volume: f32,
    looping: bool,
}

static MIXER: OnceLock<Mutex<MixerState>> = OnceLock::new();
static STREAM: OnceLock<Mutex<StreamHolder>> = OnceLock::new();

fn mixer() -> &'static Mutex<MixerState> {
    MIXER.get_or_init(|| Mutex::new(MixerState::default()))
}

/// We don't expose the `cpal::Stream` to safe Rust — it isn't
/// `Send` on macOS. Wrap it inside a Mutex on a thread-local
/// stash; we only ever touch it from the thread that called
/// `Audio.context()`, which is the Wren main thread.
struct StreamHolder {
    stream: Option<cpal::Stream>,
    sample_rate: u32,
    channels: u16,
}

unsafe impl Send for StreamHolder {}

fn stream_holder() -> &'static Mutex<StreamHolder> {
    STREAM.get_or_init(|| {
        Mutex::new(StreamHolder {
            stream: None,
            sample_rate: 44100,
            channels: 2,
        })
    })
}

// ---------------------------------------------------------------------------
// Audio context
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_context_init(vm: *mut WrenVm) {
    let mut holder = stream_holder().lock().unwrap();
    if holder.stream.is_some() {
        // Already initialised — return idempotently. The caller can
        // pump `Audio.context()` repeatedly and only the first one
        // builds the stream.
        set_return(vm, Value::bool(true));
        return;
    }

    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            runtime_error(vm, "Audio.context: no default output device.");
            return;
        }
    };
    let config = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => {
            runtime_error(vm, &format!("Audio.context: default config: {}", e));
            return;
        }
    };
    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();

    let err_fn = |err| eprintln!("wlift_audio: stream error: {}", err);
    let stream_result = match format {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                fill_buffer_f32(data, channels);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [i16], _| {
                let mut tmp = vec![0.0f32; data.len()];
                fill_buffer_f32(&mut tmp, channels);
                for (out, src) in data.iter_mut().zip(tmp.iter()) {
                    let clamped = src.clamp(-1.0, 1.0);
                    *out = (clamped * i16::MAX as f32) as i16;
                }
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [u16], _| {
                let mut tmp = vec![0.0f32; data.len()];
                fill_buffer_f32(&mut tmp, channels);
                for (out, src) in data.iter_mut().zip(tmp.iter()) {
                    let clamped = src.clamp(-1.0, 1.0);
                    let signed = (clamped * i16::MAX as f32) as i32;
                    *out = (signed + 32768) as u16;
                }
            },
            err_fn,
            None,
        ),
        other => {
            runtime_error(
                vm,
                &format!("Audio.context: unsupported sample format {:?}.", other),
            );
            return;
        }
    };
    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            runtime_error(vm, &format!("Audio.context: build_output_stream: {}", e));
            return;
        }
    };
    if let Err(e) = stream.play() {
        runtime_error(vm, &format!("Audio.context: stream.play: {}", e));
        return;
    }
    holder.stream = Some(stream);
    holder.sample_rate = sample_rate;
    holder.channels = channels;
    set_return(vm, Value::bool(true));
}

/// Mix every active voice into `data` and advance their cursors.
fn fill_buffer_f32(data: &mut [f32], device_channels: u16) {
    for s in data.iter_mut() {
        *s = 0.0;
    }
    let mut mix = mixer().lock().unwrap();
    if mix.voices.is_empty() {
        return;
    }
    let dc = device_channels as usize;
    let frame_count = data.len() / dc.max(1);
    let mut i = 0;
    while i < mix.voices.len() {
        let v = &mut mix.voices[i];
        let src = v.samples.as_slice();
        let total_frames = src.len() / 2;
        let mut local_cursor = v.cursor;
        let mut emitted = 0usize;
        while emitted < frame_count && local_cursor < src.len() {
            let l = src[local_cursor];
            let r = if local_cursor + 1 < src.len() {
                src[local_cursor + 1]
            } else {
                l
            };
            let base = emitted * dc;
            match dc {
                1 => {
                    data[base] += 0.5 * (l + r) * v.volume;
                }
                2 => {
                    data[base] += l * v.volume;
                    data[base + 1] += r * v.volume;
                }
                _ => {
                    data[base] += l * v.volume;
                    if dc > 1 {
                        data[base + 1] += r * v.volume;
                    }
                }
            }
            local_cursor += 2;
            emitted += 1;
        }
        v.cursor = local_cursor;
        let consumed_all = v.cursor >= src.len();
        if consumed_all {
            if v.looping && total_frames > 0 {
                v.cursor = 0;
                i += 1;
            } else {
                mix.voices.swap_remove(i);
            }
        } else {
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Sound — load + play
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_sound_load(vm: *mut WrenVm) {
    let bytes = match read_byte_buffer(vm, slot(vm, 1), "Sound.load") {
        Some(b) => b,
        None => return,
    };
    let mut reader = match hound::WavReader::new(Cursor::new(&bytes)) {
        Ok(r) => r,
        Err(e) => {
            runtime_error(vm, &format!("Sound.load: WAV parse: {}", e));
            return;
        }
    };
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let bits = spec.bits_per_sample;
    let format = spec.sample_format;

    let mut samples: Vec<f32> = Vec::new();
    match format {
        hound::SampleFormat::Int => {
            let max = match bits {
                8 => 127.0,
                16 => 32767.0,
                24 => 8_388_607.0,
                32 => 2_147_483_647.0,
                _ => 32767.0,
            };
            for s in reader.samples::<i32>() {
                match s {
                    Ok(v) => samples.push(v as f32 / max),
                    Err(e) => {
                        runtime_error(vm, &format!("Sound.load: sample read: {}", e));
                        return;
                    }
                }
            }
        }
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                match s {
                    Ok(v) => samples.push(v),
                    Err(e) => {
                        runtime_error(vm, &format!("Sound.load: sample read: {}", e));
                        return;
                    }
                }
            }
        }
    }
    // Mono → stereo expansion (duplicate). >2 channels gets
    // down-mixed by averaging the first two for now.
    let stereo: Vec<f32> = if channels == 1 {
        samples.iter().flat_map(|&s| [s, s]).collect()
    } else if channels == 2 {
        samples
    } else {
        let frame_count = samples.len() / channels;
        let mut out = Vec::with_capacity(frame_count * 2);
        for i in 0..frame_count {
            let base = i * channels;
            out.push(samples[base]);
            out.push(samples[base + 1]);
        }
        out
    };

    let id = next_id();
    mixer().lock().unwrap().sounds.insert(id, Arc::new(stereo));
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_sound_unload(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Sound.unload: id must be a non-negative number.");
            return;
        }
    };
    mixer().lock().unwrap().sounds.remove(&id);
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_play(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Audio.play: id must be a non-negative number.");
            return;
        }
    };
    let options = slot(vm, 2);
    let volume = map_get(options, "volume")
        .and_then(|v| v.as_num())
        .map(|n| n as f32)
        .unwrap_or(1.0);
    let looping = map_get(options, "loop")
        .map(|v| !is_falsy(v))
        .unwrap_or(false);

    let mut mix = mixer().lock().unwrap();
    let samples = match mix.sounds.get(&id) {
        Some(s) => Arc::clone(s),
        None => {
            drop(mix);
            runtime_error(vm, "Audio.play: unknown sound id.");
            return;
        }
    };
    mix.voices.push(Voice {
        sound_id: id,
        samples,
        cursor: 0,
        volume,
        looping,
    });
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_stop_all(vm: *mut WrenVm) {
    mixer().lock().unwrap().voices.clear();
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_audio_active_voices(vm: *mut WrenVm) {
    let n = mixer().lock().unwrap().voices.len();
    set_return(vm, Value::num(n as f64));
}

/// Wren truthiness — anything other than `null` / `false` is truthy.
fn is_falsy(v: Value) -> bool {
    v.is_null() || v == Value::FALSE
}
