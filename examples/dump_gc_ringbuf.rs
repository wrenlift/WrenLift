//! Post-mortem dumper for the GC ring buffer file.
//!
//! Usage: `cargo run --release --features aot --example dump_gc_ringbuf -- <path>`
//!
//! Prints the events in chronological order (oldest first) — typically
//! the last ~256 GC operations before a SIGSEGV.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_gc_ringbuf <path>");
    wren_lift::runtime::gc_ringbuf::dump_file(&path).expect("dump_file");
}
