//! The filesystem and environment a page gives the browser build, in
//! memory: what `fs` and `os` run against where there is no WASI. The
//! functions mirror the `std::fs` ones those modules call, so the same
//! code runs over either. Paths are absolute from `/`, which is also the
//! working directory; a relative path is taken from there.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

struct Tree {
    files: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
    env: BTreeMap<String, String>,
}

static TREE: Mutex<Option<Tree>> = Mutex::new(None);

fn tree() -> MutexGuard<'static, Option<Tree>> {
    let mut guard = TREE.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(|| Tree {
        files: BTreeMap::new(),
        dirs: ["/", "/home", "/tmp"]
            .iter()
            .map(|d| d.to_string())
            .collect(),
        env: [("HOME", "/home"), ("TMPDIR", "/tmp")]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    });
    guard
}

fn with<T>(f: impl FnOnce(&mut Tree) -> T) -> T {
    f(tree().as_mut().expect("initialised"))
}

/// `path` absolute and without `.`, `..` or repeated separators.
fn normal(path: &Path) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.to_str().unwrap_or("").split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    format!("/{}", parts.join("/"))
}

fn parent(path: &str) -> String {
    match path.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(i) => path[..i].to_string(),
    }
}

fn not_found(path: &str) -> Error {
    Error::new(
        ErrorKind::NotFound,
        format!("{path}: no such file or directory"),
    )
}

fn fail(kind: ErrorKind, what: String) -> Error {
    Error::new(kind, what)
}

pub fn read(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let p = normal(path.as_ref());
    with(|t| t.files.get(&p).cloned().ok_or_else(|| not_found(&p)))
}

pub fn read_to_string(path: impl AsRef<Path>) -> Result<String> {
    String::from_utf8(read(path)?).map_err(|e| fail(ErrorKind::InvalidData, e.to_string()))
}

pub fn write(path: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| {
        if t.dirs.contains(&p) {
            return Err(fail(
                ErrorKind::IsADirectory,
                format!("{p}: is a directory"),
            ));
        }
        if !t.dirs.contains(&parent(&p)) {
            return Err(not_found(&parent(&p)));
        }
        t.files.insert(p, bytes.as_ref().to_vec());
        Ok(())
    })
}

pub struct Metadata {
    len: u64,
    dir: bool,
}

// The shape of `std::fs::Metadata`, which has no `is_empty` either.
#[allow(clippy::len_without_is_empty)]
impl Metadata {
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_dir(&self) -> bool {
        self.dir
    }
    pub fn is_file(&self) -> bool {
        !self.dir
    }
}

pub fn metadata(path: impl AsRef<Path>) -> Result<Metadata> {
    let p = normal(path.as_ref());
    with(|t| {
        if t.dirs.contains(&p) {
            Ok(Metadata { len: 0, dir: true })
        } else {
            t.files
                .get(&p)
                .map(|b| Metadata {
                    len: b.len() as u64,
                    dir: false,
                })
                .ok_or_else(|| not_found(&p))
        }
    })
}

/// There are no links, so the same as [`metadata`].
pub fn symlink_metadata(path: impl AsRef<Path>) -> Result<Metadata> {
    metadata(path)
}

pub struct DirEntry(String);

impl DirEntry {
    pub fn file_name(&self) -> OsString {
        OsString::from(&self.0)
    }
}

pub fn read_dir(path: impl AsRef<Path>) -> Result<std::vec::IntoIter<Result<DirEntry>>> {
    let p = normal(path.as_ref());
    with(|t| {
        if !t.dirs.contains(&p) {
            return Err(not_found(&p));
        }
        let names = t
            .files
            .keys()
            .chain(t.dirs.iter())
            .filter(|e| *e != &p && parent(e) == p)
            .map(|e| Ok(DirEntry(e.rsplit('/').next().unwrap_or("").to_string())))
            .collect::<Vec<_>>();
        Ok(names.into_iter())
    })
}

pub fn create_dir(path: impl AsRef<Path>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| {
        if t.dirs.contains(&p) || t.files.contains_key(&p) {
            return Err(fail(ErrorKind::AlreadyExists, format!("{p}: exists")));
        }
        if !t.dirs.contains(&parent(&p)) {
            return Err(not_found(&parent(&p)));
        }
        t.dirs.insert(p);
        Ok(())
    })
}

pub fn create_dir_all(path: impl AsRef<Path>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| {
        let mut at = String::new();
        for part in p.split('/').filter(|s| !s.is_empty()) {
            at = format!("{at}/{part}");
            if t.files.contains_key(&at) {
                return Err(fail(ErrorKind::NotADirectory, format!("{at}: is a file")));
            }
            t.dirs.insert(at.clone());
        }
        Ok(())
    })
}

pub fn remove_file(path: impl AsRef<Path>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| t.files.remove(&p).map(|_| ()).ok_or_else(|| not_found(&p)))
}

pub fn remove_dir(path: impl AsRef<Path>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| {
        if !t.dirs.contains(&p) {
            return Err(not_found(&p));
        }
        let occupied = t
            .files
            .keys()
            .chain(t.dirs.iter())
            .any(|e| e != &p && parent(e) == p);
        if occupied {
            return Err(fail(
                ErrorKind::DirectoryNotEmpty,
                format!("{p}: directory not empty"),
            ));
        }
        t.dirs.remove(&p);
        Ok(())
    })
}

pub fn remove_dir_all(path: impl AsRef<Path>) -> Result<()> {
    let p = normal(path.as_ref());
    with(|t| {
        if !t.dirs.contains(&p) {
            return Err(not_found(&p));
        }
        let under = format!("{}/", p.trim_end_matches('/'));
        t.files.retain(|f, _| !f.starts_with(&under));
        t.dirs.retain(|d| d != &p && !d.starts_with(&under));
        t.dirs.insert("/".to_string());
        Ok(())
    })
}

pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    let (from, to) = (normal(from.as_ref()), normal(to.as_ref()));
    with(|t| {
        if !t.dirs.contains(&parent(&to)) {
            return Err(not_found(&parent(&to)));
        }
        if let Some(bytes) = t.files.remove(&from) {
            t.files.insert(to, bytes);
            return Ok(());
        }
        if !t.dirs.contains(&from) {
            return Err(not_found(&from));
        }
        let under = format!("{from}/");
        let moved = |e: &String| format!("{to}{}", &e[from.len()..]);
        let files: Vec<String> = t
            .files
            .keys()
            .filter(|f| f.starts_with(&under))
            .cloned()
            .collect();
        for f in files {
            let bytes = t.files.remove(&f).unwrap_or_default();
            t.files.insert(moved(&f), bytes);
        }
        let dirs: Vec<String> = t
            .dirs
            .iter()
            .filter(|d| *d == &from || d.starts_with(&under))
            .cloned()
            .collect();
        for d in dirs {
            t.dirs.remove(&d);
            t.dirs.insert(moved(&d));
        }
        Ok(())
    })
}

/// The working directory, which is the root.
pub fn current_dir() -> String {
    "/".to_string()
}

pub fn env_var(name: &str) -> Option<String> {
    with(|t| t.env.get(name).cloned())
}

pub fn set_env_var(name: &str, value: &str) {
    with(|t| t.env.insert(name.to_string(), value.to_string()));
}

pub fn remove_env_var(name: &str) {
    with(|t| t.env.remove(name));
}

pub fn env_vars() -> Vec<(String, String)> {
    with(|t| t.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_and_directories() {
        create_dir_all("/proj/src").unwrap();
        write("/proj/src/a.txt", b"hi").unwrap();
        write("proj/b.txt", b"there").unwrap();
        assert_eq!(read_to_string("/proj/src/a.txt").unwrap(), "hi");
        assert_eq!(metadata("/proj/b.txt").unwrap().len(), 5);
        assert!(metadata("/proj/src").unwrap().is_dir());
        let mut names: Vec<String> = read_dir("/proj")
            .unwrap()
            .flatten()
            .map(|e| e.file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["b.txt", "src"]);
        assert!(write("/nowhere/x", b"").is_err());
        assert!(remove_dir("/proj").is_err());
        rename("/proj/src", "/proj/lib").unwrap();
        assert_eq!(read("/proj/lib/a.txt").unwrap(), b"hi");
        remove_dir_all("/proj").unwrap();
        assert!(metadata("/proj/b.txt").is_err());
        assert!(metadata("/").unwrap().is_dir());
    }
}
