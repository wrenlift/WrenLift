//! Stopping the world: the threads of a program and how a collection
//! brings them to rest. Each thread's view of the program has a
//! record here; the record says whether the thread is running Wren
//! or is safe, and while safe, where its stack stands and what it
//! holds precisely, so the collector on another thread can scan it.
//!
//! A collector sets the request and waits for every other thread to
//! be safe. A running thread reaches a safepoint (an allocation, the
//! interpreter's poll), publishes itself and parks until the request
//! clears; a thread already safe (blocked in a native, idle in its
//! scheduler, or back in the embedder) is scanned where it stands
//! and waits at its next transition to running.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::runtime::value::Value;

pub const RUNNING: u8 = 0;
pub const SAFE: u8 = 1;

/// One thread's standing with the collector.
pub struct ThreadState {
    pub state: AtomicU8,
    /// Where the thread's stack stands while safe: the low end of the
    /// range to scan. 0 until the thread has run.
    pub sp: AtomicUsize,
    /// The high end of the thread's stack, recorded when it first
    /// becomes safe on its own thread.
    pub stack_top: AtomicUsize,
    /// The low end, recorded with it; 0 where only the top is known.
    pub stack_lo: AtomicUsize,
    /// The krio fiber the thread was inside when it became safe, or 0.
    pub fiber_id: AtomicU64,
    /// A second range to scan while safe: the registers saved when
    /// the thread was stopped at a compiled loop header. Empty when
    /// both are 0.
    pub extra_lo: AtomicUsize,
    pub extra_hi: AtomicUsize,
    /// What the thread holds outside any stack: its fiber, api stack,
    /// pools, compiled-code roots. Published when it becomes safe.
    pub roots: Mutex<Vec<Value>>,
}

impl ThreadState {
    fn new() -> Arc<ThreadState> {
        Arc::new(ThreadState {
            state: AtomicU8::new(SAFE),
            sp: AtomicUsize::new(0),
            stack_top: AtomicUsize::new(0),
            stack_lo: AtomicUsize::new(0),
            fiber_id: AtomicU64::new(0),
            extra_lo: AtomicUsize::new(0),
            extra_hi: AtomicUsize::new(0),
            roots: Mutex::new(Vec::new()),
        })
    }

    pub fn is_safe(&self) -> bool {
        self.state.load(Ordering::SeqCst) == SAFE
    }

    /// Record the thread's own stack, once, from the thread itself.
    pub fn note_stack(&self) {
        if self.stack_top.load(Ordering::Relaxed) == 0 {
            let (lo, top) = crate::runtime::stack_scan::thread_stack_bounds();
            self.stack_lo.store(lo, Ordering::Relaxed);
            self.stack_top.store(top, Ordering::Relaxed);
        }
    }
}

/// The program's threads and the collector's request to them.
pub struct World {
    threads: Mutex<Vec<Arc<ThreadState>>>,
    requested: AtomicBool,
    /// The page this program's compiled loops load from.
    pub page: poll_page::PollPage,
    /// Held by the collecting thread for the length of a collection.
    collector: Mutex<()>,
    /// Signalled when a thread becomes safe and when the request
    /// clears.
    changed: Condvar,
    gate: Mutex<()>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> World {
        World {
            threads: Mutex::new(Vec::new()),
            requested: AtomicBool::new(false),
            page: poll_page::PollPage::new(),
            collector: Mutex::new(()),
            changed: Condvar::new(),
            gate: Mutex::new(()),
        }
    }

    /// Register a thread; it starts safe with nothing to scan.
    pub fn join(&self) -> Arc<ThreadState> {
        let t = ThreadState::new();
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(t.clone());
        t
    }

    pub fn leave(&self, t: &Arc<ThreadState>) {
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|x| !Arc::ptr_eq(x, t));
    }

    /// The threads of the program other than `me`.
    pub fn others(&self, me: &Arc<ThreadState>) -> Vec<Arc<ThreadState>> {
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|x| !Arc::ptr_eq(x, me))
            .cloned()
            .collect()
    }

    pub fn thread_count(&self) -> usize {
        self.threads.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether a collection is waiting for the threads to stop.
    #[inline(always)]
    pub fn requested(&self) -> bool {
        self.requested.load(Ordering::Relaxed)
    }

    /// Mark `t` safe with its stack and roots published, and tell a
    /// waiting collector; the runtime seam hears of it too.
    pub fn become_safe(&self, t: &ThreadState, sp: usize, fiber_id: u64, roots: Vec<Value>) {
        self.become_safe_here(t, sp, fiber_id, roots);
        unsafe {
            crate::runtime::rt::thread_safe(
                sp,
                t.extra_lo.load(Ordering::Relaxed),
                t.extra_hi.load(Ordering::Relaxed),
            )
        };
    }

    /// `become_safe` for a thread a host's world goes on running: the
    /// seam hears nothing, since the thread is not in a wait.
    pub fn become_safe_here(&self, t: &ThreadState, sp: usize, fiber_id: u64, roots: Vec<Value>) {
        *t.roots.lock().unwrap_or_else(|e| e.into_inner()) = roots;
        t.sp.store(sp, Ordering::Relaxed);
        t.fiber_id.store(fiber_id, Ordering::Relaxed);
        t.state.store(SAFE, Ordering::SeqCst);
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }

    /// Wait out any collection in progress, then mark `t` running.
    /// The store and the check are both sequentially consistent
    /// against the collector's request and its read of the state, so
    /// one side always sees the other. The runtime seam hears of it
    /// last, and may hold the thread for a collection of its own.
    pub fn become_running(&self, t: &ThreadState) {
        self.become_running_here(t);
        unsafe { crate::runtime::rt::thread_running() };
    }

    /// `become_running` for a thread a host's world ran meanwhile: the
    /// seam hears nothing, as it heard nothing of the safe state.
    pub fn become_running_here(&self, t: &ThreadState) {
        loop {
            t.state.store(RUNNING, Ordering::SeqCst);
            if !self.requested.load(Ordering::SeqCst) {
                return;
            }
            t.state.store(SAFE, Ordering::SeqCst);
            let mut g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            // The collector may have seen the thread running just now.
            self.changed.notify_all();
            while self.requested() {
                g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Become the collector: every other thread is safe when this
    /// returns, and stays so until `resume`. `t` must be safe while
    /// it waits here, so a collector already at work can scan it.
    pub fn stop(&self, t: &ThreadState) -> std::sync::MutexGuard<'_, ()> {
        let guard = self.collector.lock().unwrap_or_else(|e| e.into_inner());
        self.requested.store(true, Ordering::SeqCst);
        // The program's own page for its compiled loops, and the
        // process's static one for an AOT binary's; and a host's own
        // safepoints, for a thread running the host's code.
        poll_page::install_handler();
        self.page.protect(true);
        poll_page::static_page().protect(true);
        unsafe { crate::runtime::rt::host_poll(true) };
        t.state.store(RUNNING, Ordering::SeqCst);
        let mut g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let all_safe = self
                .threads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .all(|x| std::ptr::eq(x.as_ref(), t) || x.is_safe());
            if all_safe {
                break;
            }
            g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        guard
    }

    /// Let the world go after a collection. The stopping thread went
    /// safe through the seam (`enter_safe`) and `stop` made it running
    /// for its own world alone, so the seam hears of the running state
    /// here, once, as it heard of the safe one.
    pub fn resume(&self, guard: std::sync::MutexGuard<'_, ()>) {
        self.requested.store(false, Ordering::SeqCst);
        self.page.protect(false);
        poll_page::static_page().protect(false);
        unsafe { crate::runtime::rt::host_poll(false) };
        {
            let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            self.changed.notify_all();
        }
        drop(guard);
        unsafe { crate::runtime::rt::thread_running() };
    }
}

/// A host's collector asks every thread running Wren, in any program,
/// to reach a safepoint (`on`), or lets them go. Every program's page
/// and the static one are held unreadable meanwhile, so a compiled
/// loop faults into `park_interrupted` and passes through the seam's
/// `thread_safe` and `thread_running`; a thread that polls does the
/// same at its next poll. The host's own request is what holds a
/// thread once it is safe; wren_lift's world asks nothing.
pub fn host_stop(on: bool) {
    if on {
        poll_page::install_handler();
    }
    poll_page::static_page().protect(on);
    for page in poll_page::live_pages() {
        page.protect(on);
    }
}

/// A lock a thread may take again while it holds it: the tier
/// bookkeeping's entry points call each other.
pub struct ReentrantLock {
    mutex: Mutex<()>,
    owner: AtomicU64,
    depth: AtomicUsize,
    changed: Condvar,
}

impl Default for ReentrantLock {
    fn default() -> Self {
        Self::new()
    }
}

/// A number naming the calling OS thread, stable for its lifetime.
pub fn thread_key() -> u64 {
    thread_local! {
        static KEY: u8 = const { 0 };
    }
    KEY.with(|k| k as *const u8 as u64)
}

impl ReentrantLock {
    pub fn new() -> ReentrantLock {
        ReentrantLock {
            mutex: Mutex::new(()),
            owner: AtomicU64::new(0),
            depth: AtomicUsize::new(0),
            changed: Condvar::new(),
        }
    }

    pub fn lock(&self) -> ReentrantGuard<'_> {
        let me = thread_key();
        if self.owner.load(Ordering::Acquire) == me {
            self.depth.fetch_add(1, Ordering::Relaxed);
            return ReentrantGuard(self);
        }
        let mut g = self.mutex.lock().unwrap_or_else(|e| e.into_inner());
        while self.owner.load(Ordering::Acquire) != 0 {
            g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        self.owner.store(me, Ordering::Release);
        self.depth.store(1, Ordering::Relaxed);
        ReentrantGuard(self)
    }
}

pub struct ReentrantGuard<'a>(&'a ReentrantLock);

impl Drop for ReentrantGuard<'_> {
    fn drop(&mut self) {
        let lock = self.0;
        if lock.depth.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _g = lock.mutex.lock().unwrap_or_else(|e| e.into_inner());
            lock.owner.store(0, Ordering::Release);
            lock.changed.notify_one();
        }
    }
}

/// A page compiled loop headers load a word from. Readable until a
/// collector wants the world stopped, then unreadable, so the load
/// faults and the fault handler parks the thread where it stands:
/// one plain load per iteration, and nothing else. Each program maps
/// one for the code it compiles; an AOT binary's code names the
/// static one by symbol.
pub mod poll_page {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    pub const PAGE: usize = 1 << 14;

    /// The static page, for AOT code. Aligned to the largest page
    /// size a target uses; COFF cannot express more than 8 KiB, and
    /// Windows pages are 4 KiB.
    #[cfg_attr(not(windows), repr(C, align(16384)))]
    #[cfg_attr(windows, repr(C, align(4096)))]
    pub struct Page(pub std::cell::UnsafeCell<[u8; PAGE]>);

    // SAFETY: nothing reads or writes the page's contents; only its
    // protection changes.
    unsafe impl Sync for Page {}

    #[unsafe(no_mangle)]
    pub static wlift_safepoint_page: Page = Page(std::cell::UnsafeCell::new([0; PAGE]));

    /// A page and the count of stops holding it unreadable. The
    /// state is shared with the list of live pages, so a host's stop
    /// reaches every program's page by the same count.
    pub struct PollPage {
        state: Arc<PageState>,
    }

    pub struct PageState {
        addr: usize,
        mapped: bool,
        stops: AtomicU32,
    }

    static STATIC_PAGE: OnceLock<PollPage> = OnceLock::new();

    /// The pages of the programs alive, for a host's stop.
    static LIVE: Mutex<Vec<Arc<PageState>>> = Mutex::new(Vec::new());

    pub fn live_pages() -> Vec<Arc<PageState>> {
        LIVE.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The page AOT code loads from.
    pub fn static_page() -> &'static PollPage {
        STATIC_PAGE.get_or_init(|| PollPage {
            state: Arc::new(PageState {
                addr: &wlift_safepoint_page as *const Page as usize,
                mapped: false,
                stops: AtomicU32::new(0),
            }),
        })
    }

    impl PollPage {
        /// A page of this program's own.
        pub fn new() -> PollPage {
            let state = Arc::new(PageState {
                addr: map(),
                mapped: true,
                stops: AtomicU32::new(0),
            });
            LIVE.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::clone(&state));
            PollPage { state }
        }

        pub fn address(&self) -> usize {
            self.state.addr
        }

        pub fn contains(&self, addr: usize) -> bool {
            self.state.contains(addr)
        }

        /// Whether a stop holds the page unreadable.
        pub fn is_protected(&self) -> bool {
            self.state.is_protected()
        }

        /// One more stop holds the page (`on`), or one fewer.
        pub fn protect(&self, on: bool) {
            self.state.protect(on);
        }
    }

    impl PageState {
        pub fn contains(&self, addr: usize) -> bool {
            self.addr != 0 && addr >= self.addr && addr < self.addr + PAGE
        }

        pub fn is_protected(&self) -> bool {
            self.stops.load(Ordering::Acquire) != 0
        }

        pub fn protect(&self, on: bool) {
            if self.addr == 0 {
                return;
            }
            let change = if on {
                self.stops.fetch_add(1, Ordering::SeqCst) == 0
            } else {
                self.stops.fetch_sub(1, Ordering::SeqCst) == 1
            };
            if !change {
                return;
            }
            #[cfg(all(unix, feature = "host"))]
            unsafe {
                let prot = if on {
                    libc::PROT_NONE
                } else {
                    libc::PROT_READ | libc::PROT_WRITE
                };
                libc::mprotect(self.addr as *mut libc::c_void, PAGE, prot);
            }
            #[cfg(all(windows, feature = "host"))]
            unsafe {
                use windows_sys::Win32::System::Memory::{
                    PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect,
                };
                let prot = if on { PAGE_NOACCESS } else { PAGE_READWRITE };
                let mut old = 0;
                VirtualProtect(self.addr as *const _, PAGE, prot, &mut old);
            }
        }
    }

    impl Default for PollPage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for PollPage {
        fn drop(&mut self) {
            if !self.state.mapped {
                return;
            }
            LIVE.lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|p| !Arc::ptr_eq(p, &self.state));
            // The page goes when the last stop holding it is gone too.
        }
    }

    impl Drop for PageState {
        fn drop(&mut self) {
            #[cfg(all(unix, feature = "host"))]
            if self.mapped && self.addr != 0 {
                unsafe { libc::munmap(self.addr as *mut libc::c_void, PAGE) };
            }
            #[cfg(all(windows, feature = "host"))]
            if self.mapped && self.addr != 0 {
                use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
                unsafe { VirtualFree(self.addr as *mut _, 0, MEM_RELEASE) };
            }
        }
    }

    #[cfg(all(unix, feature = "host"))]
    fn map() -> usize {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED { 0 } else { p as usize }
    }

    #[cfg(all(windows, feature = "host"))]
    fn map() -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        // Page-granular reservations are 64 KiB aligned, so the
        // 16 KiB page is whole.
        unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as usize
        }
    }

    #[cfg(not(any(all(unix, feature = "host"), all(windows, feature = "host"))))]
    fn map() -> usize {
        0
    }

    #[cfg(all(unix, feature = "host"))]
    static PREVIOUS: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

    /// Install the fault handler once.
    pub fn install_handler() {
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(install);
    }

    #[cfg(all(unix, feature = "host"))]
    fn install() {
        unsafe {
            let mut previous: [libc::sigaction; 2] = std::mem::zeroed();
            for (i, sig) in [libc::SIGSEGV, libc::SIGBUS].into_iter().enumerate() {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = on_fault as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(sig, &action, &mut previous[i]);
            }
            let _ = PREVIOUS.set(previous);
        }
    }

    /// A vectored handler sees the fault before structured exception
    /// handling does, on every thread.
    #[cfg(all(windows, feature = "host"))]
    fn install() {
        use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
        unsafe {
            AddVectoredExceptionHandler(1, Some(on_exception));
        }
    }

    #[cfg(not(any(all(unix, feature = "host"), all(windows, feature = "host"))))]
    fn install() {}

    /// The exception handler: an access violation on a safepoint page
    /// parks the thread and resumes the load; anything else goes on
    /// to the next handler.
    #[cfg(all(windows, feature = "host"))]
    unsafe extern "system" fn on_exception(
        info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::{
            EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        };
        let record = unsafe { &*(*info).ExceptionRecord };
        if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let addr = record.ExceptionInformation[1];
        if !crate::runtime::vm::fault_is_a_safepoint(addr) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let (sp, regs) = interrupted_state(unsafe { (*info).ContextRecord });
        crate::runtime::vm::park_interrupted(addr, sp, &regs);
        EXCEPTION_CONTINUE_EXECUTION
    }

    /// The fault handler: a load from a safepoint page parks the
    /// thread; any other fault goes to whoever handled it before.
    #[cfg(all(unix, feature = "host"))]
    extern "C" fn on_fault(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
        let addr = unsafe { (*info).si_addr() as usize };
        if crate::runtime::vm::fault_is_a_safepoint(addr) {
            let (sp, regs) = interrupted_state(ctx);
            crate::runtime::vm::park_interrupted(addr, sp, &regs);
            return;
        }
        let i = if sig == libc::SIGSEGV { 0 } else { 1 };
        let prev = PREVIOUS.get().map(|p| p[i]);
        unsafe {
            match prev {
                Some(p) if p.sa_sigaction == libc::SIG_DFL || p.sa_sigaction == 0 => {
                    libc::signal(sig, libc::SIG_DFL);
                }
                Some(p) if p.sa_sigaction == libc::SIG_IGN => {}
                Some(p) if p.sa_flags & libc::SA_SIGINFO != 0 => {
                    let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                        std::mem::transmute(p.sa_sigaction);
                    f(sig, info, ctx);
                }
                Some(p) => {
                    let f: extern "C" fn(libc::c_int) = std::mem::transmute(p.sa_sigaction);
                    f(sig);
                }
                None => {
                    libc::signal(sig, libc::SIG_DFL);
                }
            }
        }
    }

    /// The interrupted thread's stack pointer and general registers.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "host"))]
    fn interrupted_state(ctx: *mut libc::c_void) -> (usize, [usize; 32]) {
        let uc = ctx as *const libc::ucontext_t;
        let ss = unsafe { &(*(*uc).uc_mcontext).__ss };
        let mut regs = [0usize; 32];
        for (i, r) in ss.__x.iter().enumerate() {
            regs[i] = *r as usize;
        }
        regs[29] = ss.__fp as usize;
        regs[30] = ss.__lr as usize;
        (ss.__sp as usize, regs)
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "host"))]
    fn interrupted_state(ctx: *mut libc::c_void) -> (usize, [usize; 32]) {
        let uc = ctx as *const libc::ucontext_t;
        let mc = unsafe { &(*uc).uc_mcontext };
        let mut regs = [0usize; 32];
        for (i, r) in mc.regs.iter().enumerate() {
            regs[i] = *r as usize;
        }
        (mc.sp as usize, regs)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "host"))]
    fn interrupted_state(ctx: *mut libc::c_void) -> (usize, [usize; 32]) {
        let uc = ctx as *const libc::ucontext_t;
        let g = unsafe { &(*uc).uc_mcontext.gregs };
        let mut regs = [0usize; 32];
        for (i, r) in g.iter().enumerate().take(23) {
            regs[i] = *r as usize;
        }
        (g[libc::REG_RSP as usize] as usize, regs)
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64", feature = "host"))]
    fn interrupted_state(ctx: *mut libc::c_void) -> (usize, [usize; 32]) {
        let uc = ctx as *const libc::ucontext_t;
        let ss = unsafe { &(*(*uc).uc_mcontext).__ss };
        let regs = [
            ss.__rax as usize,
            ss.__rbx as usize,
            ss.__rcx as usize,
            ss.__rdx as usize,
            ss.__rdi as usize,
            ss.__rsi as usize,
            ss.__rbp as usize,
            ss.__rsp as usize,
            ss.__r8 as usize,
            ss.__r9 as usize,
            ss.__r10 as usize,
            ss.__r11 as usize,
            ss.__r12 as usize,
            ss.__r13 as usize,
            ss.__r14 as usize,
            ss.__r15 as usize,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        (ss.__rsp as usize, regs)
    }

    #[cfg(all(windows, target_arch = "x86_64", feature = "host"))]
    fn interrupted_state(
        ctx: *mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT,
    ) -> (usize, [usize; 32]) {
        let c = unsafe { &*ctx };
        let mut regs = [0usize; 32];
        for (slot, r) in regs.iter_mut().zip([
            c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rdi, c.Rsi, c.Rbp, c.Rsp, c.R8, c.R9, c.R10, c.R11,
            c.R12, c.R13, c.R14, c.R15,
        ]) {
            *slot = r as usize;
        }
        (c.Rsp as usize, regs)
    }
}
