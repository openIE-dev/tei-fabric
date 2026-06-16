//! teiOS Embassy binding — automatic **per-task** active-time ledgers.
//!
//! `tei-rt` prices and dispatches *primitive runs* (the work the app asks
//! for). This crate adds the orthogonal view the kernel doesn't own: how
//! much CPU-active time **every Embassy task** actually consumed — without a
//! line of instrumentation in app code.
//!
//! `embassy-executor`'s `trace` feature declares seven `_embassy_trace_*`
//! callbacks and resolves them at link time (EMBEDDED-ROADMAP §8: "Embassy
//! needs no fork or wrapper"). This crate provides them: on every task
//! `exec_begin`/`exec_end` it accumulates `now - begin` into a per-task
//! [`TaskLedgers`] entry. A board that enables `embassy-executor/trace` and
//! links this crate gets per-task active-µs for free; [`snapshot`] reads them.
//!
//! ## Honesty boundary
//!
//! The accounting logic ([`TaskLedgers`]) is pure and **host-tested**. The
//! hooks + the global behind a `critical_section` are compile-/link-verified
//! for the embedded target (they need a running Embassy executor + clock to
//! exercise — the bench step, same boundary as the rest of teiOS). Active-µs
//! is wall-clock attribution, not joules; folding it against a per-state
//! power table (→ task-level energy) is the calibration follow-on.

#![cfg_attr(not(test), no_std)]
// Not `forbid(unsafe_code)`: exporting the `#[unsafe(no_mangle)]` trace
// symbols Embassy links against is itself an unsafe attribute. The bodies
// contain no `unsafe` blocks (the deny below keeps it that way).
#![deny(unsafe_op_in_unsafe_fn)]

/// One task's accumulated accounting.
#[derive(Clone, Copy)]
struct Slot {
    task_id: u32,
    used: bool,
    /// Timestamp of the in-flight `exec_begin`, if currently RUNNING.
    begin_us: Option<u64>,
    /// Total CPU-active microseconds across all completed runs.
    active_us: u64,
    /// Number of completed `exec_begin`→`exec_end` runs.
    runs: u32,
}

impl Slot {
    const EMPTY: Slot = Slot {
        task_id: 0,
        used: false,
        begin_us: None,
        active_us: 0,
        runs: 0,
    };
}

/// A fixed-capacity, allocation-free per-task active-time table. `N` is the
/// max number of distinct tasks tracked; tasks beyond `N` are dropped (no
/// panic — `overflow()` reports it).
pub struct TaskLedgers<const N: usize> {
    slots: [Slot; N],
    overflow: u32,
}

impl<const N: usize> Default for TaskLedgers<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> TaskLedgers<N> {
    pub const fn new() -> Self {
        Self {
            slots: [Slot::EMPTY; N],
            overflow: 0,
        }
    }

    /// Find the slot for `task`, or claim a free one. `None` only when full
    /// (which bumps [`overflow`](Self::overflow)).
    fn slot(&mut self, task: u32) -> Option<&mut Slot> {
        let mut free = None;
        for (i, s) in self.slots.iter().enumerate() {
            if s.used && s.task_id == task {
                return Some(&mut self.slots[i]);
            }
            if free.is_none() && !s.used {
                free = Some(i);
            }
        }
        match free {
            Some(i) => {
                self.slots[i] = Slot {
                    task_id: task,
                    used: true,
                    ..Slot::EMPTY
                };
                Some(&mut self.slots[i])
            }
            None => {
                self.overflow = self.overflow.saturating_add(1);
                None
            }
        }
    }

    /// `exec_begin`: the task started a poll at `now_us`.
    pub fn exec_begin(&mut self, task: u32, now_us: u64) {
        if let Some(s) = self.slot(task) {
            s.begin_us = Some(now_us);
        }
    }

    /// `exec_end`: the task finished a poll at `now_us`; fold the elapsed
    /// active time in. Ignored if there was no matching `exec_begin`.
    pub fn exec_end(&mut self, task: u32, now_us: u64) {
        if let Some(s) = self.slot(task) {
            if let Some(b) = s.begin_us.take() {
                s.active_us = s.active_us.wrapping_add(now_us.wrapping_sub(b));
                s.runs = s.runs.saturating_add(1);
            }
        }
    }

    /// `task_end`: the task was destructed. We keep its accumulated stats
    /// readable (a finished task's energy still counts) but clear any
    /// dangling in-flight begin.
    pub fn end_task(&mut self, task: u32) {
        if let Some(s) = self.slot(task) {
            s.begin_us = None;
        }
    }

    /// Total active microseconds for a tracked task.
    pub fn active_us(&self, task: u32) -> Option<u64> {
        self.slots
            .iter()
            .find(|s| s.used && s.task_id == task)
            .map(|s| s.active_us)
    }

    /// Visit every tracked task: `(task_id, active_us, runs)`.
    pub fn for_each(&self, mut f: impl FnMut(u32, u64, u32)) {
        for s in self.slots.iter().filter(|s| s.used) {
            f(s.task_id, s.active_us, s.runs);
        }
    }

    /// Number of distinct tasks currently tracked.
    pub fn tracked(&self) -> usize {
        self.slots.iter().filter(|s| s.used).count()
    }

    /// How many task-claims were dropped because the table was full.
    pub fn overflow(&self) -> u32 {
        self.overflow
    }
}

// ── The Embassy trace hooks (embedded target only) ───────────────────────
//
// embassy-executor's `trace` feature imports these seven symbols; we export
// them with `#[unsafe(no_mangle)]`. exec_begin/end drive the accounting; the rest are
// no-ops (we attribute active *time*, not the full scheduler state machine).
#[cfg(all(target_arch = "arm", target_os = "none"))]
mod hooks {
    use super::TaskLedgers;
    use core::cell::RefCell;
    use critical_section::Mutex;
    use embassy_time::Instant;

    /// Max distinct tasks tracked on-device.
    pub const MAX_TASKS: usize = 16;

    static STATE: Mutex<RefCell<TaskLedgers<MAX_TASKS>>> =
        Mutex::new(RefCell::new(TaskLedgers::new()));

    fn now_us() -> u64 {
        Instant::now().as_micros()
    }

    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_task_exec_begin(_executor_id: u32, task_id: u32) {
        critical_section::with(|cs| STATE.borrow(cs).borrow_mut().exec_begin(task_id, now_us()));
    }

    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_task_exec_end(_executor_id: u32, task_id: u32) {
        critical_section::with(|cs| STATE.borrow(cs).borrow_mut().exec_end(task_id, now_us()));
    }

    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_task_end(_executor_id: u32, task_id: u32) {
        critical_section::with(|cs| STATE.borrow(cs).borrow_mut().end_task(task_id));
    }

    // Not needed for active-time attribution — provided so the trace
    // feature's whole extern block resolves at link time.
    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_task_new(_executor_id: u32, _task_id: u32) {}
    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_task_ready_begin(_executor_id: u32, _task_id: u32) {}
    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_poll_start(_executor_id: u32) {}
    #[unsafe(no_mangle)]
    pub extern "Rust" fn _embassy_trace_executor_idle(_executor_id: u32) {}

    /// Read the live per-task ledgers under a critical section.
    pub fn snapshot<R>(f: impl FnOnce(&TaskLedgers<MAX_TASKS>) -> R) -> R {
        critical_section::with(|cs| f(&STATE.borrow(cs).borrow()))
    }
}

#[cfg(all(target_arch = "arm", target_os = "none"))]
pub use hooks::{snapshot, MAX_TASKS};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_active_time_per_task() {
        let mut t: TaskLedgers<4> = TaskLedgers::new();
        // task 1 runs 100→140 (40µs) then 200→230 (30µs) = 70µs, 2 runs.
        t.exec_begin(1, 100);
        t.exec_end(1, 140);
        t.exec_begin(1, 200);
        t.exec_end(1, 230);
        // task 2 runs 150→155 (5µs).
        t.exec_begin(2, 150);
        t.exec_end(2, 155);
        assert_eq!(t.active_us(1), Some(70));
        assert_eq!(t.active_us(2), Some(5));
        assert_eq!(t.active_us(9), None);
        assert_eq!(t.tracked(), 2);
    }

    #[test]
    fn exec_end_without_begin_is_ignored() {
        let mut t: TaskLedgers<4> = TaskLedgers::new();
        t.exec_end(1, 999); // no prior begin
        assert_eq!(t.active_us(1), Some(0));
        let mut runs_seen = 0;
        t.for_each(|_, _, runs| runs_seen += runs);
        assert_eq!(runs_seen, 0);
    }

    #[test]
    fn full_table_reports_overflow_without_panic() {
        let mut t: TaskLedgers<2> = TaskLedgers::new();
        t.exec_begin(1, 0);
        t.exec_begin(2, 0);
        t.exec_begin(3, 0); // third distinct task → dropped
        assert_eq!(t.tracked(), 2);
        assert_eq!(t.overflow(), 1);
        assert_eq!(t.active_us(3), None);
    }

    #[test]
    fn end_task_keeps_stats_clears_inflight() {
        let mut t: TaskLedgers<4> = TaskLedgers::new();
        t.exec_begin(1, 10);
        t.exec_end(1, 20); // 10µs banked
        t.exec_begin(1, 30); // in-flight
        t.end_task(1); // task destructed mid-flight
        // banked time survives; the dangling begin is cleared (no later
        // exec_end double-counts).
        assert_eq!(t.active_us(1), Some(10));
        t.exec_end(1, 999);
        assert_eq!(t.active_us(1), Some(10));
    }
}
