//! Discrete-event queue with stable FIFO tie-breaking at equal times.
//!
//! Foundation for the spiking simulator (Phase 2) and the SSA paths later.
//! f64 timestamps are ordered with `total_cmp`; a monotone sequence number
//! breaks ties so insertion order is preserved — required for reproducible
//! event-driven runs.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Entry<T> {
    time: f64,
    seq: u64,
    payload: T,
}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl<T> Eq for Entry<T> {}
impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Entry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: BinaryHeap is a max-heap; we want earliest-first.
        other
            .time
            .total_cmp(&self.time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Min-priority event queue keyed by f64 time.
pub struct EventQueue<T> {
    heap: BinaryHeap<Entry<T>>,
    seq: u64,
}

impl<T> Default for EventQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> EventQueue<T> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }
    pub fn push(&mut self, time: f64, payload: T) {
        let seq = self.seq;
        self.seq += 1;
        self.heap.push(Entry { time, seq, payload });
    }
    pub fn pop(&mut self) -> Option<(f64, T)> {
        self.heap.pop().map(|e| (e.time, e.payload))
    }
    pub fn peek_time(&self) -> Option<f64> {
        self.heap.peek().map(|e| e.time)
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Events come out time-ordered.
    #[test]
    fn time_ordering() {
        let mut q = EventQueue::new();
        q.push(3.0, "c");
        q.push(1.0, "a");
        q.push(2.0, "b");
        assert_eq!(q.pop().unwrap().1, "a");
        assert_eq!(q.pop().unwrap().1, "b");
        assert_eq!(q.pop().unwrap().1, "c");
    }

    /// Equal timestamps preserve insertion order (FIFO).
    #[test]
    fn fifo_at_equal_time() {
        let mut q = EventQueue::new();
        for i in 0..100 {
            q.push(1.0, i);
        }
        for i in 0..100 {
            assert_eq!(q.pop().unwrap().1, i);
        }
    }
}
