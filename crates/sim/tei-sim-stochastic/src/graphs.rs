//! Graph constructors with closed-form Max-Cut optima (validation targets)
//! plus a seeded random-regular generator for demos.

use tei_sim_core::rng::Rng;

/// Undirected weighted graph.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(u32, u32, f64)>,
}

/// Complete graph Kₙ, unit weights. Max-Cut = ⌊n/2⌋·⌈n/2⌉ (balanced bipartition).
pub fn complete(n: usize) -> Graph {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i as u32, j as u32, 1.0));
        }
    }
    Graph { n, edges }
}

/// Cycle Cₙ, unit weights. Max-Cut = n (even n) or n−1 (odd n).
pub fn cycle(n: usize) -> Graph {
    let edges = (0..n)
        .map(|i| (i as u32, ((i + 1) % n) as u32, 1.0))
        .collect();
    Graph { n, edges }
}

/// Complete bipartite K_{a,b}, unit weights. Max-Cut = a·b (cut everything).
pub fn complete_bipartite(a: usize, b: usize) -> Graph {
    let mut edges = Vec::new();
    for i in 0..a {
        for j in 0..b {
            edges.push((i as u32, (a + j) as u32, 1.0));
        }
    }
    Graph { n: a + b, edges }
}

/// The Petersen graph (10 vertices, 15 edges, 3-regular). Max-Cut = 12
/// (known exact value for this classic instance).
pub fn petersen() -> Graph {
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(15);
    // Outer 5-cycle, inner pentagram, spokes.
    for i in 0u32..5 {
        edges.push((i, (i + 1) % 5, 1.0)); // outer C5
        edges.push((5 + i, 5 + (i + 2) % 5, 1.0)); // inner pentagram
        edges.push((i, 5 + i, 1.0)); // spokes
    }
    Graph { n: 10, edges }
}

/// Seeded random d-regular graph via the pairing model with retries.
/// Demo workload — no closed-form optimum (the UI reports cut only).
pub fn random_regular(n: usize, d: usize, seed: u64) -> Graph {
    assert!(n * d % 2 == 0, "n·d must be even");
    assert!(d < n);
    let mut rng = Rng::new(seed);
    'outer: for _attempt in 0..200 {
        // Stubs: each vertex appears d times.
        let mut stubs: Vec<u32> = (0..n)
            .flat_map(|v| std::iter::repeat(v as u32).take(d))
            .collect();
        // Fisher-Yates shuffle.
        for i in (1..stubs.len()).rev() {
            let j = rng.below(i + 1);
            stubs.swap(i, j);
        }
        let mut seen = std::collections::HashSet::new();
        let mut edges = Vec::with_capacity(n * d / 2);
        for pair in stubs.chunks_exact(2) {
            let (a, b) = (pair[0], pair[1]);
            if a == b {
                continue 'outer; // self-loop → retry
            }
            let key = (a.min(b), a.max(b));
            if !seen.insert(key) {
                continue 'outer; // multi-edge → retry
            }
            edges.push((key.0, key.1, 1.0));
        }
        return Graph { n, edges };
    }
    panic!("random_regular failed to generate a simple graph after 200 attempts");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn petersen_is_3_regular_15_edges() {
        let g = petersen();
        assert_eq!(g.edges.len(), 15);
        let mut deg = vec![0; 10];
        for &(a, b, _) in &g.edges {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        assert!(deg.iter().all(|&d| d == 3));
    }

    #[test]
    fn random_regular_is_simple_and_regular() {
        let g = random_regular(50, 4, 7);
        assert_eq!(g.edges.len(), 100);
        let mut deg = vec![0; 50];
        let mut seen = std::collections::HashSet::new();
        for &(a, b, _) in &g.edges {
            assert_ne!(a, b);
            assert!(seen.insert((a, b)));
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        assert!(deg.iter().all(|&d| d == 4));
    }
}
