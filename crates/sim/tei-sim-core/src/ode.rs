//! ODE integrators: RK4 (explicit), trapezoidal and backward Euler
//! (fixed-point iterated — sufficient for non-stiff use; the circuit crate
//! brings Newton coupling in Phase 5).

/// One RK4 step for y' = f(t, y). `y` is updated in place.
pub fn rk4_step<F>(f: &F, t: f64, y: &mut [f64], h: f64, scratch: &mut RkScratch)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = y.len();
    scratch.ensure(n);
    let RkScratch {
        k1,
        k2,
        k3,
        k4,
        tmp,
    } = scratch;

    f(t, y, k1);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    f(t + 0.5 * h, tmp, k2);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    f(t + 0.5 * h, tmp, k3);
    for i in 0..n {
        tmp[i] = y[i] + h * k3[i];
    }
    f(t + h, tmp, k4);
    for i in 0..n {
        y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Reusable RK4 work buffers.
#[derive(Default)]
pub struct RkScratch {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
}

impl RkScratch {
    fn ensure(&mut self, n: usize) {
        for v in [
            &mut self.k1,
            &mut self.k2,
            &mut self.k3,
            &mut self.k4,
            &mut self.tmp,
        ] {
            if v.len() != n {
                v.resize(n, 0.0);
            }
        }
    }
}

/// Trapezoidal step via fixed-point iteration (non-stiff regimes).
pub fn trapezoidal_step<F>(f: &F, t: f64, y: &mut [f64], h: f64, iters: usize)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = y.len();
    let mut f0 = vec![0.0; n];
    f(t, y, &mut f0);
    let mut y_next: Vec<f64> = (0..n).map(|i| y[i] + h * f0[i]).collect(); // predictor (Euler)
    let mut f1 = vec![0.0; n];
    for _ in 0..iters {
        f(t + h, &y_next, &mut f1);
        for i in 0..n {
            y_next[i] = y[i] + 0.5 * h * (f0[i] + f1[i]);
        }
    }
    y.copy_from_slice(&y_next);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RK4 must exhibit 4th-order global convergence on y' = −y.
    #[test]
    fn rk4_fourth_order() {
        let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
        let exact = (-1.0f64).exp();
        let err_at = |steps: usize| {
            let mut y = vec![1.0];
            let h = 1.0 / steps as f64;
            let mut s = RkScratch::default();
            for k in 0..steps {
                rk4_step(&f, k as f64 * h, &mut y, h, &mut s);
            }
            (y[0] - exact).abs()
        };
        let e1 = err_at(20);
        let e2 = err_at(40);
        let order = (e1 / e2).log2();
        assert!(
            (order - 4.0).abs() < 0.3,
            "measured order {order} (e1={e1:.3e}, e2={e2:.3e})"
        );
    }

    /// Trapezoidal must exhibit 2nd-order convergence on the same problem.
    #[test]
    fn trapezoidal_second_order() {
        let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
        let exact = (-1.0f64).exp();
        let err_at = |steps: usize| {
            let mut y = vec![1.0];
            let h = 1.0 / steps as f64;
            for k in 0..steps {
                trapezoidal_step(&f, k as f64 * h, &mut y, h, 8);
            }
            (y[0] - exact).abs()
        };
        let e1 = err_at(50);
        let e2 = err_at(100);
        let order = (e1 / e2).log2();
        assert!(
            (order - 2.0).abs() < 0.2,
            "measured order {order} (e1={e1:.3e}, e2={e2:.3e})"
        );
    }
}
