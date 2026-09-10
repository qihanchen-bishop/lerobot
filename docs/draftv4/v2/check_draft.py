"""Finite-example checks, not proofs or validation of robotics assumptions."""

from collections import Counter, defaultdict
from itertools import product
from math import ceil, isclose
from pathlib import Path
import re
import unittest


class DraftChecks(unittest.TestCase):
    def test_unified_risk_identity(self):
        samples = list(product((-1.0, 1.0), repeat=5))
        y = [2*a + 3*b + 4*c + 5*d + 7*n for a, b, c, d, n in samples]
        base = [s[0] for s in samples]

        def conditional_mean(indices):
            groups = defaultdict(list)
            keys = [tuple(s[i] for i in indices) for s in samples]
            for key, target in zip(keys, y):
                groups[key].append(target)
            means = {k: sum(v)/len(v) for k, v in groups.items()}
            return [means[k] for k in keys]

        def mse(left, right):
            return sum((a-b)**2 for a, b in zip(left, right))/len(left)

        m0 = conditional_mean((0,))
        m1 = conditional_mean((0, 1))
        m2 = conditional_mean((0, 1, 2, 3))
        mt = conditional_mean((0, 1, 2))
        residual = [0.8*(a-b) for a, b in zip(mt, base)]
        corrected = [a+b for a, b in zip(base, residual)]
        measured = mse(y, base) - mse(y, corrected)
        predicted = (mse(base, m0) + mse(m1, m0) + mse(m2, m1)
                     - mse(m2, mt) - mse(corrected, mt))
        self.assertTrue(isclose(measured, predicted, abs_tol=1e-10))
        self.assertGreater(measured, 0)
        alignment = sum(2*(a-b)*r-r*r for a, b, r in zip(y, base, residual))/len(y)
        self.assertTrue(isclose(measured, alignment, abs_tol=1e-10))
        self.assertGreater(mse(m2, mt), 0)  # Nonzero compression loss is included.
        hist_error_mean = [a-b for a, b in zip(m2, base)]
        predictable_energy = sum(a*a for a in hist_error_mean)/len(y)
        enc_error = mse(m2, mt)
        res_error = mse(corrected, mt)
        self.assertTrue(isclose(measured, predictable_energy-enc_error-res_error, abs_tol=1e-10))
        self.assertTrue(isclose(mse(y, corrected)-mse(y, m1),
                                enc_error+res_error-mse(m2, m1), abs_tol=1e-10))

    def test_rgb_benefit_and_fitting_counterexample(self):
        samples = list(product((-1.0, 1.0), repeat=3))
        # Current input a; RGB reveals b; n is unobserved label variability.
        for fit_bias in (0, 1, 6):
            base_risk = sum((3*b+7*n)**2 for a, b, n in samples)/len(samples)
            rgb_risk = sum((7*n-fit_bias)**2 for a, b, n in samples)/len(samples)
            gain = base_risk-rgb_risk
            self.assertTrue(isclose(gain, 9-fit_bias**2, abs_tol=1e-10))
            self.assertEqual(gain > 0, fit_bias**2 < 9)

    def test_geometry_upper_vs_baseline_lower_bound(self):
        samples = list(product((-1.0, 1.0), repeat=3))
        # Geometry g is visible in RGB but discarded by a nuisance-only baseline.
        baseline_excess = sum((2*g-0.1*n)**2 for g, n, noise in samples)/len(samples)
        query_excess = 0.2**2
        geometry_error = 0.1**2
        lipschitz = 2
        upper = lipschitz**2*geometry_error+query_excess
        baseline_lower = (2-(-2))**2/4
        self.assertLess(upper, baseline_lower)
        self.assertLessEqual(query_excess, upper)
        self.assertGreaterEqual(baseline_excess, baseline_lower)
        self.assertLess(query_excess, baseline_excess)
        # Perfect auxiliary readout does not guarantee a good action head.
        self.assertGreater(6**2, baseline_excess)

    def test_problem_definition_is_architecture_neutral(self):
        theory = (Path(__file__).resolve().parent/"theory.tex").read_text()
        problem = theory.split(r"\subsection{统一分析", 1)[0]
        for architecture in ("慢策略", "快模型", "GRU", "ACT", "QToken"):
            self.assertNotIn(architecture, problem)
        self.assertIn(r"\label{eq:taskobjective}", problem)

    def test_auxiliary_gradient_local_comparison(self):
        # A(theta) = theta^2 / 2 has gradient Lipschitz constant one.
        theta, rate, weight = 1.0, 0.1, 0.5
        theta_a = theta-rate*theta
        for geo_gradient in (0.8, -0.8, 40.0):
            theta_q = theta_a-rate*weight*geo_gradient
            difference = (theta_q**2-theta_a**2)/2
            bound = (-rate*weight*theta_a*geo_gradient
                     +(rate*weight*geo_gradient)**2/2)
            self.assertTrue(isclose(difference, bound, abs_tol=1e-10))
            self.assertEqual(difference < 0,
                             theta_a*geo_gradient > rate*weight*geo_gradient**2/2)

    def test_timing_prefix_and_overlap(self):
        for stride, horizon, delay in product(range(1, 10), range(1, 13), range(6)):
            for latency_ticks in (0, 0.2, 1, 2.5, 3, 4, 8, 20):
                expired = min(horizon, max(0, ceil(latency_ticks)-delay))
                direct = sum(delay+j < latency_ticks for j in range(horizon))
                self.assertEqual(expired, direct)
                if expired < horizon:
                    old_last = delay+horizon-1
                    new_first = stride+delay+expired
                    self.assertEqual(new_first <= old_last+1, stride+expired <= horizon)

    def test_document_references(self):
        root = Path(__file__).resolve().parent
        sources = "\n".join(p.read_text() for p in sorted(root.glob("*.tex")))
        labels = re.findall(r"\\label\{([^}]+)\}", sources)
        self.assertFalse([k for k, count in Counter(labels).items() if count > 1])
        used = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", sources)
        self.assertFalse(set(used)-set(labels))
        bibkeys = set(re.findall(r"@\w+\{([^,]+),", (root/"references.bib").read_text()))
        cites = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", sources)
                 for key in group.split(",")}
        self.assertFalse(cites-bibkeys)
        self.assertTrue({"bobu2020", "zeng2020", "ablett2025"} <= cites)
        paragraphs = [p for p in (root/"experiments.tex").read_text().split("\n\n") if p.strip()]
        self.assertEqual(len(paragraphs), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
