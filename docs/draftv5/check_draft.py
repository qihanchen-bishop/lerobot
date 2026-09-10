"""Deterministic checks of draft identities; not a robotics guarantee."""
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def average(values):
    return sum(values) / len(values)


def mse(values):
    return average([x * x for x in values])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class TheoryChecks(unittest.TestCase):
    def test_full_coverage_counterexample(self):
        risk_p, risk_q = mse([0.1, 0]), mse([0, 0.2])
        failure_p = 0.5 * min(1, 100 * 0.1**2)
        failure_q = 0.5 * min(1, 100 * 0**2)
        self.assertAlmostEqual(risk_p, 0.005)
        self.assertAlmostEqual(risk_q, 0.02)
        self.assertLess(risk_p, risk_q)
        self.assertGreater(failure_p, failure_q)

    def test_performance_difference_and_coverage_bound(self):
        horizon = 4
        transitions = [
            [[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]],
            [[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]],
        ]
        expert, candidate = [0, 0, 0], [1, 0, 1]
        terminal = [0.0, 1.0, 1.0]
        values, qs = [None] * (horizon + 1), [None] * horizon
        values[-1] = terminal
        for t in reversed(range(horizon)):
            qs[t] = [
                [dot(transitions[a][s], values[t + 1]) for a in range(2)]
                for s in range(3)
            ]
            values[t] = [qs[t][s][expert[s]] for s in range(3)]

        def occupancy(policy):
            distributions = [[0.3, 0.4, 0.3]]
            for _ in range(horizon):
                old = distributions[-1]
                distributions.append([
                    sum(old[s] * transitions[policy[s]][s][j] for s in range(3))
                    for j in range(3)
                ])
            return distributions

        de, dp = occupancy(expert), occupancy(candidate)
        gap = dot(dp[-1], terminal) - dot(de[-1], terminal)
        advantages = sum(
            dp[t][s] * (qs[t][s][candidate[s]] - values[t][s])
            for t in range(horizon) for s in range(3)
        )
        self.assertAlmostEqual(gap, advantages, places=12)
        concentration = max(
            dp[t][s] / de[t][s] for t in range(horizon) for s in range(3)
        )
        expert_risk = sum(
            de[t][s] * (candidate[s] - expert[s])**2
            for t in range(horizon) for s in range(3)
        ) / horizon
        lipschitz = max(
            abs(qs[t][s][1] - qs[t][s][0])
            for t in range(horizon) for s in range(3)
        )
        self.assertLessEqual(gap, horizon * lipschitz *
                             math.sqrt(concentration * expert_risk) + 1e-12)

    def test_projection_and_history_identities(self):
        # Independent binary current cue, history cue and unobserved noise.
        rows = [(c, h, n) for c in (-1, 1) for h in (-1, 1) for n in (-1, 1)]
        error = [c + 2*h + n for c, h, n in rows]
        current_mean = [c for c, _, _ in rows]
        history_mean = [c + 2*h for c, h, _ in rows]
        # An encoder discards half of the history information.
        encoded_mean = current_mean
        residual = [0.8*c for c, _, _ in rows]
        enc = mse([h-u for h, u in zip(history_mean, encoded_mean)])
        fit = mse([r-u for r, u in zip(residual, encoded_mean)])
        base = mse(error)
        corrected = mse([e-r for e, r in zip(error, residual)])
        best_current = mse([e-c for e, c in zip(error, current_mean)])
        history_gain = mse([h-c for h, c in zip(history_mean, current_mean)])
        self.assertAlmostEqual(base - corrected, mse(history_mean) - enc - fit)
        self.assertAlmostEqual(corrected - best_current, enc + fit - history_gain)
        self.assertAlmostEqual(base-corrected,
                               2*average([e*r for e, r in zip(error, residual)]) -
                               mse(residual))
        self.assertAlmostEqual(history_gain, 4)

    def test_semantic_determinism_has_no_extra_bayes_information(self):
        # A fixed semantic code duplicates a function already determined by RGB.
        rows = [(0, -1), (0, 1), (1, 1), (1, 3)]
        def bayes_risk(key):
            groups = {}
            for i, y in rows:
                groups.setdefault(key(i), []).append(y)
            return sum(sum((y-average(ys))**2 for y in ys)
                       for ys in groups.values()) / len(rows)
        self.assertAlmostEqual(bayes_risk(lambda i: i),
                               bayes_risk(lambda i: (i, int(i > 0))))

    def test_geometry_bound(self):
        target_mean = [-2.0, -1.0, 1.0, 2.0]
        geometry = [-1.0, -0.5, 0.5, 1.0]
        estimated = [-0.9, -0.6, 0.4, 1.2]
        sufficient_error = mse([m-2*g for m, g in zip(target_mean, geometry)])
        geo_error = mse([g-gh for g, gh in zip(geometry, estimated)])
        candidate_risk = mse([m-2*gh for m, gh in zip(target_mean, estimated)])
        bound = (math.sqrt(sufficient_error) + 2*math.sqrt(geo_error))**2
        self.assertLessEqual(candidate_risk, bound + 1e-12)

    def test_te_bias_covariance_and_convexity(self):
        weights = [0.7, 0.3]
        rows = [(n1, n2) for n1 in (-1, 1) for n2 in (-1, 1)]
        errors = [(2+n1, -1+n1+2*n2) for n1, n2 in rows]
        combined = [dot(weights, e) for e in errors]
        bias = [average([e[k] for e in errors]) for k in range(2)]
        covariance = [[average([(e[k]-bias[k])*(e[l]-bias[l]) for e in errors])
                       for l in range(2)] for k in range(2)]
        predicted = dot(weights, bias)**2 + sum(
            weights[k]*weights[l]*covariance[k][l] for k in range(2) for l in range(2)
        )
        self.assertAlmostEqual(mse(combined), predicted)
        for e, combined_error in zip(errors, combined):
            self.assertLessEqual(combined_error**2, dot(weights, [x*x for x in e]) + 1e-12)

    def test_stale_plan_can_be_worse_than_latest(self):
        old, new, alpha = 2.0, -1.0, 0.6
        te = alpha*old + (1-alpha)*new
        self.assertAlmostEqual((te-new)**2, alpha**2*(old-new)**2)
        self.assertGreater((te-new)**2, (new-new)**2)

    def test_quality_weights_can_bias_target(self):
        truth, weights = [-1.0, 1.0], [1.0, 0.2]
        estimate = dot(truth, weights) / sum(weights)
        self.assertAlmostEqual(average(truth), 0)
        self.assertNotAlmostEqual(estimate, average(truth))

    def test_local_task_direction(self):
        # Smooth cost Q(a)=a^2 on a local normalized action interval.
        base, correction, hessian_bound = 0.5, -0.1, 2.0
        actual = (base+correction)**2 - base**2
        bound = 2*base*correction + hessian_bound/2*correction**2
        self.assertAlmostEqual(actual, bound)
        self.assertLess(bound, 0)
        excessive = -1.2
        self.assertGreater((base+excessive)**2 - base**2, 0)

    def test_delay_slots(self):
        def late_prefix(latency):
            return min(9, max(0, math.ceil(latency * 30 - 1e-12) - 3))
        self.assertEqual(late_prefix(0.10), 0)
        self.assertEqual(late_prefix(0.15), 2)
        self.assertLessEqual(6 + late_prefix(0.15), 9)
        self.assertGreater(6 + late_prefix(0.24), 9)

    def test_tex_references_and_inputs(self):
        sources = "\n".join(p.read_text() for p in sorted(ROOT.glob("*.tex")))
        labels = re.findall(r"\\label\{([^}]+)\}", sources)
        self.assertEqual(len(labels), len(set(labels)))
        for label in re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", sources):
            self.assertIn(label, labels)
        bib = (ROOT / "references.bib").read_text()
        keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        for citations in re.findall(r"\\cite\{([^}]+)\}", sources):
            for key in citations.split(","):
                self.assertIn(key.strip(), keys)
        for filename in re.findall(r"\\input\{([^}]+)\}", sources):
            self.assertTrue((ROOT / (filename + ".tex")).exists())

    def test_seen_and_new_initial_positions(self):
        train = {(0, 0), (0, 3), (1, 1), (1, 2), (2, 0), (2, 3)}
        test = {(r, c) for r in range(3) for c in range(4)}
        unseen = {(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)}
        self.assertEqual(test - train, unseen)
        self.assertEqual(3 * len(test), 36)
        self.assertEqual(3 * len(unseen), 18)

    def test_completion_is_simultaneous_not_historical_latch(self):
        def complete(inside, cloth_fraction, threshold=0.6):
            return any(i and c >= threshold for i, c in zip(inside, cloth_fraction))
        self.assertFalse(complete([False], [0.9]))
        self.assertFalse(complete([True, False], [0.2, 0.9]))
        self.assertTrue(complete([False, True], [0.8, 0.7]))
        # Cloth fraction uses the image area, never the object's area.
        self.assertAlmostEqual(70 / 100, 0.7)

    def test_region_footprint_is_not_visible_region_label(self):
        footprint, object_mask = {1, 2, 3, 4}, {2, 3}
        visible_region = footprint - object_mask
        self.assertFalse(object_mask & visible_region)
        self.assertTrue(object_mask <= footprint)

    def test_abstract_theory_is_separate_from_task_instance(self):
        task = (ROOT / "task.tex").read_text()
        for concrete in ("布料", "(0,3)", "p_{\\rm restore}", "87\\%"):
            self.assertNotIn(concrete, task)
        self.assertIn(r"\input{task_instance}", (ROOT / "experiments.tex").read_text())
        self.assertIn(r"\input{evidence}", (ROOT / "experiments.tex").read_text())

    def test_task_weighted_bound_and_normalization(self):
        expert_d, deployed_d = [0.5, 0.5], [0.8, 0.2]
        weights, errors = [20.0, 1.0], [0.05, 0.2]
        advantages = [0.4, 0.1]
        concentration = max(p/q for p, q in zip(deployed_d, expert_d))
        mean_weight = dot(deployed_d, weights)
        weighted_risk = sum(q*w*e*e for q, w, e in zip(expert_d, weights, errors))
        task_gap = dot(deployed_d, advantages)
        self.assertTrue(all(a <= w*abs(e) for a, w, e in zip(advantages, weights, errors)))
        self.assertLessEqual(task_gap, math.sqrt(mean_weight*concentration*weighted_risk))
        normalizer = dot(expert_d, weights)
        weighted_d = [q*w/normalizer for q, w in zip(expert_d, weights)]
        self.assertAlmostEqual(sum(weighted_d), 1)
        self.assertAlmostEqual(weighted_risk, normalizer*dot(weighted_d, [e*e for e in errors]))

    def test_weighted_conditional_mean_must_be_recomputed(self):
        # Both samples have the same observed representation but different labels.
        labels, probabilities, weights = [-1.0, 1.0], [0.5, 0.5], [3.0, 1.0]
        normalizer = dot(probabilities, weights)
        weighted_d = [p*w/normalizer for p, w in zip(probabilities, weights)]
        old_mean = dot(probabilities, labels)
        new_mean = dot(weighted_d, labels)
        self.assertNotEqual(old_mean, new_mean)
        old_risk = dot(weighted_d, [(y-old_mean)**2 for y in labels])
        new_risk = dot(weighted_d, [(y-new_mean)**2 for y in labels])
        self.assertAlmostEqual(old_risk-new_risk, (old_mean-new_mean)**2)

    def test_compiled_pdf_and_log(self):
        pdf = ROOT / "main.pdf"
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        log = (ROOT / "main.log").read_text()
        for unwanted in ("Overfull", "Missing character:", "Undefined control sequence",
                         "There were undefined references", "LaTeX Warning: Citation",
                         "LaTeX Warning: Reference"):
            self.assertNotIn(unwanted, log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
