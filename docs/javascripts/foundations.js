/* Interactive figures for the Foundations learning path.

   Dependency-free DOM and SVG. Every figure is a progressive enhancement: the
   lesson text explains the idea without JavaScript, and each container keeps a
   short fallback paragraph until its figure is drawn. The algorithms below
   follow the NumPy reference implementation of VAMOS
   (src/vamos/foundation/kernel/numpy_backend.py and operator_primitives.py,
   and src/vamos/foundation/quality_indicators/hypervolume.py).
   tests/docs/test_foundations.py checks that every figure a lesson uses is
   implemented here and, when Node.js is installed, runs the figures'
   algorithms against VAMOS: non-dominated sorting, crowding distance,
   hypervolume and its contributions, the evolutionary loop and the families'
   survivors, the step-by-step generations of every algorithm lesson (NSGA-II,
   SPEA2, MOEA/D, SMS-EMOA, IBEA, SMPSO, NSGA-III, RVEA, AGE-MOEA, CCMO,
   C-TAEA, ToP and CMOEA-MS), the variation operators, the benchmark fronts
   and CTP violations, two-layer reference directions, the constraint-handling
   rankings, F-race and the tuning budgets. The distance indicators and the
   Das-Dennis lattices were checked by hand against MooCore and VAMOS when the
   path was added. */
(function (root) {
  "use strict";

  /* ------------------------------------------------------------------ */
  /* Pure algorithms                                                     */
  /* ------------------------------------------------------------------ */

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** a Pareto-dominates b (minimization): no worse everywhere, better somewhere. */
  function dominates(a, b) {
    let better = false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > b[i]) return false;
      if (a[i] < b[i]) better = true;
    }
    return better;
  }

  /** Fast non-dominated sorting (Deb et al., 2002) with its bookkeeping. */
  function nonDominatedSort(F, relation) {
    const better = relation || dominates;
    const n = F.length;
    const S = [];
    const count = new Array(n).fill(0);
    for (let p = 0; p < n; p++) S.push([]);
    for (let p = 0; p < n; p++) {
      for (let q = 0; q < n; q++) {
        if (p === q) continue;
        if (better(F[p], F[q], p, q)) S[p].push(q);
        else if (better(F[q], F[p], q, p)) count[p] += 1;
      }
    }
    const rank = new Array(n).fill(-1);
    const fronts = [];
    const remaining = count.slice();
    let current = [];
    for (let p = 0; p < n; p++) if (remaining[p] === 0) current.push(p);
    let level = 0;
    while (current.length) {
      fronts.push(current);
      const next = [];
      for (const p of current) rank[p] = level;
      for (const p of current) {
        for (const q of S[p]) {
          remaining[q] -= 1;
          if (remaining[q] === 0) next.push(q);
        }
      }
      next.sort((a, b) => a - b);
      current = next;
      level += 1;
    }
    return { fronts, rank, count, dominatedSets: S };
  }

  /** Crowding distance of the rows listed in `front`, aligned with `front`. */
  function crowdingDistance(F, front) {
    const k = front.length;
    const d = new Array(k).fill(0);
    if (k === 0) return d;
    if (k <= 2) return d.fill(Infinity);
    const m = F[front[0]].length;
    for (let j = 0; j < m; j++) {
      const order = front.map((_, i) => i).sort((a, b) => F[front[a]][j] - F[front[b]][j] || a - b);
      const lo = F[front[order[0]]][j];
      const hi = F[front[order[k - 1]]][j];
      const span = hi - lo;
      if (!(span > 0)) continue;
      d[order[0]] = Infinity;
      d[order[k - 1]] = Infinity;
      for (let i = 1; i < k - 1; i++) {
        d[order[i]] += (F[front[order[i + 1]]][j] - F[front[order[i - 1]]][j]) / span;
      }
    }
    return d;
  }

  /** Ranks and crowding distances of every row, as VAMOS' nsga2_ranking. */
  function rankAndCrowding(F, relation) {
    const sorted = nonDominatedSort(F, relation);
    const crowding = new Array(F.length).fill(0);
    for (const front of sorted.fronts) {
      const d = crowdingDistance(F, front);
      front.forEach((row, i) => {
        crowding[row] = d[i];
      });
    }
    return { rank: sorted.rank, crowding, fronts: sorted.fronts };
  }

  /** vamos.select_survivors: lower rank, then larger crowding, then lower index. */
  function selectSurvivors(F, count) {
    const r = rankAndCrowding(F);
    const order = F.map((_, i) => i).sort((a, b) => r.rank[a] - r.rank[b] || cmpDesc(r.crowding[a], r.crowding[b]) || a - b);
    return { selected: order.slice(0, count), rank: r.rank, crowding: r.crowding, fronts: r.fronts };
  }

  function cmpDesc(a, b) {
    if (a === b) return 0;
    return a > b ? -1 : 1;
  }

  function nonDominatedIndices(F) {
    const out = [];
    for (let i = 0; i < F.length; i++) {
      let dominated = false;
      for (let j = 0; j < F.length && !dominated; j++) {
        if (j !== i && dominates(F[j], F[i])) dominated = true;
      }
      if (!dominated) out.push(i);
    }
    return out;
  }

  /** Exact two-objective hypervolume; points outside the reference box are ignored. */
  function hypervolume2d(points, ref) {
    const inside = points.filter((p) => p[0] < ref[0] && p[1] < ref[1]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let area = 0;
    let ceiling = ref[1];
    for (const p of inside) {
      if (p[1] < ceiling) {
        area += (ref[0] - p[0]) * (ceiling - p[1]);
        ceiling = p[1];
      }
    }
    return area;
  }

  /** Exclusive hypervolume contributions, as VAMOS's hypervolume_contributions and MooCore's
      hv_contributions (ignore_dominated) compute them: points outside the reference box are
      left out, a dominated point gives 0 and is left out of the others' contributions, and
      equal points give 0 each while still covering their neighbours. */
  function hvContributions2d(points, ref) {
    const out = new Array(points.length).fill(0);
    const boxes = new Array(points.length).fill(null);
    const groups = new Map();
    points.forEach((p, i) => {
      if (!(p[0] < ref[0] && p[1] < ref[1])) return;
      const key = p[0] + "," + p[1];
      if (!groups.has(key)) groups.set(key, { p, members: [] });
      groups.get(key).members.push(i);
    });
    const unique = [...groups.values()].sort((a, b) => a.p[0] - b.p[0] || a.p[1] - b.p[1]);
    const front = [];
    let lowest = Infinity;
    for (const group of unique) {
      if (group.p[1] < lowest) {
        front.push(group);
        lowest = group.p[1];
      }
    }
    front.forEach((group, k) => {
      if (group.members.length > 1) return;
      const p = group.p;
      const right = k + 1 < front.length ? front[k + 1].p[0] : ref[0];
      const top = k > 0 ? front[k - 1].p[1] : ref[1];
      out[group.members[0]] = (right - p[0]) * (top - p[1]);
      boxes[group.members[0]] = [p[0], p[1], right, top];
    });
    return { contributions: out, boxes };
  }

  /** The contour max(w·f₁, (1 − w)·f₂) = c of the Tchebycheff scalarization with the ideal point at the origin:
      the upper and right edges of the rectangle where the value is at most c, which meet at (c / w, c / (1 − w)). */
  function tchebycheffContour(w, c) {
    const cx = w > 1e-9 ? c / w : 3;
    const cy = 1 - w > 1e-9 ? c / (1 - w) : 3;
    return [[0, cy], [cx, cy], [cx, 0]];
  }

  function euclidean(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return Math.sqrt(s);
  }

  function dPlus(a, r) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const v = Math.max(a[i] - r[i], 0);
      s += v * v;
    }
    return Math.sqrt(s);
  }

  function nearest(from, to, distance) {
    let best = Infinity;
    let index = -1;
    to.forEach((t, i) => {
      const d = distance(t, from);
      if (d < best) {
        best = d;
        index = i;
      }
    });
    return { distance: best, index };
  }

  /** GD with p = 1: mean distance from each approximation point to the reference set. */
  function generationalDistance(A, R) {
    return A.reduce((sum, a) => sum + nearest(a, R, (r, x) => euclidean(x, r)).distance, 0) / A.length;
  }

  /** IGD with p = 1: mean distance from each reference point to the approximation set. */
  function igd(A, R) {
    return R.reduce((sum, r) => sum + nearest(r, A, (a, x) => euclidean(a, x)).distance, 0) / R.length;
  }

  /** IGD+ (Ishibuchi et al., 2015): only the objectives where a is worse than r count. */
  function igdPlus(A, R) {
    return R.reduce((sum, r) => sum + nearest(r, A, (a, x) => dPlus(a, x)).distance, 0) / R.length;
  }

  /** Additive epsilon: smallest shift that makes A weakly dominate every point of R. */
  function epsilonAdditive(A, R) {
    let worst = -Infinity;
    for (const r of R) {
      let best = Infinity;
      for (const a of A) {
        let shift = -Infinity;
        for (let i = 0; i < a.length; i++) shift = Math.max(shift, a[i] - r[i]);
        best = Math.min(best, shift);
      }
      worst = Math.max(worst, best);
    }
    return worst;
  }

  /** Schott's spacing: standard deviation of nearest-neighbour L1 distances. */
  function spacing(A) {
    if (A.length < 2) return NaN;
    const d = A.map((a, i) => {
      let best = Infinity;
      A.forEach((b, j) => {
        if (i === j) return;
        let s = 0;
        for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
        best = Math.min(best, s);
      });
      return best;
    });
    const mean = d.reduce((s, v) => s + v, 0) / d.length;
    return Math.sqrt(d.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (d.length - 1));
  }

  /** Deb's spread Delta for two objectives, given the extremes of the true front. */
  function spreadDelta(A, extremes) {
    if (A.length < 2) return NaN;
    const P = A.slice().sort((a, b) => a[0] - b[0]);
    const gaps = [];
    for (let i = 0; i + 1 < P.length; i++) gaps.push(euclidean(P[i], P[i + 1]));
    const mean = gaps.reduce((s, v) => s + v, 0) / gaps.length;
    const df = euclidean(extremes[0], P[0]);
    const dl = euclidean(extremes[1], P[P.length - 1]);
    const dev = gaps.reduce((s, v) => s + Math.abs(v - mean), 0);
    return (df + dl + dev) / (df + dl + gaps.length * mean);
  }

  function sbxPair(p1, p2, lower, upper, eta, probVar, rand) {
    const c1 = p1.slice();
    const c2 = p2.slice();
    const inv = 1 / (eta + 1);
    for (let j = 0; j < p1.length; j++) {
      const y1 = Math.min(p1[j], p2[j]);
      const y2 = Math.max(p1[j], p2[j]);
      const diff = y2 - y1;
      if (!(diff > 1e-14)) continue;
      if (probVar < 1 && !(rand() <= probVar)) continue;
      const u = rand();
      let beta = Math.max(1 + (2 * (y1 - lower)) / diff, 1e-14);
      let alpha = Math.max(2 - Math.pow(beta, -(eta + 1)), 1e-14);
      let bq = u <= 1 / alpha ? Math.pow(u * alpha, inv) : Math.pow(1 / (2 - u * alpha), inv);
      let a = 0.5 * (y1 + y2 - bq * diff);
      beta = Math.max(1 + (2 * (upper - y2)) / diff, 1e-14);
      alpha = Math.max(2 - Math.pow(beta, -(eta + 1)), 1e-14);
      bq = u <= 1 / alpha ? Math.pow(u * alpha, inv) : Math.pow(1 / (2 - u * alpha), inv);
      let b = 0.5 * (y1 + y2 + bq * diff);
      if (rand() <= 0.5) {
        const t = a;
        a = b;
        b = t;
      }
      c1[j] = a;
      c2[j] = b;
    }
    return [c1, c2];
  }

  function polynomialMutation(x, lower, upper, prob, eta, rand) {
    const span = upper - lower;
    const power = 1 / (eta + 1);
    for (let j = 0; j < x.length; j++) {
      if (!(rand() <= prob)) continue;
      const y = x[j];
      const u = rand();
      let dq;
      if (u <= 0.5) {
        const val = 2 * u + (1 - 2 * u) * Math.pow(1 - (y - lower) / span, eta + 1);
        dq = Math.pow(val, power) - 1;
      } else {
        const val = 2 * (1 - u) + 2 * (u - 0.5) * Math.pow(1 - (upper - y) / span, eta + 1);
        dq = 1 - Math.pow(val, power);
      }
      x[j] = Math.min(upper, Math.max(lower, y + dq * span));
    }
    return x;
  }

  function zdtG(x) {
    let s = 0;
    for (let i = 1; i < x.length; i++) s += x[i];
    return 1 + (9 * s) / (x.length - 1);
  }

  const PROBLEMS = {
    zdt1: {
      label: "ZDT1 (convex)",
      evaluate: (x) => {
        const g = zdtG(x);
        return [x[0], g * (1 - Math.sqrt(x[0] / g))];
      },
      curve: (t) => 1 - Math.sqrt(t),
    },
    zdt2: {
      label: "ZDT2 (non-convex)",
      evaluate: (x) => {
        const g = zdtG(x);
        return [x[0], g * (1 - (x[0] / g) * (x[0] / g))];
      },
      curve: (t) => 1 - t * t,
    },
    zdt3: {
      label: "ZDT3 (disconnected)",
      evaluate: (x) => {
        const g = zdtG(x);
        const r = x[0] / g;
        return [x[0], g * (1 - Math.sqrt(r) - r * Math.sin(10 * Math.PI * x[0]))];
      },
      curve: (t) => 1 - Math.sqrt(t) - t * Math.sin(10 * Math.PI * t),
    },
  };

  function sampledFront(problem, n) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      pts.push([t, problem.curve(t)]);
    }
    return nonDominatedIndices(pts).map((i) => pts[i]);
  }

  function nsga2Truncate(F, count) {
    const r = rankAndCrowding(F);
    const selected = [];
    for (const front of r.fronts) {
      if (selected.length + front.length <= count) {
        selected.push(...front);
        if (selected.length === count) break;
      } else {
        const rest = front.slice().sort((a, b) => cmpDesc(r.crowding[a], r.crowding[b]) || a - b);
        selected.push(...rest.slice(0, count - selected.length));
        break;
      }
    }
    return selected;
  }

  /** A compact NSGA-II with VAMOS' default real-coded operators. */
  function createNSGAII(problem, popSize, nVar, seed) {
    const rand = mulberry32(seed);
    const st = { X: [], F: [], generation: 1, evaluations: 0 };
    for (let i = 0; i < popSize; i++) {
      const x = [];
      for (let j = 0; j < nVar; j++) x.push(rand());
      st.X.push(x);
      st.F.push(problem.evaluate(x));
    }
    st.evaluations = popSize;
    let metrics = rankAndCrowding(st.F);
    st.rank = metrics.rank;
    function tournament() {
      const n = st.X.length;
      const a = Math.floor(rand() * n);
      let b = Math.floor(rand() * (n - 1));
      if (b >= a) b += 1;
      if (metrics.rank[a] !== metrics.rank[b]) return metrics.rank[a] < metrics.rank[b] ? a : b;
      if (metrics.crowding[a] !== metrics.crowding[b]) return metrics.crowding[a] > metrics.crowding[b] ? a : b;
      return rand() < 0.5 ? a : b;
    }
    st.step = function () {
      const offspring = [];
      while (offspring.length < popSize) {
        const p1 = st.X[tournament()];
        const p2 = st.X[tournament()];
        const children = rand() <= 0.9 ? sbxPair(p1, p2, 0, 1, 20, 0.5, rand) : [p1.slice(), p2.slice()];
        for (const child of children) {
          polynomialMutation(child, 0, 1, 1 / nVar, 20, rand);
          offspring.push(child);
        }
      }
      const offF = offspring.map((x) => problem.evaluate(x));
      const allX = st.X.concat(offspring);
      const allF = st.F.concat(offF);
      const keep = nsga2Truncate(allF, popSize);
      st.X = keep.map((i) => allX[i]);
      st.F = keep.map((i) => allF[i]);
      st.generation += 1;
      st.evaluations += offspring.length;
      metrics = rankAndCrowding(st.F);
      st.rank = metrics.rank;
    };
    return st;
  }

  /* CTP2 of Deb, Pratap and Meyarivan (2001), written in objective space. */
  const CTP2 = { theta: -0.2 * Math.PI, a: 0.2, b: 10, c: 1, d: 6, e: 1 };

  function ctpViolation(f, p) {
    const prm = p || CTP2;
    const lhs = Math.cos(prm.theta) * (f[1] - prm.e) - Math.sin(prm.theta) * f[0];
    const inner = Math.sin(prm.theta) * (f[1] - prm.e) + Math.cos(prm.theta) * f[0];
    const rhs = prm.a * Math.pow(Math.abs(Math.sin(prm.b * Math.PI * Math.pow(inner, prm.c))), prm.d);
    return Math.max(0, rhs - lhs);
  }

  /** Constrained domination (Deb, 2000), with violations passed alongside. */
  function constrainedDominates(cv) {
    return function (a, b, i, j) {
      const fa = cv[i] <= 0;
      const fb = cv[j] <= 0;
      if (fa && !fb) return true;
      if (!fa && fb) return false;
      if (!fa && !fb) return cv[i] < cv[j];
      return dominates(a, b);
    };
  }

  function binomial(n, k) {
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
    return Math.round(r);
  }

  function dasDennis(m, p) {
    const out = [];
    const rec = (prefix, left, depth) => {
      if (depth === m - 1) {
        out.push(prefix.concat([left / p]));
        return;
      }
      for (let i = 0; i <= left; i++) rec(prefix.concat([i / p]), left - i, depth + 1);
    };
    rec([], p, 0);
    return out;
  }

  /** Points z attained by at least k of the runs, as a 2-D staircase. */
  function attainmentSurface(runs, k) {
    const xs = Array.from(new Set(runs.flat().map((p) => p[0]))).sort((a, b) => a - b);
    return xs.map((u) => {
      const heights = runs
        .map((run) => run.reduce((best, p) => (p[0] <= u ? Math.min(best, p[1]) : best), Infinity))
        .sort((a, b) => a - b);
      return [u, heights[k - 1]];
    });
  }

  function normalCdf(z) {
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  /** Two-sided Mann-Whitney U test with the normal approximation. */
  function mannWhitney(a, b) {
    const all = a.map((v) => [v, 0]).concat(b.map((v) => [v, 1])).sort((x, y) => x[0] - y[0]);
    const ranks = new Array(all.length);
    for (let i = 0; i < all.length; ) {
      let j = i;
      while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j += 1;
      for (let k = i; k <= j; k++) ranks[k] = (i + j) / 2 + 1;
      i = j + 1;
    }
    let r1 = 0;
    all.forEach((item, i) => {
      if (item[1] === 0) r1 += ranks[i];
    });
    const n1 = a.length;
    const n2 = b.length;
    const u = r1 - (n1 * (n1 + 1)) / 2;
    const mean = (n1 * n2) / 2;
    const sd = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
    const z = (Math.abs(u - mean) - 0.5) / sd;
    return { u, p: Math.min(1, 2 * (1 - normalCdf(Math.max(z, 0)))) };
  }

  /** Lexicographic comparison of two ascending distance lists. */
  function lexLess(a, b) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (Math.abs(a[i] - b[i]) > 1e-12) return a[i] < b[i];
    }
    return false;
  }

  /** SPEA2 strength S, raw fitness R, distance σᵏ to the k-th nearest neighbour, density D = 1 / (σᵏ + 2)
      and fitness F = R + D. */
  function spea2Fitness(F, k) {
    const n = F.length;
    const S = F.map((a) => F.filter((b) => dominates(a, b)).length);
    const R = F.map((b, i) => F.reduce((sum, a, j) => (j !== i && dominates(a, b) ? sum + S[j] : sum), 0));
    const kk = Math.max(1, Math.min(n - 1, k || Math.floor(Math.sqrt(n))));
    const sigma = F.map((a, i) => F.map((b, j) => (j === i ? Infinity : euclidean(a, b))).sort((x, y) => x - y)[kk - 1]);
    const D = sigma.map((d) => 1 / (d + 2));
    return { S, R, sigma, D, fitness: R.map((r, i) => r + D[i]), k: kk };
  }

  /** SPEA2 environmental selection: non-dominated first, then truncation by nearest neighbours.
      `truncation` records, before each removal, the members left and their sorted distances to the others. */
  function spea2Select(F, size, k) {
    const fit = spea2Fitness(F, k);
    const nd = F.map((_, i) => i).filter((i) => fit.R[i] === 0);
    let archive = nd.slice();
    let filled = [];
    if (F.length <= size) archive = F.map((_, i) => i);
    else if (archive.length < size) {
      const rest = F.map((_, i) => i)
        .filter((i) => fit.R[i] > 0)
        .sort((a, b) => fit.fitness[a] - fit.fitness[b] || a - b);
      filled = rest.slice(0, size - archive.length);
      archive = archive.concat(filled);
    }
    const removed = [];
    const truncation = [];
    while (archive.length > size) {
      const lists = archive.map((i) =>
        archive
          .filter((j) => j !== i)
          .map((j) => euclidean(F[i], F[j]))
          .sort((a, b) => a - b)
      );
      let worst = 0;
      for (let a = 1; a < archive.length; a++) if (lexLess(lists[a], lists[worst])) worst = a;
      truncation.push({ members: archive.slice(), lists });
      removed.push(archive[worst]);
      archive.splice(worst, 1);
    }
    return { kept: archive, removed, filled, nd, truncation, ...fit };
  }

  /** One SPEA2 generation of the loop figure's problem, recorded for the SPEA2 lesson as VAMOS runs it with
      archive_size = pop_size and k = 1. The archive holds the initial population, which the first environmental
      selection keeps whole, with the raw fitness R and neighbour distance σᵏ measured there. N pairs of binary
      tournaments on the archive (lower R, then larger σᵏ, then a coin) give one child per pair: the first child of
      SBX (probability 0.9, η = 20), mutated by polynomial mutation (probability 1/2, η = 20). Archive and children
      form the union, and spea2Select gives its S, R, σᵏ, D and F, the non-dominated members, the members added by F
      or the truncation with its distance lists, and the new archive, whose R and σᵏ the next tournaments use. */
  function spea2Generation(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    const k = 1;
    let nextId = 1;
    const archive = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      archive.push({ id: nextId++, x, f: gaEvaluate(x) });
    }
    const start = spea2Select(archive.map((p) => p.f), N, k);
    const tournaments = [];
    const tournament = () => {
      const a = Math.floor(rand() * N);
      let b = Math.floor(rand() * (N - 1));
      if (b >= a) b += 1;
      let w = a;
      let reason = "coin";
      if (start.R[a] !== start.R[b]) {
        w = start.R[a] < start.R[b] ? a : b;
        reason = "raw";
      } else if (start.sigma[a] !== start.sigma[b]) {
        w = start.sigma[a] > start.sigma[b] ? a : b;
        reason = "density";
      } else if (rand() >= 0.5) w = b;
      tournaments.push({ a: archive[a].id, b: archive[b].id, winner: archive[w].id, reason });
      return archive[w];
    };
    const pairs = [];
    for (let c = 0; c < N; c++) pairs.push([tournament(), tournament()]);
    const children = pairs.map(([p1, p2]) => {
      const x = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 20, 0.5, rand)[0] : p1.x.slice();
      polynomialMutation(x, 0, 1, 0.5, 20, rand);
      return { id: nextId++, x, f: gaEvaluate(x), parents: [p1.id, p2.id] };
    });
    const union = archive.concat(children);
    const sel = spea2Select(union.map((p) => p.f), N, k);
    return {
      popSize: N,
      k,
      archive,
      archiveS: start.S,
      archiveR: start.R,
      archiveSigma: start.sigma,
      archiveFitness: start.fitness,
      tournaments,
      children,
      union,
      S: sel.S,
      R: sel.R,
      sigma: sel.sigma,
      D: sel.D,
      fitness: sel.fitness,
      nd: sel.nd,
      filled: sel.filled,
      removed: sel.removed,
      truncation: sel.truncation,
      kept: sel.kept,
    };
  }

  /** Additive epsilon indicator I(a, b): smallest shift of a that weakly dominates b. */
  function epsilonPair(a, b) {
    let m = -Infinity;
    for (let i = 0; i < a.length; i++) m = Math.max(m, a[i] - b[i]);
    return m;
  }

  /** IBEA environmental selection (VAMOS' ibea/helpers.py): additive epsilon or hypervolume difference.
      Z holds the normalized objectives, and history the fitness of every row after each removal. */
  function ibeaSelect(F, size, kappa, indicator) {
    const m = F[0].length;
    const lo = [];
    const hi = [];
    for (let j = 0; j < m; j++) {
      lo.push(Math.min(...F.map((f) => f[j])));
      hi.push(Math.max(...F.map((f) => f[j])));
    }
    const Z = F.map((f) => f.map((v, j) => (hi[j] > lo[j] ? (v - lo[j]) / (hi[j] - lo[j]) : 0)));
    let M;
    if (indicator === "hv") {
      const ref = Z[0].map((_, j) => Math.max(...Z.map((z) => z[j])) + 1);
      const H = (p) => p.reduce((prod, v, j) => prod * Math.max(ref[j] - v, 0), 1);
      const Hc = (p, q) => p.reduce((prod, v, j) => prod * Math.max(ref[j] - Math.max(v, q[j]), 0), 1);
      M = Z.map((x, i) => Z.map((y, k) => (i === k ? 0 : dominates(y, x) ? H(x) - H(y) : H(x) - Hc(x, y))));
    } else {
      M = Z.map((x) => Z.map((y) => epsilonPair(y, x)));
    }
    let c = 0;
    M.forEach((row) => row.forEach((v) => (c = Math.max(c, Math.abs(v)))));
    if (c > 0) M = M.map((row) => row.map((v) => v / c));
    const T = M.map((row, x) => row.map((v, y) => (x === y ? 0 : Math.exp(-v / kappa))));
    const fitness = T.map((row) => -row.reduce((sum, v) => sum + v, 0));
    const initial = fitness.slice();
    const alive = Z.map((_, i) => i);
    const removed = [];
    const history = [];
    while (alive.length > size) {
      let worst = alive[0];
      for (const i of alive) if (fitness[i] < fitness[worst]) worst = i;
      alive.splice(alive.indexOf(worst), 1);
      removed.push(worst);
      for (const x of alive) fitness[x] += T[x][worst];
      history.push(fitness.slice());
    }
    return { fitness: initial, final: fitness, kept: alive, removed, history, Z, c: c || 1, terms: T };
  }

  /** One IBEA generation of the loop figure's problem, recorded for the IBEA lesson as VAMOS runs it with the
      additive ε indicator and κ = 0.05. The population carries the fitness of the initial environmental
      selection, which removes nobody. N binary tournaments (the higher fitness wins) give N / 2 pairs, and SBX
      (probability 0.9, η = 20) with polynomial mutation (probability 1/2, η = 20) gives two children per pair.
      Population and children form the merged set, and ibeaSelect gives its normalized objectives, the scaling c,
      the fitness, the removals in order with the fitness after each one, and the survivors with their final fitness,
      which the next tournaments compare. */
  function ibeaGeneration(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    const kappa = 0.05;
    let nextId = 1;
    const pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push({ id: nextId++, x, f: gaEvaluate(x) });
    }
    const start = ibeaSelect(pop.map((p) => p.f), N, kappa, "eps");
    const tournaments = [];
    const tournament = () => {
      const a = Math.floor(rand() * N);
      let b = Math.floor(rand() * (N - 1));
      if (b >= a) b += 1;
      // Equal fitness: the lower row wins, as VAMOS ranks a small population.
      const w = start.final[a] > start.final[b] || (start.final[a] === start.final[b] && a < b) ? a : b;
      tournaments.push({ a: pop[a].id, b: pop[b].id, winner: pop[w].id });
      return pop[w];
    };
    const pairs = [];
    for (let c = 0; c < Math.ceil(N / 2); c++) pairs.push([tournament(), tournament()]);
    const children = [];
    for (const [p1, p2] of pairs) {
      const xs = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 20, 0.5, rand) : [p1.x.slice(), p2.x.slice()];
      for (const x of xs) {
        polynomialMutation(x, 0, 1, 0.5, 20, rand);
        if (children.length < N) children.push({ id: nextId++, x, f: gaEvaluate(x), parents: [p1.id, p2.id] });
      }
    }
    const merged = pop.concat(children);
    const sel = ibeaSelect(merged.map((p) => p.f), N, kappa, "eps");
    return {
      popSize: N,
      kappa,
      pop,
      popFitness: start.final,
      tournaments,
      children,
      merged,
      Z: sel.Z,
      c: sel.c,
      fitness: sel.fitness,
      removed: sel.removed,
      history: sel.history,
      kept: sel.kept,
      final: sel.final,
    };
  }

  /** RVEA angle-penalized distance of translated objectives Fp against unit reference vectors V. */
  function rveaSelect(Fp, V, progress, alpha, penaltyM) {
    const M = penaltyM || V[0].length;
    const norm = (a) => Math.max(1e-64, Math.sqrt(a.reduce((s, v) => s + v * v, 0)));
    const cosine = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0) / (norm(a) * norm(b) || 1);
    const gamma = V.map((v, j) => Math.min(...V.map((u, k) => (k === j ? Infinity : Math.acos(Math.min(1, cosine(v, u)))))));
    const assoc = [];
    const theta = [];
    const apd = [];
    Fp.forEach((f) => {
      let best = 0;
      V.forEach((v, j) => {
        if (cosine(f, v) > cosine(f, V[best])) best = j;
      });
      const t = Math.acos(Math.min(1, cosine(f, V[best])));
      assoc.push(best);
      theta.push(t);
      apd.push((1 + M * Math.pow(progress, alpha) * (t / gamma[best])) * norm(f));
    });
    const selected = V.map((_, j) => {
      let pick = -1;
      Fp.forEach((_, i) => {
        if (assoc[i] === j && (pick < 0 || apd[i] < apd[pick])) pick = i;
      });
      return pick;
    });
    return { assoc, theta, apd, gamma, selected };
  }

  /** RVEA on the loop figure's problem with VAMOS' defaults, recorded for the RVEA lesson: N = popSize unit
      reference vectors (Das-Dennis, p = popSize − 1), parents from random permutations of the population, SBX
      (probability 0.9, η = 30) and polynomial mutation (probability 1/2, η = 20), the merge, the ideal point
      accumulated over the run, the angle-penalized distance with M = 1 and α = 2 at generation t of t_max,
      one survivor per non-empty vector, and the adaptation of the vectors to the survivors' nadir point minus the
      ideal point after every generation that is a multiple of ⌈0.1·t_max⌉. Generations 1 to `generation` − 1 run
      first; `history` keeps the input and the selection of every generation, and the other fields describe the
      last one, with the population numbered 1 to n and the children after it. */
  function rveaGeneration(seed, popSize, generation, maxGen) {
    const rand = mulberry32(seed);
    const N = popSize;
    const alpha = 2;
    const unit = (w) => {
      const n = Math.hypot(w[0], w[1]) || 1;
      return [w[0] / n, w[1] / n];
    };
    const V0 = dasDennis(2, N - 1).map(unit);
    const interval = Math.max(1, Math.ceil(maxGen * 0.1));
    const shuffle = (a) => {
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };
    let pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push({ x, f: gaEvaluate(x) });
    }
    let V = V0;
    let ideal = [Infinity, Infinity];
    const history = [];
    let last = null;
    for (let t = 1; t <= generation; t++) {
      pop = pop.map((p, i) => ({ id: i + 1, x: p.x, f: p.f }));
      const n = pop.length;
      const count = 2 * Math.ceil(N / 2);
      const draws = [];
      while (draws.length < count) draws.push(...shuffle(pop.map((_, i) => i)));
      draws.length = count;
      const children = [];
      for (let k = 0; k < count; k += 2) {
        const p1 = pop[draws[k]];
        const p2 = pop[draws[k + 1]];
        const xs = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 30, 0.5, rand) : [p1.x.slice(), p2.x.slice()];
        for (const x of xs) {
          polynomialMutation(x, 0, 1, 0.5, 20, rand);
          if (children.length < N) children.push({ id: n + children.length + 1, x, f: gaEvaluate(x), parents: [p1.id, p2.id] });
        }
      }
      const merged = pop.concat(children);
      const F = merged.map((p) => p.f);
      const idealBefore = ideal;
      ideal = [0, 1].map((m) => Math.min(idealBefore[m], ...F.map((f) => f[m])));
      const translated = F.map((f) => [f[0] - ideal[0], f[1] - ideal[1]]);
      const sel = rveaSelect(translated, V, t / maxGen, alpha, 1);
      const survivors = sel.selected.filter((i) => i >= 0);
      const nadir = [0, 1].map((m) => Math.max(...survivors.map((i) => F[i][m])));
      const adapted = t % interval === 0;
      const next = adapted ? V0.map((v) => unit(v.map((c, m) => c * Math.max(nadir[m] - ideal[m], 1e-64)))) : V;
      history.push({ generation: t, F, idealBefore, V, survivors, ideal, nadir, adapted, next });
      last = { pop, draws, children, merged, idealBefore, ideal, translated, V, select: sel, survivors, nadir, adapted, next };
      pop = survivors.map((i) => merged[i]);
      V = next;
    }
    return Object.assign({ popSize: N, generation, maxGen, alpha, M: 1, interval, V0, history }, last, {
      progress: Math.pow(generation / maxGen, alpha),
      gammaNext: rveaSelect([[1, 1]], last.next, 0, alpha, 1).gamma,
    });
  }

  /** MOEA/D weights for two objectives, as numpy.linspace builds them. */
  function moeadWeights(n) {
    const step = 1 / (n - 1);
    return Array.from({ length: n }, (_, k) => {
      const v = k === n - 1 ? 1 : k * step;
      return [v, 1 - v];
    });
  }

  /** Neighbourhoods: the T closest weight vectors (stable argsort of Euclidean distances). */
  function moeadNeighbours(W, T) {
    return W.map((w) =>
      W.map((u, j) => [Math.sqrt(w.reduce((sum, v, m) => sum + (v - u[m]) * (v - u[m]), 0)), j])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1])
        .slice(0, T)
        .map((e) => e[1])
    );
  }

  /** MOEA/D aggregation functions of VAMOS: Tchebycheff, weighted sum and PBI. */
  function moeadAggregate(mode, f, w, z, theta) {
    if (mode === "ws") return f.reduce((sum, v, m) => sum + w[m] * v, 0);
    if (mode === "pbi") {
      const norm = Math.sqrt(w.reduce((sum, v) => sum + v * v, 0));
      const u = w.map((v) => v / norm);
      const d = f.map((v, m) => v - z[m]);
      const d1 = Math.abs(d.reduce((sum, v, m) => sum + v * u[m], 0));
      const d2 = Math.sqrt(d.reduce((sum, v, m) => sum + (v - d1 * u[m]) * (v - d1 * u[m]), 0));
      return d1 + (theta === undefined ? 5 : theta) * d2;
    }
    return Math.max(...f.map((v, m) => (w[m] === 0 ? 1e-4 : w[m]) * Math.abs(v - z[m])));
  }

  /** One MOEA/D replacement step: strict improvement, at most nr replacements, in the given order. The ideal
      point is the minimum of y and `ideal`, the ideal point before y, or of y and the current solutions. */
  function moeadReplace(current, W, y, order, nr, mode, theta, ideal) {
    const before = ideal ? [ideal] : current;
    const z = y.map((v, m) => Math.min(v, ...before.map((f) => f[m])));
    let replaced = 0;
    const rows = order.map((j) => {
      const gx = moeadAggregate(mode, current[j], W[j], z, theta);
      const gy = moeadAggregate(mode, y, W[j], z, theta);
      let result = "kept";
      if (gy < gx) {
        if (replaced < nr) {
          result = "replaced";
          replaced += 1;
        } else result = "better (limit n_r reached)";
      }
      return { j, gx, gy, result };
    });
    return { z, rows, replaced };
  }

  /** One MOEA/D generation of the loop figure's problem, recorded for the MOEA/D lesson: one pass over the N
      subproblems in a random permutation, each creating one offspring with VAMOS' defaults. The mating pool is
      the neighbourhood of T subproblems when a uniform draw is below δ = 0.9, otherwise the whole population;
      two distinct parents r₁ and r₂ come from it; DE rand/1/bin from the current solution (F = 0.5, CR = 1.0)
      and polynomial mutation (probability 1/2, η = 20), each clipped to the bounds, create the offspring; the
      ideal point takes its objectives into account; and the offspring replaces, visiting the pool in a random
      order, the solutions whose PBI value (θ = 5) it improves, until n_r = 2 are replaced. The rows of a step
      after the n_r-th replacement are not visited. */
  function moeadGeneration(seed, popSize, T) {
    const rand = mulberry32(seed);
    const N = popSize;
    const delta = 0.9;
    const nr = 2;
    const theta = 5;
    const W = moeadWeights(N);
    const B = moeadNeighbours(W, T);
    const shuffle = (a) => {
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };
    const pop = [];
    for (let k = 0; k < N; k++) {
      const x = [rand(), rand()];
      pop.push({ id: k + 1, x, f: gaEvaluate(x) });
    }
    const holder = pop.slice();
    const z0 = [0, 1].map((m) => Math.min(...pop.map((p) => p.f[m])));
    let z = z0;
    const order = shuffle(W.map((_, k) => k));
    const steps = order.map((k, s) => {
      const u = rand();
      const neighbourhood = u < delta;
      const pool = neighbourhood ? B[k].slice() : W.map((_, j) => j);
      const a = Math.floor(rand() * pool.length);
      let b = Math.floor(rand() * (pool.length - 1));
      if (b >= a) b += 1;
      const r1 = pool[a];
      const r2 = pool[b];
      const before = holder.slice();
      const v = OPERATORS.deMating(before[k].x, before[r1].x, before[r2].x, 0.5, 1.0, rand)[0].map((t) => Math.min(1, Math.max(0, t)));
      const x = polynomialMutation(v.slice(), 0, 1, 0.5, 20, rand);
      const child = { id: N + s + 1, subproblem: k, x, v, f: gaEvaluate(x) };
      const candidates = shuffle(pool.slice());
      const res = moeadReplace(
        before.map((p) => p.f),
        W,
        child.f,
        candidates,
        nr,
        "pbi",
        theta,
        z
      );
      let count = 0;
      const rows = res.rows.map((r) => {
        const visited = count < nr;
        const replaced = visited && r.result === "replaced";
        if (replaced) count += 1;
        return { j: r.j, gx: r.gx, gy: r.gy, visited, better: r.gy < r.gx, replaced };
      });
      const replaced = rows.filter((r) => r.replaced).map((r) => r.j);
      replaced.forEach((j) => (holder[j] = child));
      const step = { k, u, neighbourhood, pool, r1, r2, before, child, zBefore: z, z: res.z, candidates, rows, replaced, after: holder.slice() };
      z = res.z;
      return step;
    });
    return { popSize: N, T, delta, nr, theta, W, B, pop, z0, order, steps, final: holder.slice() };
  }

  /** NSGA-III association: the direction of largest cosine and the perpendicular distance to it. */
  function nsga3Associate(f, W) {
    const norm = Math.sqrt(f.reduce((s2, v) => s2 + v * v, 0));
    let best = 0;
    let bestCos = -Infinity;
    W.forEach((w, j) => {
      const wn = Math.sqrt(w.reduce((s2, v) => s2 + v * v, 0));
      const c = norm > 0 ? f.reduce((s2, v, m) => s2 + (v / norm) * (w[m] / wn), 0) : 0;
      if (c > bestCos) {
        bestCos = c;
        best = j;
      }
    });
    return { dir: best, dperp: norm * Math.sqrt(Math.max(0, 1 - bestCos * bestCos)) };
  }

  /** NSGA-III niche preservation, returning every pick with its round and reason: the directions tied at the
      smallest count (`tied`) and those of them served in the round's random order (`Jmin`). */
  function nsga3Niching(rho0, candidates, K, rand) {
    const rho = rho0.slice();
    const left = candidates.slice();
    const picks = [];
    let round = 0;
    const shuffle = (a) => {
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };
    while (picks.length < K) {
      round += 1;
      const J = rho.map((_, j) => j).filter((j) => left.some((c) => c.dir === j));
      if (!J.length) break;
      const low = Math.min(...J.map((j) => rho[j]));
      const tied = J.filter((j) => rho[j] === low);
      const Jmin = shuffle(tied.slice()).slice(0, K - picks.length);
      for (const j of Jmin) {
        const C = shuffle(left.filter((c) => c.dir === j));
        let chosen = C[0];
        if (rho[j] === 0) for (const c of C) if (c.dperp < chosen.dperp) chosen = c;
        picks.push({ id: chosen.id, dir: j, round, count: rho[j], low, tied, Jmin: Jmin.slice(), closest: rho[j] === 0 });
        left.splice(left.indexOf(chosen), 1);
        rho[j] += 1;
        if (picks.length >= K) break;
      }
    }
    return { picks, rho };
  }

  /** One NSGA-III generation of the loop figure's problem, recorded for the NSGA-III lesson, with VAMOS' defaults:
      random parents, SBX (probability 0.9, η = 30) and polynomial mutation (probability 1/2, η = 20), the merge,
      the fronts, the whole fronts and the last front (null when the whole fronts fill the places exactly), the
      normalization (the ideal point, the extreme points of the first front by the ASF, the intercepts of the line
      through them or, when that hyperplane is rejected, the worst point of the first front), the association of
      every merged solution with the H = popSize Das-Dennis directions (p = popSize − 1), the niche counts of the
      whole fronts and the niching of the last front. It is the first generation of a run, so the ideal and worst
      points are those of the merged population and no earlier extreme points compete. */
  function nsga3Generation(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    const W = dasDennis(2, N - 1);
    const pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push({ id: i + 1, x, f: gaEvaluate(x) });
    }
    const draws = Array.from({ length: 2 * Math.ceil(N / 2) }, () => Math.floor(rand() * N));
    const children = [];
    for (let k = 0; k < draws.length; k += 2) {
      const p1 = pop[draws[k]];
      const p2 = pop[draws[k + 1]];
      const xs = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 30, 0.5, rand) : [p1.x.slice(), p2.x.slice()];
      for (const x of xs) {
        polynomialMutation(x, 0, 1, 0.5, 20, rand);
        if (children.length < N) children.push({ id: N + children.length + 1, x, f: gaEvaluate(x), parents: [p1.id, p2.id] });
      }
    }
    const merged = pop.concat(children);
    const F = merged.map((p) => p.f);
    const fronts = nonDominatedSort(F).fronts;
    const rank = new Array(F.length).fill(0);
    fronts.forEach((front, k) => front.forEach((i) => (rank[i] = k)));
    const whole = [];
    let used = 0;
    let last = null;
    for (const front of fronts) {
      if (used === N) break;
      if (used + front.length > N) {
        last = front;
        break;
      }
      whole.push(front);
      used += front.length;
    }
    const ideal = [0, 1].map((m) => Math.min(...F.map((f) => f[m])));
    const worst = [0, 1].map((m) => Math.max(...F.map((f) => f[m])));
    const first = fronts[0];
    const extremes = [0, 1].map((axis) => {
      let best = first[0];
      let bestValue = Infinity;
      for (const i of first) {
        const value = Math.max(...F[i].map((v, m) => (v - ideal[m] < 1e-3 ? 0 : v - ideal[m]) * (m === axis ? 1 : 1e6)));
        if (value < bestValue) {
          bestValue = value;
          best = i;
        }
      }
      return best;
    });
    const shifted = extremes.map((i) => F[i].map((v, m) => v - ideal[m]));
    const plane = solveLinear(shifted, [1, 1]);
    let intercepts = plane ? plane.map((v) => 1 / v) : null;
    const hyperplane =
      !!plane &&
      shifted.every((row) => Math.abs(row[0] * plane[0] + row[1] * plane[1] - 1) <= 1e-8 + 1e-5) &&
      intercepts.every((a) => a > 1e-6) &&
      intercepts.every((a, m) => !(ideal[m] + a > worst[m]));
    if (!hyperplane) intercepts = null;
    const frontWorst = [0, 1].map((m) => Math.max(...first.map((i) => F[i][m])));
    const estimate = hyperplane ? intercepts.map((a, m) => ideal[m] + a) : frontWorst;
    const fromPopulation = estimate.map((v, m) => v - ideal[m] <= 1e-6);
    const nadir = estimate.map((v, m) => (fromPopulation[m] ? worst[m] : v));
    const scale = nadir.map((v, m) => (v - ideal[m] === 0 ? 1e-12 : v - ideal[m]));
    const normalized = F.map((f) => f.map((v, m) => (v - ideal[m]) / scale[m]));
    const association = normalized.map((f) => nsga3Associate(f, W));
    const rho = W.map(() => 0);
    for (const i of whole.flat()) rho[association[i].dir] += 1;
    const places = N - used;
    const niching = last && places > 0 ? nsga3Niching(rho, last.map((i) => ({ id: i, dir: association[i].dir, dperp: association[i].dperp })), places, rand) : null;
    return {
      popSize: N,
      W,
      pop,
      draws,
      children,
      merged,
      fronts,
      rank,
      whole,
      last: places > 0 ? last : null,
      places,
      ideal,
      worst,
      extremes,
      hyperplane,
      intercepts,
      frontWorst,
      fromPopulation,
      nadir,
      normalized,
      association,
      rho,
      niching,
      keep: whole.flat().concat(niching ? niching.picks.map((p) => p.id) : []),
    };
  }

  /** One SMPSO velocity and position update, as VAMOS computes it (bounds [lower, upper] per variable). */
  function smpsoStep(o) {
    const phi = o.c1 + o.c2;
    const chi = phi <= 4 ? 1 : 2 / (2 - phi - Math.sqrt(phi * phi - 4 * phi));
    const inertia = o.v.map((v) => o.w * v);
    const cognitive = o.x.map((x, j) => o.c1 * o.r1 * (o.p[j] - x));
    const social = o.x.map((x, j) => o.c2 * o.r2 * (o.g[j] - x));
    const raw = o.x.map((_, j) => inertia[j] + cognitive[j] + social[j]);
    const constricted = raw.map((v) => chi * v);
    const delta = o.x.map((_, j) => o.vmax * (o.upper[j] - o.lower[j]));
    const clamped = constricted.map((v, j) => (o.clamp === false ? v : Math.min(delta[j], Math.max(-delta[j], v))));
    const tentative = o.x.map((x, j) => x + clamped[j]);
    const hit = [];
    const position = tentative.slice();
    const velocity = clamped.slice();
    tentative.forEach((x, j) => {
      if (x < o.lower[j]) {
        position[j] = o.lower[j];
        velocity[j] = -velocity[j];
        hit.push([j, "lower"]);
      } else if (x > o.upper[j]) {
        position[j] = o.upper[j];
        velocity[j] = -velocity[j];
        hit.push([j, "upper"]);
      }
    });
    return { phi, chi, inertia, cognitive, social, raw, constricted, clamped, tentative, position, velocity, delta, hit };
  }

  /** One SMPSO iteration of the loop figure's problem, recorded for the SMPSO lesson, with VAMOS' defaults: w = 0.1,
      C₁ and C₂ uniform in [1.5, 2.5], vmax_fraction 0.5, polynomial-mutation turbulence (probability 1/n = 1/2,
      η = 20) on every sixth particle, and a leaders archive as large as the swarm, which receives the particles one at
      a time. The swarm starts at random positions with zero velocity; its first iteration (`previous`) is not shown, so
      that the recorded one has velocities and personal bests: the swarm and the leaders with their crowding
      distances, the leader tournaments, the velocity and position updates (smpsoStep), the turbulence, the
      evaluation, the personal bests and the insertion of the new positions into the archive. */
  function smpsoGeneration(seed, swarmSize) {
    const rand = mulberry32(seed);
    const N = swarmSize;
    const lower = [0, 0];
    const upper = [1, 1];
    const prob = 1 / 2;
    let names = 0;
    const leaderSnapshot = (archive) => {
      const d = crowdingDistance(
        archive.map((a) => a.f),
        archive.map((_, i) => i)
      );
      return archive.map((a, i) => ({ name: a.name, from: a.from, x: a.x.slice(), f: a.f.slice(), crowding: d[i] }));
    };
    // jMetal's leaders.add, as VAMOS inserts one particle: rejected when a leader dominates or equals it; otherwise
    // the leaders it dominates leave, it joins at the end, and on overflow the most crowded leader is removed.
    const insert = (archive, q) => {
      const by = archive.find((a) => a.f.every((v, j) => v <= q.f[j]));
      if (by) return { archive, event: { id: q.id, accepted: false, by: by.name, removed: [], pruned: null, crowding: null } };
      const removed = archive.filter((a) => a.f.every((v, j) => v >= q.f[j]));
      const leader = { name: String.fromCharCode(65 + names++), from: q.id, x: q.x.slice(), f: q.f.slice() };
      let next = archive.filter((a) => !removed.includes(a)).concat([leader]);
      let pruned = null;
      let crowding = null;
      if (next.length > N) {
        const d = crowdingDistance(
          next.map((a) => a.f),
          next.map((_, i) => i)
        );
        let k = 0;
        d.forEach((v, i) => {
          if (v < d[k]) k = i;
        });
        crowding = next.map((a, i) => ({ name: a.name, crowding: d[i] }));
        pruned = next[k].name;
        next = next.filter((_, i) => i !== k);
      }
      return { archive: next, event: { id: q.id, accepted: true, leader: leader.name, removed: removed.map((a) => a.name), pruned, crowding } };
    };
    const swarm = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      const f = gaEvaluate(x);
      swarm.push({ id: i + 1, x, v: [0, 0], f, p: x.slice(), pf: f.slice() });
    }
    let archive = [];
    const initial = swarm.map((q) => {
      const r = insert(archive, q);
      archive = r.archive;
      return r.event;
    });
    const snapshot = () => swarm.map((q) => ({ id: q.id, x: q.x.slice(), v: q.v.slice(), f: q.f.slice(), p: q.p.slice(), pf: q.pf.slice() }));
    const start = { swarm: snapshot(), leaders: leaderSnapshot(archive), inserts: initial };
    function iterate() {
      const before = snapshot();
      const leaders = leaderSnapshot(archive);
      const tournaments = swarm.map((q) => {
        if (leaders.length <= 2) return { id: q.id, a: 0, b: null, winner: 0, reason: "first" };
        const a = Math.floor(rand() * leaders.length);
        let b = Math.floor(rand() * (leaders.length - 1));
        if (b >= a) b += 1;
        let winner = a;
        let reason = "crowding";
        if (leaders[b].crowding > leaders[a].crowding) winner = b;
        else if (leaders[b].crowding === leaders[a].crowding) {
          reason = "coin";
          if (rand() >= 0.5) winner = b;
        }
        return { id: q.id, a, b, winner, reason };
      });
      const moves = swarm.map((q, i) => {
        const draws = { c1: 1.5 + rand(), c2: 1.5 + rand(), r1: rand(), r2: rand() };
        const g = leaders[tournaments[i].winner].x;
        return Object.assign({ id: q.id, g: g.slice() }, draws, smpsoStep(Object.assign({ x: q.x, v: q.v, p: q.p, g, w: 0.1, vmax: 0.5, lower, upper }, draws)));
      });
      const turbulence = [];
      moves.forEach((m, i) => {
        if (i % 6 !== 0) return;
        const draws = [];
        const after = polynomialMutation(m.position.slice(), 0, 1, prob, 20, () => {
          const r = rand();
          draws.push(r);
          return r;
        });
        const mask = [];
        const u = [];
        let k = 0;
        for (let j = 0; j < after.length; j++) {
          mask.push(draws[k++]);
          u.push(mask[j] <= prob ? draws[k++] : null);
        }
        turbulence.push({ id: m.id, index: i, before: m.position.slice(), after, mask, u, mutated: u.map((v, j) => (v === null ? -1 : j)).filter((j) => j >= 0) });
      });
      const pbest = [];
      swarm.forEach((q, i) => {
        const t = turbulence.find((e) => e.index === i);
        q.x = (t ? t.after : moves[i].position).slice();
        q.v = moves[i].velocity.slice();
        q.f = gaEvaluate(q.x);
        const kept = dominates(q.pf, q.f);
        if (!kept) {
          q.p = q.x.slice();
          q.pf = q.f.slice();
        }
        pbest.push({ id: q.id, replaced: !kept, p: q.p.slice(), pf: q.pf.slice() });
      });
      const after = snapshot();
      const inserts = swarm.map((q) => {
        const r = insert(archive, q);
        archive = r.archive;
        return r.event;
      });
      return { before, leaders, tournaments, moves, turbulence, after, pbest, inserts, leadersAfter: leaderSnapshot(archive) };
    }
    const previous = iterate();
    return Object.assign({ swarmSize: N, start, previous }, iterate());
  }

  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-14) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const k = M[r][c] / M[c][c];
        for (let q = c; q <= n; q++) M[r][q] -= k * M[c][q];
      }
    }
    return M.map((row, i) => row[n] / row[i]);
  }

  /** AGE-MOEA geometry estimate and greedy survival score of one front (VAMOS' geometry.py). */
  function agemoeaSurvival(F, forceP) {
    const m = F[0].length;
    const z = F[0].map((_, i) => Math.min(...F.map((f) => f[i])));
    const T = F.map((f) => f.map((v, i) => Math.round((v - z[i]) * 1e12) / 1e12));
    if (F.length < m) {
      // Fewer points than objectives: no hyperplane, p = 1, every point scores 0, and the spread above the ideal
      // point normalizes the later fronts.
      const spread = z.map((_, i) => Math.max(...T.map((t) => t[i])) || 1);
      const zeros = F.map(() => 0);
      return { p: 1, estimated: 1, central: -1, corners: [], normalized: T.map((t) => t.map((v, i) => v / spread[i])), intercepts: spread, ideal: z, proximity: F.map(() => NaN), score: zeros, order: [] };
    }
    const dot = (a, b) => a.reduce((s2, v, i) => s2 + v * b[i], 0);
    const corners = [];
    for (let i = 0; i < m; i++) {
      const d = F[0].map((_, k) => (k === i ? 1 : 0) + 1e-6);
      const dd = dot(d, d);
      let best = -1;
      let bestDist = Infinity;
      T.forEach((t, s2) => {
        if (corners.includes(s2)) return;
        const dist = dot(t, t) - (dot(t, d) * dot(t, d)) / dd;
        if (dist < bestDist) {
          bestDist = dist;
          best = s2;
        }
      });
      corners.push(best);
    }
    let a = null;
    if (new Set(corners).size === m) {
      const b = solveLinear(corners.map((c) => T[c]), corners.map(() => 1));
      if (b && b.every((v) => Number.isFinite(v) && v > 0)) a = b.map((v) => 1 / v);
    }
    if (!a) a = F[0].map((_, i) => Math.max(...T.map((t) => t[i])) || 1);
    const Nf = T.map((t) => t.map((v, i) => v / a[i]));
    let central = -1;
    let centralDist = Infinity;
    Nf.forEach((t, s2) => {
      if (corners.includes(s2)) return;
      const dist = dot(t, t) - (t.reduce((x, v) => x + v, 0) ** 2) / m;
      if (dist < centralDist) {
        centralDist = dist;
        central = s2;
      }
    });
    let p = 1;
    if (central >= 0) {
      const mean = Nf[central].reduce((x, v) => x + v, 0) / m;
      if (mean > 0) {
        const est = Math.log(m) / Math.log(1 / mean);
        p = Number.isNaN(est) || est <= 0.1 ? 1 : Math.min(est, 20);
      }
    }
    const estimated = p;
    if (forceP) p = forceP;
    const lp = (v) => Math.pow(v.reduce((x, u) => x + Math.pow(Math.abs(u), p), 0), 1 / p);
    const proximity = Nf.map((t) => {
      const v = lp(t);
      return v < 1e-8 ? 1 : v;
    });
    const dist = (s2, q) => Math.max(1e-8, lp(Nf[s2].map((v, i) => v - Nf[q][i]))) / proximity[s2];
    const score = new Array(F.length).fill(0);
    const selected = [];
    const order = [];
    corners
      .slice()
      .sort((x, y) => x - y)
      .forEach((c) => {
        score[c] = Infinity;
        selected.push(c);
        order.push(c);
      });
    while (selected.length < F.length) {
      let best = -1;
      let bestScore = -Infinity;
      F.forEach((_, s2) => {
        if (selected.includes(s2)) return;
        const ds = selected.map((q) => dist(s2, q)).sort((x, y) => x - y);
        const sc = ds.length === 1 ? ds[0] : ds[0] + ds[1];
        if (sc > bestScore) {
          bestScore = sc;
          best = s2;
        }
      });
      score[best] = bestScore;
      selected.push(best);
      order.push(best);
    }
    return { p, estimated, central, corners, normalized: Nf, intercepts: a, ideal: z, proximity, score, order };
  }

  /** The N survivors of a last front by survival score (descending; later index first on ties). */
  function agemoeaCut(score, N) {
    return score
      .map((v, i) => i)
      .sort((x, y) => (score[y] === score[x] ? y - x : score[y] > score[x] ? 1 : -1))
      .slice(0, N);
  }

  /** One AGE-MOEA generation of the loop figure's problem, recorded for the AGE-MOEA lesson, as VAMOS runs the
      default variant: binary tournaments on front and survival score, SBX (probability 0.9, η = 15) and polynomial
      mutation (probability 1/n = 1/2, η = 20), the merge of parents and children, non-dominated sorting, the geometry
      of the first front (agemoeaSurvival), the inverse proximities of the later fronts up to the last one, and the cut
      of the last front by score (agemoeaCut). `last` is the front that fills the population; it is cut only when it
      has more members than `places`. Indices of the geometry refer to `merged`. */
  function agemoeaGeneration(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    let nextId = 1;
    // VAMOS' age_survival: the survivors among F, and the score of every member of the fronts up to the last one.
    const survive = (F, n) => {
      const fronts = nonDominatedSort(F).fronts;
      const rank = new Array(F.length).fill(-1);
      fronts.forEach((front, k) => front.forEach((i) => (rank[i] = k)));
      let used = 0;
      let last = fronts.length - 1;
      for (let k = 0; k < fronts.length; k++) {
        if (used + fronts[k].length >= n) {
          last = k;
          break;
        }
        used += fronts[k].length;
      }
      const first = fronts[0];
      const res = agemoeaSurvival(first.map((i) => F[i]));
      const score = new Array(F.length).fill(0);
      const proximity = new Array(F.length).fill(NaN);
      first.forEach((i, k) => {
        score[i] = res.score[k];
        proximity[i] = res.proximity[k];
      });
      // Later fronts: the inverse Lp distance to the ideal point in the first front's normalized coordinates.
      for (let k = 1; k <= last; k++) {
        for (const i of fronts[k]) {
          const t = F[i].map((v, j) => (v - res.ideal[j]) / res.intercepts[j]);
          proximity[i] = Math.pow(t.reduce((sum, v) => sum + Math.pow(Math.abs(v), res.p), 0), 1 / res.p);
          score[i] = 1 / Math.max(proximity[i], 1e-12);
        }
      }
      const places = n - used;
      const cut = agemoeaCut(
        fronts[last].map((i) => score[i]),
        places
      ).map((k) => fronts[last][k]);
      const keep = fronts
        .slice(0, last)
        .flat()
        .concat(cut)
        .sort((a, b) => a - b);
      const local = (list) => list.map((k) => first[k]);
      const geometry = Object.assign({}, res, { first, corners: local(res.corners), central: res.central >= 0 ? first[res.central] : -1, order: local(res.order) });
      return { rank, fronts, last, places, geometry, score, proximity, keep };
    };
    const pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push({ id: nextId++, x, f: gaEvaluate(x) });
    }
    const start = survive(
      pop.map((p) => p.f),
      N
    );
    const tournaments = [];
    const tournament = () => {
      const a = Math.floor(rand() * N);
      let b = Math.floor(rand() * (N - 1));
      if (b >= a) b += 1;
      let w = a;
      let reason = "coin";
      if (start.rank[a] !== start.rank[b]) {
        w = start.rank[a] < start.rank[b] ? a : b;
        reason = "front";
      } else if (start.score[a] !== start.score[b]) {
        w = start.score[a] > start.score[b] ? a : b;
        reason = "score";
      } else if (rand() >= 0.5) w = b;
      tournaments.push({ a, b, winner: w, reason });
      return pop[w];
    };
    const pairs = [];
    for (let k = 0; k < Math.ceil(N / 2); k++) pairs.push([tournament(), tournament()]);
    const children = [];
    for (const [p1, p2] of pairs) {
      const xs = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 15, 0.5, rand) : [p1.x.slice(), p2.x.slice()];
      for (const x of xs) {
        polynomialMutation(x, 0, 1, 0.5, 20, rand);
        if (children.length < N) children.push({ id: nextId++, x, f: gaEvaluate(x), parents: [p1.id, p2.id] });
      }
    }
    const merged = pop.concat(children);
    const end = survive(
      merged.map((p) => p.f),
      N
    );
    return {
      popSize: N,
      pop,
      rank: start.rank,
      score: start.score,
      proximity: start.proximity,
      fronts0: start.fronts,
      geometry0: start.geometry,
      tournaments,
      children,
      merged,
      mergedRank: end.rank,
      fronts: end.fronts,
      last: end.last,
      places: end.places,
      geometry: end.geometry,
      mergedScore: end.score,
      mergedProximity: end.proximity,
      keep: end.keep,
    };
  }

  /* ---- Constrained algorithms: shared candidates and the selection rules of CCMO, C-TAEA, ToP, CMOEA-MS ---- */

  const CS_CANDIDATES = [
    ["P1", 0.15, 0.88],
    ["P2", 0.2, 1.25],
    ["P3", 0.32, 0.775],
    ["P4", 0.45, 0.6],
    ["P5", 0.565, 0.6],
    ["P6", 0.7, 0.8],
    ["P7", 0.81, 0.42],
    ["P8", 0.95, 0.2],
    ["O1", 0.08, 0.95],
    ["O2", 0.3, 1.1],
    ["O3", 0.25, 0.76],
    ["O4", 0.52, 0.5],
    ["O5", 0.6, 0.42],
    ["O6", 0.645, 0.54],
    ["O7", 0.88, 0.15],
    ["O8", 0.97, 0.3],
  ];

  /** Strength-based fitness R + density with a custom dominance relation. */
  function strengthFitness(n, dom, density) {
    const S = [];
    for (let i = 0; i < n; i++) {
      let s2 = 0;
      for (let j = 0; j < n; j++) if (i !== j && dom(i, j)) s2 += 1;
      S.push(s2);
    }
    const R = [];
    const by = [];
    for (let j = 0; j < n; j++) {
      let r = 0;
      const d = [];
      for (let i = 0; i < n; i++)
        if (i !== j && dom(i, j)) {
          r += S[i];
          d.push(i);
        }
      R.push(r);
      by.push(d);
    }
    const D = Array.from({ length: n }, (_, i) => density(i));
    return { S, R, D, by, fitness: R.map((r, i) => r + D[i]) };
  }

  function kthOther(row, i, k) {
    return row.filter((_, j) => j !== i).sort((a, b) => a - b)[k - 1];
  }

  /** Keep fitness < 1; fill with the lowest fitness (stable) or truncate by lexicographic distance lists. */
  function strengthSelect(fitness, n, dist) {
    const idx = fitness.map((_, i) => i);
    let keep = idx.filter((i) => fitness[i] < 1);
    if (keep.length < n) {
      keep = idx.slice().sort((a, b) => (Number.isNaN(fitness[a]) ? 1 : 0) - (Number.isNaN(fitness[b]) ? 1 : 0) || fitness[a] - fitness[b] || a - b).slice(0, n);
    } else {
      while (keep.length > n) {
        const lists = keep.map((i) =>
          keep
            .filter((j) => j !== i)
            .map((j) => dist(i, j))
            .sort((a, b) => a - b)
        );
        let worst = 0;
        for (let a = 1; a < keep.length; a++) if (lexLess(lists[a], lists[worst])) worst = a;
        keep.splice(worst, 1);
      }
    }
    return keep;
  }

  /** CCMO: SPEA2 fitness with violation-first dominance (population 1) or zero violations (population 2). */
  function ccmoSelect(F, cv, n) {
    const m = F.length;
    const dist = (i, j) => euclidean(F[i], F[j]);
    const k = Math.floor(Math.sqrt(m));
    const dk = F.map((_, i) => kthOther(F.map((_, j) => dist(i, j)), i, k));
    const dom = (i, j) => cv[i] < cv[j] || (cv[i] === cv[j] && dominates(F[i], F[j]));
    const fit = strengthFitness(m, dom, (i) => 1 / (dk[i] + 2));
    return { ...fit, dk, k, kept: strengthSelect(fit.fitness, n, dist) };
  }

  /** One CCMO generation of the loop figure's problem with CTP2's constraint, ctpViolation(gaEvaluate(x)), recorded
      for the CCMO lesson. Population 1 (ids 1 to N) and population 2 (ids N + 1 to 2N) are drawn uniformly,
      population 1 first; each one's fitness is ccmoSelect's on its own members, with the violations in
      population 1 and zeros in population 2. Each population in turn draws N binary tournaments on its fitness
      (two members at random, possibly the same one; the lower fitness wins and a tie goes to the first), mates the
      i-th winner of the first half with the i-th of the second half and keeps the first child of SBX
      (probability 1, η = 20), mutated by polynomial mutation (probability 1/n = 1/2, η = 20): VAMOS' default GA
      variation for CCMO. The N offspring, population 1's first, are evaluated once, and each population selects
      N survivors from itself and all the offspring with ccmoSelect; the survivors are stored by increasing fitness. */
  function ccmoGeneration(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    const half = Math.floor(N / 2);
    let nextId = 1;
    const born = (x, extra) => {
      const f = gaEvaluate(x);
      return Object.assign({ id: nextId++, x, f, cv: ctpViolation(f) }, extra);
    };
    const pops = [0, 1].map((k) => {
      const pop = [];
      for (let i = 0; i < N; i++) pop.push(born([rand(), rand()], { population: k + 1 }));
      return pop;
    });
    const violations = (rows, k) => rows.map((p) => (k === 0 ? p.cv : 0));
    const own = pops.map((pop, k) =>
      ccmoSelect(
        pop.map((p) => p.f),
        violations(pop, k),
        N
      )
    );
    const fitness = own.map((res) => res.fitness);
    const mating = [];
    const offspring = [];
    pops.forEach((pop, k) => {
      const fit = fitness[k];
      const tournaments = [];
      for (let t = 0; t < N; t++) {
        const a = Math.floor(rand() * N);
        const b = Math.floor(rand() * N);
        const w = fit[a] < fit[b] || (fit[a] === fit[b] && a <= b) ? a : b;
        tournaments.push({ a: pop[a].id, b: pop[b].id, winner: pop[w].id, draws: [a, b], tie: fit[a] === fit[b] });
      }
      const byId = new Map(pop.map((p) => [p.id, p]));
      const pairs = [];
      for (let i = 0; i < half; i++) pairs.push([tournaments[i].winner, tournaments[half + i].winner]);
      for (const [a, b] of pairs) {
        const child = sbxPair(byId.get(a).x, byId.get(b).x, 0, 1, 20, 0.5, rand)[0];
        polynomialMutation(child, 0, 1, 0.5, 20, rand);
        offspring.push({ x: child, population: k + 1, parents: [a, b] });
      }
      mating.push({ tournaments, pairs });
    });
    const children = offspring.map((c) => born(c.x, { population: c.population, parents: c.parents, child: true }));
    const selections = pops.map((pop, k) => {
      const candidates = pop.concat(children);
      const res = ccmoSelect(
        candidates.map((p) => p.f),
        violations(candidates, k),
        N
      );
      const next = res.kept.slice().sort((a, b) => res.fitness[a] - res.fitness[b] || a - b);
      const below = res.fitness.filter((v) => v < 1).length;
      return {
        candidates,
        S: res.S,
        R: res.R,
        D: res.D,
        dk: res.dk,
        k: res.k,
        by: res.by.map((rows) => rows.map((i) => candidates[i].id)),
        fitness: res.fitness,
        kept: next.map((i) => candidates[i].id),
        rule: below < N ? "fill" : below > N ? "truncate" : "exact",
        below,
      };
    });
    return { popSize: N, pops, fitness, R: own.map((res) => res.R), D: own.map((res) => res.D), mating, children, selections };
  }

  const CTAEA_W = (count) =>
    Array.from({ length: count }, (_, k) => {
      const a = k / (count - 1);
      return [Math.max(a, 1e-6), Math.max((count - 1 - k) / (count - 1), 1e-6)];
    });

  function ctaeaAssociate(f, W) {
    const nf = Math.hypot(...f) || 1;
    let best = 0;
    let bestCos = -Infinity;
    W.forEach((w, k) => {
      const c = (f[0] * w[0] + f[1] * w[1]) / (nf * Math.hypot(...w));
      if (c > bestCos) {
        bestCos = c;
        best = k;
      }
    });
    return best;
  }

  function ctaeaTchebycheff(rows, W) {
    const z = [0, 1].map((j) => Math.min(...rows.map((f) => f[j])));
    return rows.map((f) => {
      const w = W[ctaeaAssociate(f, W)];
      return Math.max(...f.map((v, j) => Math.abs(v - z[j]) / w[j]));
    });
  }

  /** C-TAEA's convergence archive update (ctaea/archives.py) of the candidates F with violations cv: the rows of the
      new CA in VAMOS' order, the feasible rows, and a record of the case that applied (the fronts and each removal
      with more feasible candidates than weight vectors, the ranking of the infeasible ones with fewer). */
  function ctaeaUpdateCA(F, cv, W) {
    const n = W.length;
    const idx = F.map((_, i) => i);
    const feasible = idx.filter((i) => cv[i] <= 0);
    const ca = { removals: [], fronts: null, normalized: null, infeasible: null };
    let CA;
    if (feasible.length === n) CA = feasible.slice();
    else if (feasible.length > n) {
      const sorted = nonDominatedSort(feasible.map((i) => F[i]));
      ca.fronts = sorted.fronts.map((f) => f.map((r) => feasible[r]));
      CA = [];
      for (const front of ca.fronts) {
        if (CA.length >= n) break;
        CA.push(...front);
      }
      while (CA.length > n) {
        const rows = CA.map((i) => F[i]);
        const lo = [0, 1].map((j) => Math.min(...rows.map((f) => f[j])));
        const hi = [0, 1].map((j) => Math.max(...rows.map((f) => f[j])));
        const norm = rows.map((f) => f.map((v, j) => (hi[j] > lo[j] ? (v - lo[j]) / (hi[j] - lo[j]) : 0)));
        const region = norm.map((f) => ctaeaAssociate(f, W));
        const counts = W.map((_, k) => region.filter((r) => r === k).length);
        const crowded = counts.indexOf(Math.max(...counts));
        const members = CA.filter((_, a) => region[a] === crowded);
        let dmin = Infinity;
        members.forEach((i) => members.forEach((j) => i !== j && euclidean(F[i], F[j]) > 0 && (dmin = Math.min(dmin, euclidean(F[i], F[j])))));
        // Equal points are at no distance; when every member is the same point, every member is a candidate.
        const closest = dmin === Infinity ? members.slice() : members.filter((i) => members.some((j) => j !== i && Math.abs(euclidean(F[i], F[j]) - dmin) < 1e-15));
        const tch = ctaeaTchebycheff(
          closest.map((i) => F[i]),
          W
        );
        let worst = 0;
        tch.forEach((v, a) => {
          if (v > tch[worst]) worst = a;
        });
        ca.normalized = ca.normalized || CA.map((i, a) => ({ i, f: norm[a], region: region[a] }));
        ca.removals.push({ counts, crowded, closest, tch, removed: closest[worst], ideal: [0, 1].map((j) => Math.min(...closest.map((i) => F[i][j]))), rows: CA.slice(), region, lo, hi });
        CA.splice(CA.indexOf(closest[worst]), 1);
      }
    } else {
      const infeasible = idx.filter((i) => cv[i] > 0);
      const tch = ctaeaTchebycheff(
        infeasible.map((i) => F[i]),
        W
      );
      const fronts = nonDominatedSort(infeasible.map((i, a) => [cv[i], tch[a]])).fronts.map((f) => f.map((r) => infeasible[r]));
      CA = feasible.slice();
      let removed = [];
      for (const front of fronts) {
        if (CA.length >= n) break;
        if (CA.length + front.length <= n) CA.push(...front);
        else {
          // The last front keeps its order and loses its largest violations, the first in a tie.
          removed = front
            .slice()
            .sort((a, b) => cv[b] - cv[a] || a - b)
            .slice(0, CA.length + front.length - n);
          CA.push(...front.filter((i) => !removed.includes(i)));
        }
      }
      ca.infeasible = { rows: infeasible, tch, fronts, removed };
    }
    return { CA, ca, feasible };
  }

  /** C-TAEA's diversity archive update (ctaea/archives.py) of the candidates F, the constraints ignored, with the
      objective vectors caF of the new CA: the rows of the new DA in selection order, the CA's count in each
      subregion, the subregion of each candidate and a trace of the picks. */
  function ctaeaUpdateDA(F, caF, W) {
    const n = W.length;
    const idx = F.map((_, i) => i);
    const regions = F.map((f) => ctaeaAssociate(f, W));
    const caRegions = caF.map((f) => ctaeaAssociate(f, W));
    const caCounts = W.map((_, k) => caRegions.filter((r) => r === k).length);
    const left = W.map((_, k) => regions.filter((r) => r === k).length);
    const available = new Set(idx);
    const DA = [];
    const trace = [];
    let iteration = 1;
    while (DA.length < n && available.size) {
      for (let r = 0; r < W.length; r++) {
        if (DA.length >= n) break;
        for (let q = 0; q < iteration - caCounts[r]; q++) {
          if (!left[r]) break;
          const cands = idx.filter((i) => available.has(i) && regions[i] === r);
          const nd = cands.filter((i) => !cands.some((j) => j !== i && dominates(F[j], F[i])));
          const tch = ctaeaTchebycheff(
            nd.map((i) => F[i]),
            W
          );
          let best = 0;
          tch.forEach((v, a) => {
            if (v < tch[best]) best = a;
          });
          available.delete(nd[best]);
          left[r] -= 1;
          if (DA.length < n) {
            DA.push(nd[best]);
            trace.push({ iteration, region: r, ca: caCounts[r], cands, nd, tch, chosen: nd[best] });
          }
        }
      }
      iteration += 1;
    }
    return { DA, caCounts, regions, trace };
  }

  /** C-TAEA's convergence archive and diversity archive updates of the same candidates, with n weight vectors. */
  function ctaeaSelect(F, cv, n) {
    const W = CTAEA_W(n);
    const { CA, ca, feasible } = ctaeaUpdateCA(F, cv, W);
    const { DA, caCounts, regions, trace } = ctaeaUpdateDA(
      F,
      CA.map((i) => F[i]),
      W
    );
    return { W, CA, DA, ca, caCounts, regions, trace, feasible };
  }

  /** C-TAEA on the loop figure's problem with CTP2's constraint, ctpViolation(gaEvaluate(x)), with popSize weight
      vectors, recorded generation by generation for the C-TAEA lesson. As in VAMOS, the N random solutions of the
      initial population (ids 1 to N) are the candidates of the first update of both archives, and each generation
      creates one offspring per weight vector. A generation records the CA and the DA; the shares of non-dominated
      solutions of CA ∪ DA in each half and of the CA alone, which choose the parents' archives; the binary
      tournaments (three members drawn with replacement: if only one of the first two is feasible it wins, if both
      are the first wins unless the second dominates it, if neither is the third wins); the first child of SBX
      (probability 1, η = 20) on the second parent and the first, mutated by polynomial mutation
      (probability 1/n = 1/2, η = 20); and the CA update of CA ∪ offspring, then the DA update of DA ∪ offspring with
      the new CA. Solutions are referred to by id; history lists the archives after each update. */
  function ctaeaGeneration(seed, popSize, generations) {
    const rand = mulberry32(seed);
    const W = CTAEA_W(popSize);
    const N = W.length;
    const solutions = [];
    const born = (x, extra) => {
      const f = gaEvaluate(x);
      const sol = Object.assign({ id: solutions.length + 1, x, f, cv: ctpViolation(f), region: ctaeaAssociate(f, W) }, extra);
      solutions.push(sol);
      return sol;
    };
    const objectives = (rows) => rows.map((sol) => sol.f);
    const idsOf = (rows, cands) => rows.map((i) => cands[i].id);
    const nonDominated = (rows) => rows.map((f, i) => !rows.some((g, j) => j !== i && dominates(g, f)));
    const tournament = (archive) => {
      const draws = [0, 1, 2].map(() => Math.floor(rand() * archive.length));
      const [A, B, C] = draws.map((d) => archive[d]);
      let winner = A;
      let reason = "first";
      if (A.cv === 0 && B.cv === 0) {
        if (dominates(B.f, A.f)) {
          winner = B;
          reason = "dominates";
        }
      } else if (A.cv === 0) reason = "feasible";
      else if (B.cv === 0) {
        winner = B;
        reason = "feasible";
      } else {
        winner = C;
        reason = "third";
      }
      return { a: A.id, b: B.id, c: C.id, winner: winner.id, reason, draws };
    };
    const update = (CA, DA, offspring) => {
      const caCands = CA.concat(offspring);
      const ca = ctaeaUpdateCA(
        objectives(caCands),
        caCands.map((sol) => sol.cv),
        W
      );
      const newCA = ca.CA.map((i) => caCands[i]);
      const daCands = DA.concat(offspring);
      const da = ctaeaUpdateDA(objectives(daCands), objectives(newCA), W);
      const rec = ca.ca;
      const caUpdate = {
        candidates: caCands.map((sol) => sol.id),
        feasible: idsOf(ca.feasible, caCands),
        fronts: rec.fronts && rec.fronts.map((front) => idsOf(front, caCands)),
        removals: rec.removals.map((r) => ({ rows: idsOf(r.rows, caCands), region: r.region, lo: r.lo, hi: r.hi, counts: r.counts, crowded: r.crowded, closest: idsOf(r.closest, caCands), tch: r.tch, ideal: r.ideal, removed: caCands[r.removed].id })),
        infeasible: rec.infeasible && { rows: idsOf(rec.infeasible.rows, caCands), tch: rec.infeasible.tch, fronts: rec.infeasible.fronts.map((front) => idsOf(front, caCands)), removed: idsOf(rec.infeasible.removed, caCands) },
        kept: newCA.map((sol) => sol.id),
      };
      const daUpdate = {
        candidates: daCands.map((sol) => sol.id),
        caCounts: da.caCounts,
        trace: da.trace.map((t) => ({ iteration: t.iteration, region: t.region, ca: t.ca, cands: idsOf(t.cands, daCands), nd: idsOf(t.nd, daCands), tch: t.tch, chosen: daCands[t.chosen].id })),
        kept: da.DA.map((i) => daCands[i].id),
      };
      return { CA: newCA, DA: da.DA.map((i) => daCands[i]), caUpdate, daUpdate };
    };
    const initial = [];
    for (let i = 0; i < N; i++) initial.push(born([rand(), rand()], { generation: 0 }));
    let state = update([], [], initial);
    const history = [{ offspring: initial.map((sol) => sol.id), CA: state.caUpdate.kept, DA: state.daUpdate.kept }];
    const records = [];
    for (let gen = 1; gen <= generations; gen++) {
      const CA = state.CA;
      const DA = state.DA;
      const union = CA.concat(DA);
      const nd = nonDominated(objectives(union));
      const half = Math.floor((union.length + 1) / 2);
      const ndCA = nonDominated(objectives(CA));
      const shares = {
        nd,
        inFirst: nd.slice(0, half).filter(Boolean).length,
        inSecond: nd.slice(half).filter(Boolean).length,
        size: union.length,
        ndCA,
        inCA: ndCA.filter(Boolean).length,
      };
      shares.rhoC = shares.inFirst / union.length;
      shares.rhoD = shares.inSecond / union.length;
      shares.PC = shares.inCA / CA.length;
      const firstFrom = shares.rhoC > shares.rhoD ? "CA" : "DA";
      const first = [];
      for (let i = 0; i < N; i++) first.push(tournament(firstFrom === "CA" ? CA : DA));
      const uniforms = [];
      for (let i = 0; i < N; i++) uniforms.push(rand());
      const fromCA = uniforms.map((u) => u < shares.PC);
      const caGames = [];
      const daGames = [];
      for (let i = 0; i < N; i++) caGames.push(tournament(CA));
      for (let i = 0; i < N; i++) daGames.push(tournament(DA));
      const second = fromCA.map((c, i) => (c ? caGames[i] : daGames[i]));
      const offspring = [];
      for (let i = 0; i < N; i++) {
        const p2 = solutions[second[i].winner - 1];
        const p1 = solutions[first[i].winner - 1];
        const child = sbxPair(p2.x, p1.x, 0, 1, 20, 0.5, rand)[0];
        polynomialMutation(child, 0, 1, 0.5, 20, rand);
        offspring.push(child);
      }
      const children = offspring.map((x, i) => born(x, { generation: gen, parents: [second[i].winner, first[i].winner] }));
      const next = update(CA, DA, children);
      records.push({
        generation: gen,
        CA: CA.map((sol) => sol.id),
        DA: DA.map((sol) => sol.id),
        shares,
        firstFrom,
        first,
        uniforms,
        fromCA,
        caGames,
        daGames,
        second,
        children: children.map((sol) => sol.id),
        caUpdate: next.caUpdate,
        daUpdate: next.daUpdate,
      });
      history.push({ offspring: children.map((sol) => sol.id), CA: next.caUpdate.kept, DA: next.daUpdate.kept });
      state = next;
    }
    return { popSize: N, W, solutions, history, records };
  }

  /** ToP's constrained fronts (the feasible rows by Pareto dominance, then one front per violation value) and crowding distances. */
  function topFronts(F, cv) {
    const idx = F.map((_, i) => i);
    const feasible = idx.filter((i) => cv[i] <= 0);
    const fronts = nonDominatedSort(feasible.map((i) => F[i])).fronts.map((f) => f.map((r) => feasible[r]));
    const infeasible = idx.filter((i) => cv[i] > 0);
    Array.from(new Set(infeasible.map((i) => cv[i])))
      .sort((a, b) => a - b)
      .forEach((v) => fronts.push(infeasible.filter((i) => cv[i] === v)));
    const crowd = new Array(F.length).fill(0);
    fronts.forEach((front) => {
      const d = crowdingDistance(F, front);
      front.forEach((i, a) => (crowd[i] = d[a]));
    });
    return { fronts, crowd };
  }

  /** ToP phase 1 (pairwise replacement on the mean objective) and phase 2 (constrained NSGA-II survival).
      sums lists [row, sum of normalized objectives] of the feasible survivors of phase 1, lo and hi their bounds. */
  function topSelect(F, cv, n) {
    const mean = (f) => f.reduce((s2, v) => s2 + v, 0) / f.length;
    const pairs = [];
    const next = [];
    for (let i = 0; i < n; i++) {
      const p = i;
      const o = i + n;
      const wins = cv[o] < cv[p] || (cv[o] === cv[p] && mean(F[o]) < mean(F[p]));
      pairs.push({ p, o, winner: wins ? o : p, pm: mean(F[p]), om: mean(F[o]) });
      next.push(wins ? o : p);
    }
    const feas = next.filter((i) => cv[i] <= 0);
    const Pf = feas.length / n;
    let delta = 1;
    let lo = null;
    let hi = null;
    let sums = [];
    if (feas.length >= 3) {
      lo = [0, 1].map((j) => Math.min(...feas.map((i) => F[i][j])));
      hi = [0, 1].map((j) => Math.max(...feas.map((i) => F[i][j])));
      sums = feas.map((i) => [i, F[i].reduce((s2, v, j) => s2 + (v - lo[j]) / (hi[j] - lo[j]), 0)]);
      const sorted = sums.map((e) => e[1]).sort((a, b) => a - b);
      delta = sorted[Math.floor(feas.length / 3) - 1] - sorted[0];
    }
    const { fronts, crowd } = topFronts(F, cv);
    const kept = [];
    for (const front of fronts) {
      if (kept.length + front.length <= n) kept.push(...front);
      else {
        kept.push(
          ...front
            .slice()
            .sort((a, b) => cmpDesc(crowd[a], crowd[b]) || a - b)
            .slice(0, n - kept.length)
        );
        break;
      }
      if (kept.length === n) break;
    }
    return { pairs, next, Pf, delta, sums, lo, hi, phaseOneContinues: delta >= 0.2 || Pf <= 1 / 3, fronts, crowd, kept: kept.sort((a, b) => a - b) };
  }

  /** One generation of each phase of ToP, recorded for the ToP lesson, on the loop figure's problem with CTP2's
      constraint on its objectives (gaEvaluate, then ctpViolation). popSize random solutions (ids 1 to N) start
      phase 1 with δ = 1. Phase 1 gives each solution a trial vector, DE/current-to-rand/1 or DE/rand-to-best/1/bin
      with probability 0.5 each (r1, r2 and r3 drawn with replacement, x_best the first lowest mean objective,
      F = 0.5, CR = 1), clamped and mutated (polynomial mutation, probability 1/2, η = 20); topSelect replaces each
      solution by its trial vector or not and updates Pf and δ. If they end phase 1, phase 2 follows: binary
      tournaments on ToP's fronts and crowding distances choose x₁ (a full tie goes to the first row), x₂ and x₃ are
      random, each offspring x₁ + F (x₂ − x₃) is clamped and mutated, and topSelect keeps the best N of the
      population and the offspring. phase2 is null when phase 1 continues. Random numbers are drawn in VAMOS'
      blocks: the strategies, the indices, the crossover, the mutation. */
  function topGeneration(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    const CR = 1;
    const SCALE = 0.5;
    let nextId = 1;
    const born = (x, extra) => {
      const f = gaEvaluate(x);
      return Object.assign({ id: nextId++, x, f, cv: ctpViolation(f), mean: (f[0] + f[1]) / 2 }, extra);
    };
    const draw = () => Math.floor(rand() * N);
    // Binomial crossover without j_rand, then clamping and polynomial mutation, as ToP's trial vectors.
    const trialVectors = (bases, mutants) => {
      const crossed = bases.map((b, i) => b.map((v, j) => (rand() < CR ? mutants[i][j] : v)));
      return crossed.map((x) =>
        polynomialMutation(
          x.map((v) => Math.min(1, Math.max(0, v))),
          0,
          1,
          0.5,
          20,
          rand
        )
      );
    };
    const pop = [];
    for (let i = 0; i < N; i++) pop.push(born([rand(), rand()]));
    let best = 0;
    pop.forEach((p, i) => {
      if (p.mean < pop[best].mean) best = i;
    });
    const strategy = pop.map(() => (rand() < 0.5 ? "current-to-rand" : "rand-to-best"));
    const drawn = pop.map(() => [draw(), draw(), draw()]);
    const bases = pop.map((p, i) => (strategy[i] === "current-to-rand" ? p.x : pop[drawn[i][0]].x));
    const mutants = pop.map((p, i) => {
      const [r1, r2, r3] = drawn[i].map((k) => pop[k].x);
      const b = pop[best].x;
      return p.x.map((v, j) => (strategy[i] === "current-to-rand" ? v + SCALE * (r1[j] - v) + SCALE * (r2[j] - r3[j]) : r1[j] + SCALE * (b[j] - r1[j]) + SCALE * (r2[j] - r3[j])));
    });
    const trials = trialVectors(bases, mutants).map((x, i) =>
      born(x, { parent: pop[i].id, strategy: strategy[i], r: drawn[i].map((k) => pop[k].id), base: strategy[i] === "current-to-rand" ? pop[i].id : pop[drawn[i][0]].id, mutant: mutants[i] })
    );
    const merged1 = pop.concat(trials);
    const one = topSelect(
      merged1.map((p) => p.f),
      merged1.map((p) => p.cv),
      N
    );
    const pop1 = one.next.map((i) => merged1[i]);
    const phase1 = { pop, best, trials, merged: merged1, pairs: one.pairs, next: one.next, Pf: one.Pf, delta: one.delta, sums: one.sums, lo: one.lo, hi: one.hi, continues: one.phaseOneContinues };
    if (one.phaseOneContinues) return { popSize: N, phase1, phase2: null };
    const ranked = topFronts(
      pop1.map((p) => p.f),
      pop1.map((p) => p.cv)
    );
    const front = new Array(N).fill(0);
    ranked.fronts.forEach((members, k) => members.forEach((i) => (front[i] = k)));
    const tournaments = pop1
      .map(() => [draw(), draw()])
      .map(([a, b]) => {
        let w = Math.min(a, b);
        let reason = a === b ? "same" : "tie";
        if (front[a] !== front[b]) {
          w = front[a] < front[b] ? a : b;
          reason = "front";
        } else if (ranked.crowd[a] !== ranked.crowd[b]) {
          w = ranked.crowd[a] > ranked.crowd[b] ? a : b;
          reason = "crowding";
        }
        return { a: pop1[a].id, b: pop1[b].id, winner: pop1[w].id, reason, row: w };
      });
    const others = pop1.map(() => [draw(), draw()]);
    const bases2 = tournaments.map((t) => pop1[t.row].x);
    const mutants2 = bases2.map((b, i) => b.map((v, j) => v + SCALE * (pop1[others[i][0]].x[j] - pop1[others[i][1]].x[j])));
    const children = trialVectors(bases2, mutants2).map((x, i) => born(x, { parents: [tournaments[i].winner, pop1[others[i][0]].id, pop1[others[i][1]].id], mutant: mutants2[i] }));
    const merged2 = pop1.concat(children);
    const two = topSelect(
      merged2.map((p) => p.f),
      merged2.map((p) => p.cv),
      N
    );
    const whole = [];
    let used = 0;
    let split = null;
    for (const members of two.fronts) {
      if (used === N) break;
      if (used + members.length > N) {
        split = members;
        break;
      }
      whole.push(members);
      used += members.length;
    }
    const phase2 = { pop: pop1, fronts: ranked.fronts, crowd: ranked.crowd, front, tournaments, children, merged: merged2, mergedFronts: two.fronts, mergedCrowd: two.crowd, whole, split, places: N - used, kept: two.kept };
    return { popSize: N, phase1, phase2 };
  }

  /** CMOEA-MS stage A and stage B fitness and selection (cmoeams/fitness.py, selection.py). */
  function cmoeamsSelect(F, cv, n) {
    const m = F.length;
    const top = Math.max(...cv);
    const cvn = cv.map((v) => (top > 0 ? Math.max(v, 0) / top : 0));
    const lo = [0, 1].map((j) => Math.min(...F.map((f) => f[j])));
    const hi = [0, 1].map((j) => Math.max(...F.map((f) => f[j])));
    const Fn = F.map((f) => f.map((v, j) => (hi[j] > lo[j] ? (v - lo[j]) / (hi[j] - lo[j]) : 0)));
    const kk = Math.floor(Math.sqrt(m));
    const dSde = Fn.map((fi) =>
      Fn.map((fj) => Math.hypot(...fi.map((v, t) => v - Math.max(fj[t], v))))
        .sort((a, b) => a - b)[kk]
    );
    const sde = dSde.map((d) => 1 / (d + 2));
    const cosd = (a, b) => {
      const na = Math.hypot(...a);
      const nb = Math.hypot(...b);
      return na && nb ? 1 - (a[0] * b[0] + a[1] * b[1]) / (na * nb) : 1;
    };
    const fitnessOf = (points, viol) => {
      const dom = (i, j) => viol[i] < viol[j] || (viol[i] === viol[j] && dominates(points[i], points[j]));
      const dcos = points.map((p, i) => kthOther(points.map((q) => cosd(p, q)), i, kk));
      const fit = strengthFitness(m, dom, (i) => 1 / (dcos[i] + 2));
      return { ...fit, dcos, kept: strengthSelect(fit.fitness, n, (i, j) => cosd(F[i], F[j])).sort((a, b) => a - b) };
    };
    const zeros = new Array(m).fill(0);
    const A = fitnessOf(
      sde.map((v, i) => [v, cvn[i]]),
      zeros
    );
    const B = fitnessOf(F, cvn);
    const share = cvn.filter((v) => v <= 0).length / m;
    return { cvn, dSde, sde, A, B, share };
  }

  /** Two generations of CMOEA-MS, recorded for the CMOEA-MS lesson, on the loop figure's problem with CTP2's
      constraint on its objectives (gaEvaluate, then ctpViolation), with a budget of `budget` evaluations and
      λ = `lambda` (0.5 by default). popSize random solutions (ids 1 to N) get stage A's fitness. Each generation
      selects N parents by binary tournaments on the fitness the population got in its union (two rows drawn with
      replacement; the lower fitness wins, a tie goes to the first row), crosses the i-th of the first half with the
      i-th of the second half, passing the second one first, with SBX (probability 1, η = 20), keeps the first children
      before the second ones and mutates them (polynomial mutation, probability 1/2, η = 20). cmoeamsSelect ranks the
      union Q of the population and the offspring; Q is in stage B when more than λ of it is feasible and 10% of the
      budget is spent, counting the offspring, and in stage A otherwise. */
  function cmoeamsGeneration(seed, popSize, budget, lambda) {
    const rand = mulberry32(seed);
    const N = popSize;
    const half = Math.floor(N / 2);
    const threshold = lambda === undefined ? 0.5 : lambda;
    let nextId = 1;
    const born = (x, extra) => {
      const f = gaEvaluate(x);
      return Object.assign({ id: nextId++, x, f, cv: ctpViolation(f) }, extra);
    };
    let pop = [];
    for (let i = 0; i < N; i++) pop.push(born([rand(), rand()]));
    const initial = cmoeamsSelect(
      pop.map((p) => p.f),
      pop.map((p) => p.cv),
      N
    );
    let fitness = initial.A.fitness.slice();
    let evaluations = N;
    const generations = [];
    for (let generation = 1; generation <= 2; generation++) {
      const tournaments = pop
        .map(() => [Math.floor(rand() * N), Math.floor(rand() * N)])
        .map(([a, b]) => {
          const w = fitness[b] < fitness[a] || (fitness[b] === fitness[a] && b < a) ? b : a;
          return { a: pop[a].id, b: pop[b].id, winner: pop[w].id, row: w };
        });
      const pairs = [];
      const firsts = [];
      const seconds = [];
      for (let i = 0; i < half; i++) {
        const first = pop[tournaments[i].row];
        const second = pop[tournaments[half + i].row];
        const [c1, c2] = sbxPair(second.x, first.x, 0, 1, 20, 0.5, rand);
        pairs.push([first.id, second.id]);
        firsts.push(c1);
        seconds.push(c2);
      }
      const children = firsts
        .concat(seconds)
        .map((x) => polynomialMutation(x, 0, 1, 0.5, 20, rand))
        .map((x, k) => born(x, { parents: pairs[k % half] }));
      evaluations += children.length;
      const Q = pop.concat(children);
      const sel = cmoeamsSelect(
        Q.map((q) => q.f),
        Q.map((q) => q.cv),
        N
      );
      const stageB = sel.share > threshold && evaluations >= 0.1 * budget;
      const ranked = stageB ? sel.B : sel.A;
      generations.push({ generation, pop, fitness, tournaments, pairs, children, Q, evaluations, share: sel.share, stageB, cvn: sel.cvn, dSde: sel.dSde, sde: sel.sde, A: sel.A, B: sel.B, ranked, kept: ranked.kept });
      pop = ranked.kept.map((i) => Q[i]);
      fitness = ranked.kept.map((i) => ranked.fitness[i]);
    }
    return { popSize: N, budget, lambda: threshold, initial, generations, pop, fitness };
  }

  /* ---- Tuning: F-race, Hyperband brackets and fidelity slices (vamos/engine/tuning) ---- */

  /** ln Γ(x) for x > 0 (Lanczos approximation, g = 7). */
  function lnGamma(x) {
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
    const y = x - 1;
    let a = c[0];
    for (let i = 1; i < 9; i++) a += c[i] / (y + i);
    const t = y + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
  }

  /** The regularized upper incomplete gamma function Q(a, x) (series or continued fraction). */
  function gammaQ(a, x) {
    if (x <= 0) return 1;
    const front = Math.exp(-x + a * Math.log(x) - lnGamma(a));
    if (x < a + 1) {
      let term = 1 / a;
      let sum = term;
      for (let n = 1; n < 1000 && Math.abs(term) > Math.abs(sum) * 1e-16; n++) {
        term *= x / (a + n);
        sum += term;
      }
      return Math.max(0, 1 - sum * front);
    }
    const tiny = 1e-300;
    let b = x + 1 - a;
    let c = 1 / tiny;
    let d = 1 / b;
    let f = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b;
      if (Math.abs(d) < tiny) d = tiny;
      c = b + an / c;
      if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      f *= d * c;
      if (Math.abs(d * c - 1) < 1e-16) break;
    }
    return front * f;
  }

  /** Survival function of the χ² distribution with k degrees of freedom. */
  function chi2Sf(x, k) {
    return gammaQ(k / 2, x / 2);
  }

  /** Survival function of the standard normal distribution. */
  function normalSf(z) {
    const erfc = (y) => (y >= 0 ? gammaQ(0.5, y * y) : 2 - gammaQ(0.5, y * y));
    return 0.5 * erfc(z / Math.SQRT2);
  }

  /** The regularized incomplete beta function I_x(a, b) (continued fraction). */
  function betaInc(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const fraction = (p, q, y) => {
      const tiny = 1e-300;
      let c = 1;
      let d = 1 - ((p + q) * y) / (p + 1);
      if (Math.abs(d) < tiny) d = tiny;
      d = 1 / d;
      let f = d;
      for (let m = 1; m < 1000; m++) {
        const m2 = 2 * m;
        let num = (m * (q - m) * y) / ((p + m2 - 1) * (p + m2));
        d = 1 + num * d;
        if (Math.abs(d) < tiny) d = tiny;
        c = 1 + num / c;
        if (Math.abs(c) < tiny) c = tiny;
        d = 1 / d;
        f *= d * c;
        num = (-(p + m) * (p + q + m) * y) / ((p + m2) * (p + m2 + 1));
        d = 1 + num * d;
        if (Math.abs(d) < tiny) d = tiny;
        c = 1 + num / c;
        if (Math.abs(c) < tiny) c = tiny;
        d = 1 / d;
        f *= d * c;
        if (Math.abs(d * c - 1) < 1e-16) break;
      }
      return f;
    };
    const front = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return (front * fraction(a, b, x)) / a;
    return 1 - (front * fraction(b, a, 1 - x)) / b;
  }

  /** Quantile of Student's t distribution with df degrees of freedom, for p > 0.5 (bisection). */
  function tQuantile(p, df) {
    const cdf = (t) => 1 - 0.5 * betaInc(df / 2, 0.5, df / (df + t * t));
    let lo = 0;
    let hi = 1;
    while (cdf(hi) < p) hi *= 2;
    for (let i = 0; i < 200 && hi - lo > 1e-13 * hi; i++) {
      const mid = 0.5 * (lo + hi);
      if (cdf(mid) < p) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  /** Ranks 1..n of the values, with average ranks for ties (scipy's rankdata). */
  function averageRanks(values) {
    const order = values.map((v, i) => i).sort((i, j) => values[i] - values[j]);
    const ranks = new Array(values.length);
    for (let i = 0; i < order.length; ) {
      let j = i;
      while (j + 1 < order.length && values[order[j + 1]] === values[order[i]]) j += 1;
      for (let k = i; k <= j; k++) ranks[order[k]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return ranks;
  }

  /** Sizes of the groups of equal values. */
  function tieCounts(values) {
    const counts = new Map();
    values.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
    return Array.from(counts.values());
  }

  /** Paired Wilcoxon signed-rank test, as racing/stats.py's _wilcoxon_signed_rank. */
  function wilcoxonSignedRank(differences) {
    const d = differences.filter((v) => v !== 0);
    const n = d.length;
    if (n === 0) return { p: 1, signedRank: 0 };
    const ranks = averageRanks(d.map(Math.abs));
    let signedRank = 0;
    d.forEach((v, i) => (signedRank += v > 0 ? ranks[i] : -ranks[i]));
    const ties = tieCounts(ranks).reduce((acc, c) => acc + c * c * c - c, 0);
    const variance = (n * (n + 1) * (2 * n + 1)) / 24 - ties / 48;
    if (variance <= 0) return { p: 1, signedRank };
    const centred = 0.5 * signedRank;
    const z = centred ? (centred - 0.5 * Math.sign(centred)) / Math.sqrt(variance) : 0;
    return { p: Math.min(1, 2 * normalSf(Math.abs(z))), signedRank, z };
  }

  /** One F-race test, as racing/stats.py's select_configs_by_frace: scores[configuration][block]. */
  function fraceTest(scores, maximize, alpha) {
    const nConfigs = scores.length;
    const nBlocks = nConfigs ? scores[0].length : 0;
    const out = { keep: new Array(nConfigs).fill(true), test: "none", p: 1, statistic: null, rankSums: null, critical: null, best: null };
    if (nConfigs <= 1 || nBlocks <= 1) return out;
    const oriented = scores.map((row) => row.map((v) => (maximize ? -v : v)));
    if (nConfigs === 2) {
      const w = wilcoxonSignedRank(oriented[0].map((v, b) => v - oriented[1][b]));
      Object.assign(out, { test: "wilcoxon", p: w.p, statistic: w.signedRank });
      if (w.p < alpha && w.signedRank !== 0) out.keep[w.signedRank > 0 ? 0 : 1] = false;
      return out;
    }
    const ranks = [];
    for (let b = 0; b < nBlocks; b++) ranks.push(averageRanks(oriented.map((row) => row[b])));
    const rankSums = oriented.map((_, c) => ranks.reduce((acc, row) => acc + row[c], 0));
    const ties = ranks.reduce((acc, row) => acc + tieCounts(row).reduce((t, c) => t + c * c * c - c, 0), 0);
    const expected = (nBlocks * (nConfigs + 1)) / 2;
    const denominator = nBlocks * nConfigs * (nConfigs + 1) - ties / (nConfigs - 1);
    Object.assign(out, { test: "friedman", rankSums });
    if (denominator <= 0) return out;
    const statistic = (12 * rankSums.reduce((acc, r) => acc + (r - expected) * (r - expected), 0)) / denominator;
    out.statistic = statistic;
    out.p = chi2Sf(statistic, nConfigs - 1);
    if (!(out.p < alpha)) return out;
    const df = (nBlocks - 1) * (nConfigs - 1);
    const squares = ranks.reduce((acc, row) => acc + row.reduce((t, r) => t + r * r, 0), 0);
    const spread = (2 * (nBlocks * squares - rankSums.reduce((acc, r) => acc + r * r, 0))) / df;
    const critical = tQuantile(1 - alpha / 2, df) * Math.sqrt(Math.max(spread, 0));
    let best = 0;
    rankSums.forEach((r, c) => {
      if (r < rankSums[best]) best = c;
    });
    out.keep = rankSums.map((r, c) => c === best || r - rankSums[best] <= critical);
    Object.assign(out, { critical, best });
    return out;
  }

  /** A statistical race over fixed scores, as RacingTuner with use_statistical_tests and no restarts.

      scores[c][b] is the score of configuration c on the b-th block played. After each block the
      alive configurations are tested once they have max(firstTest, minBlocks) blocks; the race ends
      when minSurvivors or fewer are alive, the experiments run out, or the blocks do. */
  function fraceRace(scores, o) {
    const opt = Object.assign({ maximize: true, alpha: 0.05, firstTest: 5, minBlocks: 3, minSurvivors: 1, maxExperiments: Infinity }, o || {});
    const n = scores.length;
    const nBlocks = n ? scores[0].length : 0;
    const alive = new Array(n).fill(true);
    const played = new Array(n).fill(0);
    const eliminatedAt = new Array(n).fill(null);
    const steps = [];
    let experiments = 0;
    for (let b = 0; b < nBlocks; b++) {
      const before = alive.map((a, c) => (a ? c : -1)).filter((c) => c >= 0);
      if (!before.length || experiments >= opt.maxExperiments) break;
      before.forEach((c) => (played[c] += 1));
      experiments += before.length;
      const step = { block: b, alive: before.slice(), test: null, eliminated: [] };
      const blocks = b + 1;
      if (before.length > 1 && blocks >= Math.max(opt.firstTest, opt.minBlocks) && blocks > 1) {
        const result = fraceTest(
          before.map((c) => scores[c].slice(0, blocks)),
          opt.maximize,
          opt.alpha
        );
        step.test = result;
        result.keep.forEach((k, row) => {
          if (!k) {
            alive[before[row]] = false;
            eliminatedAt[before[row]] = b;
            step.eliminated.push(before[row]);
          }
        });
      }
      steps.push(step);
      if (experiments >= opt.maxExperiments || alive.filter(Boolean).length <= opt.minSurvivors) break;
    }
    const survivors = alive.map((a, c) => (a ? c : -1)).filter((c) => c >= 0);
    const mean = (c) => scores[c].slice(0, played[c]).reduce((acc, v) => acc + v, 0) / Math.max(played[c], 1);
    let best = null;
    survivors.forEach((c) => {
      if (best === null || (opt.maximize ? mean(c) > mean(best) : mean(c) < mean(best))) best = c;
    });
    return { steps, alive, played, eliminatedAt, experiments, survivors, best, mean: scores.map((_, c) => mean(c)) };
  }

  /** Configurations per rung of each BOHB iteration, as _model_backend_utils.hyperband_brackets. */
  function hyperbandBrackets(maxShIter, eta, maxConfigs) {
    const brackets = [];
    let remaining = Math.max(1, maxConfigs);
    while (remaining > 0) {
      const sMax = maxShIter - 1 - (brackets.length % maxShIter);
      const n0 = Math.min(remaining, Math.floor(maxShIter / (sMax + 1)) * Math.pow(eta, sMax));
      const rungs = [];
      for (let i = 0; i <= sMax; i++) rungs.push(Math.max(Math.floor(n0 * Math.pow(eta, -i)), 1));
      brackets.push(rungs);
      remaining -= n0;
    }
    return brackets;
  }

  /** Python's round(): halves go to the even neighbour. */
  /**
   * The budgets of Hyperband between lo and hi with factor eta: geometric, as hpbandster and SMAC3 space them,
   * rounded half to even and clipped to [lo, hi], as run_smac3 and run_bohb_native round each budget before a run.
   */
  function hyperbandBudgets(lo, hi, eta) {
    const rungs = Math.floor(Math.log(hi / lo) / Math.log(eta) + 1e-9) + 1;
    return Array.from({ length: rungs }, (_, i) => Math.min(hi, Math.max(lo, roundHalfEven(hi / Math.pow(eta, rungs - 1 - i)))));
  }

  function roundHalfEven(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  /** How many instances and seeds a model-based backend runs at a budget (_model_backend_fidelity.resolve_fidelity_slice). */
  function fidelitySlice(budget, levels, o) {
    const full = levels[levels.length - 1];
    const b = Math.min(full, Math.max(1, budget));
    const frac = full <= levels[0] ? 1 : Math.min(1, Math.max(0, (b - levels[0]) / (full - levels[0])));
    const nInst = o.instances;
    const nSeeds = Math.max(1, o.seeds);
    const instFrac = o.minInstanceFrac + frac * (1 - o.minInstanceFrac);
    const instances = Math.max(1, Math.min(nInst, Math.trunc(roundHalfEven(nInst * instFrac))));
    const maxCount = Math.min(nSeeds, Math.max(1, o.maxSeedCount || nSeeds));
    const minCount = Math.min(maxCount, Math.max(1, o.minSeedCount || maxCount));
    const seeds = Math.max(1, Math.min(nSeeds, Math.trunc(roundHalfEven(minCount + frac * (maxCount - minCount)))));
    return { budget: b, fraction: frac, instances, seeds };
  }

  /* ---- Advanced extras: two-layer reference directions and constraint-handling techniques ---- */

  /** The integer points of a Das and Dennis lattice: m non-negative integers that sum to p, in VAMOS's order. */
  function latticeIntegers(m, p) {
    const out = [];
    const rec = (prefix, left, depth) => {
      if (depth === m - 1) {
        out.push(prefix.concat([left]));
        return;
      }
      for (let i = 0; i <= left; i++) rec(prefix.concat([i]), left - i, depth + 1);
    };
    rec([], p, 0);
    return out;
  }

  /** Deb and Jain's two layers, as VAMOS's two_layer_lattice: the lattice of p1, then that of p2 shrunk by half towards the centre. */
  function twoLayerLattice(m, p1, p2) {
    return {
      boundary: dasDennis(m, p1),
      inside: dasDennis(m, p2).map((w) => w.map((v) => v / 2 + 1 / (2 * m))),
    };
  }

  /** How many inside directions repeat a boundary direction, compared exactly on a common grid (VAMOS's two_layer_shared). */
  function twoLayerShared(m, p1, p2) {
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const q = 2 * p2 * m;
    const grid = (p1 / gcd(p1, q)) * q;
    const boundary = new Set(latticeIntegers(m, p1).map((k) => k.map((v) => (v * grid) / p1).join(",")));
    // An inside coordinate is (k / p2) / 2 + 1 / (2 m) = (k m + p2) / q.
    return latticeIntegers(m, p2).filter((k) => boundary.has(k.map((v) => ((v * m + p2) * grid) / q).join(","))).length;
  }

  /** The overall violation of each row: the sum of its positive constraint values (G <= 0 is feasible). */
  function overallViolation(G) {
    return G.map((g) => g.reduce((acc, v) => acc + Math.max(0, v), 0));
  }

  function violatedCount(G) {
    return G.map((g) => g.filter((v) => v > 0).length);
  }

  function paretoRanks(F) {
    return F.length ? nonDominatedSort(F).rank : [];
  }

  /** Deb's feasibility rule over a measure, as VAMOS's ranks_by_violation: feasible fronts first, then one group per value. */
  function ranksByViolation(F, measure, objectivesOnTies) {
    const n = F.length;
    const ranks = new Array(n).fill(0);
    const rows = Array.from({ length: n }, (_, i) => i);
    const feasible = rows.filter((i) => measure[i] <= 0);
    let offset = 0;
    if (feasible.length) {
      const r = paretoRanks(feasible.map((i) => F[i]));
      feasible.forEach((row, a) => (ranks[row] = r[a]));
      offset = Math.max(...r) + 1;
    }
    const infeasible = rows.filter((i) => !(measure[i] <= 0)).sort((a, b) => measure[a] - measure[b] || a - b);
    for (let start = 0; start < infeasible.length; ) {
      let end = start + 1;
      while (end < infeasible.length && measure[infeasible[end]] === measure[infeasible[start]]) end += 1;
      const members = infeasible.slice(start, end);
      if (objectivesOnTies && members.length > 1) {
        const r = paretoRanks(members.map((i) => F[i]));
        members.forEach((row, a) => (ranks[row] = offset + r[a]));
        offset += Math.max(...r) + 1;
      } else {
        members.forEach((row) => (ranks[row] = offset));
        offset += 1;
      }
      start = end;
    }
    return ranks;
  }

  /** Runarsson and Yao's stochastic bubble sort, as VAMOS's stochastic_ranking_order, with the uniforms drawn in order. */
  function stochasticRankingOrder(paretoRank, feasible, measure, uniforms, probability) {
    const n = paretoRank.length;
    const order = Array.from({ length: n }, (_, i) => i);
    let drawn = 0;
    for (let sweep = 0; sweep < n; sweep++) {
      let swapped = false;
      for (let j = 0; j < n - 1; j++) {
        const first = order[j];
        const second = order[j + 1];
        let byRank = true;
        if (!(feasible[first] && feasible[second])) {
          byRank = uniforms[drawn] < probability;
          drawn += 1;
        }
        if (byRank ? paretoRank[second] < paretoRank[first] : measure[second] < measure[first]) {
          order[j] = second;
          order[j + 1] = first;
          swapped = true;
        }
      }
      if (!swapped) break;
    }
    return order;
  }

  /** The fronts of a stochastic ranking: runs of consecutive feasible rows with the same Pareto rank. */
  function stochasticRankingRanks(order, paretoRank, feasible) {
    const ranks = new Array(order.length).fill(0);
    let level = 0;
    order.forEach((row, a) => {
      const prev = order[a - 1];
      if (a > 0 && !(feasible[prev] && feasible[row] && paretoRank[prev] === paretoRank[row])) level += 1;
      ranks[row] = level;
    });
    return ranks;
  }

  /** Woldesenbet, Yen and Tessema's modified objectives, as VAMOS's self_adaptive_penalty_objectives. */
  function selfAdaptivePenaltyObjectives(F, G) {
    const n = F.length;
    const m = F[0].length;
    const c = G[0].length;
    const excess = G.map((g) => g.map((v) => Math.max(0, v)));
    const largest = Array.from({ length: c }, (_, j) => Math.max(...excess.map((e) => e[j])));
    const penalty = excess.map((e) => (c > 0 ? e.reduce((acc, v, j) => acc + (largest[j] > 0 ? v / largest[j] : 0), 0) / c : 0));
    const feasible = excess.map((e) => e.every((v) => v <= 0));
    const ratio = feasible.filter(Boolean).length / n;
    const low = Array.from({ length: m }, (_, i) => Math.min(...F.map((f) => f[i])));
    const high = Array.from({ length: m }, (_, i) => Math.max(...F.map((f) => f[i])));
    return F.map((f, r) =>
      f.map((v, i) => {
        const span = high[i] - low[i];
        const normalized = span > 0 ? (v - low[i]) / span : 0;
        const distance = ratio === 0 ? penalty[r] : Math.sqrt(normalized * normalized + penalty[r] * penalty[r]);
        const violationPenalty = ratio === 0 ? 0 : penalty[r];
        const objectivePenalty = feasible[r] ? 0 : normalized;
        return distance + (1 - ratio) * violationPenalty + ratio * objectivePenalty;
      })
    );
  }

  /** The fronts of a population under a constraint-handling technique of VAMOS, and the objectives it ranks, if modified. */
  function constraintRanks(name, F, G, options) {
    const o = options || {};
    const cv = overallViolation(G);
    if (name === "cdp") return { ranks: ranksByViolation(F, cv, true) };
    if (name === "violated_constraints") return { ranks: ranksByViolation(F, violatedCount(G), true) };
    if (name === "push_pull" && o.pushing) return { ranks: paretoRanks(F) };
    if (name === "epsilon_constrained" || name === "push_pull") {
      const epsilon = o.epsilon || 0;
      return { ranks: ranksByViolation(F, cv.map((v) => (v <= epsilon ? 0 : v)), false) };
    }
    if (name === "stochastic_ranking") {
      const pareto = paretoRanks(F);
      const feasible = cv.map((v) => v <= 0);
      const order = stochasticRankingOrder(pareto, feasible, cv, o.uniforms, o.probability);
      return { ranks: stochasticRankingRanks(order, pareto, feasible), order };
    }
    if (name === "self_adaptive_penalty") {
      const objectives = selfAdaptivePenaltyObjectives(F, G);
      return { ranks: paretoRanks(objectives), objectives };
    }
    throw new Error("Unknown constraint-handling technique " + name);
  }

  /* ---- Benchmarks: two-objective fronts, position bias and CTP feasibility ---- */

  /* The two-objective Pareto fronts of the Benchmarks lessons, written in closed form from the
     definitions in src/vamos/foundation/problem/ (zdt*.py, dtlz.py and wfg.py): point(t) maps a
     position parameter t in [lo, hi] to the objective vector at the optimal distance variables.
     tests/docs/test_foundations.py compares every front with VAMOS' packaged reference front. */
  const HALF_PI = Math.PI / 2;
  const WFG_CONCAVE = "Concave: the quarter ellipse (f₁/2)² + (f₂/4)² = 1";
  const BENCHMARK_FRONTS = {
    zdt1: {
      label: "ZDT1",
      point: (t) => [t, 1 - Math.sqrt(t)],
      shape: "Convex",
      tests: "Convergence and spread on a simple convex front; the usual first test",
      separable: "Yes",
      objectives: "2",
      n: "30",
      files: "2 objectives",
    },
    zdt2: {
      label: "ZDT2",
      point: (t) => [t, 1 - t * t],
      shape: "Concave",
      tests: "A concave front, whose inner points no weighted sum of the objectives can reach",
      separable: "Yes",
      objectives: "2",
      n: "30",
      files: "2 objectives",
    },
    zdt3: {
      label: "ZDT3",
      point: (t) => [t, 1 - Math.sqrt(t) - t * Math.sin(10 * Math.PI * t)],
      shape: "Disconnected: five pieces",
      tests: "Keeping solutions on every piece of a disconnected front",
      separable: "Yes",
      objectives: "2",
      n: "30",
      files: "2 objectives",
    },
    zdt4: {
      label: "ZDT4",
      point: (t) => [t, 1 - Math.sqrt(t)],
      shape: "Convex (ZDT1's front)",
      tests: "Multimodality: a Rastrigin-type g with 21⁹ local fronts for n = 10",
      separable: "Yes",
      objectives: "2",
      n: "10",
      files: "2 objectives",
    },
    zdt6: {
      label: "ZDT6",
      point: (t) => [t, 1 - t * t],
      lo: 0.2807753191,
      shape: "Concave, from f₁ ≈ 0.281",
      tests: "Bias: uniform x₁ crowds near f₁ = 1, and the density of solutions falls towards the front",
      separable: "Yes",
      objectives: "2",
      n: "10",
      files: "2 objectives",
    },
    dtlz1: {
      label: "DTLZ1",
      point: (t) => [0.5 * t, 0.5 * (1 - t)],
      shape: "Linear: f₁ + … + f_M = 0.5",
      tests: "Multimodality: 11 local minima per distance variable, 11ᵏ − 1 local fronts",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 4",
      files: "2, 3, 4, 6 and 8 objectives",
    },
    dtlz2: {
      label: "DTLZ2",
      point: (t) => [Math.cos(t * HALF_PI), Math.sin(t * HALF_PI)],
      shape: "Concave: the unit sphere, f₁² + … + f_M² = 1",
      tests: "Scaling the number of objectives on a smooth unimodal problem",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 9",
      files: "2, 3, 4, 6 and 8 objectives",
    },
    dtlz3: {
      label: "DTLZ3",
      point: (t) => [Math.cos(t * HALF_PI), Math.sin(t * HALF_PI)],
      shape: "Concave (DTLZ2's front)",
      tests: "Multimodality on the sphere: DTLZ1's distance function with DTLZ2's shape",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 9",
      files: "2 and 3 objectives; DTLZ2's for 4, 6 and 8",
    },
    dtlz4: {
      label: "DTLZ4",
      point: (t) => [Math.cos(t * HALF_PI), Math.sin(t * HALF_PI)],
      shape: "Concave (DTLZ2's front)",
      tests: "Bias: the position variables are raised to α = 100, which crowds solutions at one end",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 9",
      files: "2 and 3 objectives; DTLZ2's for 4, 6 and 8",
    },
    dtlz7: {
      label: "DTLZ7",
      point: (t) => [t, 4 - t * (1 + Math.sin(3 * Math.PI * t))],
      shape: "Disconnected: 2^(M−1) pieces, two with M = 2",
      tests: "A disconnected front in any number of objectives",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 19",
      files: "2, 3, 4, 6 and 8 objectives",
    },
    idtlz2: {
      label: "IDTLZ2",
      point: (t) => [1 - Math.cos(t * HALF_PI), 1 - Math.sin(t * HALF_PI)],
      shape: "Convex: DTLZ2's sphere turned over",
      tests: "An inverted front, which many evenly spread weight vectors miss",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "M + 9",
      files: "2 and 3 objectives",
    },
    wfg1: {
      label: "WFG1",
      point: (t) => [2 * (1 - Math.cos(t * HALF_PI)), 4 * (1 - t - Math.cos(10 * Math.PI * t + HALF_PI) / (10 * Math.PI))],
      shape: "Convex and mixed",
      tests: "Bias: flat regions and a polynomial bias with exponent 0.02",
      separable: "Yes",
      objectives: "Any M ≥ 2",
      n: "24",
      files: "2 objectives",
    },
    wfg2: {
      label: "WFG2",
      point: (t) => [2 * (1 - Math.cos(t * HALF_PI)), 4 * (1 - t * Math.cos(5 * Math.PI * t) ** 2)],
      shape: "Convex and disconnected",
      tests: "Non-separable distance variables and a disconnected front",
      separable: "No",
      objectives: "Any M ≥ 2",
      n: "24",
      files: "2 objectives",
    },
    wfg3: {
      label: "WFG3",
      point: (t) => [2 * t, 4 * (1 - t)],
      shape: "Linear; designed to be degenerate for M ≥ 3",
      tests: "Non-separable variables and a front of lower dimension than M − 1",
      separable: "No",
      objectives: "Any M ≥ 2",
      n: "24",
      files: "2 objectives",
    },
    wfg4: { label: "WFG4", tests: "Multimodality: many local optima in every variable", separable: "Yes" },
    wfg5: { label: "WFG5", tests: "Deception: a broad local optimum that leads away from the global one", separable: "Yes" },
    wfg6: { label: "WFG6", tests: "Non-separable variables", separable: "No" },
    wfg7: { label: "WFG7", tests: "Parameter-dependent bias: the position variables depend on the distance variables", separable: "Yes" },
    wfg8: { label: "WFG8", tests: "Parameter-dependent bias: the distance variables depend on the position variables", separable: "No" },
    wfg9: { label: "WFG9", tests: "Parameter-dependent bias, multimodality and deception together", separable: "No" },
  };
  for (const key of ["wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]) {
    Object.assign(BENCHMARK_FRONTS[key], {
      point: (t) => [2 * Math.sin(t * HALF_PI), 4 * Math.cos(t * HALF_PI)],
      shape: WFG_CONCAVE,
      objectives: "Any M ≥ 2",
      n: "24",
      files: "2 objectives",
    });
  }

  /** The non-dominated points of a two-objective set, sorted by f₁. */
  function nonDominated2d(points) {
    const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const out = [];
    for (const p of sorted) if (!out.length || p[1] < out[out.length - 1][1]) out.push(p);
    return out;
  }

  /** n evenly spaced values of a front's position parameter, mapped and filtered for non-dominance. */
  function benchmarkFront(key, n) {
    const front = BENCHMARK_FRONTS[key];
    const lo = front.lo || 0;
    const pts = [];
    for (let i = 0; i < n; i++) pts.push(front.point(lo + ((1 - lo) * i) / (n - 1)));
    return nonDominated2d(pts);
  }

  /** The pieces of a sorted two-objective front: a gap in f₁ wider than `gap` times its range starts a new one. */
  function frontPieces(pts, gap) {
    const span = pts[pts.length - 1][0] - pts[0][0] || 1;
    const pieces = [[pts[0]]];
    for (let i = 1; i < pts.length; i++) {
      if (pts[i][0] - pts[i - 1][0] > gap * span) pieces.push([]);
      pieces[pieces.length - 1].push(pts[i]);
    }
    return pieces;
  }

  /* Position bias: where a Pareto-optimal solution lands on the front, from its position variables u in
     [0, 1] (the distance variables at their optimum), for M = 2. position(u, alpha) returns the front
     parameter t of BENCHMARK_FRONTS[front].point(t): DTLZ4 raises its position variable to α, 100 by
     default (dtlz.py; α = 1 is DTLZ2); ZDT6 maps x₁ through f₁ = 1 − exp(−4x₁) sin⁶(6πx₁) (zdt6.py); WFG1
     applies its polynomial bias y^0.02 to its four position variables and reduces them by a weighted sum
     with weights 2, 4, 6 and 8 (wfg.py). */
  const BIAS_PROBLEMS = {
    dtlz4: { label: "DTLZ4 (DTLZ2 when α = 1)", positions: 1, front: "dtlz4", position: (u, alpha) => Math.pow(u[0], alpha) },
    zdt6: { label: "ZDT6", positions: 1, front: "zdt6", position: (u) => 1 - Math.exp(-4 * u[0]) * Math.pow(Math.sin(6 * Math.PI * u[0]), 6) },
    wfg1: {
      label: "WFG1",
      positions: 4,
      front: "wfg1",
      position(u) {
        let num = 0;
        let den = 0;
        u.forEach((v, i) => {
          num += 2 * (i + 1) * Math.pow(v, 0.02);
          den += 2 * (i + 1);
        });
        return num / den;
      },
    },
  };

  function biasPoint(key, u, alpha) {
    const problem = BIAS_PROBLEMS[key];
    return BENCHMARK_FRONTS[problem.front].point(problem.position(u, alpha));
  }

  /* CTP1-CTP7 of Deb, Pratap and Meyarivan (2001) in objective space, with the parameters of
     src/vamos/foundation/problem/ctp.py. Every design has g ≥ 1, so the attainable region lies on or above
     the unconstrained front: f₂ ≥ exp(−f₁) for CTP1 and f₂ ≥ 1 − f₁ for CTP2-CTP7. */
  const CTP_PROBLEMS = {
    ctp1: { a: [0.858266, 0.728234], b: [0.541475, 0.295039], ymax: 1.2 },
    ctp2: { theta: -0.2 * Math.PI, a: 0.2, b: 10, c: 1, d: 6, e: 1, ymax: 1.6 },
    ctp3: { theta: -0.2 * Math.PI, a: 0.1, b: 10, c: 1, d: 0.5, e: 1, ymax: 1.6 },
    ctp4: { theta: -0.2 * Math.PI, a: 0.75, b: 10, c: 1, d: 0.5, e: 1, ymax: 2.4 },
    ctp5: { theta: -0.2 * Math.PI, a: 0.1, b: 10, c: 2, d: 0.5, e: 1, ymax: 1.6 },
    ctp6: { theta: 0.1 * Math.PI, a: 40, b: 0.5, c: 1, d: 2, e: -2, ymax: 20 },
    ctp7: { theta: -0.05 * Math.PI, a: 40, b: 5, c: 1, d: 6, e: 0, ymax: 2 },
  };

  function ctpBase(key, f1) {
    return key === "ctp1" ? Math.exp(-f1) : 1 - f1;
  }

  /** The total violation of CTP1-CTP7 at an objective vector, as VAMOS' G with its positive parts summed. */
  function ctpTotalViolation(key, f) {
    const p = CTP_PROBLEMS[key];
    if (key === "ctp1") return p.a.reduce((sum, a, j) => sum + Math.max(0, a * Math.exp(-p.b[j] * f[0]) - f[1]), 0);
    return ctpViolation(f, p);
  }

  /** The constrained front of a CTP problem: for n values of f₁, the least feasible f₂ on or above the
      unconstrained front (scanned, then refined by bisection), filtered for non-dominance. */
  function ctpFront(key, n) {
    const top = CTP_PROBLEMS[key].ymax;
    const lowest = [];
    for (let i = 0; i < n; i++) {
      const f1 = i / (n - 1);
      const base = ctpBase(key, f1);
      const step = (top - base) / 2000;
      if (ctpTotalViolation(key, [f1, base]) <= 0) {
        lowest.push([f1, base]);
        continue;
      }
      for (let k = 1; k <= 2000; k++) {
        const f2 = base + k * step;
        if (ctpTotalViolation(key, [f1, f2]) > 0) continue;
        let lo = f2 - step;
        let hi = f2;
        for (let it = 0; it < 40; it++) {
          const mid = 0.5 * (lo + hi);
          if (ctpTotalViolation(key, [f1, mid]) > 0) lo = mid;
          else hi = mid;
        }
        lowest.push([f1, hi]);
        break;
      }
    }
    if (key !== "ctp1") {
      // Add the isolated optima that a scan can miss: on the line cos(θ) (f₂ − e) = sin(θ) f₁, where the
      // left side of the constraint is 0, sin(θ) (f₂ − e) + cos(θ) f₁ = f₁ / cos(θ), and the sine vanishes
      // where (f₁ / cos(θ))^c = j / b.
      const p = CTP_PROBLEMS[key];
      for (let j = 0; j <= 200; j++) {
        const f1 = Math.cos(p.theta) * Math.pow(j / p.b, 1 / p.c);
        const f2 = p.e + f1 * Math.tan(p.theta);
        if (f1 <= 1 && f2 >= ctpBase(key, f1) && ctpTotalViolation(key, [f1, f2]) <= 1e-6) lowest.push([f1, f2]);
      }
    }
    return nonDominated2d(lowest);
  }

  /* ---- Operators: ports of the VAMOS variation operators (engine/operators/impl) ---- */

  /** Gamma function (Lanczos approximation), for the Lévy flight scale of Mantegna's algorithm. */
  function gammaFn(z) {
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gammaFn(1 - z));
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    const x = z - 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
  }

  /** ``v`` with its component along ``d`` removed (multi_parent._remove_component). */
  function removeComponent(v, d) {
    const dd = Math.max(d.reduce((s, x) => s + x * x, 0), 1e-32);
    const k = v.reduce((s, x, i) => s + x * d[i], 0) / dd;
    return v.map((x, i) => x - k * d[i]);
  }

  const norm = (v) => Math.hypot(...v);

  /** NumPy's rint: round half to even. */
  function rint(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  const OPERATORS = {
    /* Real-coded crossovers: two parents in, two children out (impl/real/crossover_blend.py, crossover_stochastic.py). */
    blxAlphaBeta(p1, p2, alpha, beta, rand) {
      const c1 = [];
      const c2 = [];
      for (let j = 0; j < p1.length; j++) {
        const lo = Math.min(p1[j], p2[j]);
        const span = Math.max(p1[j], p2[j]) - lo;
        const a = lo - alpha * span;
        const width = lo + span + beta * span - a;
        c1.push(a + rand() * width);
        c2.push(a + rand() * width);
      }
      return [c1, c2];
    },
    arithmetic(p1, p2, rand) {
      const lam = rand();
      return [p1.map((v, j) => lam * v + (1 - lam) * p2[j]), p1.map((v, j) => (1 - lam) * v + lam * p2[j])];
    },
    wholeArithmetic(p1, p2, alpha) {
      return [p1.map((v, j) => alpha * v + (1 - alpha) * p2[j]), p1.map((v, j) => (1 - alpha) * v + alpha * p2[j])];
    },
    laplace(p1, p2, a, b, rand) {
      const beta = () => {
        const u = rand();
        return a + b * Math.sign(u - 0.5) * Math.log(Math.max(1 - 2 * Math.abs(u - 0.5), 1e-30));
      };
      return [p1.map((v, j) => v + beta() * Math.abs(p1[j] - p2[j])), p2.map((v, j) => v + beta() * Math.abs(p1[j] - p2[j]))];
    },
    fuzzy(p1, p2, d, lower, upper, rand) {
      const c1 = [];
      const c2 = [];
      for (let j = 0; j < p1.length; j++) {
        const lo = Math.min(p1[j], p2[j]);
        const hi = Math.max(p1[j], p2[j]);
        const span = hi - lo;
        const tlo = Math.min(upper, Math.max(lower, lo - d * span));
        const thi = Math.min(upper, Math.max(lower, hi + d * span));
        const mode = Math.min(thi, Math.max(tlo, 0.5 * (p1[j] + p2[j])));
        for (const c of [c1, c2]) {
          const u = rand();
          if (thi - tlo < 1e-30) {
            c.push(mode);
            continue;
          }
          const w = thi - tlo;
          const fc = (mode - tlo) / w;
          c.push(u < fc ? tlo + Math.sqrt(u * w * fc * w) : thi - Math.sqrt((1 - u) * w * (1 - fc) * w));
        }
      }
      return [c1, c2];
    },
    /* Three-parent crossovers (impl/real/crossover_differential.py, crossover_multi_parent.py). */
    deMating(x, a, b, F, CR, rand) {
      const n = x.length;
      const jr = Math.min(n - 1, Math.floor(rand() * n));
      return [x.map((v, j) => (rand() < CR || j === jr ? v + F * (a[j] - b[j]) : v))];
    },
    pcx(group, sigmaEta, sigmaZeta, rand) {
      const n = group[0].length;
      const g = group[0].map((_, i) => group.reduce((s, p) => s + p[i], 0) / group.length);
      return group.map((xj, j) => {
        const d = g.map((v, i) => v - xj[i]);
        const others = group.filter((_, q) => q !== j).map((p) => removeComponent(p.map((v, i) => v - xj[i]), d));
        const D = others.reduce((s, v) => s + norm(v), 0) / others.length;
        const noise = removeComponent(
          Array.from({ length: n }, () => gaussian(rand) * sigmaEta * D),
          d
        );
        const w = gaussian(rand) * sigmaZeta;
        return xj.map((v, i) => v + w * d[i] + noise[i]);
      });
    },
    undx(group, zeta, eta, rand) {
      const [p1, p2, p3] = group;
      const n = p1.length;
      const d = p2.map((v, i) => v - p1[i]);
      if (norm(d) < 1e-10) return [p1.slice(), p2.slice()];
      const m = p1.map((v, i) => 0.5 * (v + p2[i]));
      const D = norm(removeComponent(p3.map((v, i) => v - m[i]), d));
      const xi = gaussian(rand) * zeta;
      const orth = removeComponent(
        Array.from({ length: n }, () => gaussian(rand) * (eta / Math.sqrt(n)) * D),
        d
      );
      const step = d.map((v, i) => xi * v + orth[i]);
      return [m.map((v, i) => v + step[i]), m.map((v, i) => v - step[i])];
    },
    spx(group, epsilon, rand) {
      const k = group.length;
      const eps = epsilon === undefined || epsilon === null ? Math.sqrt(k + 1) : epsilon;
      const o = group[0].map((_, i) => group.reduce((s, p) => s + p[i], 0) / k);
      return group.map(() => {
        const e = group.map(() => -Math.log(Math.max(rand(), 1e-300)));
        const total = e.reduce((s, v) => s + v, 0);
        return o.map((ov, i) => ov + eps * (group.reduce((s, p, q) => s + (e[q] / total) * p[i], 0) - ov));
      });
    },
    /* Real-coded mutations: each variable is mutated with probability prob (impl/real/mutation.py). */
    gaussianMutation(x, lower, upper, prob, sigma, rand) {
      return x.map((v) => (rand() <= prob ? v + gaussian(rand) * sigma * (upper - lower) : v));
    },
    uniformMutation(x, lower, upper, prob, perturb, rand) {
      return x.map((v) => (rand() <= prob ? v + (rand() * 2 - 1) * perturb * (upper - lower) : v));
    },
    uniformReset(x, lower, upper, prob, rand) {
      return x.map((v) => (rand() <= prob ? lower + rand() * (upper - lower) : v));
    },
    nonUniform(x, lower, upper, prob, rand) {
      // The step y (1 - r^((1 - t/T)^b)) with the run's progress t/T = 0, as in VAMOS: y (1 - r).
      return x.map((v) => {
        if (!(rand() <= prob)) return v;
        const delta = 1 - rand();
        return v + delta * (rand() <= 0.5 ? upper - v : lower - v);
      });
    },
    cauchyMutation(x, lower, upper, prob, gamma, rand) {
      return x.map((v) => (rand() <= prob ? v + Math.tan(Math.PI * (rand() - 0.5)) * gamma * (upper - lower) : v));
    },
    levyFlight(x, lower, upper, prob, beta, scale, rand) {
      const num = gammaFn(1 + beta) * Math.sin((Math.PI * beta) / 2);
      const den = gammaFn((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2);
      const sigmaU = Math.pow(num / den, 1 / beta);
      return x.map((v) => {
        if (!(rand() <= prob)) return v;
        const step = (gaussian(rand) * sigmaU) / (Math.pow(Math.abs(gaussian(rand)), 1 / beta) + 1e-30);
        return v + scale * step * (upper - lower);
      });
    },
    powerLaw(x, lower, upper, prob, index, rand) {
      return x.map((v) => {
        if (!(rand() <= prob)) return v;
        const s = Math.pow(rand(), index);
        const t = (v - lower) / (upper - lower);
        const y = t < rand() ? v - s * (v - lower) : v + s * (upper - v);
        return Math.min(upper, Math.max(lower, y));
      });
    },
    linkedPolynomial(x, lower, upper, prob, eta, rand) {
      // One draw u per individual, shared by all of its mutated variables.
      const mask = x.map(() => rand() <= prob);
      const u = rand();
      return x.map((v, j) => (mask[j] ? polynomialMutation([v], lower, upper, 1, eta, () => u)[0] : v));
    },

    /* Binary operators (impl/binary.py); the draws follow VAMOS's ranges. */
    onePoint(p1, p2, cut) {
      return [p1.slice(0, cut).concat(p2.slice(cut)), p2.slice(0, cut).concat(p1.slice(cut))];
    },
    twoPoint(p1, p2, lo, hi) {
      return [p1.map((v, j) => (j >= lo && j < hi ? p2[j] : v)), p2.map((v, j) => (j >= lo && j < hi ? p1[j] : v))];
    },
    uniformBits(p1, p2, keep) {
      return [p1.map((v, j) => (keep[j] ? v : p2[j])), p2.map((v, j) => (keep[j] ? v : p1[j]))];
    },
    hux(p1, p2, chosen) {
      const set = new Set(chosen);
      return [p1.map((v, j) => (set.has(j) ? p2[j] : v)), p2.map((v, j) => (set.has(j) ? p1[j] : v))];
    },
    bitFlip(x, mask) {
      return x.map((v, j) => (mask[j] ? 1 - v : v));
    },
    segmentInversion(x, a, b) {
      return x.map((v, j) => (j >= a && j <= b ? 1 - v : v));
    },
    drawOnePoint(n, rand) {
      return 1 + Math.floor(rand() * (n - 1));
    },
    drawTwoPoint(n, rand) {
      const a = Math.floor(rand() * n);
      const b = Math.floor(rand() * n);
      const lo = Math.min(a, b);
      return [lo, Math.min(n, Math.max(Math.max(a, b), lo + 1))];
    },
    drawHux(p1, p2, rand) {
      const diff = p1.map((v, j) => (v !== p2[j] ? j : -1)).filter((j) => j >= 0);
      if (!diff.length) return [];
      const count = Math.max(1, Math.floor(diff.length / 2));
      const pool = diff.slice();
      for (let i = pool.length - 1; i > 0; i--) {
        const k = Math.floor(rand() * (i + 1));
        [pool[i], pool[k]] = [pool[k], pool[i]];
      }
      return pool.slice(0, count).sort((x, y) => x - y);
    },
    /** Two distinct positions a < b (permutation two_cut_points and segment inversion). */
    drawSegment(n, rand) {
      const a = Math.floor(rand() * n);
      let b = Math.floor(rand() * (n - 1));
      if (b >= a) b += 1;
      return [Math.min(a, b), Math.max(a, b)];
    },

    /* Integer SBX: the bounded SBX of the real-coded lesson, clamped and rounded (impl/integer.py). */
    integerSbx(p1, p2, lower, upper, eta, probVar, rand, truncate) {
      const round = truncate ? Math.trunc : rint;
      return sbxPair(p1, p2, lower, upper, eta, probVar, rand).map((c) => c.map((v) => round(Math.min(upper, Math.max(lower, v)))));
    },

    /* Permutation operators (impl/_permutation_crossovers.py, _permutation_mutations.py). */
    pmxChild(a, b, cut1, cut2) {
      const map = new Map();
      for (let i = cut1; i <= cut2; i++) map.set(b[i], a[i]);
      return a.map((gene, i) => {
        if (i >= cut1 && i <= cut2) return b[i];
        let g = gene;
        let steps = 0;
        while (map.has(g) && steps <= map.size) {
          g = map.get(g);
          steps += 1;
        }
        return g;
      });
    },
    oxChild(donor, filler, lo, hi) {
      const n = donor.length;
      const out = new Array(n).fill(-1);
      const used = new Set();
      for (let i = lo; i < hi; i++) {
        out[i] = donor[i];
        used.add(donor[i]);
      }
      const rest = filler.filter((g) => !used.has(g));
      const positions = [];
      for (let i = hi; i < n; i++) positions.push(i);
      for (let i = 0; i < lo; i++) positions.push(i);
      positions.forEach((pos, k) => (out[pos] = rest[k]));
      return out;
    },
    cycle(p1, p2, start) {
      const pos = [];
      p1.forEach((g, i) => (pos[g] = i));
      const cyc = [];
      let idx = start;
      do {
        cyc.push(idx);
        idx = pos[p2[idx]];
      } while (idx !== start);
      const inCycle = new Set(cyc);
      return { cycle: cyc, children: [p2.map((g, i) => (inCycle.has(i) ? p1[i] : g)), p1.map((g, i) => (inCycle.has(i) ? p2[i] : g))] };
    },
    positionBased(own, other, keep) {
      const child = own.map((g, i) => (keep[i] ? g : -1));
      const used = new Set(child.filter((g) => g >= 0));
      const free = child.map((g, i) => (g < 0 ? i : -1)).filter((i) => i >= 0);
      let k = 0;
      for (const g of other) {
        if (k >= free.length) break;
        if (!used.has(g)) {
          child[free[k++]] = g;
          used.add(g);
        }
      }
      return child;
    },
    invert(row, lo, hi) {
      return row.slice(0, lo).concat(row.slice(lo, hi + 1).reverse(), row.slice(hi + 1));
    },
    swap(row, i, j) {
      const out = row.slice();
      [out[i], out[j]] = [out[j], out[i]];
      return out;
    },
    /** Move the gene at position i to position j (insert mutation). */
    insert(row, i, j) {
      const out = row.slice();
      const [gene] = out.splice(i, 1);
      out.splice(j, 0, gene);
      return out;
    },
    tourLength(cities, perm) {
      let total = 0;
      for (let k = 0; k < perm.length; k++) {
        const a = cities[perm[k]];
        const b = cities[perm[(k + 1) % perm.length]];
        total += Math.hypot(a[0] - b[0], a[1] - b[1]);
      }
      return total;
    },
    tourCrossings(cities, perm) {
      const n = perm.length;
      const seg = (k) => [cities[perm[k]], cities[perm[(k + 1) % n]]];
      const orient = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
      let count = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 2; j < n; j++) {
          if (i === 0 && j === n - 1) continue;
          const [a, b] = seg(i);
          const [c, d] = seg(j);
          if (orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0) count += 1;
        }
      }
      return count;
    },
    tourEdges(perm) {
      const n = perm.length;
      return perm.map((g, k) => {
        const h2 = perm[(k + 1) % n];
        return g < h2 ? g + "-" + h2 : h2 + "-" + g;
      });
    },
  };

  /* ---- Genetic algorithms and families of algorithms: the evolutionary loop and four survival rules ---- */

  /** The two-variable problem of the loop figure: ZDT1's front f₂ = 1 − √f₁, with g = 1 + x₂. */
  function gaEvaluate(x) {
    const g = 1 + x[1];
    return [x[0], g * (1 - Math.sqrt(x[0] / g))];
  }

  /** A small genetic algorithm recorded phase by phase: select, vary, evaluate, survive.
      mode "plus" is generational (μ + λ) with λ = μ, "comma" generational (μ, λ), "steady" steady-state (μ + 1).
      Parents come from binary tournaments on (front, crowding distance); VAMOS' default real-coded operators,
      SBX (probability 0.9, η = 20) and polynomial mutation (probability 1/n = 1/2, η = 20), create the
      children, and "plus" and "steady" keep the survivors of NSGA-II's elitist rule. */
  function gaLoop(mode, seed, popSize, budget) {
    const rand = mulberry32(seed);
    const N = popSize;
    let nextId = 1;
    const born = (x, f) => ({ id: nextId++, x, f });
    let pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push(born(x, gaEvaluate(x)));
    }
    let evaluations = N;
    const steps = [{ phase: "init", cycle: 0, pop, evaluations }];
    const lambda = mode === "steady" ? 1 : N;
    let cycle = 0;
    while (evaluations < budget) {
      cycle += 1;
      const count = Math.min(lambda, budget - evaluations);
      const r = rankAndCrowding(pop.map((p) => p.f));
      const tournaments = [];
      const tournament = () => {
        const a = Math.floor(rand() * N);
        let b = Math.floor(rand() * (N - 1));
        if (b >= a) b += 1;
        let w = a;
        let reason = "coin";
        if (r.rank[a] !== r.rank[b]) {
          w = r.rank[a] < r.rank[b] ? a : b;
          reason = "front";
        } else if (r.crowding[a] !== r.crowding[b]) {
          w = r.crowding[a] > r.crowding[b] ? a : b;
          reason = "crowding";
        } else if (rand() >= 0.5) w = b;
        tournaments.push({ a: pop[a].id, b: pop[b].id, winner: pop[w].id, reason });
        return pop[w];
      };
      const pairs = [];
      for (let k = 0; k < Math.ceil(count / 2); k++) pairs.push([tournament(), tournament()]);
      steps.push({ phase: "select", cycle, pop, rank: r.rank, crowding: r.crowding, tournaments, pairs, evaluations });
      const children = [];
      for (const [p1, p2] of pairs) {
        const xs = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 20, 0.5, rand) : [p1.x.slice(), p2.x.slice()];
        for (const x of xs) {
          polynomialMutation(x, 0, 1, 0.5, 20, rand);
          if (children.length < count) children.push({ id: nextId++, x, f: null, parents: [p1.id, p2.id] });
        }
      }
      steps.push({ phase: "vary", cycle, pop, rank: r.rank, pairs, children, evaluations });
      const evaluated = children.map((c) => Object.assign({}, c, { f: gaEvaluate(c.x) }));
      evaluations += evaluated.length;
      steps.push({ phase: "evaluate", cycle, pop, rank: r.rank, pairs, children: evaluated, evaluations });
      const merged = pop.concat(evaluated);
      const mr = rankAndCrowding(merged.map((p) => p.f));
      let keep;
      if (mode === "comma") {
        keep = evaluated.map((_, k) => N + k);
        if (keep.length < N) keep = keep.concat(nsga2Truncate(pop.map((p) => p.f), N - keep.length));
      } else keep = nsga2Truncate(merged.map((p) => p.f), N);
      const kept = keep.slice().sort((a, b) => a - b).map((i) => merged[i]);
      const removed = merged.filter((_, i) => !keep.includes(i));
      steps.push({ phase: "survive", cycle, pop, children: evaluated, merged, mergedRank: mr.rank, mergedCrowding: mr.crowding, kept, removed, evaluations });
      pop = kept;
    }
    steps.push({ phase: "done", cycle, pop, rank: rankAndCrowding(pop.map((p) => p.f)).rank, evaluations });
    return steps;
  }

  /** One NSGA-II generation of the loop figure's problem, recorded for the NSGA-II lesson: the population with
      its fronts and crowding distances, the binary tournaments, the children, the fronts of the merged
      population, the whole fronts that fit, the front that does not fit (null when the whole fronts fill the
      places exactly) with its crowding distances, and the survivors. It is gaLoop's first elitist cycle. */
  function nsga2Generation(seed, popSize) {
    const steps = gaLoop("plus", seed, popSize, 2 * popSize);
    const phase = (name) => steps.find((st) => st.phase === name);
    const chosen = phase("select");
    const merged = phase("survive").merged;
    const F = merged.map((p) => p.f);
    const sorted = rankAndCrowding(F);
    const whole = [];
    let used = 0;
    let split = null;
    for (const front of sorted.fronts) {
      if (used === popSize) break;
      if (used + front.length > popSize) {
        split = front;
        break;
      }
      whole.push(front);
      used += front.length;
    }
    return {
      popSize,
      pop: chosen.pop,
      rank: chosen.rank,
      crowding: chosen.crowding,
      tournaments: chosen.tournaments,
      children: phase("evaluate").children,
      merged,
      mergedRank: sorted.rank,
      mergedCrowding: sorted.crowding,
      fronts: sorted.fronts,
      whole,
      split,
      places: popSize - used,
      keep: nsga2Truncate(F, popSize),
    };
  }

  /** Which members of one population each family keeps, with N places:
      dominance-based, NSGA-II's fronts and crowding distance;
      decomposition-based, for each of N evenly spread weight vectors the member with the best Tchebycheff value,
      as MOEA/D's subproblems would if every member competed for every subproblem;
      indicator-based, SMS-EMOA's removal of the least hypervolume contributor of the last front, repeated
      (reference point nadir + 1.0 · (nadir − ideal) of the remaining members, VAMOS' default);
      swarm-based, every particle stays, and the leaders are the non-dominated ones. */
  function familySurvivors(F, N) {
    const n = F.length;
    const dominance = nsga2Truncate(F, N).sort((a, b) => a - b);
    const W = moeadWeights(N);
    const z = F[0].map((_, j) => Math.min(...F.map((f) => f[j])));
    const g = W.map((w) => F.map((f) => moeadAggregate("tchebycheff", f, w, z)));
    const best = g.map((row) => row.reduce((k, v, i) => (v < row[k] ? i : k), 0));
    const decomposition = [...new Set(best)].sort((a, b) => a - b);
    const alive = F.map((_, i) => i);
    const removed = [];
    let ref = null;
    while (alive.length > N) {
      const Q = alive.map((i) => F[i]);
      const top = [0, 1].map((j) => Math.max(...Q.map((q) => q[j])));
      const low = [0, 1].map((j) => Math.min(...Q.map((q) => q[j])));
      ref = top.map((t, j) => t + (t - low[j] > 0 ? t - low[j] : 1));
      const fronts = nonDominatedSort(Q).fronts;
      const last = fronts[fronts.length - 1];
      let out = last[0];
      if (last.length > 1) {
        const c = hvContributions2d(
          last.map((i) => Q[i]),
          ref
        ).contributions;
        out = last[c.reduce((k, v, i) => (v < c[k] ? i : k), 0)];
      }
      removed.push(alive[out]);
      alive.splice(out, 1);
    }
    const leaders = nonDominatedIndices(F);
    return {
      dominance,
      decomposition: { kept: decomposition, best, W, z, g },
      indicator: { kept: alive.slice(), removed, ref },
      swarm: { kept: F.map((_, i) => i), leaders },
      n,
    };
  }

  const algorithms = {
    fraceTest,
    fraceRace,
    hyperbandBrackets,
    hyperbandBudgets,
    fidelitySlice,
    operators: OPERATORS,
    ccmoSelect,
    ccmoGeneration,
    ctaeaSelect,
    ctaeaGeneration,
    topSelect,
    topGeneration,
    cmoeamsSelect,
    cmoeamsGeneration,
    CS_CANDIDATES,
    agemoeaSurvival,
    agemoeaCut,
    agemoeaGeneration,
    smpsoStep,
    smpsoGeneration,
    nsga3Associate,
    nsga3Niching,
    nsga3Generation,
    moeadWeights,
    moeadNeighbours,
    moeadAggregate,
    moeadReplace,
    moeadGeneration,
    spea2Fitness,
    spea2Select,
    spea2Generation,
    epsilonPair,
    ibeaSelect,
    ibeaGeneration,
    rveaSelect,
    rveaGeneration,
    mulberry32,
    dominates,
    nonDominatedSort,
    crowdingDistance,
    rankAndCrowding,
    selectSurvivors,
    nonDominatedIndices,
    hypervolume2d,
    hvContributions2d,
    smsemoaStep,
    smsemoaGeneration,
    tchebycheffContour,
    generationalDistance,
    igd,
    igdPlus,
    epsilonAdditive,
    spacing,
    spreadDelta,
    sbxPair,
    polynomialMutation,
    nsga2Truncate,
    createNSGAII,
    ctpViolation,
    constrainedDominates,
    binomial,
    dasDennis,
    attainmentSurface,
    mannWhitney,
    PROBLEMS,
    twoLayerLattice,
    twoLayerShared,
    overallViolation,
    constraintRanks,
    stochasticRankingOrder,
    BENCHMARK_FRONTS,
    BIAS_PROBLEMS,
    CTP_PROBLEMS,
    benchmarkFront,
    biasPoint,
    ctpTotalViolation,
    ctpFront,
    gaEvaluate,
    gaLoop,
    nsga2Generation,
    familySurvivors,
  };
  root.VamosFoundations = algorithms;

  if (typeof document === "undefined") return;

  /* ------------------------------------------------------------------ */
  /* DOM and SVG helpers                                                 */
  /* ------------------------------------------------------------------ */

  const SVG_NS = "http://www.w3.org/2000/svg";
  const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const RANK_CLASSES = ["vf-r0", "vf-r1", "vf-r2", "vf-r3", "vf-r4"];
  const reducedMotion = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  let uid = 0;

  function s(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const key of Object.keys(attrs)) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function h(tag, attrs, parent, text) {
    const node = document.createElement(tag);
    if (attrs) for (const key of Object.keys(attrs)) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function txt(parent, x, y, content, attrs) {
    const node = s("text", Object.assign({ x, y }, attrs || {}), parent);
    node.textContent = content;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmt(value, digits) {
    if (value === Infinity) return "∞";
    if (value === -Infinity) return "−∞";
    if (!Number.isFinite(value)) return "–";
    return value.toFixed(digits === undefined ? 3 : digits);
  }

  function pair(p, digits) {
    return "(" + fmt(p[0], digits === undefined ? 2 : digits) + ", " + fmt(p[1], digits === undefined ? 2 : digits) + ")";
  }

  function rankClass(rank) {
    return RANK_CLASSES[Math.min(rank, RANK_CLASSES.length - 1)];
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const err = raw / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  function tickFormat(v) {
    if (v === 0) return "0";
    if (Math.abs(v) >= 1000) return v.toLocaleString("en-US");
    return String(+v.toFixed(3));
  }

  function marker(layer, shape, x, y, r, cls) {
    if (shape === "triangle") {
      const k = r * 1.3;
      return s("path", { d: `M${x},${y - k}L${x + k * 0.866},${y + k * 0.5}L${x - k * 0.866},${y + k * 0.5}Z`, class: cls }, layer);
    }
    if (shape === "square") {
      return s("rect", { x: x - r * 0.88, y: y - r * 0.88, width: r * 1.76, height: r * 1.76, rx: 1.5, class: cls }, layer);
    }
    if (shape === "diamond") {
      const k = r * 1.25;
      return s("path", { d: `M${x},${y - k}L${x + k},${y}L${x},${y + k}L${x - k},${y}Z`, class: cls }, layer);
    }
    if (shape === "cross") {
      return s("path", { d: `M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`, class: cls }, layer);
    }
    return s("circle", { cx: x, cy: y, r, class: cls }, layer);
  }

  function legend(parent, items) {
    const list = h("ul", { class: "vf-legend" }, parent);
    for (const item of items) {
      const li = h("li", {}, list);
      const icon = s("svg", { viewBox: "0 0 14 14", "aria-hidden": "true" }, li);
      if (item.shape === "line") s("line", { x1: 1, y1: 7, x2: 13, y2: 7, class: item.cls }, icon);
      else if (item.shape === "region") s("rect", { x: 1, y: 1, width: 12, height: 12, rx: 2, class: item.cls }, icon);
      else marker(icon, item.shape, 7, 7, 4.2, item.cls);
      h("span", {}, li, item.label);
    }
    return list;
  }

  function controls(parent) {
    return h("div", { class: "vf-controls" }, parent);
  }

  function button(parent, label, onClick, primary) {
    const b = h("button", { type: "button", class: "vf-button" + (primary ? " vf-button--primary" : "") }, parent, label);
    b.addEventListener("click", onClick);
    return b;
  }

  function slider(parent, o) {
    const label = h("label", {}, parent);
    h("span", {}, label, o.label);
    const input = h("input", { type: "range", min: o.min, max: o.max, step: o.step }, label);
    input.value = o.value;
    const show = (v) => (o.format ? o.format(v) : String(v));
    const out = h("output", {}, label, show(o.value));
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      out.textContent = show(v);
      o.onInput(v);
    });
    return {
      input,
      set(v) {
        input.value = v;
        out.textContent = show(v);
      },
    };
  }

  function select(parent, o) {
    const label = h("label", {}, parent);
    h("span", {}, label, o.label);
    const node = h("select", {}, label);
    for (const [value, text] of o.options) {
      const opt = h("option", { value: String(value) }, node, text);
      if (String(value) === String(o.value)) opt.selected = true;
    }
    node.addEventListener("change", () => o.onChange(node.value));
    return node;
  }

  function checkbox(parent, o) {
    const label = h("label", {}, parent);
    const box = h("input", { type: "checkbox" }, label);
    box.checked = !!o.checked;
    h("span", {}, label, o.label);
    box.addEventListener("change", () => o.onChange(box.checked));
    return box;
  }

  function createPlot(parent, o) {
    const W = o.width || 360;
    const H = o.height || 280;
    const m = Object.assign({ l: 50, r: 14, t: 12, b: 42 }, o.margin || {});
    const wrap = h("div", { class: "vf-plot-wrap" }, parent);
    if (o.title) h("p", { class: "vf-title" }, wrap, o.title);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "vf-plot", role: "group", "aria-label": o.label || o.title || "" }, wrap);
    const clipId = "vf-clip-" + ++uid;
    const clipRect = s("rect", {}, s("clipPath", { id: clipId }, s("defs", {}, svg)));
    const axes = s("g", { "aria-hidden": "true" }, svg);
    const clipped = s("g", { "clip-path": `url(#${clipId})` }, svg);
    const plot = { svg, wrap, W, H, m, xd: o.x, yd: o.y, pw: W - m.l - m.r, ph: H - m.t - m.b };
    plot.layers = {
      back: s("g", {}, clipped),
      mid: s("g", {}, clipped),
      marks: s("g", {}, svg),
      points: s("g", {}, svg),
      top: s("g", {}, svg),
    };
    plot.x = (v) => m.l + ((v - plot.xd[0]) / (plot.xd[1] - plot.xd[0])) * plot.pw;
    plot.y = (v) => m.t + plot.ph - ((v - plot.yd[0]) / (plot.yd[1] - plot.yd[0])) * plot.ph;
    plot.xi = (px) => plot.xd[0] + ((px - m.l) / plot.pw) * (plot.xd[1] - plot.xd[0]);
    plot.yi = (py) => plot.yd[0] + ((m.t + plot.ph - py) / plot.ph) * (plot.yd[1] - plot.yd[0]);
    plot.drawAxes = function () {
      clear(axes);
      clipRect.setAttribute("x", m.l);
      clipRect.setAttribute("y", m.t);
      clipRect.setAttribute("width", plot.pw);
      clipRect.setAttribute("height", plot.ph);
      if (o.axes === false) return;
      const xt = o.xTicks || niceTicks(plot.xd[0], plot.xd[1], o.xCount || 5);
      const yt = o.yTicks || niceTicks(plot.yd[0], plot.yd[1], o.yCount || 5);
      for (const v of xt) {
        if (v < plot.xd[0] - 1e-9 || v > plot.xd[1] + 1e-9) continue;
        const px = plot.x(v);
        s("line", { x1: px, x2: px, y1: m.t, y2: m.t + plot.ph, class: "vf-grid-line" }, axes);
        txt(axes, px, m.t + plot.ph + 15, o.xFormat ? o.xFormat(v) : tickFormat(v), { "text-anchor": "middle" });
      }
      for (const v of yt) {
        if (v < plot.yd[0] - 1e-9 || v > plot.yd[1] + 1e-9) continue;
        const py = plot.y(v);
        s("line", { x1: m.l, x2: m.l + plot.pw, y1: py, y2: py, class: "vf-grid-line" }, axes);
        txt(axes, m.l - 6, py + 4, o.yFormat ? o.yFormat(v) : tickFormat(v), { "text-anchor": "end" });
      }
      s("path", { d: `M${m.l},${m.t}V${m.t + plot.ph}H${m.l + plot.pw}`, class: "vf-axis-line" }, axes);
      if (o.xLabel) txt(axes, m.l + plot.pw / 2, H - 6, o.xLabel, { "text-anchor": "middle", class: "vf-axis-label" });
      if (o.yLabel) {
        txt(axes, 0, 0, o.yLabel, {
          "text-anchor": "middle",
          class: "vf-axis-label",
          transform: `translate(12 ${m.t + plot.ph / 2}) rotate(-90)`,
        });
      }
    };
    plot.setDomain = function (x, y) {
      plot.xd = x;
      plot.yd = y;
      plot.drawAxes();
    };
    plot.pointer = function (evt) {
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const pt = svg.createSVGPoint();
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      const p = pt.matrixTransform(ctm.inverse());
      return { x: plot.xi(p.x), y: plot.yi(p.y) };
    };
    plot.line = function (layer, points, cls) {
      const d = points.map((p, i) => (i ? "L" : "M") + plot.x(p[0]).toFixed(2) + "," + plot.y(p[1]).toFixed(2)).join("");
      return s("path", { d, class: cls }, layer);
    };
    plot.rect = function (layer, x0, y0, x1, y1, cls) {
      const xa = plot.x(Math.min(x0, x1));
      const xb = plot.x(Math.max(x0, x1));
      const ya = plot.y(Math.max(y0, y1));
      const yb = plot.y(Math.min(y0, y1));
      return s("rect", { x: xa, y: ya, width: Math.max(0, xb - xa), height: Math.max(0, yb - ya), class: cls }, layer);
    };
    plot.drawAxes();
    return plot;
  }

  function tooltip(plot) {
    const tip = h("div", { class: "vf-tooltip", "aria-hidden": "true" }, plot.wrap);
    tip.hidden = true;
    return {
      show(content, px, py) {
        tip.textContent = content;
        tip.hidden = false;
        const box = plot.svg.getBoundingClientRect();
        const outer = plot.wrap.getBoundingClientRect();
        const scale = box.width / plot.W || 1;
        const x = box.left - outer.left + px * scale;
        let left = x + 12;
        if (left + tip.offsetWidth > plot.wrap.clientWidth) left = Math.max(0, x - tip.offsetWidth - 12);
        tip.style.left = left + "px";
        tip.style.top = Math.max(0, box.top - outer.top + py * scale - 34) + "px";
      },
      hide() {
        tip.hidden = true;
      },
    };
  }

  function attachTip(node, tip, content, px, py) {
    node.addEventListener("pointerenter", () => tip.show(typeof content === "function" ? content() : content, px, py));
    node.addEventListener("pointerleave", () => tip.hide());
  }

  /** A focusable point the reader can drag with a pointer or move with the arrow keys. */
  function handle(plot, o) {
    const g = s("g", { class: "vf-handle", tabindex: "0", role: "button", "aria-roledescription": "draggable point" }, plot.layers.points);
    s("circle", { r: 15, class: "vf-hit" }, g);
    s("circle", { r: (o.r || 6) + 4, class: "vf-focus" }, g);
    const mark = marker(g, o.shape || "circle", 0, 0, o.r || 6, o.cls);
    const bx = o.boundsX || plot.xd;
    const by = o.boundsY || plot.yd;
    const clamp = (v, b) => Math.min(b[1], Math.max(b[0], v));
    let dragging = false;
    const api = {
      node: g,
      mark,
      place() {
        const p = o.get();
        g.setAttribute("transform", `translate(${plot.x(p[0]).toFixed(2)} ${plot.y(p[1]).toFixed(2)})`);
        g.setAttribute("aria-label", (o.label ? o.label() : "Point") + " at " + pair(p) + ". Use the arrow keys to move it.");
      },
      setClass(cls) {
        mark.setAttribute("class", cls);
      },
    };
    g.addEventListener("pointerdown", (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      dragging = true;
      if (g.setPointerCapture) g.setPointerCapture(e.pointerId);
      if (g.focus) g.focus({ preventScroll: true });
      if (o.onSelect) o.onSelect();
    });
    g.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const p = plot.pointer(e);
      if (!p) return;
      o.set(clamp(p.x, bx), clamp(p.y, by));
    });
    const stop = () => {
      dragging = false;
    };
    g.addEventListener("pointerup", stop);
    g.addEventListener("pointercancel", stop);
    g.addEventListener("focus", () => {
      if (o.onSelect) o.onSelect();
    });
    g.addEventListener("keydown", (e) => {
      const keys = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
      const move = keys[e.key];
      if (!move) return;
      e.preventDefault();
      const p = o.get();
      const factor = e.shiftKey ? 0.1 : 0.02;
      o.set(clamp(p[0] + move[0] * factor * (bx[1] - bx[0]), bx), clamp(p[1] + move[1] * factor * (by[1] - by[0]), by));
    });
    api.place();
    return api;
  }

  function table(parent, headers, rows, numeric) {
    const t = h("table", {}, h("div", { class: "vf-table-wrap" }, parent));
    const tr = h("tr", {}, h("thead", {}, t));
    headers.forEach((label, i) => h("th", numeric && numeric[i] ? { class: "vf-num" } : {}, tr, label));
    const body = h("tbody", {}, t);
    for (const row of rows) {
      const r = h("tr", {}, body);
      row.forEach((cell, i) => h("td", numeric && numeric[i] ? { class: "vf-num" } : {}, r, cell));
    }
    return t;
  }

  /** Draw a sampled front, breaking the line where the front is disconnected. */
  function drawFront(plot, layer, pts, cls, maxGap) {
    let seg = [];
    const gap = maxGap || 3 / Math.max(pts.length, 2);
    pts.forEach((p, i) => {
      if (i && p[0] - pts[i - 1][0] > gap) {
        if (seg.length > 1) plot.line(layer, seg, cls);
        seg = [];
      }
      seg.push(p);
    });
    if (seg.length > 1) plot.line(layer, seg, cls);
  }

  /** Make a mark keyboard-focusable and give it back focus after a re-render. */
  function focusable(mark, label, state, index) {
    mark.setAttribute("tabindex", "0");
    mark.setAttribute("class", mark.getAttribute("class") + " vf-focusable");
    mark.setAttribute("aria-label", label);
    if (state.refocus === index) {
      state.refocus = -1;
      mark.focus({ preventScroll: true });
    }
  }

  function frontPoints(curve, lo, hi, n) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = lo + ((hi - lo) * i) / (n - 1);
      pts.push([t, curve(t)]);
    }
    return pts;
  }

  /* ------------------------------------------------------------------ */
  /* Figures                                                             */
  /* ------------------------------------------------------------------ */

  const widgets = {};

  /* B1: one design in decision space and its image in objective space. */
  widgets["decision-map"] = function (node) {
    const toy = (x1, x2) => {
      const g = 1 + x2;
      return [x1, g * (1 - Math.sqrt(x1 / g))];
    };
    const rand = mulberry32(7);
    const cloud = Array.from({ length: 260 }, () => [rand(), rand()]);
    const state = { x: [0.3, 0.6] };
    const ctl = controls(node);
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const dec = createPlot(body, { title: "Decision space", x: [0, 1], y: [-0.04, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision space with a draggable design" });
    const obj = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2], xLabel: "f₁(x)", yLabel: "f₂(x)", label: "Objective space showing the image of the design" });
    for (const c of cloud) {
      s("circle", { cx: dec.x(c[0]), cy: dec.y(c[1]), r: 1.8, class: "vf-cloud" }, dec.layers.back);
      const f = toy(c[0], c[1]);
      s("circle", { cx: obj.x(f[0]), cy: obj.y(f[1]), r: 1.8, class: "vf-cloud" }, obj.layers.back);
    }
    dec.line(dec.layers.mid, [[0, 0], [1, 0]], "vf-line-1");
    obj.line(obj.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 101), "vf-line-1");
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Random designs and their images", shape: "circle", cls: "vf-cloud" },
      { label: "Pareto set (left) and Pareto front (right)", shape: "line", cls: "vf-line-1" },
      { label: "Your design x and F(x)", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Same x₁ with x₂ = 0", shape: "circle", cls: "vf-pt-hollow" },
    ]);
    const hd = handle(dec, {
      cls: "vf-pt vf-pt-2",
      r: 7,
      label: () => "Design x",
      get: () => state.x,
      set: (a, b) => {
        state.x = [Math.max(0, Math.min(1, a)), Math.max(0, Math.min(1, b))];
        render();
      },
      boundsY: [0, 1],
    });
    function render() {
      hd.place();
      clear(obj.layers.marks);
      clear(dec.layers.top);
      const f = toy(state.x[0], state.x[1]);
      const best = toy(state.x[0], 0);
      if (state.x[1] > 0.005) {
        s("line", { x1: obj.x(f[0]), y1: obj.y(f[1]), x2: obj.x(best[0]), y2: obj.y(best[1]), class: "vf-line-thin vf-dashed" }, obj.layers.marks);
        marker(obj.layers.marks, "circle", obj.x(best[0]), obj.y(best[1]), 5, "vf-pt-hollow");
        marker(dec.layers.top, "circle", dec.x(state.x[0]), dec.y(0), 5, "vf-pt-hollow");
      }
      marker(obj.layers.marks, "circle", obj.x(f[0]), obj.y(f[1]), 7, "vf-pt vf-pt-2");
      txt(obj.layers.marks, obj.x(f[0]) + 10, obj.y(f[1]) - 8, "F(x)", { class: "vf-mark-label" });
      clear(readout);
      h("p", {}, readout, `x = ${pair(state.x)}  →  F(x) = ${pair(f)}.`);
      if (state.x[1] > 0.005) {
        h(
          "p",
          {},
          readout,
          `The design (${fmt(state.x[0], 2)}, 0.00) reaches F = ${pair(best)}: the same f₁ and a lower f₂. ` +
            "So x is dominated. Every design with x₂ > 0 is dominated in this way, which is why the Pareto set is the segment x₂ = 0."
        );
      } else {
        h("p", {}, readout, "x lies in the Pareto set: no other design lowers one objective without raising the other.");
      }
    }
    button(ctl, "Move to the Pareto set", () => {
      state.x = [state.x[0], 0];
      render();
    });
    button(ctl, "Random design", () => {
      state.x = [Math.round(Math.random() * 100) / 100, Math.round(Math.random() * 100) / 100];
      render();
    });
    render();
  };

  /* B1 and I6: weighted sum versus Tchebycheff scalarization. */
  widgets.scalarization = function (node) {
    const modes = (node.dataset.modes || "ws").split(",");
    const shapes = { convex: (t) => 1 - Math.sqrt(t), concave: (t) => 1 - t * t };
    const state = { shape: node.dataset.shape || "concave", mode: modes[0], w: 0.5, trail: [] };
    const ctl = controls(node);
    select(ctl, {
      label: "Front",
      options: [
        ["convex", "Convex (ZDT1)"],
        ["concave", "Non-convex (ZDT2)"],
      ],
      value: state.shape,
      onChange: (v) => {
        state.shape = v;
        state.trail = [];
        render();
      },
    });
    if (modes.length > 1) {
      select(ctl, {
        label: "Method",
        options: [
          ["ws", "Weighted sum"],
          ["tch", "Tchebycheff"],
        ],
        value: state.mode,
        onChange: (v) => {
          state.mode = v;
          state.trail = [];
          render();
        },
      });
    }
    slider(ctl, {
      label: "Weight w₁",
      min: 0,
      max: 1,
      step: 0.01,
      value: state.w,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.w = v;
        render();
      },
    });
    const sweepButton = button(ctl, "Sweep 21 weights", sweep, true);
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1.15], y: [0, 1.15], xLabel: "f₁", yLabel: "f₂", label: "Pareto front and the solution selected by a weight vector" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Pareto front", shape: "line", cls: "vf-front" },
      { label: "Contour of the scalarized objective", shape: "line", cls: "vf-line-2" },
      { label: "Selected solution", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Solutions found by the sweep", shape: "circle", cls: "vf-pt vf-pt-1" },
    ]);
    function front() {
      return frontPoints(shapes[state.shape], 0, 1, 401);
    }
    function solve(w, mode, pts) {
      let best = Infinity;
      let arg = null;
      for (const p of pts) {
        const v = mode === "ws" ? w * p[0] + (1 - w) * p[1] : Math.max(w * p[0], (1 - w) * p[1]);
        if (v < best - 1e-12) {
          best = v;
          arg = p;
        }
      }
      return { value: best, point: arg };
    }
    function render() {
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      const pts = front();
      plot.line(plot.layers.mid, pts, "vf-front");
      const w = state.w;
      const sol = solve(w, state.mode, pts);
      const c = sol.value;
      if (state.mode === "ws") {
        if (1 - w > 1e-9) {
          const yAt = (x) => (c - w * x) / (1 - w);
          plot.line(plot.layers.mid, [[-0.5, yAt(-0.5)], [2, yAt(2)]], "vf-line-2");
        } else {
          plot.line(plot.layers.mid, [[c / w, -1], [c / w, 3]], "vf-line-2");
        }
      } else {
        const contour = tchebycheffContour(w, c);
        const [cx, cy] = contour[1];
        plot.line(plot.layers.mid, contour, "vf-line-2");
        plot.line(plot.layers.mid, [[0, 0], [cx * 3, cy * 3]], "vf-line-thin vf-dashed");
      }
      for (const p of state.trail) marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 4.5, "vf-pt vf-pt-1");
      marker(plot.layers.marks, "circle", plot.x(sol.point[0]), plot.y(sol.point[1]), 7, "vf-pt vf-pt-2");
      clear(readout);
      const name = state.mode === "ws" ? "weighted sum w₁f₁ + w₂f₂" : "Tchebycheff max(w₁f₁, w₂f₂)";
      h("p", {}, readout, `Minimizing the ${name} with w = (${w.toFixed(2)}, ${(1 - w).toFixed(2)}) selects f = ${pair(sol.point)}.`);
      if (state.trail.length) {
        const distinct = new Set(state.trail.map((p) => p[0].toFixed(3))).size;
        h("p", {}, readout, `21 evenly spaced weights found ${distinct} distinct Pareto-optimal solution${distinct === 1 ? "" : "s"}.`);
        if (state.mode === "ws" && state.shape === "concave") {
          h("p", {}, readout, "On a non-convex front the weighted sum only ever returns the two extremes: the interior points are never minimizers (Das and Dennis, 1997).");
        }
      } else {
        h("p", {}, readout, "Move the weight, or sweep 21 weights to see which part of the front a scalarizing method can reach.");
      }
    }
    function sweep() {
      const weights = Array.from({ length: 21 }, (_, i) => i / 20);
      const pts = front();
      state.trail = [];
      if (reducedMotion) {
        state.trail = weights.map((w) => solve(w, state.mode, pts).point);
        render();
        return;
      }
      sweepButton.disabled = true;
      let i = 0;
      const tick = () => {
        state.w = weights[i];
        state.trail.push(solve(state.w, state.mode, pts).point);
        render();
        i += 1;
        if (i < weights.length) setTimeout(tick, 90);
        else sweepButton.disabled = false;
      };
      tick();
    }
    render();
  };

  /* B2: Pareto dominance explorer. */
  widgets.dominance = function (node) {
    const preset = [
      [0.1, 0.9],
      [0.22, 0.62],
      [0.35, 0.7],
      [0.4, 0.4],
      [0.55, 0.52],
      [0.62, 0.28],
      [0.75, 0.45],
      [0.86, 0.14],
      [0.3, 0.95],
      [0.68, 0.8],
    ];
    const state = { P: preset.map((p) => p.slice()), selected: 3, showFront: false };
    const ctl = controls(node);
    const picker = select(ctl, {
      label: "Selected point",
      options: state.P.map((_, i) => [i, LETTERS[i]]),
      value: state.selected,
      onChange: (v) => {
        state.selected = +v;
        render();
      },
    });
    checkbox(ctl, {
      label: "Show the non-dominated set, ideal and nadir points",
      checked: false,
      onChange: (v) => {
        state.showFront = v;
        render();
      },
    });
    button(ctl, "Random points", () => {
      state.P = state.P.map(() => [0.05 + Math.random() * 0.9, 0.05 + Math.random() * 0.9]);
      render();
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Ten objective vectors; drag them or move them with the arrow keys" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Region dominated by the selected point", shape: "region", cls: "vf-region-1" },
      { label: "Region that dominates it", shape: "region", cls: "vf-region-2" },
      { label: "Dominated by the selected point", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Dominates the selected point", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Incomparable", shape: "circle", cls: "vf-pt vf-pt-muted" },
    ]);
    const handles = state.P.map((_, i) =>
      handle(plot, {
        cls: "vf-pt vf-pt-muted",
        label: () => "Point " + LETTERS[i],
        get: () => state.P[i],
        set: (a, b) => {
          state.P[i] = [a, b];
          render();
        },
        onSelect: () => {
          if (state.selected !== i) {
            state.selected = i;
            picker.value = String(i);
            render();
          }
        },
      })
    );
    handles.forEach((hd, i) => {
      hd.node.addEventListener("pointerenter", () => {
        const p = state.P[i];
        tip.show(`${LETTERS[i]} ${pair(p)}`, plot.x(p[0]), plot.y(p[1]));
      });
      hd.node.addEventListener("pointerleave", () => tip.hide());
    });
    function render() {
      clear(plot.layers.back);
      clear(plot.layers.mid);
      clear(plot.layers.top);
      const P = state.P;
      const sel = P[state.selected];
      plot.rect(plot.layers.back, sel[0], sel[1], 1.5, 1.5, "vf-region-1");
      plot.rect(plot.layers.back, -0.5, -0.5, sel[0], sel[1], "vf-region-2");
      const dominatedBy = [];
      const dominating = [];
      const incomparable = [];
      P.forEach((p, i) => {
        const hd = handles[i];
        hd.place();
        hd.node.classList.toggle("vf-selected", i === state.selected);
        if (i === state.selected) {
          hd.setClass("vf-pt vf-pt-ink");
          return;
        }
        if (dominates(sel, p)) {
          dominatedBy.push(LETTERS[i]);
          hd.setClass("vf-pt vf-pt-1");
        } else if (dominates(p, sel)) {
          dominating.push(LETTERS[i]);
          hd.setClass("vf-pt vf-pt-2");
        } else {
          incomparable.push(LETTERS[i]);
          hd.setClass("vf-pt vf-pt-muted");
        }
      });
      P.forEach((p, i) => txt(plot.layers.top, plot.x(p[0]) + 9, plot.y(p[1]) - 7, LETTERS[i], { class: "vf-mark-label" }));
      const nd = nonDominatedIndices(P);
      if (state.showFront) {
        const F = nd.map((i) => P[i]).sort((a, b) => a[0] - b[0]);
        const stair = [[F[0][0], 1.5]];
        F.forEach((p, k) => {
          stair.push([p[0], p[1]]);
          stair.push([k + 1 < F.length ? F[k + 1][0] : 1.5, p[1]]);
        });
        plot.line(plot.layers.mid, stair, "vf-line-1");
        const ideal = [Math.min(...F.map((p) => p[0])), Math.min(...F.map((p) => p[1]))];
        const nadir = [Math.max(...F.map((p) => p[0])), Math.max(...F.map((p) => p[1]))];
        marker(plot.layers.top, "cross", plot.x(ideal[0]), plot.y(ideal[1]), 5, "vf-line-ink");
        txt(plot.layers.top, plot.x(ideal[0]) + 8, plot.y(ideal[1]) - 6, "ideal", { class: "vf-mark-label" });
        marker(plot.layers.top, "cross", plot.x(nadir[0]), plot.y(nadir[1]), 5, "vf-line-ink");
        txt(plot.layers.top, plot.x(nadir[0]) + 8, plot.y(nadir[1]) - 6, "nadir", { class: "vf-mark-label" });
      }
      clear(readout);
      const name = LETTERS[state.selected];
      h("p", {}, readout, `${name} = ${pair(sel)}.`);
      h("p", {}, readout, `${name} dominates: ${dominatedBy.join(", ") || "none"}.`);
      h("p", {}, readout, `${name} is dominated by: ${dominating.join(", ") || "none"}.`);
      h("p", {}, readout, `Incomparable with ${name}: ${incomparable.join(", ") || "none"}.`);
      h("p", {}, readout, `Non-dominated set: ${nd.map((i) => LETTERS[i]).join(", ")}.`);
    }
    render();
  };

  const QUALITY_INFO = {
    hv: { name: "HV ↑", better: "higher", aspects: "all four aspects" },
    igd: { name: "IGD ↓", better: "lower", aspects: "convergence, spread, uniformity" },
    igdp: { name: "IGD+ ↓", better: "lower", aspects: "convergence, spread, uniformity" },
    gd: { name: "GD ↓", better: "lower", aspects: "convergence only" },
    eps: { name: "Additive ε ↓", better: "lower", aspects: "convergence, worst-case spread" },
    spacing: { name: "Spacing ↓", better: "lower", aspects: "uniformity only" },
    spread: { name: "Spread Δ ↓", better: "lower", aspects: "spread and uniformity" },
    n: { name: "Points", better: "context", aspects: "cardinality" },
  };

  /* B3 and B4: approximation sets with controllable quality aspects. */
  widgets.quality = function (node) {
    const keys = (node.dataset.indicators || "hv,igd,igdp,gd,eps,spacing,spread,n").split(",");
    const curve = (t) => 1 - Math.sqrt(t);
    const R = frontPoints(curve, 0, 1, 201);
    const ref = [1.1, 1.1];
    const hvTrue = hypervolume2d(R, ref);
    const state = { gap: 0.05, extent: 1, cluster: 0, n: 12 };
    const ctl = controls(node);
    slider(ctl, { label: "Convergence gap", min: 0, max: 0.5, step: 0.01, value: state.gap, format: (v) => v.toFixed(2), onInput: (v) => ((state.gap = v), render()) });
    slider(ctl, { label: "Extent covered", min: 0.1, max: 1, step: 0.05, value: state.extent, format: (v) => Math.round(v * 100) + "%", onInput: (v) => ((state.extent = v), render()) });
    slider(ctl, { label: "Clustering", min: 0, max: 1, step: 0.05, value: state.cluster, format: (v) => v.toFixed(2), onInput: (v) => ((state.cluster = v), render()) });
    slider(ctl, { label: "Points", min: 2, max: 40, step: 1, value: state.n, format: (v) => String(v), onInput: (v) => ((state.n = v), render()) });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const plot = createPlot(body, { x: [0, 1.6], y: [0, 1.6], xLabel: "f₁", yLabel: "f₂", label: "Approximation set compared with the ZDT1 Pareto front" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Pareto front (ZDT1)", shape: "line", cls: "vf-front" },
      { label: "Approximation set A", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "HV reference point r = (1.1, 1.1)", shape: "cross", cls: "vf-line-ink" },
      { label: "Region dominated by A inside r", shape: "region", cls: "vf-region-1" },
    ]);
    plot.line(plot.layers.mid, R, "vf-front");
    function approximation() {
      const A = [];
      const start = (1 - state.extent) / 2;
      for (let i = 0; i < state.n; i++) {
        const t = state.n === 1 ? 0.5 : i / (state.n - 1);
        const warped = Math.pow(t, 1 + 4 * state.cluster);
        const f1 = start + state.extent * warped;
        A.push([f1 + state.gap, curve(f1) + state.gap]);
      }
      return A;
    }
    function render() {
      clear(plot.layers.back);
      clear(plot.layers.marks);
      const A = approximation();
      const inside = A.filter((p) => p[0] < ref[0] && p[1] < ref[1]).sort((a, b) => a[0] - b[0]);
      if (inside.length) {
        const poly = [[inside[0][0], ref[1]]];
        inside.forEach((p, k) => {
          poly.push(p);
          poly.push([k + 1 < inside.length ? inside[k + 1][0] : ref[0], p[1]]);
        });
        poly.push([ref[0], ref[1]]);
        plot.line(plot.layers.back, poly, "vf-region-1");
      }
      marker(plot.layers.marks, "cross", plot.x(ref[0]), plot.y(ref[1]), 5, "vf-line-ink");
      txt(plot.layers.marks, plot.x(ref[0]) + 8, plot.y(ref[1]) - 6, "r", { class: "vf-mark-label" });
      for (const p of A) marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 4.5, "vf-pt vf-pt-1");
      const values = {
        hv: `${fmt(hypervolume2d(A, ref))} (${fmt((100 * hypervolume2d(A, ref)) / hvTrue, 1)}%)`,
        igd: fmt(igd(A, R)),
        igdp: fmt(igdPlus(A, R)),
        gd: fmt(generationalDistance(A, R)),
        eps: fmt(epsilonAdditive(A, R)),
        spacing: fmt(spacing(A)),
        spread: fmt(spreadDelta(A, [[0, 1], [1, 0]])),
        n: String(A.length),
      };
      clear(readout);
      table(
        readout,
        ["Indicator", "Value", "Reacts to"],
        keys.map((k) => [QUALITY_INFO[k].name, values[k], QUALITY_INFO[k].aspects]),
        [false, true, false]
      );
      h("p", {}, readout, "↑ higher is better, ↓ lower is better. The HV percentage is relative to the hypervolume of the Pareto front itself.");
    }
    render();
  };

  /* B4 and I6: hypervolume and exclusive contributions. */
  widgets.hypervolume = function (node) {
    const smsemoa = node.dataset.smsemoa === "true";
    const state = {
      P: [
        [0.1, 0.85],
        [0.3, 0.52],
        [0.42, 0.44],
        [0.65, 0.22],
        [0.9, 0.08],
      ],
      ref: [1.05, 1.05],
      active: -1,
    };
    const names = state.P.map((_, i) => "P" + (i + 1));
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1.2], y: [0, 1.2], xLabel: "f₁", yLabel: "f₂", label: "Hypervolume of five draggable points and a draggable reference point" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Dominated region (the hypervolume)", shape: "region", cls: "vf-region-1" },
      { label: "Exclusive contribution of the focused point", shape: "region", cls: "vf-region-strong" },
      { label: "Point", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Contributes nothing", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Reference point r", shape: "diamond", cls: "vf-pt vf-pt-3" },
    ]);
    const handles = state.P.map((_, i) =>
      handle(plot, {
        cls: "vf-pt vf-pt-1",
        label: () => names[i],
        get: () => state.P[i],
        set: (a, b) => {
          state.P[i] = [a, b];
          render();
        },
        onSelect: () => {
          state.active = i;
          render();
        },
      })
    );
    handles.forEach((hd, i) => {
      hd.node.addEventListener("pointerenter", () => {
        state.active = i;
        render();
        const p = state.P[i];
        tip.show(`${names[i]}: contribution ${fmt(hvContributions2d(state.P, state.ref).contributions[i])}`, plot.x(p[0]), plot.y(p[1]));
      });
      hd.node.addEventListener("pointerleave", () => tip.hide());
    });
    const refHandle = handle(plot, {
      cls: "vf-pt vf-pt-3",
      shape: "diamond",
      r: 6,
      label: () => "Reference point r",
      get: () => state.ref,
      set: (a, b) => {
        state.ref = [a, b];
        render();
      },
    });
    function render() {
      clear(plot.layers.back);
      clear(plot.layers.mid);
      clear(plot.layers.top);
      const ref = state.ref;
      const inside = state.P.filter((p) => p[0] < ref[0] && p[1] < ref[1]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const nd = inside.filter((p) => !inside.some((q) => dominates(q, p)));
      if (nd.length) {
        const poly = [[nd[0][0], ref[1]]];
        nd.forEach((p, k) => {
          poly.push(p);
          poly.push([k + 1 < nd.length ? nd[k + 1][0] : ref[0], p[1]]);
        });
        poly.push([ref[0], ref[1]]);
        plot.line(plot.layers.back, poly, "vf-region-1");
      }
      const contrib = hvContributions2d(state.P, ref);
      if (state.active >= 0 && contrib.boxes[state.active]) {
        const b = contrib.boxes[state.active];
        plot.rect(plot.layers.mid, b[0], b[1], b[2], b[3], "vf-region-strong");
      }
      handles.forEach((hd, i) => {
        hd.place();
        hd.setClass(contrib.contributions[i] > 0 ? "vf-pt vf-pt-1" : "vf-pt-hollow");
        hd.node.classList.toggle("vf-selected", i === state.active);
        const p = state.P[i];
        txt(plot.layers.top, plot.x(p[0]) + 9, plot.y(p[1]) - 7, names[i], { class: "vf-mark-label" });
      });
      refHandle.place();
      txt(plot.layers.top, plot.x(ref[0]) + 9, plot.y(ref[1]) - 7, "r", { class: "vf-mark-label" });
      clear(readout);
      h("p", {}, readout, `HV = ${fmt(hypervolume2d(state.P, ref))} with r = ${pair(ref)}.`);
      table(
        readout,
        ["Point", "Objectives", "Exclusive contribution"],
        state.P.map((p, i) => [names[i], pair(p), fmt(contrib.contributions[i])]),
        [false, true, true]
      );
      if (smsemoa) {
        let worst = 0;
        contrib.contributions.forEach((c, i) => {
          if (c < contrib.contributions[worst]) worst = i;
        });
        h("p", {}, readout, `SMS-EMOA would discard ${names[worst]}: removing it loses the least hypervolume (${fmt(contrib.contributions[worst])}).`);
      } else {
        h("p", {}, readout, "Focus or hover a point to see the area that only it dominates. Drag r to see how the reference point changes the value and the weight of the extreme points.");
      }
    }
    render();
  };

  /* B4: GD, IGD, IGD+ and additive epsilon drawn as distances. */
  widgets["distance-indicators"] = function (node) {
    const R = frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 11);
    const state = {
      A: [
        [0.06, 0.9],
        [0.3, 0.6],
        [0.52, 0.42],
        [0.85, 0.2],
      ],
      mode: "igd",
    };
    const ctl = controls(node);
    select(ctl, {
      label: "Indicator",
      options: [
        ["gd", "GD: from A to the reference set"],
        ["igd", "IGD: from the reference set to A"],
        ["igdp", "IGD+: only where A is worse"],
        ["eps", "Additive ε: shift needed"],
      ],
      value: state.mode,
      onChange: (v) => {
        state.mode = v;
        render();
      },
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1.2], y: [0, 1.2], xLabel: "f₁", yLabel: "f₂", label: "Distance-based indicators between a draggable approximation set and a reference set" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Reference set R (11 points of the ZDT1 front)", shape: "diamond", cls: "vf-pt vf-pt-3" },
      { label: "Approximation set A", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Distance used by the indicator", shape: "line", cls: "vf-line-2" },
      { label: "A shifted by ε", shape: "circle", cls: "vf-pt-hollow" },
    ]);
    const handles = state.A.map((_, i) =>
      handle(plot, {
        cls: "vf-pt vf-pt-1",
        label: () => "Approximation point a" + (i + 1),
        get: () => state.A[i],
        set: (a, b) => {
          state.A[i] = [a, b];
          render();
        },
      })
    );
    function render() {
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      const A = state.A;
      const seg = (p, q) => s("line", { x1: plot.x(p[0]), y1: plot.y(p[1]), x2: plot.x(q[0]), y2: plot.y(q[1]), class: "vf-line-2" }, plot.layers.mid);
      let value;
      let text;
      if (state.mode === "gd") {
        A.forEach((a) => seg(a, R[nearest(a, R, (r, x) => euclidean(x, r)).index]));
        value = generationalDistance(A, R);
        text = "GD averages, over the points of A, the distance to the nearest reference point. A set with one well-converged point scores well even if it misses most of the front.";
      } else if (state.mode === "igd") {
        R.forEach((r) => seg(r, A[nearest(r, A, (a, x) => euclidean(a, x)).index]));
        value = igd(A, R);
        text = "IGD averages, over the reference points, the distance to the nearest point of A. Gaps in A leave some reference points far away.";
      } else if (state.mode === "igdp") {
        R.forEach((r) => {
          const a = A[nearest(r, A, (x, y) => dPlus(x, y)).index];
          seg(r, [r[0] + Math.max(a[0] - r[0], 0), r[1] + Math.max(a[1] - r[1], 0)]);
        });
        value = igdPlus(A, R);
        text = "IGD+ measures, from each reference point, only how much the nearest point of A is worse. A point that is better than a reference point adds nothing.";
      } else {
        value = epsilonAdditive(A, R);
        A.forEach((a) => {
          const shifted = [a[0] - value, a[1] - value];
          seg(a, shifted);
          marker(plot.layers.marks, "circle", plot.x(shifted[0]), plot.y(shifted[1]), 4, "vf-pt-hollow");
        });
        text = "Additive ε is the smallest amount every point of A must be moved towards the origin so that each reference point is weakly dominated. It is a worst-case measure.";
      }
      for (const r of R) marker(plot.layers.marks, "diamond", plot.x(r[0]), plot.y(r[1]), 4, "vf-pt vf-pt-3");
      handles.forEach((hd) => hd.place());
      clear(readout);
      h("p", {}, readout, `${QUALITY_INFO[state.mode].name.replace(" ↓", "")} = ${fmt(value)} (lower is better).`);
      h("p", {}, readout, text);
      h("p", {}, readout, "Drag the points of A, or focus one and use the arrow keys.");
    }
    render();
  };

  /* NSGA-II at a glance: one generation in eight steps, from the population to the next one. */
  widgets["nsga2-steps"] = function (node) {
    const N = 6;
    const TITLES = ["the population Pₜ", "mating selection", "variation", "evaluation and merge", "non-dominated sorting", "whole fronts", "crowding distance", "the next population Pₜ₊₁"];
    const state = { seed: Number(node.dataset.seed || 4679), step: 0 };
    let g = null;
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw populations until one needs the crowding distance to decide who survives.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        const candidate = nsga2Generation(seed, N);
        if (candidate.split && candidate.places > 0 && candidate.split.length > candidate.places) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the children" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, fronts and crowding distances of one NSGA-II generation" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another generation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Tournament winner, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Cuboid of a crowding distance", shape: "region", cls: "vf-cuboid" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = nsga2Generation(state.seed, N);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const ids = (rows) => rows.map((i) => g.merged[i].id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const frontsText = (fronts) => fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ");
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      txt(plot.layers.top, px + 9, py - 7, label, { class: "vf-mark-label" });
    }
    function render() {
      const step = state.step;
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const rows = [];
      const wins = new Map();
      g.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
      if (step <= 2) {
        const byId = new Map(g.pop.map((p) => [p.id, p]));
        if (step === 1) {
          for (const t of g.tournaments) fPlot.line(fPlot.layers.mid, [byId.get(t.a).f, byId.get(t.b).f], "vf-line-thin vf-dashed");
        }
        g.pop.forEach((p, i) => {
          const k = wins.get(p.id) || 0;
          const cls = "vf-pt " + rankClass(g.rank[i]);
          const label = String(p.id) + (step === 1 && k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", cls, label, step === 1 && k > 0);
          draw(fPlot, p.f, "circle", cls, label, step === 1 && k > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(g.rank[i] + 1), fmt(g.crowding[i], 2), step === 1 ? (k ? `won ${k}×` : "–") : "–"]);
        });
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((id) => xPlot.line(xPlot.layers.mid, [byId.get(id).x, c.x], "vf-line-thin vf-dashed"));
            draw(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "–", "–", "not evaluated"]);
          });
        }
      } else {
        const keep = new Set(g.keep);
        const whole = new Set(g.whole.flat());
        const split = new Set(g.split || []);
        if (step === 4) {
          g.fronts.forEach((front, k) => {
            const pts = front.map((i) => g.merged[i].f).sort((a, b) => a[0] - b[0]);
            if (pts.length > 1) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(k));
          });
        }
        if (step === 6 && g.split) {
          const order = g.split.slice().sort((a, b) => g.merged[a].f[0] - g.merged[b].f[0] || a - b);
          for (let k = 1; k < order.length - 1; k++) {
            const a = g.merged[order[k - 1]].f;
            const b = g.merged[order[k + 1]].f;
            fPlot.rect(fPlot.layers.back, a[0], a[1], b[0], b[1], "vf-cuboid");
          }
        }
        g.merged.forEach((p, i) => {
          const isChild = i >= N;
          const sorted = step >= 4;
          const out = (step === 5 || step === 6) && !whole.has(i) && !split.has(i);
          const removed = step === 7 && !keep.has(i);
          const cls = "vf-pt " + (sorted ? rankClass(g.mergedRank[i]) : "vf-pt-muted") + (out || removed ? " vf-faded" : "");
          const ring = ((step === 5 || step === 6) && whole.has(i)) || (step === 7 && keep.has(i));
          let label = String(p.id);
          if (step === 6 && split.has(i)) label += " · " + fmt(g.mergedCrowding[i], 2);
          if (removed) label += " ✕";
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, ring);
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, ring);
          let status = isChild ? "child" : "parent";
          if (step === 5 || step === 6) status = whole.has(i) ? "kept, whole front" : split.has(i) ? `competes for ${g.places} place${g.places === 1 ? "" : "s"}` : "discarded";
          if (step === 7) status = keep.has(i) ? "kept" : "removed";
          const crowd = step >= 6 && split.has(i) ? fmt(g.mergedCrowding[i], 2) : "–";
          rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), sorted ? String(g.mergedRank[i] + 1) : "?", crowd, status]);
        });
      }
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Crowding", "Status"], rows, [false, false, true, true, true, true, true, true, false]);
    }
    function describe(step) {
      const popIndex = (front) => front.map((i) => g.pop[i].id).sort((a, b) => a - b);
      if (step === 0) {
        const fronts = rankAndCrowding(g.pop.map((p) => p.f)).fronts;
        const text = fronts.map((front, k) => `front ${k + 1} = {${popIndex(front).join(", ")}}`).join(", ");
        return `${N} solutions of a two-variable problem whose Pareto front is ZDT1's: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂. They are already sorted into fronts, ${text}, and each has a crowding distance inside its front.`;
      }
      if (step === 1) {
        const why = { front: "better front", crowding: "same front, more isolated", coin: "full tie, coin" };
        const games = g.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner} (${why[t.reason]})`).join("; ");
        return `${N} binary tournaments pick ${N / 2} pairs of parents: ${games}. The better front wins; within a front, the larger crowding distance. A member can win several times, and ringed members won at least once.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 20) recombines each pair and polynomial mutation (probability 1/2 per variable, η = 20) perturbs each child: children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents in the decision space. Their objectives are not known yet.`;
      }
      if (step === 3) {
        return `The ${N} children are evaluated and merged with the ${N} parents: Rₜ = Pₜ ∪ Qₜ holds ${2 * N} solutions, which compete for ${N} places on equal terms. Keeping the parents in the competition is NSGA-II's elitism: a good solution is never lost to worse children.`;
      }
      if (step === 4) return `The merged population falls into ${g.fronts.length} fronts: ${frontsText(g.fronts)}. Lines join the members of each front; each front is non-dominated once the fronts before it are removed.`;
      if (step === 5) {
        const used = N - g.places;
        const fit = g.whole.length === 0 ? "No front fits entirely" : g.whole.length === 1 ? `Front 1 fits entirely, ${used} of ${N} places` : `Fronts 1 to ${g.whole.length} fit entirely, ${used} of ${N} places`;
        if (!g.split) return `${fit}: the whole fronts fill the places exactly, and the fronts after them are discarded.`;
        return `${fit}. Front ${g.whole.length + 1} has ${g.split.length} members for the ${g.places} remaining place${g.places === 1 ? "" : "s"}; the fronts after it are discarded without further work.`;
      }
      if (step === 6) {
        if (!g.split) return `The whole fronts filled the ${N} places exactly, so the crowding distance decides nothing in this generation.`;
        const k = g.whole.length + 1;
        const ext = g.split.filter((i) => g.mergedCrowding[i] === Infinity);
        const inner = g.split.filter((i) => g.mergedCrowding[i] !== Infinity).sort((a, b) => cmpDesc(g.mergedCrowding[a], g.mergedCrowding[b]) || a - b);
        const winners = g.split.filter((i) => g.keep.includes(i));
        const extText = ext.length ? `Its extremes, ${idList(ids(ext))}, get ∞. ` : "";
        const innerText = inner.length ? `Every other member gets the sum, over the two objectives, of a side of its cuboid (the box spanned by its two neighbours in the front) divided by the front's range in that objective: ${inner.map((i) => `${g.merged[i].id} · ${fmt(g.mergedCrowding[i], 2)}`).join(", ")}. ` : "";
        return `Inside front ${k} only. ${extText}${innerText}The ${g.places} largest distances win the remaining places: ${idList(ids(winners))}.`;
      }
      const kept = g.keep.slice().sort((a, b) => a - b);
      const parents = kept.filter((i) => i < N).length;
      const children = kept.length - parents;
      const removed = g.merged.map((_, i) => i).filter((i) => !g.keep.includes(i));
      return `Pₜ₊₁ = ${set(kept)}: ${parents} parent${parents === 1 ? "" : "s"} and ${children} child${children === 1 ? "" : "ren"} survive, and ${idList(ids(removed))} leave. Pₜ₊₁ starts the next generation, which the figure below repeats until the budget is spent.`;
    }
    build();
  };

  /* I1: NSGA-II running in the page. */
  widgets["nsga2-live"] = function (node) {
    const state = { problem: "zdt1", pop: 100, seed: 2026, playing: false, timer: null, history: [] };
    const nVar = 30;
    const maxGenerations = 250;
    const ref = [1.1, 1.1];
    let algo;
    let hvTrue;
    const ctl = controls(node);
    select(ctl, {
      label: "Problem",
      options: Object.keys(PROBLEMS).map((k) => [k, PROBLEMS[k].label]),
      value: state.problem,
      onChange: (v) => {
        state.problem = v;
        reset();
      },
    });
    select(ctl, {
      label: "Population",
      options: [
        [20, "20"],
        [50, "50"],
        [100, "100"],
      ],
      value: state.pop,
      onChange: (v) => {
        state.pop = +v;
        reset();
      },
    });
    const playButton = button(ctl, "Play", () => (state.playing ? pause() : play()), true);
    button(ctl, "One generation", () => {
      pause();
      advance();
    });
    button(ctl, "Reset", () => reset());
    button(ctl, "New seed", () => {
      state.seed = Math.floor(Math.random() * 1e6);
      reset();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const plot = createPlot(body, { title: "Population in objective space", x: [0, 1], y: [0, 6], xLabel: "f₁", yLabel: "f₂", label: "NSGA-II population in objective space" });
    const hvPlot = createPlot(body, {
      title: "Hypervolume relative to the true front",
      x: [0, maxGenerations * 100],
      y: [0, 1],
      xLabel: "Evaluations",
      yLabel: "HV / HV(front)",
      xCount: 4,
      yFormat: (v) => Math.round(v * 100) + "%",
      label: "Hypervolume of the non-dominated population as a share of the hypervolume of the true front",
    });
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Pareto front", shape: "line", cls: "vf-front" },
      { label: "Rank 1 (non-dominated)", shape: "circle", cls: "vf-pt vf-r0" },
      { label: "Ranks 2 and higher", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Hypervolume ratio", shape: "line", cls: "vf-line-1" },
    ]);
    function hvRatio() {
      const F = algo.F.filter((_, i) => algo.rank[i] === 0);
      return hypervolume2d(F, ref) / hvTrue;
    }
    const fronts = {};
    function trueFront() {
      if (!fronts[state.problem]) fronts[state.problem] = sampledFront(PROBLEMS[state.problem], 2001);
      return fronts[state.problem];
    }
    function reset() {
      pause();
      algo = createNSGAII(PROBLEMS[state.problem], state.pop, nVar, state.seed);
      hvTrue = hypervolume2d(trueFront(), ref);
      plot.yd = [0, 0];
      hvPlot.setDomain([0, maxGenerations * state.pop], [0, 1]);
      state.history = [[algo.evaluations, hvRatio()]];
      draw();
    }
    function draw() {
      const ymin = state.problem === "zdt3" ? -1 : 0;
      const top = Math.max(...algo.F.map((f) => f[1]));
      const ymax = top > 1.5 ? Math.min(8, Math.ceil(top * 1.05)) : 1.5;
      if (ymax !== plot.yd[1] || ymin !== plot.yd[0]) {
        plot.setDomain([0, 1], [ymin, ymax]);
        clear(plot.layers.mid);
        drawFront(plot, plot.layers.mid, trueFront(), "vf-front");
      }
      clear(plot.layers.marks);
      algo.F.forEach((f, i) => {
        if (algo.rank[i] > 0 && f[1] <= ymax) marker(plot.layers.marks, "circle", plot.x(f[0]), plot.y(f[1]), 3.5, "vf-pt vf-pt-muted");
      });
      algo.F.forEach((f, i) => {
        if (algo.rank[i] === 0 && f[1] <= ymax) marker(plot.layers.marks, "circle", plot.x(f[0]), plot.y(f[1]), 4, "vf-pt vf-r0");
      });
      clear(hvPlot.layers.mid);
      hvPlot.line(hvPlot.layers.mid, [[0, 1], [maxGenerations * state.pop, 1]], "vf-front");
      if (state.history.length > 1) hvPlot.line(hvPlot.layers.mid, state.history, "vf-line-1");
      const last = state.history[state.history.length - 1];
      clear(hvPlot.layers.marks);
      marker(hvPlot.layers.marks, "circle", hvPlot.x(last[0]), hvPlot.y(last[1]), 4, "vf-pt vf-pt-1");
      const nd = algo.rank.filter((r) => r === 0).length;
      clear(readout);
      h(
        "p",
        {},
        readout,
        `Generation ${algo.generation}: ${algo.evaluations.toLocaleString("en-US")} evaluations, ${nd} of ${state.pop} solutions in the first front, ` +
          `hypervolume ${fmt(100 * last[1], 1)}% of the true front's (r = (1.1, 1.1)). Seed ${state.seed}.`
      );
    }
    function advance() {
      if (algo.generation >= maxGenerations) {
        pause();
        return;
      }
      algo.step();
      state.history.push([algo.evaluations, hvRatio()]);
      draw();
    }
    function play() {
      if (algo.generation >= maxGenerations) reset();
      state.playing = true;
      playButton.textContent = "Pause";
      const loop = () => {
        if (!state.playing) return;
        advance();
        if (state.playing) state.timer = setTimeout(loop, 70);
      };
      loop();
    }
    function pause() {
      state.playing = false;
      playButton.textContent = "Play";
      if (state.timer) clearTimeout(state.timer);
      state.timer = null;
    }
    reset();
  };

  function randomFrontSet(rand, n, offsets) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const level = offsets[i % offsets.length];
      const t = 0.03 + 0.94 * rand();
      pts.push([+(t + level * 0.6).toFixed(2), +(1 - Math.sqrt(t) + level + 0.04 * rand()).toFixed(2)]);
    }
    return pts.map((p) => [Math.min(0.98, p[0]), Math.min(0.98, Math.max(0.02, p[1]))]);
  }

  /* I2: fast non-dominated sorting, one bookkeeping step at a time. */
  widgets.sorting = function (node) {
    const preset = [
      [0.08, 0.78],
      [0.2, 0.5],
      [0.45, 0.3],
      [0.8, 0.1],
      [0.18, 0.92],
      [0.32, 0.62],
      [0.55, 0.45],
      [0.9, 0.3],
      [0.45, 0.85],
      [0.66, 0.62],
      [0.95, 0.55],
      [0.75, 0.9],
    ];
    const state = { P: preset, step: 0, focus: -1, refocus: -1 };
    let steps = [];
    const ctl = controls(node);
    const back = button(ctl, "Back", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New points", () => {
      const rand = mulberry32(Math.floor(Math.random() * 1e6));
      state.P = randomFrontSet(rand, 12, [0, 0.15, 0.3]);
      build();
      go(0);
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "f₁", yLabel: "f₂", label: "Twelve points sorted into non-domination fronts" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Front 1, 2, 3, … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Not yet assigned", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Front being extracted", shape: "circle", cls: "vf-ring" },
      { label: "S_p: points dominated by the focused point", shape: "line", cls: "vf-line-thin" },
    ]);
    function build() {
      const sorted = nonDominatedSort(state.P);
      const counts = sorted.count.slice();
      const rank = new Array(state.P.length).fill(-1);
      steps = [
        {
          counts: counts.slice(),
          rank: rank.slice(),
          current: [],
          text: "Compare every pair once. For each point p, n_p (the number next to it) counts the points that dominate p, and S_p stores the points that p dominates. Focus a point to see its S_p.",
        },
      ];
      sorted.fronts.forEach((front, k) => {
        front.forEach((p) => {
          rank[p] = k;
        });
        steps.push({
          counts: counts.slice(),
          rank: rank.slice(),
          current: front,
          text: `Front ${k + 1}: the unassigned points with n_p = 0 are ${front.map((p) => LETTERS[p]).join(", ")}. Nothing left dominates them.`,
        });
        for (const p of front) for (const q of sorted.dominatedSets[p]) counts[q] -= 1;
        const last = k === sorted.fronts.length - 1;
        steps.push({
          counts: counts.slice(),
          rank: rank.slice(),
          current: [],
          text: last
            ? `Done: ${sorted.fronts.length} fronts. Each pair was compared once, so the cost is O(M N²) for N points and M objectives.`
            : `Remove front ${k + 1}: for each of its points p, decrease n_q by one for every q in S_p. The points that reach 0 form front ${k + 2}.`,
        });
      });
      state.dominated = sorted.dominatedSets;
    }
    function go(i) {
      state.step = Math.max(0, Math.min(steps.length - 1, i));
      render();
    }
    function render() {
      const st = steps[state.step];
      back.disabled = state.step === 0;
      next.disabled = state.step === steps.length - 1;
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      if (state.focus >= 0) {
        const p = state.P[state.focus];
        for (const q of state.dominated[state.focus]) {
          s("line", { x1: plot.x(p[0]), y1: plot.y(p[1]), x2: plot.x(state.P[q][0]), y2: plot.y(state.P[q][1]), class: "vf-line-thin" }, plot.layers.mid);
        }
      }
      state.P.forEach((p, i) => {
        const px = plot.x(p[0]);
        const py = plot.y(p[1]);
        const assigned = st.rank[i] >= 0;
        const hit = s("circle", { cx: px, cy: py, r: 13, class: "vf-hit" }, plot.layers.marks);
        const mark = marker(plot.layers.marks, "circle", px, py, 6, assigned ? "vf-pt " + rankClass(st.rank[i]) : "vf-pt-hollow");
        const show = (fromKeyboard) => {
          tip.show(`${LETTERS[i]} ${pair(p)} · dominates ${state.dominated[i].map((q) => LETTERS[q]).join(", ") || "nothing"}`, px, py);
          if (state.focus !== i) {
            state.focus = i;
            if (fromKeyboard) state.refocus = i;
            render();
          }
        };
        hit.addEventListener("pointerenter", () => show(false));
        hit.addEventListener("pointerleave", () => tip.hide());
        mark.addEventListener("pointerenter", () => show(false));
        mark.addEventListener("focus", () => show(true));
        mark.addEventListener("blur", () => tip.hide());
        focusable(mark, `${LETTERS[i]} ${pair(p)}, n_p = ${st.counts[i]}${assigned ? ", front " + (st.rank[i] + 1) : ""}, dominates ${state.dominated[i].map((q) => LETTERS[q]).join(", ") || "nothing"}`, state, i);
        if (st.current.includes(i)) marker(plot.layers.marks, "circle", px, py, 10, "vf-ring");
        const counter = assigned && !st.current.includes(i) ? "" : " · " + st.counts[i];
        txt(plot.layers.top, px + 10, py - 8, LETTERS[i] + counter, { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", { class: "vf-step-text" }, readout, `Step ${state.step + 1} of ${steps.length}. ${st.text}`);
      const fronts = [];
      st.rank.forEach((r, i) => {
        if (r < 0) return;
        (fronts[r] = fronts[r] || []).push(LETTERS[i]);
      });
      if (fronts.length) table(readout, ["Front", "Points"], fronts.map((f, k) => [String(k + 1), f.join(", ")]));
    }
    build();
    go(0);
  };

  /* I3: crowding distance and one-shot truncation. */
  widgets.crowding = function (node) {
    const curve = (t) => 1 - Math.sqrt(t);
    const preset = [0, 0.04, 0.09, 0.16, 0.38, 0.47, 0.55, 0.8, 1].map((t) => [t, curve(t)]);
    const state = { P: preset, focus: 4, keep: 9, refocus: -1 };
    const ctl = controls(node);
    const keepSlider = slider(ctl, {
      label: "Places left for this front",
      min: 2,
      max: 9,
      step: 1,
      value: 9,
      format: (v) => String(v),
      onInput: (v) => {
        state.keep = v;
        render();
      },
    });
    button(ctl, "New front", () => {
      const rand = mulberry32(Math.floor(Math.random() * 1e6));
      const ts = [0, 1].concat(Array.from({ length: 7 }, () => +rand().toFixed(2))).sort((a, b) => a - b);
      state.P = ts.map((t) => [t, curve(t)]);
      state.keep = 9;
      keepSlider.set(9);
      render();
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [-0.05, 1.08], y: [-0.05, 1.08], xLabel: "f₁", yLabel: "f₂", label: "Crowding distance in one front" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Kept", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Discarded by crowding", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Cuboid of the focused point", shape: "region", cls: "vf-region-2" },
    ]);
    function render() {
      const P = state.P;
      const front = P.map((_, i) => i);
      const d = crowdingDistance(P, front);
      const order = front.slice().sort((a, b) => cmpDesc(d[a], d[b]) || a - b);
      const kept = new Set(order.slice(0, state.keep));
      clear(plot.layers.back);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      const byF1 = front.slice().sort((a, b) => P[a][0] - P[b][0] || a - b);
      const pos = byF1.indexOf(state.focus);
      const f1span = P[byF1[byF1.length - 1]][0] - P[byF1[0]][0];
      const f2span = Math.max(...P.map((p) => p[1])) - Math.min(...P.map((p) => p[1]));
      let detail;
      if (pos > 0 && pos < byF1.length - 1) {
        const a = P[byF1[pos - 1]];
        const b = P[byF1[pos + 1]];
        plot.rect(plot.layers.back, a[0], a[1], b[0], b[1], "vf-region-2");
        const c1 = (b[0] - a[0]) / f1span;
        const c2 = (a[1] - b[1]) / f2span;
        detail = `${LETTERS[state.focus]}: (${fmt(b[0], 2)} − ${fmt(a[0], 2)}) / ${fmt(f1span, 2)} + (${fmt(a[1], 2)} − ${fmt(b[1], 2)}) / ${fmt(f2span, 2)} = ${fmt(c1)} + ${fmt(c2)} = ${fmt(c1 + c2)}.`;
      } else {
        detail = `${LETTERS[state.focus]} is a boundary point: its distance is ∞, so the extremes of the front are always kept.`;
      }
      P.forEach((p, i) => {
        const px = plot.x(p[0]);
        const py = plot.y(p[1]);
        const hit = s("circle", { cx: px, cy: py, r: 13, class: "vf-hit" }, plot.layers.marks);
        const mark = marker(plot.layers.marks, "circle", px, py, 6, kept.has(i) ? "vf-pt vf-pt-1" : "vf-pt-hollow");
        const pick = (fromKeyboard) => {
          if (state.focus !== i) {
            state.focus = i;
            if (fromKeyboard) state.refocus = i;
            render();
          }
        };
        hit.addEventListener("pointerenter", () => pick(false));
        mark.addEventListener("pointerenter", () => pick(false));
        mark.addEventListener("focus", () => pick(true));
        focusable(mark, `${LETTERS[i]}, crowding distance ${fmt(d[i])}${kept.has(i) ? ", kept" : ", discarded"}`, state, i);
        if (i === state.focus) marker(plot.layers.marks, "circle", px, py, 10, "vf-ring");
        txt(plot.layers.top, px + 9, py - 8, LETTERS[i], { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", { class: "vf-step-text" }, readout, detail);
      table(
        readout,
        ["Point", "Crowding distance", "Status"],
        order.map((i) => [LETTERS[i], fmt(d[i]), kept.has(i) ? "kept" : "discarded"]),
        [false, true, false]
      );
    }
    render();
  };

  /* I4: offspring distributions of SBX and polynomial mutation. */
  widgets.variation = function (node) {
    const mode = node.dataset.mode || "sbx";
    const state = { p1: 0.3, p2: 0.6, x: 0.3, eta: 20 };
    const samples = 20000;
    const bins = 50;
    const ctl = controls(node);
    if (mode === "sbx") {
      slider(ctl, { label: "Parent 1", min: 0, max: 1, step: 0.01, value: state.p1, format: (v) => v.toFixed(2), onInput: (v) => ((state.p1 = v), render()) });
      slider(ctl, { label: "Parent 2", min: 0, max: 1, step: 0.01, value: state.p2, format: (v) => v.toFixed(2), onInput: (v) => ((state.p2 = v), render()) });
    } else {
      slider(ctl, { label: "Parent", min: 0, max: 1, step: 0.01, value: state.x, format: (v) => v.toFixed(2), onInput: (v) => ((state.x = v), render()) });
    }
    slider(ctl, {
      label: mode === "sbx" ? "Distribution index η_c" : "Distribution index η_m",
      min: 0,
      max: 100,
      step: 1,
      value: state.eta,
      format: (v) => String(v),
      onInput: (v) => ((state.eta = v), render()),
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, {
      x: [0, 1],
      y: [0, 1],
      xLabel: "Value of the variable (bounds [0, 1])",
      yLabel: "Share of offspring",
      yFormat: (v) => Math.round(v * 100) + "%",
      label: mode === "sbx" ? "Distribution of SBX offspring" : "Distribution of polynomial mutation results",
    });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: mode === "sbx" ? "Offspring (20,000 crossovers)" : "Mutated values (20,000 mutations)", shape: "region", cls: "vf-bar-1" },
      { label: mode === "sbx" ? "Parents" : "Parent", shape: "line", cls: "vf-line-2" },
    ]);
    function render() {
      const rand = mulberry32(11);
      const counts = new Array(bins).fill(0);
      const parents = mode === "sbx" ? [state.p1, state.p2] : [state.x];
      let near = 0;
      for (let i = 0; i < samples; i++) {
        const values = mode === "sbx" ? sbxPair([state.p1], [state.p2], 0, 1, state.eta, 1, rand).map((c) => c[0]) : [polynomialMutation([state.x], 0, 1, 1, state.eta, rand)[0]];
        for (const v of values) {
          counts[Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))] += 1;
          if (parents.some((p) => Math.abs(v - p) <= 0.05)) near += 1;
        }
      }
      const total = samples * (mode === "sbx" ? 2 : 1);
      const shares = counts.map((c) => c / total);
      const top = Math.max(0.05, Math.ceil(Math.max(...shares) * 20) / 20);
      plot.setDomain([0, 1], [0, top]);
      clear(plot.layers.mid);
      clear(plot.layers.top);
      shares.forEach((v, i) => {
        if (v <= 0) return;
        const x0 = plot.x(i / bins) + 0.5;
        const x1 = plot.x((i + 1) / bins) - 0.5;
        s("rect", { x: x0, y: plot.y(v), width: Math.max(0.5, x1 - x0), height: plot.y(0) - plot.y(v), class: "vf-bar-1" }, plot.layers.mid);
      });
      parents.forEach((p, k) => {
        s("line", { x1: plot.x(p), x2: plot.x(p), y1: plot.y(0), y2: plot.y(top), class: "vf-line-2" }, plot.layers.top);
        txt(plot.layers.top, plot.x(p) + 4, plot.m.t + 12 + 14 * k, mode === "sbx" ? "p" + (k + 1) : "parent", { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", {}, readout, `${fmt((100 * near) / total, 1)}% of the ${mode === "sbx" ? "offspring" : "mutated values"} lie within ±0.05 of ${mode === "sbx" ? "a parent" : "the parent"} (η = ${state.eta}).`);
      h(
        "p",
        {},
        readout,
        mode === "sbx"
          ? "A large η_c keeps children close to their parents; a small η_c spreads them out. The spread also scales with the distance between the parents, and the bounded version never leaves [0, 1]."
          : "A large η_m makes small perturbations; a small η_m makes large jumps. Near a bound the distribution is squeezed so that the result stays inside it."
      );
    }
    render();
  };

  /* I5: one generation of elitist survival, step by step. */
  widgets.survival = function (node) {
    const preset = {
      parents: [
        [0.2, 0.3],
        [0.25, 0.7],
        [0.4, 0.48],
        [0.9, 0.12],
        [0.5, 0.65],
        [0.95, 0.35],
      ],
      offspring: [
        [0.45, 0.1],
        [0.22, 0.9],
        [0.32, 0.55],
        [0.6, 0.3],
        [0.35, 0.9],
        [0.7, 0.5],
      ],
    };
    const state = { data: preset, step: 0 };
    let steps = [];
    const ctl = controls(node);
    const back = button(ctl, "Back", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New candidates", () => {
      const rand = mulberry32(Math.floor(Math.random() * 1e6));
      const pts = randomFrontSet(rand, 12, [0, 0.12, 0.24, 0.06, 0.18, 0.3]);
      state.data = { parents: pts.slice(0, 6), offspring: pts.slice(6) };
      build();
      go(0);
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "f₁", yLabel: "f₂", label: "Parents and offspring competing for the next population" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Parent (P)", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Offspring (O)", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Survivor", shape: "circle", cls: "vf-ring" },
    ]);
    const N = 6;
    function build() {
      const F = state.data.parents.concat(state.data.offspring);
      const names = state.data.parents.map((_, i) => "P" + (i + 1)).concat(state.data.offspring.map((_, i) => "O" + (i + 1)));
      const r = rankAndCrowding(F);
      const survivors = nsga2Truncate(F, N);
      const list = (rows) => rows.map((i) => names[i]).join(", ");
      steps = [];
      steps.push({ F, names, rank: null, kept: [], crowd: null, faded: [], text: `Parents P (circles) and offspring O (triangles): ${F.length} candidates compete for N = ${N} places. Both groups are treated alike: this is what makes NSGA-II elitist.` });
      steps.push({
        F,
        names,
        rank: r.rank,
        kept: [],
        crowd: null,
        faded: [],
        text: "Sort the merged population into fronts: " + r.fronts.map((f, k) => `front ${k + 1} = {${list(f)}}`).join("; ") + ".",
      });
      const kept = [];
      let k = 0;
      for (; k < r.fronts.length; k++) {
        const front = r.fronts[k];
        if (kept.length + front.length > N) break;
        kept.push(...front);
        steps.push({ F, names, rank: r.rank, kept: kept.slice(), crowd: null, faded: [], text: `Front ${k + 1} fits entirely: ${kept.length} of ${N} places are now used.` });
        if (kept.length === N) break;
      }
      if (kept.length < N && k < r.fronts.length) {
        const front = r.fronts[k];
        const left = N - kept.length;
        steps.push({
          F,
          names,
          rank: r.rank,
          kept: kept.slice(),
          crowd: front,
          faded: [],
          text: `Front ${k + 1} has ${front.length} points but only ${left} place${left === 1 ? "" : "s"} remain. Compute the crowding distance inside front ${k + 1} (shown next to each point).`,
        });
        const chosen = front.filter((i) => survivors.includes(i));
        const dropped = front.filter((i) => !survivors.includes(i));
        steps.push({
          F,
          names,
          rank: r.rank,
          kept: survivors.slice(),
          crowd: front,
          faded: dropped,
          text: `Keep the ${left} largest distances: ${list(chosen)}. Discard ${list(dropped)}, the most crowded.`,
        });
      }
      const parentsKept = survivors.filter((i) => i < 6).length;
      steps.push({
        F,
        names,
        rank: r.rank,
        kept: survivors.slice(),
        crowd: null,
        faded: F.map((_, i) => i).filter((i) => !survivors.includes(i)),
        text: `New population: ${list(survivors.slice().sort((a, b) => a - b))}. ${parentsKept} parent${parentsKept === 1 ? "" : "s"} and ${N - parentsKept} offspring survive; later fronts were never needed.`,
      });
      state.crowding = r.crowding;
    }
    function go(i) {
      state.step = Math.max(0, Math.min(steps.length - 1, i));
      render();
    }
    function render() {
      const st = steps[state.step];
      back.disabled = state.step === 0;
      next.disabled = state.step === steps.length - 1;
      clear(plot.layers.marks);
      clear(plot.layers.top);
      st.F.forEach((p, i) => {
        const px = plot.x(p[0]);
        const py = plot.y(p[1]);
        const cls = (st.rank ? "vf-pt " + rankClass(st.rank[i]) : "vf-pt vf-pt-muted") + (st.faded.includes(i) ? " vf-faded" : "");
        marker(plot.layers.marks, i < 6 ? "circle" : "triangle", px, py, 6, cls);
        if (st.kept.includes(i)) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
        let label = st.names[i];
        if (st.crowd && st.crowd.includes(i)) label += " · " + fmt(state.crowding[i], 2);
        txt(plot.layers.top, px + 10, py - 8, label, { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", { class: "vf-step-text" }, readout, `Step ${state.step + 1} of ${steps.length}. ${st.text}`);
    }
    build();
    go(0);
  };

  /* A1: the share of non-dominated points as objectives are added. */
  widgets["dominance-resistance"] = function (node) {
    const state = { n: 100, results: [], token: 0 };
    const ms = Array.from({ length: 14 }, (_, i) => i + 2);
    const ctl = controls(node);
    select(ctl, {
      label: "Random points",
      options: [
        [50, "50"],
        [100, "100"],
        [200, "200"],
      ],
      value: state.n,
      onChange: (v) => {
        state.n = +v;
        compute();
      },
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, {
      x: [2, 15],
      y: [0, 1],
      xLabel: "Number of objectives m",
      yLabel: "Non-dominated share",
      yFormat: (v) => Math.round(v * 100) + "%",
      xTicks: [2, 4, 6, 8, 10, 12, 14],
      label: "Share of mutually non-dominated points among uniform random points as the number of objectives grows",
    });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    function compute() {
      const token = ++state.token;
      state.results = [];
      const trials = 10;
      let index = 0;
      const work = () => {
        if (token !== state.token) return;
        const m = ms[index];
        const rand = mulberry32(1000 + m);
        let share = 0;
        for (let t = 0; t < trials; t++) {
          const P = Array.from({ length: state.n }, () => Array.from({ length: m }, rand));
          share += nonDominatedIndices(P).length / state.n;
        }
        state.results.push([m, share / trials]);
        render();
        index += 1;
        if (index < ms.length) setTimeout(work, 0);
      };
      work();
    }
    function render() {
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      if (state.results.length > 1) plot.line(plot.layers.mid, state.results, "vf-line-1");
      state.results.forEach((r) => {
        const mark = marker(plot.layers.marks, "circle", plot.x(r[0]), plot.y(r[1]), 4, "vf-pt vf-pt-1");
        const hit = s("circle", { cx: plot.x(r[0]), cy: plot.y(r[1]), r: 12, class: "vf-hit" }, plot.layers.marks);
        attachTip(hit, tip, `m = ${r[0]}: ${fmt(100 * r[1], 1)}% non-dominated`, plot.x(r[0]), plot.y(r[1]));
        mark.setAttribute("aria-hidden", "true");
      });
      clear(readout);
      h("p", {}, readout, `Average over 10 samples of ${state.n} points drawn uniformly in [0, 1]^m.`);
      const pick = state.results.filter((r) => [2, 3, 5, 8, 10, 15].includes(r[0]));
      table(readout, ["m", "Non-dominated"], pick.map((r) => [String(r[0]), fmt(100 * r[1], 1) + "%"]), [false, true]);
    }
    compute();
  };

  /* A1: Das and Dennis reference directions. */
  widgets["reference-directions"] = function (node) {
    const state = { m: 3, p: 12 };
    const ctl = controls(node);
    select(ctl, {
      label: "Objectives m",
      options: Array.from({ length: 9 }, (_, i) => [i + 2, String(i + 2)]),
      value: state.m,
      onChange: (v) => {
        state.m = +v;
        render();
      },
    });
    slider(ctl, { label: "Divisions p", min: 1, max: 30, step: 1, value: state.p, format: (v) => String(v), onInput: (v) => ((state.p = v), render()) });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const holder = h("div", {}, body);
    const readout = h("div", { class: "vf-readout" }, body);
    function render() {
      clear(holder);
      const H = binomial(state.p + state.m - 1, state.m - 1);
      if (state.m === 2) {
        const plot = createPlot(holder, { x: [0, 1.05], y: [0, 1.05], xLabel: "w₁", yLabel: "w₂", label: "Reference directions for two objectives" });
        for (const w of dasDennis(2, state.p)) {
          plot.line(plot.layers.mid, [[0, 0], [w[0] * 1.5, w[1] * 1.5]], "vf-line-thin");
          marker(plot.layers.marks, "circle", plot.x(w[0]), plot.y(w[1]), 4, "vf-pt vf-pt-1");
        }
      } else if (state.m === 3) {
        const plot = createPlot(holder, { x: [-0.08, 1.08], y: [-0.1, 0.95], axes: false, margin: { l: 10, r: 10, t: 10, b: 10 }, width: 340, height: 300, label: "Reference points on the unit simplex for three objectives" });
        const V = [
          [0, 0],
          [1, 0],
          [0.5, 0.866],
        ];
        plot.line(plot.layers.mid, [V[0], V[1], V[2], V[0]], "vf-line-ink");
        ["w₁ = 1", "w₂ = 1", "w₃ = 1"].forEach((label, i) => txt(plot.layers.top, plot.x(V[i][0]) + (i === 1 ? -36 : i === 0 ? 0 : 8), plot.y(V[i][1]) + (i === 2 ? 4 : 16), label, { class: "vf-mark-label" }));
        for (const w of dasDennis(3, state.p)) {
          const x = w[0] * V[0][0] + w[1] * V[1][0] + w[2] * V[2][0];
          const y = w[0] * V[0][1] + w[1] * V[1][1] + w[2] * V[2][1];
          marker(plot.layers.marks, "circle", plot.x(x), plot.y(y), state.p > 20 ? 2.5 : 3.5, "vf-pt vf-pt-1");
        }
      } else {
        h("p", { class: "vf-fallback" }, holder, `With ${state.m} objectives the simplex cannot be drawn in the plane; the table gives the number of directions.`);
      }
      clear(readout);
      h("p", {}, readout, `H = C(p + m − 1, m − 1) = C(${state.p + state.m - 1}, ${state.m - 1}) = ${H.toLocaleString("en-US")} reference directions.`);
      h("p", {}, readout, `NSGA-III in VAMOS needs a population of at least H = ${H.toLocaleString("en-US")} so that every direction can keep a solution.`);
      const rows = [1, 2, 3, 4, 6, 8, 12].map((p) => [String(p), binomial(p + state.m - 1, state.m - 1).toLocaleString("en-US")]);
      table(readout, ["p", `H for m = ${state.m}`], rows, [false, true]);
    }
    render();
  };

  /** CTP2 background: infeasible region, unconstrained front and constrained front. */
  function drawCtp2(plot) {
    const cols = 120;
    for (let c = 0; c < cols; c++) {
      const f1 = (c + 0.5) / cols;
      let start = null;
      const step = 0.004;
      for (let f2 = Math.max(0, 1 - f1); f2 <= 1.6 + step; f2 += step) {
        const bad = f2 <= 1.6 && ctpViolation([f1, f2]) > 0;
        if (bad && start === null) start = f2;
        if (!bad && start !== null) {
          plot.rect(plot.layers.back, c / cols, start, (c + 1) / cols + 0.0005, f2, "vf-region-2");
          start = null;
        }
      }
    }
    plot.line(plot.layers.back, [[0, 1], [1, 0]], "vf-front");
    const lowest = [];
    for (let i = 0; i <= 1000; i++) {
      const f1 = i / 1000;
      for (let f2 = 1 - f1; f2 <= 1.6; f2 += 0.0005) {
        if (ctpViolation([f1, f2]) <= 0) {
          lowest.push([f1, f2]);
          break;
        }
      }
    }
    const constrainedFront = nonDominatedIndices(lowest).map((i) => lowest[i]);
    constrainedFront.forEach((p) => marker(plot.layers.back, "circle", plot.x(p[0]), plot.y(p[1]), 1.6, "vf-pt-1"));
    drawFront(plot, plot.layers.back, constrainedFront, "vf-line-1", 0.0015);
  }

  /* A2: constrained domination on CTP2. */
  widgets.constraints = function (node) {
    const state = { rule: "cdp", seed: 5, pop: [] };
    const ctl = controls(node);
    select(ctl, {
      label: "Ranking rule",
      options: [
        ["cdp", "Constrained domination (Deb, 2000)"],
        ["pareto", "Pareto dominance, constraints ignored"],
      ],
      value: state.rule,
      onChange: (v) => {
        state.rule = v;
        render();
      },
    });
    button(ctl, "New population", () => {
      state.seed = Math.floor(Math.random() * 1e6);
      sample();
      render();
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1.6], xLabel: "f₁", yLabel: "f₂", label: "CTP2 objective space with its infeasible regions and a ranked population" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Unconstrained front f₂ = 1 − f₁", shape: "line", cls: "vf-front" },
      { label: "Constrained Pareto front", shape: "line", cls: "vf-line-1" },
      { label: "Feasible solution (colour = front)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Infeasible solution (colour = front)", shape: "square", cls: "vf-pt vf-r1" },
    ]);
    drawCtp2(plot);
    function sample() {
      const rand = mulberry32(state.seed);
      state.pop = Array.from({ length: 24 }, () => {
        const f1 = rand();
        const g = 1 + 0.55 * rand();
        return [f1, g - f1];
      });
    }
    function render() {
      clear(plot.layers.marks);
      const F = state.pop;
      const cv = F.map((f) => ctpViolation(f));
      const sorted = nonDominatedSort(F, state.rule === "cdp" ? constrainedDominates(cv) : null);
      F.forEach((f, i) => {
        const px = plot.x(f[0]);
        const py = plot.y(f[1]);
        marker(plot.layers.marks, cv[i] > 0 ? "square" : "circle", px, py, 5.5, "vf-pt " + rankClass(sorted.rank[i]));
        const hit = s("circle", { cx: px, cy: py, r: 11, class: "vf-hit" }, plot.layers.marks);
        attachTip(hit, tip, `f = ${pair(f)} · violation ${fmt(cv[i])} · front ${sorted.rank[i] + 1}`, px, py);
      });
      const feasible = cv.filter((v) => v <= 0).length;
      const first = sorted.fronts[0];
      const firstFeasible = first.filter((i) => cv[i] <= 0).length;
      clear(readout);
      h("p", {}, readout, `${feasible} of ${F.length} solutions are feasible. Front 1 has ${first.length} solution${first.length === 1 ? "" : "s"}, ${firstFeasible} of them feasible.`);
      h(
        "p",
        {},
        readout,
        state.rule === "cdp"
          ? "Under constrained domination every feasible solution beats every infeasible one, infeasible solutions are ordered by their violation, and Pareto dominance only decides between feasible solutions."
          : "Ignoring the constraints, infeasible solutions close to the unconstrained front take the best ranks: selection would push the population into regions it can never use."
      );
    }
    sample();
    render();
  };

  /* A3: empirical attainment surfaces and a per-run indicator. */
  widgets.attainment = function (node) {
    const curve = (t) => 1 - Math.sqrt(t);
    const ref = [1.1, 1.1];
    const runsPerAlgorithm = 11;
    const state = { seed: 31, show: "median", runs: { A: [], B: [] } };
    const profiles = {
      A: { gap: [0.08, 0.025], extent: [0.92, 1] },
      B: { gap: [0.02, 0.01], extent: [0.45, 0.7] },
    };
    const ctl = controls(node);
    select(ctl, {
      label: "Surfaces",
      options: [
        ["median", "Median (6 of 11 runs)"],
        ["A", "Algorithm A: best, median, worst"],
        ["B", "Algorithm B: best, median, worst"],
      ],
      value: state.show,
      onChange: (v) => {
        state.show = v;
        render();
      },
    });
    button(ctl, "New runs", () => {
      state.seed = Math.floor(Math.random() * 1e6);
      generate();
      render();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const plot = createPlot(body, { title: "Attainment surfaces", x: [0, 1.15], y: [0, 1.15], xLabel: "f₁", yLabel: "f₂", label: "Empirical attainment surfaces of two algorithms" });
    const strip = createPlot(body, {
      title: "Hypervolume of each run",
      x: [0, 3],
      y: [0.6, 0.9],
      xTicks: [1, 2],
      xFormat: (v) => (v === 1 ? "A" : "B"),
      xLabel: "Algorithm",
      yLabel: "HV (r = 1.1, 1.1)",
      label: "Hypervolume of every run of each algorithm",
    });
    const tip = tooltip(strip);
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Algorithm A", shape: "line", cls: "vf-line-1" },
      { label: "Algorithm B", shape: "line", cls: "vf-line-2" },
      { label: "Best and worst surfaces (thin)", shape: "line", cls: "vf-line-thin" },
      { label: "Pareto front", shape: "line", cls: "vf-front" },
    ]);
    function generate() {
      const rand = mulberry32(state.seed);
      for (const key of ["A", "B"]) {
        const prof = profiles[key];
        state.runs[key] = Array.from({ length: runsPerAlgorithm }, () => {
          const gap = Math.max(0.002, prof.gap[0] + prof.gap[1] * gaussian(rand));
          const extent = prof.extent[0] + (prof.extent[1] - prof.extent[0]) * rand();
          const start = key === "A" ? rand() * (1 - extent) * 0.5 : 0.1 + rand() * (0.9 - extent);
          const pts = Array.from({ length: 10 }, (_, i) => {
            const t = Math.min(1, start + (extent * (i + 0.2 * rand())) / 9);
            return [t + 0.2 * gap * rand(), curve(t) + gap * (0.5 + rand())];
          });
          return nonDominatedIndices(pts).map((i) => pts[i]);
        });
      }
    }
    function stairs(surface) {
      const pts = [];
      let prev = 3;
      for (const [u, v] of surface) {
        if (!Number.isFinite(v)) continue;
        pts.push([u, Math.min(prev, 3)]);
        pts.push([u, v]);
        prev = v;
      }
      if (pts.length) pts.push([3, prev]);
      return pts;
    }
    function render() {
      clear(plot.layers.mid);
      plot.line(plot.layers.mid, frontPoints(curve, 0, 1, 101), "vf-front");
      const median = Math.ceil(runsPerAlgorithm / 2);
      const keys = state.show === "median" ? ["A", "B"] : [state.show];
      for (const key of keys) {
        const cls = key === "A" ? "vf-line-1" : "vf-line-2";
        if (state.show !== "median") {
          plot.line(plot.layers.mid, stairs(attainmentSurface(state.runs[key], 1)), "vf-line-thin");
          plot.line(plot.layers.mid, stairs(attainmentSurface(state.runs[key], runsPerAlgorithm)), "vf-line-thin vf-dashed");
        }
        plot.line(plot.layers.mid, stairs(attainmentSurface(state.runs[key], median)), cls);
      }
      const hv = { A: state.runs.A.map((r) => hypervolume2d(r, ref)), B: state.runs.B.map((r) => hypervolume2d(r, ref)) };
      const all = hv.A.concat(hv.B);
      const lo = Math.floor(Math.min(...all) * 20) / 20 - 0.025;
      const hi = Math.ceil(Math.max(...all) * 20) / 20 + 0.025;
      strip.setDomain([0, 3], [lo, hi]);
      clear(strip.layers.mid);
      clear(strip.layers.marks);
      const med = (v) => {
        const sorted = v.slice().sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      };
      const jitter = mulberry32(3);
      ["A", "B"].forEach((key, k) => {
        const cx = k + 1;
        hv[key].forEach((v, i) => {
          const x = cx + (jitter() - 0.5) * 0.5;
          marker(strip.layers.marks, "circle", strip.x(x), strip.y(v), 4.5, "vf-pt " + (key === "A" ? "vf-pt-1" : "vf-pt-2"));
          const hit = s("circle", { cx: strip.x(x), cy: strip.y(v), r: 10, class: "vf-hit" }, strip.layers.marks);
          attachTip(hit, tip, `${key}, run ${i + 1}: HV ${fmt(v)}`, strip.x(x), strip.y(v));
        });
        strip.line(strip.layers.mid, [[cx - 0.35, med(hv[key])], [cx + 0.35, med(hv[key])]], "vf-line-ink");
      });
      const test = mannWhitney(hv.A, hv.B);
      clear(readout);
      h("p", {}, readout, `Median HV: A ${fmt(med(hv.A))}, B ${fmt(med(hv.B))}. Mann–Whitney U = ${fmt(test.u, 1)}, two-sided p ≈ ${fmt(test.p, 3)} (normal approximation, ${runsPerAlgorithm} runs each).`);
      h(
        "p",
        {},
        readout,
        "The data are synthetic: A covers the whole front but converges less, B converges closer on part of it. The surfaces show where each one is better; a single indicator value cannot."
      );
    }
    generate();
    render();
  };

  /** Shared scaffold: draggable labelled points, a plot and a readout. */
  function pointSetFigure(node, o) {
    const state = { P: o.points.map((p) => p.slice()), active: -1 };
    const ctl = controls(node);
    const body = h("div", { class: "vf-body " + (o.layout === undefined ? "vf-body--side" : o.layout) }, node);
    const plot = createPlot(body, { x: o.x || [0, 1], y: o.y || [0, 1], xLabel: o.xLabel || "f₁", yLabel: o.yLabel || "f₂", label: o.label });
    const readout = h("div", { class: "vf-readout" }, body);
    if (o.legend) legend(node, o.legend);
    const fig = { state, ctl, plot, readout, render: () => {} };
    fig.name = (i) => (o.names ? o.names[i] : LETTERS[i]);
    fig.handles = state.P.map((_, i) =>
      handle(plot, {
        cls: "vf-pt vf-pt-1",
        label: () => "Point " + fig.name(i),
        get: () => state.P[i],
        set: (a, b) => {
          state.P[i] = [a, b];
          fig.render();
        },
        onSelect: () => {
          if (state.active !== i) {
            state.active = i;
            fig.render();
          }
        },
      })
    );
    fig.labels = () => {
      clear(plot.layers.top);
      state.P.forEach((p, i) => txt(plot.layers.top, plot.x(p[0]) + 9, plot.y(p[1]) - 7, fig.name(i), { class: "vf-mark-label" }));
    };
    return fig;
  }

  /* I7: SPEA2 strength, raw fitness, density and archive truncation. */
  widgets["spea2-fitness"] = function (node) {
    const defaults = [
      [0.05, 0.92],
      [0.12, 0.7],
      [0.16, 0.64],
      [0.3, 0.45],
      [0.45, 0.33],
      [0.5, 0.3],
      [0.85, 0.08],
      [0.4, 0.62],
      [0.7, 0.55],
      [0.55, 0.32],
    ];
    const names = defaults.map((_, i) => "P" + (i + 1));
    const fig = pointSetFigure(node, {
      points: defaults,
      names,
      layout: "",
      label: "Ten draggable points with their SPEA2 fitness and the archive built by environmental selection",
      legend: [
        { label: "Non-dominated, in the archive", shape: "circle", cls: "vf-pt vf-pt-1" },
        { label: "Dominated, added by fitness", shape: "square", cls: "vf-pt vf-pt-1" },
        { label: "Removed by truncation (order shown)", shape: "circle", cls: "vf-pt vf-pt-2" },
        { label: "Rejected", shape: "circle", cls: "vf-pt-hollow" },
        { label: "Distance to the k-th neighbour of the focused point", shape: "line", cls: "vf-line-thin vf-dashed" },
      ],
    });
    const state = fig.state;
    state.size = 5;
    state.k = 1;
    const ordinal = (n) => n + (n === 1 ? "st" : n === 2 ? "nd" : n === 3 ? "rd" : "th");
    slider(fig.ctl, { label: "Archive size", min: 1, max: 10, step: 1, value: state.size, format: (v) => String(v), onInput: (v) => ((state.size = v), fig.render()) });
    select(fig.ctl, {
      label: "k",
      options: [
        [1, "1, VAMOS default"],
        [2, "2"],
        [3, "3 ≈ √10, the paper"],
      ],
      value: 1,
      onChange: (v) => ((state.k = +v), fig.render()),
    });
    button(fig.ctl, "Reset", () => {
      defaults.forEach((p, i) => (state.P[i] = p.slice()));
      fig.render();
    });
    fig.render = function () {
      const P = state.P;
      const sel = spea2Select(P, state.size, state.k);
      const kept = new Set(sel.kept);
      clear(fig.plot.layers.mid);
      fig.labels();
      const sigma = (i) => 1 / sel.D[i] - 2;
      if (state.active >= 0) {
        const a = P[state.active];
        const r = sigma(state.active);
        s("ellipse", { cx: fig.plot.x(a[0]), cy: fig.plot.y(a[1]), rx: fig.plot.x(a[0] + r) - fig.plot.x(a[0]), ry: fig.plot.y(a[1]) - fig.plot.y(a[1] + r), class: "vf-line-thin vf-dashed" }, fig.plot.layers.mid);
      }
      const status = (i) => {
        const order = sel.removed.indexOf(i);
        if (order >= 0) return "removed " + ordinal(order + 1);
        const fill = sel.filled.indexOf(i);
        if (fill >= 0) return `added by F (${ordinal(fill + 1)})`;
        return kept.has(i) ? "kept" : "rejected (dominated)";
      };
      P.forEach((p, i) => {
        const hd = fig.handles[i];
        hd.place();
        hd.node.classList.toggle("vf-selected", i === state.active);
        const order = sel.removed.indexOf(i);
        hd.setClass(kept.has(i) ? "vf-pt vf-pt-1" : order >= 0 ? "vf-pt vf-pt-2" : "vf-pt-hollow");
        if (sel.filled.includes(i)) hd.setClass("vf-pt vf-pt-1 vf-faded");
        if (order >= 0) txt(fig.plot.layers.top, fig.plot.x(p[0]) - 6, fig.plot.y(p[1]) + 20, "×" + (order + 1), { class: "vf-mark-label" });
      });
      clear(fig.readout);
      const ndn = sel.nd.length;
      let summary;
      if (P.length <= state.size) summary = `${P.length} points for ${state.size} places: every point is kept.`;
      else if (ndn > state.size) summary = `${ndn} non-dominated points for ${state.size} places: the truncation removes ${sel.removed.map((i) => names[i]).join(", then ")}.`;
      else if (ndn < state.size) summary = `${ndn} non-dominated points for ${state.size} places: the archive is completed with ${sel.filled.map((i) => names[i]).join(" and ")}, the dominated points with the lowest F.`;
      else summary = `Exactly ${ndn} non-dominated points for ${ndn} places.`;
      h("p", {}, fig.readout, summary);
      if (state.active >= 0) {
        const i = state.active;
        const doms = P.map((_, j) => j).filter((j) => j !== i && dominates(P[j], P[i]));
        h("p", {}, fig.readout, doms.length ? `${names[i]} is dominated by ${doms.map((j) => names[j]).join(", ")}: R = ${doms.map((j) => sel.S[j]).join(" + ")} = ${sel.R[i]}.` : `${names[i]} is non-dominated: R = 0, and its strength is S = ${sel.S[i]}.`);
      }
      table(
        fig.readout,
        ["Point", "S", "R", "σ_k", "D", "F", "Status"],
        P.map((_, i) => [names[i], String(sel.S[i]), String(sel.R[i]), fmt(sigma(i), 4), fmt(sel.D[i], 4), fmt(sel.fitness[i], 4), status(i)]),
        [false, true, true, true, true, true, false]
      );
    };
    fig.render();
  };

  /** Labels of the step figures' marks. Each item is { x, y, text, r } in SVG units; an item without text is only
      an obstacle. Each label goes beside its point where it covers the fewest other points, labels and tick labels:
      right and above, right and below, left and above, then left and below. */
  function placeMarkLabels(plot, items) {
    const placed = [];
    const spots = [
      [9, -7, "start"],
      [9, 15, "start"],
      [-9, -7, "end"],
      [-9, 15, "end"],
    ];
    const hitsCircle = (b, p) => Math.hypot(Math.max(b[0] - p.x, 0, p.x - b[2]), Math.max(b[1] - p.y, 0, p.y - b[3])) < (p.r || 7);
    const hitsBox = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
    for (const item of items) {
      if (!item.text) continue;
      const width = 6.3 * [...item.text].length + 2;
      let best = null;
      for (const [dx, dy, anchor] of spots) {
        const x0 = anchor === "start" ? item.x + dx : item.x + dx - width;
        const box = [x0, item.y + dy - 10, x0 + width, item.y + dy + 2];
        let cost = 0;
        for (const other of items) if (other !== item && hitsCircle(box, other)) cost += 2;
        for (const b of placed) if (hitsBox(box, b)) cost += 3;
        if (box[0] < 0 || box[2] > plot.W || box[1] < 0) cost += 5;
        if (box[0] < plot.m.l - 2 || box[3] > plot.m.t + plot.ph + 2) cost += 4;
        if (!best || cost < best.cost) best = { cost, dx, dy, anchor, box };
      }
      placed.push(best.box);
      txt(plot.layers.top, item.x + best.dx, item.y + best.dy, item.text, { class: "vf-mark-label", "text-anchor": best.anchor });
    }
  }

  /* SPEA2, one generation in steps: binary tournaments on the archive, one child per pair, the union with its
     strength, raw fitness and nearest-neighbour distances, the non-dominated members, then the truncation (one
     removal per step) or the filling, and the new archive. */
  widgets["spea2-steps"] = function (node) {
    const N = 6;
    const state = { seed: Number(node.dataset.seed || 1693), step: 0 };
    let g = null;
    let titles = [];
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw generations until one needs the truncation operator, with at most three removals.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        const removals = spea2Generation(seed, N).removed.length;
        if (removals >= 1 && removals <= 3) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the archive and of the children" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, dominance, nearest neighbours and the truncation of one SPEA2 generation" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another generation that needs the truncation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Archive member", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Non-dominated, R = 0", shape: "circle", cls: "vf-pt vf-r0" },
      { label: "Dominated, R > 0", shape: "circle", cls: "vf-pt vf-r2" },
      { label: "Tournament winner, member of the new archive", shape: "circle", cls: "vf-ring" },
      { label: "Dominated by, nearest neighbour", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Distances that decide a removal", shape: "line", cls: "vf-line-2" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    const NTH = ["nearest", "second-nearest", "third-nearest", "fourth-nearest", "fifth-nearest", "sixth-nearest", "seventh-nearest"];
    function build() {
      g = spea2Generation(state.seed, N);
      titles = ["the archive Aₜ", "mating selection", "variation", "evaluation and union", "strength and raw fitness", "density", "the non-dominated members"];
      const T = g.truncation.length;
      if (T) g.truncation.forEach((_, j) => titles.push(`truncation, removal ${j + 1} of ${T}`));
      else titles.push(g.filled.length ? "filling" : "neither truncation nor filling");
      titles.push("the new archive Aₜ₊₁");
      const top = Math.max(...g.union.map((p) => p.f[1]));
      fPlot.setDomain([0, 1], [0, Math.max(1, Math.ceil((top + 0.08) * 10) / 10)]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(titles.length - 1, i));
      render();
    }
    // Rows of the union: the archive first (ids 1 to N), then the children (ids N + 1 to 2N).
    const id = (i) => g.union[i].id;
    const ids = (rows) => rows.map(id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const dist = (i, j) => euclidean(g.union[i].f, g.union[j].f);
    const neighbours = (i, among) => among.filter((j) => j !== i).sort((a, b) => dist(i, a) - dist(i, b) || a - b);
    const dominators = (i) => g.union.map((_, j) => j).filter((j) => j !== i && dominates(g.union[j].f, g.union[i].f));
    const all = () => g.union.map((_, i) => i);
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      plot.labels.push({ x: px, y: py, text: label, r: ring ? 11 : 7 });
    }
    /** The members that tie on the first distances of the lexicographic comparison, and the level that splits them. */
    function decider(cut) {
      let tied = cut.members.map((_, a) => a);
      for (let level = 0; level < cut.lists[0].length; level++) {
        const low = Math.min(...tied.map((a) => cut.lists[a][level]));
        const best = tied.filter((a) => Math.abs(cut.lists[a][level] - low) <= 1e-12);
        if (level > 0 && best.length < tied.length) return { tied: tied.map((a) => cut.members[a]), level };
        tied = best;
      }
      return { tied: tied.map((a) => cut.members[a]), level: -1 };
    }
    function render() {
      const step = state.step;
      const last = titles.length - 1;
      const T = g.truncation.length;
      back.disabled = step === 0;
      next.disabled = step === last;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
        plot.labels = [];
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const rows = [];
      if (step <= 2) {
        const wins = new Map();
        g.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
        if (step === 1) for (const t of g.tournaments) fPlot.line(fPlot.layers.mid, [g.union[t.a - 1].f, g.union[t.b - 1].f], "vf-line-thin vf-dashed");
        g.archive.forEach((p, i) => {
          const k = wins.get(p.id) || 0;
          const cls = "vf-pt " + rankClass(g.archiveR[i] === 0 ? 0 : 2);
          const label = String(p.id) + (step === 1 && k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", cls, label, step === 1 && k > 0);
          draw(fPlot, p.f, "circle", cls, label, step === 1 && k > 0);
          rows.push([String(p.id), "archive", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(g.archiveS[i]), String(g.archiveR[i]), fmt(g.archiveSigma[i], 3), fmt(g.archiveFitness[i], 3), step === 1 ? (k ? `won ${k}×` : "–") : "–"]);
        });
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((pid) => xPlot.line(xPlot.layers.mid, [g.union[pid - 1].x, c.x], "vf-line-thin vf-dashed"));
            draw(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "?", "?", "?", "?", "not evaluated"]);
          });
        }
      } else {
        const nd = new Set(g.nd);
        const kept = new Set(g.kept);
        const j = step - 7;
        const cut = T && j >= 0 && j < T ? g.truncation[j] : null;
        const gone = new Set(cut ? g.removed.slice(0, j + 1) : []);
        let chosen = new Set();
        if (step === 6) chosen = nd;
        else if (cut) chosen = new Set(cut.members.filter((i) => i !== g.removed[j]));
        else if (step === 7 && step < last) chosen = new Set(g.nd.concat(g.filled));
        else if (step === last) chosen = kept;
        if (step === 4) {
          for (const i of all()) for (const d of dominators(i)) fPlot.line(fPlot.layers.mid, [g.union[d].f, g.union[i].f], "vf-line-thin vf-dashed");
        }
        if (step === 5) {
          for (const i of all()) fPlot.line(fPlot.layers.mid, [g.union[i].f, g.union[neighbours(i, all())[g.k - 1]].f], "vf-line-thin vf-dashed");
        }
        if (cut) {
          for (const i of cut.members) fPlot.line(fPlot.layers.mid, [g.union[i].f, g.union[neighbours(i, cut.members)[0]].f], "vf-line-thin vf-dashed");
          const why = decider(cut);
          for (const i of why.tied) {
            fPlot.line(fPlot.layers.mid, [g.union[i].f, g.union[neighbours(i, cut.members)[0]].f], "vf-line-2");
            if (why.level > 0) fPlot.line(fPlot.layers.mid, [g.union[i].f, g.union[neighbours(i, cut.members)[why.level]].f], "vf-line-2 vf-dashed");
          }
        }
        g.union.forEach((p, i) => {
          const isChild = i >= N;
          let out = false;
          if (step === 6) out = T > 0 && !nd.has(i);
          // The member that this truncation step removes keeps its colour; it fades from the next step on.
          if (step >= 7) out = !chosen.has(i) && !(cut && i === g.removed[j]);
          let label = String(p.id);
          if (gone.has(i) || (step === last && !kept.has(i))) label += " ✕";
          const cls = "vf-pt " + (step >= 4 ? rankClass(g.R[i] === 0 ? 0 : 2) : "vf-pt-muted") + (out ? " vf-faded" : "");
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, chosen.has(i));
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, chosen.has(i));
          let status = "–";
          if (step === 4) status = g.R[i] === 0 ? "non-dominated" : "dominated by " + idList(ids(dominators(i)));
          else if (step === 5) status = "nearest: " + id(neighbours(i, all())[0]);
          else if (step === 6) status = nd.has(i) ? "copied to Aₜ₊₁" : T ? "discarded" : g.filled.length ? "competes by F" : "discarded";
          else if (cut) status = gone.has(i) ? (i === g.removed[j] ? "removed now" : `removed (${g.removed.indexOf(i) + 1})`) : chosen.has(i) ? "in Aₜ₊₁ so far" : "discarded";
          else if (step === 7 && step < last) status = nd.has(i) ? "non-dominated" : g.filled.includes(i) ? `added ${g.filled.indexOf(i) + 1}${["st", "nd", "rd"][g.filled.indexOf(i)] || "th"}` : "discarded";
          else if (step === last) status = kept.has(i) ? "kept" : "removed";
          const assessed = step >= 4;
          const measured = step >= 5;
          rows.push([String(p.id), isChild ? "child" : "archive", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), assessed ? String(g.S[i]) : "?", assessed ? String(g.R[i]) : "?", measured ? fmt(g.sigma[i], 3) : "?", measured ? fmt(g.fitness[i], 3) : "?", status]);
        });
      }
      for (const plot of [xPlot, fPlot]) placeMarkLabels(plot, plot.labels);
      stepText.textContent = `Step ${step + 1} of ${titles.length}: ${titles[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "S", "R", "σᵏ", "F", "Status"], rows, [false, false, true, true, true, true, true, true, true, true, false]);
    }
    function describe(step) {
      const last = titles.length - 1;
      const T = g.truncation.length;
      const archiveRows = g.archive.map((_, i) => i);
      if (step === 0) {
        const free = archiveRows.filter((i) => g.archiveR[i] === 0);
        const held = archiveRows.filter((i) => g.archiveR[i] > 0);
        const split = held.length ? `R = 0 for ${set(free)}, and ${set(held)} ${held.length === 1 ? "is" : "are"} dominated.` : "All six are non-dominated, R = 0.";
        return `${N} solutions of a two-variable problem whose Pareto front is ZDT1's: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂. In the first generation the archive is the initial population: VAMOS' default archive size equals the population size, so the first environmental selection keeps all ${N}, with the strength S, raw fitness R and nearest-neighbour distance σᵏ (k = 1) they have among themselves. ${split}`;
      }
      if (step === 1) {
        const why = { raw: "lower R", density: "same R, larger σᵏ", coin: "full tie, coin" };
        const games = g.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner} (${why[t.reason]})`).join("; ");
        return `${2 * N} binary tournaments between two different archive members pick ${N} pairs of parents: ${games}. The lower R wins; on equal R, the larger σᵏ, the less crowded member, which is the same as the lower F; a full tie, as between two members that are each other's nearest neighbour, goes to a coin. Ringed members won at least once.`;
      }
      if (step === 2) {
        return `Each pair gives one child: SBX crossover (probability 0.9, η = 20) creates two, VAMOS keeps the first, and polynomial mutation (probability 1/2 per variable, η = 20) perturbs it. Children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents in the decision space, form the population Pₜ; their objectives are not known yet.`;
      }
      if (step === 3) {
        return `The ${N} children are evaluated and joined to the archive: Q = Aₜ ∪ Pₜ holds M = ${2 * N} solutions. Everything that follows is computed afresh on Q, archive members included; the archive members' old values are discarded.`;
      }
      if (step === 4) {
        const held = all().filter((i) => g.R[i] > 0);
        const sums = held.map((i) => {
          const d = dominators(i);
          return `${id(i)} is dominated by ${idList(ids(d))}: R = ${d.length > 1 ? d.map((k) => g.S[k]).join(" + ") + " = " : ""}${g.R[i]}`;
        });
        return `S counts the members of Q that a solution dominates, and R sums the strengths of its dominators. R = 0 for the ${g.nd.length} non-dominated members, ${set(g.nd)}. ${sums.join("; ")}. Dashed lines join each dominated member to its dominators.`;
      }
      if (step === 5) {
        const low = all().reduce((a, i) => (g.sigma[i] < g.sigma[a] ? i : a), 0);
        const mate = neighbours(low, all())[0];
        return `σᵏ is the distance in objective space to the k-th nearest neighbour in Q; VAMOS' default k = 1 makes it the nearest, joined by the dashed lines. The smallest, ${fmt(g.sigma[low], 3)}, is shared by ${id(Math.min(low, mate))} and ${id(Math.max(low, mate))}, each other's nearest neighbours: D = 1 / (σᵏ + 2) = ${fmt(g.D[low], 3)}, the largest density. F = R + D stays below 1 for the non-dominated members, so density only orders members of equal R.`;
      }
      if (step === 6) {
        const n = g.nd.length;
        let then = `exactly fill the ${N} places of the archive.`;
        if (n > N) then = `are more than the ${N} places: the truncation operator removes ${n - N} of them, one at a time, and the dominated members are discarded.`;
        else if (n < N) then = N - n === 1 ? "leave 1 place, which goes to the dominated member with the lowest F." : `leave ${N - n} places, which go to the dominated members with the lowest F.`;
        return `R = 0 for ${n} members, ${set(g.nd)}: they are copied into Aₜ₊₁ and ${then}`;
      }
      if (step < last && T) {
        const j = step - 7;
        const cut = g.truncation[j];
        const out = g.removed[j];
        const why = decider(cut);
        const left = cut.members.length;
        const near = cut.lists[cut.members.indexOf(why.tied[0])][0];
        let text = `${left} members are left for ${N} places. `;
        text += why.tied.length === 2 ? `The closest pair is ${id(why.tied[0])} and ${id(why.tied[1])}, ${fmt(near, 3)} apart: they tie on the nearest distance. ` : `${idList(ids(why.tied))} share the smallest nearest distance, ${fmt(near, 3)}. `;
        if (why.level > 0) {
          const parts = why.tied.map((i) => `${id(i)}'s is ${fmt(cut.lists[cut.members.indexOf(i)][why.level], 3)} (to ${id(neighbours(i, cut.members)[why.level])})`);
          text += `Their ${NTH[why.level]} distances decide: ${parts.join(", ")}. ${id(out)} has the ${why.tied.length === 2 ? "smaller" : "smallest"} one, the more crowded surroundings, and is removed.`;
        } else text += `They tie on every distance, and the first of them, ${id(out)}, is removed.`;
        text += j < T - 1 ? ` The distances are measured again among the ${left - 1} members left.` : ` ${left - 1} members are left: the archive is full.`;
        return text;
      }
      if (step < last) {
        if (!g.filled.length) return `Exactly ${N} members are non-dominated: they fill the archive, and neither the truncation nor the filling is needed.`;
        const parts = g.filled.map((i) => `${id(i)} (R = ${g.R[i]}, F = ${fmt(g.fitness[i], 3)})`);
        const places = g.filled.length === 1 ? "the place left goes to the dominated member" : `the ${g.filled.length} places left go to the dominated members`;
        return `Only ${g.nd.length} members are non-dominated: ${places} with the lowest F = R + D, in order: ${parts.join(", ")}. R decides first; D only separates equal R.`;
      }
      const kept = g.kept.slice().sort((a, b) => a - b);
      const fromArchive = kept.filter((i) => i < N).length;
      const children = kept.length - fromArchive;
      const removed = all().filter((i) => !g.kept.includes(i));
      const free = kept.every((i) => g.R[i] === 0);
      const top = Math.max(...kept.map((i) => g.sigma[i]));
      const isolated = kept.filter((i) => g.sigma[i] === top);
      const mating = free ? `All of them have R = 0, so the next tournaments compare the σᵏ each had in Q, the larger winning: the most isolated ${isolated.length === 1 ? "member is" : "members are"} ${idList(ids(isolated))} (σᵏ = ${fmt(top, 3)}).` : "The next tournaments compare the R and σᵏ each had in Q.";
      return `Aₜ₊₁ = ${set(kept)}: ${fromArchive} archive member${fromArchive === 1 ? "" : "s"} and ${children} child${children === 1 ? "" : "ren"}; ${idList(ids(removed))} leave. ${mating} Their ${N} children form the next population.`;
    }
    build();
  };

  /* I10: IBEA fitness with the additive epsilon indicator. */
  widgets["ibea-fitness"] = function (node) {
    const defaults = [
      [0.05, 0.95],
      [0.15, 0.6],
      [0.2, 0.55],
      [0.4, 0.38],
      [0.6, 0.2],
      [0.95, 0.05],
      [0.5, 0.6],
      [0.75, 0.45],
    ];
    const names = defaults.map((_, i) => "P" + (i + 1));
    const fig = pointSetFigure(node, {
      points: defaults,
      names,
      layout: "",
      label: "Eight draggable points with their IBEA fitness and the order in which environmental selection removes them",
      legend: [
        { label: "Survives", shape: "circle", cls: "vf-pt vf-pt-1" },
        { label: "Removed (order shown)", shape: "circle", cls: "vf-pt-hollow" },
      ],
    });
    const state = fig.state;
    state.kappa = 0.05;
    state.size = 5;
    state.indicator = "eps";
    const ordinal = (n) => n + (n === 1 ? "st" : n === 2 ? "nd" : n === 3 ? "rd" : "th");
    const kappaOf = (t) => +(0.01 * Math.pow(200, t)).toPrecision(2);
    slider(fig.ctl, { label: "κ (log scale)", min: 0, max: 1, step: 0.01, value: Math.log(0.05 / 0.01) / Math.log(200), format: (t) => String(kappaOf(t)), onInput: (t) => ((state.kappa = kappaOf(t)), fig.render()) });
    select(fig.ctl, {
      label: "Indicator",
      options: [
        ["eps", "Additive ε (default)"],
        ["hv", "Hypervolume difference"],
      ],
      value: "eps",
      onChange: (v) => ((state.indicator = v), fig.render()),
    });
    slider(fig.ctl, { label: "Population size", min: 1, max: 7, step: 1, value: state.size, format: (v) => String(v), onInput: (v) => ((state.size = v), fig.render()) });
    button(fig.ctl, "Reset", () => {
      defaults.forEach((p, i) => (state.P[i] = p.slice()));
      fig.render();
    });
    const big = (v) => (Math.abs(v) >= 1e6 ? v.toExponential(3) : fmt(v));
    fig.render = function () {
      const P = state.P;
      const sel = ibeaSelect(P, state.size, state.kappa, state.indicator);
      const kept = new Set(sel.kept);
      fig.labels();
      P.forEach((p, i) => {
        const hd = fig.handles[i];
        hd.place();
        hd.node.classList.toggle("vf-selected", i === state.active);
        const order = sel.removed.indexOf(i);
        hd.setClass(kept.has(i) ? "vf-pt vf-pt-1" : "vf-pt-hollow");
        if (order >= 0) txt(fig.plot.layers.top, fig.plot.x(p[0]) - 6, fig.plot.y(p[1]) + 20, "×" + (order + 1), { class: "vf-mark-label" });
      });
      clear(fig.readout);
      h("p", {}, fig.readout, `Population size ${state.size}: IBEA removes ${sel.removed.map((i) => names[i]).join(", then ") || "nothing"}. c = ${fmt(sel.c)} (largest |I| on normalized objectives), κ = ${state.kappa}.`);
      if (state.active >= 0) {
        const x = state.active;
        const terms = sel.terms[x]
          .map((v, y) => [y, -v])
          .filter(([y]) => y !== x)
          .sort((a, b) => a[1] - b[1])
          .slice(0, 3);
        h("p", {}, fig.readout, `${names[x]}: ${terms.map(([y, v]) => `${names[y]} contributes ${big(v)}`).join(", ")}.`);
      }
      table(
        fig.readout,
        ["Point", "f₁", "f₂", "Fitness (initial)", "Status", "Fitness after selection"],
        P.map((p, i) => {
          const order = sel.removed.indexOf(i);
          return [names[i], fmt(p[0], 2), fmt(p[1], 2), big(sel.fitness[i]), kept.has(i) ? "survives" : "removed " + ordinal(order + 1), kept.has(i) ? big(sel.final[i]) : "–"];
        }),
        [false, true, true, true, false, true]
      );
    };
    fig.render();
  };

  /* IBEA, one generation in steps: binary tournaments on fitness, two children per pair, the merged set with its
     normalized objectives, the ε indicator and the fitness, the removals one at a time with their fitness updates,
     and the next population. */
  widgets["ibea-steps"] = function (node) {
    const N = 6;
    const TITLES = ["the population Pₜ", "mating selection", "variation", "evaluation and merge", "normalization and the ε indicator", "fitness"]
      .concat(Array.from({ length: N }, (_, j) => `removal ${j + 1} of ${N}`))
      .concat(["the next population Pₜ₊₁"]);
    const state = { seed: Number(node.dataset.seed || 31040), step: 0 };
    let g = null;
    // The N members with the best initial fitness: the survivors if the fitness were never updated.
    const byInitial = (gen) =>
      gen.fitness
        .map((_, i) => i)
        .sort((a, b) => gen.fitness[b] - gen.fitness[a] || a - b)
        .slice(0, N);
    const updatesMatter = (gen) => {
      const once = byInitial(gen);
      return gen.kept.some((i) => !once.includes(i));
    };
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw generations until the fitness updates change who survives.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (updatesMatter(ibeaGeneration(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the children" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, normalization box and removals of one IBEA generation" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another generation in which the fitness updates change who survives.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-1" },
      { label: "No fitness yet", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Tournament winner, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Normalization box: minimum to maximum of each objective", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Largest fitness rise after a removal", shape: "line", cls: "vf-line-2" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = ibeaGeneration(state.seed, N);
      const top = Math.max(...g.merged.map((p) => p.f[1]));
      fPlot.setDomain([0, 1], [0, Math.max(1, Math.ceil((top + 0.08) * 10) / 10)]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    // Rows of the merged set: the parents first (ids 1 to N), then the children (ids N + 1 to 2N).
    const id = (i) => g.merged[i].id;
    const ids = (rows) => rows.map(id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const num = (v) => (Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 5e-4) ? v.toExponential(2) : fmt(v, 3)).replace(/-/g, "−");
    const before = (j) => (j === 0 ? g.fitness : g.history[j - 1]);
    const ordinal = (n) => n + (n === 1 ? "st" : n === 2 ? "nd" : n === 3 ? "rd" : "th");
    /** The survivor whose fitness rose most at removal j, with the rise. */
    function lifted(j) {
      const alive = g.merged.map((_, i) => i).filter((i) => !g.removed.slice(0, j + 1).includes(i));
      const rise = (i) => g.history[j][i] - before(j)[i];
      const best = alive.reduce((a, i) => (rise(i) > rise(a) ? i : a), alive[0]);
      return { row: best, rise: rise(best) };
    }
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      plot.labels.push({ x: px, y: py, text: label, r: ring ? 11 : 7 });
    }
    function render() {
      const step = state.step;
      const last = TITLES.length - 1;
      back.disabled = step === 0;
      next.disabled = step === last;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
        plot.labels = [];
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const rows = [];
      if (step <= 2) {
        const wins = new Map();
        g.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
        if (step === 1) for (const t of g.tournaments) fPlot.line(fPlot.layers.mid, [g.merged[t.a - 1].f, g.merged[t.b - 1].f], "vf-line-thin vf-dashed");
        g.pop.forEach((p, i) => {
          const k = wins.get(p.id) || 0;
          const label = String(p.id) + (step === 1 && k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", "vf-pt vf-pt-1", label, step === 1 && k > 0);
          draw(fPlot, p.f, "circle", "vf-pt vf-pt-1", label, step === 1 && k > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), "–", "–", num(g.popFitness[i]), step === 1 ? (k ? `won ${k}×` : "–") : "–"]);
        });
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((pid) => xPlot.line(xPlot.layers.mid, [g.merged[pid - 1].x, c.x], "vf-line-thin vf-dashed"));
            draw(xPlot, c.x, "triangle", "vf-pt vf-pt-muted", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "?", "?", "?", "not evaluated"]);
          });
        }
      } else {
        const j = step - 6;
        const removing = j >= 0 && j < N;
        const gone = new Set(g.removed.slice(0, removing ? j + 1 : step === last ? N : 0));
        const kept = new Set(g.kept);
        if (step === 4) {
          const lo = [0, 1].map((m) => Math.min(...g.merged.map((p) => p.f[m])));
          const hi = [0, 1].map((m) => Math.max(...g.merged.map((p) => p.f[m])));
          fPlot.rect(fPlot.layers.back, lo[0], lo[1], hi[0], hi[1], "vf-line-thin vf-dashed");
        }
        if (removing) {
          const up = lifted(j);
          if (up.rise > 1e-3) fPlot.line(fPlot.layers.mid, [g.merged[g.removed[j]].f, g.merged[up.row].f], "vf-line-2");
        }
        const fitnessNow = removing ? g.history[j] : step === last ? g.final : g.fitness;
        g.merged.forEach((p, i) => {
          const isChild = i >= N;
          const out = gone.has(i);
          // The member that this removal step removes keeps its colour; it fades from the next step on.
          const now = removing && i === g.removed[j];
          const cls = "vf-pt " + (step >= 5 ? "vf-pt-1" : "vf-pt-muted") + (out && !now ? " vf-faded" : "");
          const label = String(p.id) + (out ? " ✕" : "");
          const ring = step === last && kept.has(i);
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, ring);
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, ring);
          let status = "–";
          if (removing) {
            const k = g.removed.indexOf(i);
            const rise = g.history[j][i] - before(j)[i];
            status = out ? (k === j ? "removed now" : `removed ${ordinal(k + 1)}`) : rise > 5e-4 ? `rose by ${num(rise)}` : "–";
          } else if (step === last) status = kept.has(i) ? "survives" : `removed ${ordinal(g.removed.indexOf(i) + 1)}`;
          const z = step >= 4;
          const shown = step >= 5 && !out;
          rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), z ? fmt(g.Z[i][0], 3) : "?", z ? fmt(g.Z[i][1], 3) : "?", shown ? num(fitnessNow[i]) : step >= 5 ? "–" : "?", status]);
        });
      }
      for (const plot of [xPlot, fPlot]) placeMarkLabels(plot, plot.labels);
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "z₁", "z₂", "Fitness", "Status"], rows, [false, false, true, true, true, true, true, true, true, false]);
    }
    function describe(step) {
      const last = TITLES.length - 1;
      const rowsOf = (n) => Array.from({ length: n }, (_, i) => i);
      if (step === 0) {
        const order = rowsOf(N).sort((a, b) => g.popFitness[b] - g.popFitness[a]);
        return `${N} solutions of a two-variable problem whose Pareto front is ZDT1's: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂. Each carries the fitness of the initial environmental selection, computed among the ${N}, which removed nobody: ${N} solutions fit ${N} places. The highest, ${id(order[0])}'s (${num(g.popFitness[order[0]])}), belongs to the solution the others cover the least; the lowest is ${id(order[N - 1])}'s (${num(g.popFitness[order[N - 1]])}).`;
      }
      if (step === 1) {
        const games = g.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner}`).join("; ");
        return `${N} binary tournaments between two different members pick ${N / 2} pairs of parents: ${games}. The higher fitness always wins. Ringed members won at least once.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 20) turns each pair into two children, and polynomial mutation (probability 1/2 per variable, η = 20) perturbs each child: children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents in the decision space. Their objectives are not known yet.`;
      }
      if (step === 3) {
        return `The ${N} children are evaluated and merged with the ${N} parents: Q = P ∪ P′ holds ${2 * N} solutions, which compete for ${N} places. The parents' old fitness is discarded; IBEA computes it afresh on Q.`;
      }
      if (step === 4) {
        const all = rowsOf(2 * N);
        const ext = (m, sign) => all.reduce((a, i) => (sign * g.merged[i].f[m] < sign * g.merged[a].f[m] ? i : a), 0);
        const [a1, b1, a2, b2] = [ext(0, 1), ext(0, -1), ext(1, 1), ext(1, -1)];
        return `Each objective is mapped to [0, 1] with its minimum and maximum over Q, the dashed box: f₁ from ${id(a1)} (${fmt(g.merged[a1].f[0], 2)}) to ${id(b1)} (${fmt(g.merged[b1].f[0], 2)}), f₂ from ${id(a2)} (${fmt(g.merged[a2].f[1], 2)}) to ${id(b2)} (${fmt(g.merged[b2].f[1], 2)}). For every ordered pair, I_ε+(y, x) = max_i (z_i(y) − z_i(x)) is the shift that y needs to cover x. Every value is divided by c = max |I| = ${fmt(g.c, 3)}: no value can exceed 1 on normalized objectives, and ${id(b1)}, with the largest f₁, needs a shift of exactly 1 to cover ${id(a1)}.`;
      }
      if (step === 5) {
        const order = rowsOf(2 * N).sort((a, b) => g.fitness[a] - g.fitness[b]);
        const low = order.slice(0, 3).map((i) => `${id(i)} (${num(g.fitness[i])})`);
        const best = order[order.length - 1];
        return `F(x) = Σ_{y ≠ x} −exp(−I(y, x) / (c·κ)), with κ = 0.05. A term is below −1 when y dominates x, close to −1 when y nearly covers x, and close to 0 when y is far from covering x. The lowest fitness values are ${low.join(", ")}; the highest is ${id(best)}'s (${num(g.fitness[best])}), the member the others cover the least.`;
      }
      if (step < last) {
        const j = step - 6;
        const out = g.removed[j];
        const left = 2 * N - j - 1;
        const up = lifted(j);
        const what = up.rise > 1e-3 ? `Every survivor's fitness rises by exp(−I(${id(out)}, y) / (c·κ)), the term ${id(out)} contributed to it: most for ${id(up.row)}, the member ${id(out)} covered best, from ${num(before(j)[up.row])} to ${num(g.history[j][up.row])}.` : `${id(out)} covered none of the others closely: no survivor's fitness changes by more than 0.001.`;
        return `${id(out)} has the lowest fitness, ${num(before(j)[out])}, and leaves; ${left} members are left for ${N} places. ${what} Nothing else is recomputed: normalization and c stay those of Q.`;
      }
      const kept = g.kept.slice().sort((a, b) => a - b);
      const parents = kept.filter((i) => i < N).length;
      const children = kept.length - parents;
      const once = byInitial(g);
      const rescued = kept.filter((i) => !once.includes(i));
      const displaced = once.filter((i) => !g.kept.includes(i)).sort((a, b) => a - b);
      let lesson = ` Here the updates do not change the outcome: the ${N} best initial fitness values belong to the same members.`;
      if (rescued.length) {
        const r = rescued[0];
        const k = rowsOf(N).reduce((a, j) => (g.history[j][r] - before(j)[r] > g.history[a][r] - before(a)[r] ? j : a), 0);
        lesson = ` Ranking the initial fitness once would have kept ${idList(ids(displaced))} instead of ${idList(ids(rescued))}: ${id(r)}'s fitness, ${num(g.fitness[r])} in Q, rose most when ${id(g.removed[k])} left, which covered it closely, and reached ${num(g.final[r])}.`;
      }
      return `Pₜ₊₁ = ${set(kept)}: ${parents} parent${parents === 1 ? "" : "s"} and ${children} child${children === 1 ? "" : "ren"} survive, and ${idList(ids(g.removed.slice().sort((a, b) => a - b)))} leave.${lesson} The next tournaments compare the survivors' final fitness.`;
    }
    build();
  };

  /* A3: RVEA angle-penalized distance. */
  widgets["rvea-apd"] = function (node) {
    const V = [0, 1, 2, 3, 4].map((k) => {
      const w = [k / 4, 1 - k / 4];
      const n = Math.hypot(w[0], w[1]);
      return [w[0] / n, w[1] / n];
    });
    const F = [
      [0.2, 1.2],
      [0.305, 1.095],
      [0.387, 0.98],
      [0.494, 1.004],
      [0.688, 1.196],
      [0.576, 0.806],
      [0.862, 0.452],
      [1.197, 0.565],
      [1.046, 0.408],
      [1.031, 0.277],
      [1.246, 0.192],
      [1.6, 0.1],
    ];
    const z = [Math.min(...F.map((f) => f[0])), Math.min(...F.map((f) => f[1]))];
    const Fp = F.map((f) => [f[0] - z[0], f[1] - z[1]]);
    const state = { progress: 0, alpha: 2, M: 1, focus: -1 };
    const ctl = controls(node);
    slider(ctl, { label: "t / t_max", min: 0, max: 1, step: 0.005, value: 0, format: (v) => v.toFixed(3), onInput: (v) => ((state.progress = v), render()) });
    slider(ctl, { label: "α", min: 0.5, max: 10, step: 0.5, value: 2, format: (v) => v.toFixed(1), onInput: (v) => ((state.alpha = v), render()) });
    checkbox(ctl, { label: "M = 2, as in the article", checked: false, onChange: (v) => ((state.M = v ? 2 : 1), render()) });
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, { x: [0, 1.8], y: [0, 1.4], xLabel: "f₁", yLabel: "f₂", width: 400, height: 320, label: "RVEA reference vectors from the ideal point and the solution each one keeps" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Reference vector (from z)", shape: "line", cls: "vf-line-ink" },
      { label: "Kept by its vector", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Not kept", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Radius stretched by the penalty, ‖f′‖·(1 + P(θ))", shape: "line", cls: "vf-line-thin vf-dashed" },
    ]);
    function render() {
      const sel = rveaSelect(Fp, V, state.progress, state.alpha, state.M);
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      plot.line(plot.layers.mid, [[z[0], 3], [z[0], z[1]], [3, z[1]]], "vf-line-thin");
      V.forEach((v, j) => {
        plot.line(plot.layers.mid, [z, [z[0] + 3 * v[0], z[1] + 3 * v[1]]], "vf-line-ink");
        const t = Math.min((1.75 - z[0]) / (v[0] || 1e-9), (1.36 - z[1]) / (v[1] || 1e-9));
        txt(plot.layers.top, plot.x(z[0] + t * v[0]) - (j === 4 ? 18 : 4), plot.y(z[1] + t * v[1]) + (j === 4 ? -6 : 12), "v" + (j + 1), { class: "vf-mark-label" });
      });
      F.forEach((f, i) => {
        const nrm = Math.hypot(Fp[i][0], Fp[i][1]);
        const ghost = [z[0] + (sel.apd[i] * Fp[i][0]) / nrm, z[1] + (sel.apd[i] * Fp[i][1]) / nrm];
        if (nrm > 0) plot.line(plot.layers.mid, [f, ghost], "vf-line-thin vf-dashed");
        const winner = sel.selected.includes(i);
        const px = plot.x(f[0]);
        const py = plot.y(f[1]);
        const hit = s("circle", { cx: px, cy: py, r: 11, class: "vf-hit" }, plot.layers.marks);
        marker(plot.layers.marks, "circle", px, py, winner ? 6.5 : 5, winner ? "vf-pt vf-pt-2" : "vf-pt-hollow");
        attachTip(hit, tip, `${i + 1} · v${sel.assoc[i] + 1} · θ ${fmt((sel.theta[i] * 180) / Math.PI, 1)}° · APD ${fmt(sel.apd[i])}`, px, py);
        txt(plot.layers.top, px + 8, py - 6, String(i + 1), { class: "vf-mark-label" });
      });
      marker(plot.layers.marks, "cross", plot.x(z[0]), plot.y(z[1]), 4, "vf-line-ink");
      clear(readout);
      const sv = Math.pow(state.progress, state.alpha);
      const kept = sel.selected.filter((i) => i >= 0).map((i) => i + 1);
      const empty = sel.selected.map((i, j) => (i < 0 ? "v" + (j + 1) : null)).filter(Boolean);
      h("p", {}, readout, `s = (t/t_max)^α = ${fmt(sv)} · M = ${state.M}. Survivors: ${kept.join(", ")}${empty.length ? ` (${empty.join(", ")} ${empty.length === 1 ? "has" : "have"} no solution)` : ""}.`);
      const rows = F.map((_, i) => i).sort((a, b) => sel.assoc[a] - sel.assoc[b] || a - b);
      table(
        readout,
        ["Point", "Vector", "θ (°)", "‖f′‖", "P(θ)", "APD", "Kept"],
        rows.map((i) => {
          const pen = state.M * sv * (sel.theta[i] / sel.gamma[sel.assoc[i]]);
          return [String(i + 1), "v" + (sel.assoc[i] + 1), fmt((sel.theta[i] * 180) / Math.PI, 1), fmt(Math.hypot(Fp[i][0], Fp[i][1])), fmt(pen), fmt(sel.apd[i]), sel.selected.includes(i) ? "✓" : ""];
        }),
        [true, false, true, true, true, true, false]
      );
    }
    render();
  };

  /* RVEA at a glance: one mid-run generation in nine steps, from the population to the adapted vectors. */
  widgets["rvea-steps"] = function (node) {
    const N = 6;
    const GEN = 6;
    const TMAX = 12;
    const BUDGET = N + TMAX * N;
    const TITLES = [
      "the population Pₜ",
      "mating selection",
      "variation",
      "evaluation and merge",
      "translation by the ideal point",
      "association by angle",
      "angle-penalized distance",
      "one survivor per vector",
      "adaptation of the vectors",
    ];
    const state = { seed: Number(node.dataset.seed || 24637), step: 0 };
    let g = null;
    const deg =(a) => (a * 180) / Math.PI;
    const nearest = (c, j) => {
      const members = c.merged.map((_, i) => i).filter((i) => c.select.assoc[i] === j);
      const norm = (i) => Math.hypot(c.translated[i][0], c.translated[i][1]);
      return members.length ? members.reduce((b, i) => (norm(i) < norm(b) ? i : b), members[0]) : -1;
    };
    const switched = (c) => c.V.map((_, j) => j).filter((j) => c.select.selected[j] >= 0 && nearest(c, j) !== c.select.selected[j]);
    const interesting = (c) => switched(c).length > 0 && c.select.selected.some((i) => i < 0);
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw runs until the penalty changes a vector's survivor and a vector is left empty.
      for (let tries = 0; tries < 300; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(rveaGeneration(seed, N, GEN, TMAX))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the children" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 1.4], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors and RVEA's reference vectors drawn from the ideal point", margin: { t: 22, r: 28 } });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another run.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-1" },
      { label: "Survivor of its vector", shape: "circle", cls: "vf-ring" },
      { label: "Reference vector from z*", shape: "line", cls: "vf-line-ink" },
      { label: "Empty vector; vector before adaptation", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "To its vector (association); radius stretched by the penalty (APD)", shape: "line", cls: "vf-line-2 vf-dashed" },
      { label: "Ideal point z*, largest values z_max", shape: "cross", cls: "vf-line-ink" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = rveaGeneration(state.seed, N, GEN, TMAX);
      const top = Math.max(1.2, Math.ceil(10 * (Math.max(...g.merged.map((p) => p.f[1])) + 0.12)) / 10);
      fPlot.setDomain([0, 1], [0, top]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const n = () => g.pop.length;
    const ids = (rows) => rows.map((i) => g.merged[i].id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const vname = (j) => "v" + (j + 1);
    const norm = (i) => Math.hypot(g.translated[i][0], g.translated[i][1]);
    const penalty = (i) => g.M * g.progress * (g.select.theta[i] / g.select.gamma[g.select.assoc[i]]);
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(Math.min(v[0], plot.xd[1]));
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      if (label) txt(plot.layers.top, px + 9, py - 7, label, { class: "vf-mark-label" });
    }
    function cross(v, label) {
      const px = fPlot.x(Math.min(v[0], fPlot.xd[1]));
      const py = fPlot.y(Math.min(v[1], fPlot.yd[1]));
      marker(fPlot.layers.marks, "cross", px, py, 4.5, "vf-line-ink");
      const right = px > fPlot.m.l + fPlot.pw - 40;
      txt(fPlot.layers.top, px + (right ? -7 : 7), py + 15, label, { class: "vf-mark-label", "text-anchor": right ? "end" : "start" });
    }
    function vectors(z, V, cls, labelled) {
      V.forEach((v, j) => {
        fPlot.line(fPlot.layers.mid, [z, [z[0] + 3 * v[0], z[1] + 3 * v[1]]], typeof cls === "function" ? cls(j) : cls);
        if (!labelled) return;
        // The labels sit in the widened top and right margins, clear of the points.
        const toRight = v[0] > 1e-9 ? (fPlot.xd[1] - z[0]) / v[0] : Infinity;
        const toTop = v[1] > 1e-9 ? (fPlot.yd[1] - z[1]) / v[1] : Infinity;
        const reach = Math.min(toRight, toTop);
        const px = fPlot.x(z[0] + reach * v[0]);
        const py = fPlot.y(z[1] + reach * v[1]);
        if (toTop < toRight) txt(fPlot.layers.top, px, py - 6, vname(j), { class: "vf-mark-label", "text-anchor": "middle" });
        else txt(fPlot.layers.top, px + 4, py + 4, vname(j), { class: "vf-mark-label" });
      });
    }
    function render() {
      const step = state.step;
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.back, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.back, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const z = step >= 4 ? g.ideal : g.idealBefore;
      const sizes = g.V.map((_, j) => g.select.assoc.filter((a) => a === j).length);
      let rows = [];
      let headers;
      let numeric;
      if (step === 8) {
        vectors(g.ideal, g.V, "vf-line-thin vf-dashed", false);
        vectors(g.ideal, g.next, "vf-line-ink", true);
        cross(g.ideal, "z*");
        cross(g.nadir, "z_max");
      } else {
        vectors(z, g.V, (j) => (step >= 5 && !sizes[j] ? "vf-line-thin vf-dashed" : "vf-line-ink"), true);
        cross(z, "z*");
        if (step === 4 && (g.ideal[0] !== g.idealBefore[0] || g.ideal[1] !== g.idealBefore[1])) marker(fPlot.layers.marks, "cross", fPlot.x(g.idealBefore[0]), fPlot.y(g.idealBefore[1]), 4.5, "vf-line-ink vf-faded");
        if (step === 4) fPlot.line(fPlot.layers.mid, [[z[0], 3], z, [3, z[1]]], "vf-line-thin");
      }
      if (step <= 2) {
        const byId = new Map(g.pop.map((p) => [p.id, p]));
        const counts = new Map();
        g.draws.forEach((d) => counts.set(d, (counts.get(d) || 0) + 1));
        if (step === 1) {
          for (let k = 0; k < g.draws.length; k += 2) fPlot.line(fPlot.layers.mid, [g.pop[g.draws[k]].f, g.pop[g.draws[k + 1]].f], "vf-line-thin vf-dashed");
        }
        g.pop.forEach((p, i) => {
          const k = counts.get(i) || 0;
          const label = String(p.id) + (step >= 1 && k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", "vf-pt vf-pt-1", label, step === 1 && k > 0);
          draw(fPlot, p.f, "circle", "vf-pt vf-pt-1", label, step === 1 && k > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), step >= 1 ? `drawn ${k}×` : "–"]);
        });
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((id) => xPlot.line(xPlot.layers.mid, [byId.get(id).x, c.x], "vf-line-thin vf-dashed"));
            draw(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "not evaluated"]);
          });
        }
        headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Status"];
        numeric = [false, false, true, true, true, true, false];
      } else if (step === 8) {
        g.survivors.forEach((i) => {
          const p = g.merged[i];
          draw(xPlot, p.x, i >= n() ? "triangle" : "circle", "vf-pt vf-pt-1", String(p.id), false);
          draw(fPlot, p.f, i >= n() ? "triangle" : "circle", "vf-pt vf-pt-1", String(p.id), false);
        });
        headers = ["Vector", "Initial v⁰", "Before", "γ before (°)", "After", "γ after (°)"];
        numeric = [false, false, false, true, false, true];
        rows = g.V.map((v, j) => [vname(j), pair(g.V0[j], 3), pair(v, 3), fmt(deg(g.select.gamma[j]), 1), pair(g.next[j], 3), fmt(deg(g.gammaNext[j]), 1)]);
      } else {
        const keep = new Set(g.survivors);
        g.merged.forEach((p, i) => {
          const isChild = i >= n();
          const j = g.select.assoc[i];
          const v = g.V[j];
          const fp = g.translated[i];
          if (step === 5) {
            const t = fp[0] * v[0] + fp[1] * v[1];
            fPlot.line(fPlot.layers.mid, [p.f, [z[0] + t * v[0], z[1] + t * v[1]]], "vf-line-2 vf-dashed");
          }
          if (step === 6 && norm(i) > 0) {
            const r = g.select.apd[i] / norm(i);
            fPlot.line(fPlot.layers.mid, [p.f, [z[0] + r * fp[0], z[1] + r * fp[1]]], "vf-line-2 vf-dashed");
          }
          const removed = step === 7 && !keep.has(i);
          const cls = "vf-pt vf-pt-1" + (removed ? " vf-faded" : "");
          const label = String(p.id) + (removed ? " ✕" : "");
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, step === 7 && keep.has(i));
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, step === 7 && keep.has(i));
          const known = (s) => step >= s;
          let status = isChild ? "child" : "parent";
          if (step === 7) status = keep.has(i) ? `survives for ${vname(j)}` : "removed";
          rows.push([
            String(p.id),
            status,
            fmt(fp[0], 3),
            fmt(fp[1], 3),
            fmt(norm(i), 3),
            known(5) ? vname(j) : "?",
            known(5) ? fmt(deg(g.select.theta[i]), 1) : "?",
            known(6) ? fmt(penalty(i), 3) : "?",
            known(6) ? fmt(g.select.apd[i], 4) : "?",
          ]);
        });
        headers = ["Id", "Status", "f′₁", "f′₂", "‖f′‖", "Vector", "θ (°)", "P(θ)", "APD"];
        numeric = [false, false, true, true, true, false, true, true, true];
      }
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, headers, rows, numeric);
    }
    function describe(step) {
      if (step === 0) {
        const last = Math.floor((GEN - 1) / g.interval) * g.interval;
        const adapted = last > 0 ? `were last adapted after generation ${last}` : "are still the initial unit vectors";
        return `Generation t = ${GEN} of a run on the loop figure's problem with N = 6 and max_evaluations = ${BUDGET}, so t_max = ⌈(${BUDGET} − 6) / 6⌉ = ${TMAX}. The population has ${n()} members${n() < N ? `, fewer than N = 6 because an earlier generation left vectors empty` : ""}. Its six reference vectors (p = 5), drawn from the ideal point z* = ${pair(g.idealBefore)} found so far, ${adapted}.`;
      }
      if (step === 1) {
        const pairs = [];
        for (let k = 0; k < g.draws.length; k += 2) pairs.push(`(${g.pop[g.draws[k]].id}, ${g.pop[g.draws[k + 1]].id})`);
        const extra = g.draws.length - n();
        const take = extra > 0 ? `With ${n()} members, the ${g.draws.length} parents take one permutation and the first ${extra === 1 ? "member" : extra + " members"} of a second one` : `The ${g.draws.length} parents are one permutation`;
        return `VAMOS draws RVEA's parents from random permutations of the population, as pymoo does, so no member is drawn twice before every member is drawn once. ${take}: the pairs ${idList(pairs)}. Neither dominance nor the vectors play a part.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 30) recombines each pair and polynomial mutation (probability 1/2 per variable, η = 20) perturbs each child, VAMOS' defaults for RVEA: children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents.${n() < N ? ` The population has ${n()} members, but RVEA still creates N = 6 children.` : ""}`;
      }
      if (step === 3) return `The ${N} children are evaluated and merged with the ${n()} parents: ${g.merged.length} candidates. RVEA sorts nothing; the vectors alone decide who survives.`;
      if (step === 4) {
        const moved = g.ideal[0] !== g.idealBefore[0] || g.ideal[1] !== g.idealBefore[1];
        return `${moved ? `The ideal point moves from ${pair(g.idealBefore)} to ${pair(g.ideal)}, the smallest values seen during the run, Rₜ included.` : `The ideal point stays at ${pair(g.ideal)}: no candidate improves on the smallest values seen during the run.`} The objectives are translated, f′ = f − z*, and ‖f′‖, the distance to z*, measures convergence.`;
      }
      if (step === 5) {
        const groups = g.V.map((_, j) => {
          const members = g.merged.map((_, i) => i).filter((i) => g.select.assoc[i] === j);
          return members.length ? `${vname(j)} ← ${set(members)}` : null;
        }).filter(Boolean);
        const empty = g.V.map((_, j) => j).filter((j) => !g.select.assoc.includes(j)).map(vname);
        return `Each candidate joins the vector at the smallest angle θ (the dashed lines lead to it): ${groups.join("; ")}.${empty.length ? ` ${idList(empty)} (dashed) ${empty.length === 1 ? "has" : "have"} no candidate.` : ""} The angle is measured on the translated objectives, without rescaling.`;
      }
      if (step === 6) {
        const gammas = g.select.gamma.map(deg);
        const cases = switched(g).map((j) => {
          const a = nearest(g, j);
          const b = g.select.selected[j];
          return `in ${vname(j)}, ${g.merged[a].id} is closer to z* (‖f′‖ ${fmt(norm(a), 2)} against ${fmt(norm(b), 2)}), but its angle of ${fmt(deg(g.select.theta[a]), 1)}° raises its APD to ${fmt(g.select.apd[a], 3)}, above ${g.merged[b].id}'s ${fmt(g.select.apd[b], 3)} (${fmt(deg(g.select.theta[b]), 1)}°)`;
        });
        return `The progress factor is (t / t_max)^α = (${GEN}/${TMAX})² = ${fmt(g.progress, 2)} and M = 1 with two objectives (VAMOS' default variant). So P(θ) = ${fmt(g.progress, 2)}·θ/γ, where γ, the angle from a vector to its nearest neighbour, is between ${fmt(Math.min(...gammas), 1)}° and ${fmt(Math.max(...gammas), 1)}°, and APD = (1 + P(θ))·‖f′‖, the dashed radius.${cases.length ? " " + cases.map((c, k) => (k ? c : c[0].toUpperCase() + c.slice(1))).join("; ") + "." : ""}`;
      }
      if (step === 7) {
        const kept = g.V.map((_, j) => (g.select.selected[j] >= 0 ? `${vname(j)} → ${g.merged[g.select.selected[j]].id}` : null)).filter(Boolean);
        const empty = g.V.map((_, j) => j).filter((j) => g.select.selected[j] < 0).map(vname);
        const dominance = g.survivors
          .map((i) => [i, g.survivors.filter((k) => k !== i && dominates(g.merged[k].f, g.merged[i].f))])
          .filter(([, by]) => by.length)
          .map(([i, by]) => `${g.merged[i].id} survives although ${idList(ids(by))} dominate${by.length === 1 ? "s" : ""} it`);
        return `Each vector keeps the candidate with the smallest APD: ${kept.join(", ")}.${empty.length ? ` ${idList(empty)} keep${empty.length === 1 ? "s" : ""} nothing, so Pₜ₊₁ has ${g.survivors.length} members.` : " Every vector keeps one solution."}${dominance.length ? ` RVEA never checks dominance: ${idList(dominance)}.` : ""}`;
      }
      const scale = g.nadir.map((v, m) => v - g.ideal[m]);
      if (!g.adapted) return `t = ${GEN} is not a multiple of ⌈0.1 · ${TMAX}⌉ = ${g.interval}, so the vectors do not adapt in this generation.`;
      return `t = ${GEN} is a multiple of ⌈0.1 · ${TMAX}⌉ = ${g.interval}, so the vectors adapt after the survival. z_max = ${pair(g.nadir)} holds the largest values of the survivors, and each initial vector v⁰ᵢ is multiplied by z_max − z* = ${pair(scale)}, element by element, and normalized again (solid; the dashed ones are the vectors before). The spacings γ are recomputed, and generation ${GEN + 1} selects with these vectors.`;
    }
    build();
  };

  /* I8: MOEA/D neighbourhoods and replacement. */
  widgets["moead-neighbourhood"] = function (node) {
    const N = 11;
    const W = moeadWeights(N);
    const P = [
      [1.1, 0.1],
      [0.925, 0.192],
      [0.786, 0.272],
      [0.671, 0.345],
      [0.571, 0.414],
      [0.482, 0.482],
      [0.401, 0.551],
      [0.325, 0.625],
      [0.252, 0.71],
      [0.18, 0.818],
      [0.1, 1.1],
    ];
    const state = { i: 5, T: 5, nr: 2, mode: "tch", pool: "neighbourhood", y: [0.39, 0.39], order: [6, 3, 5, 7, 4] };
    const ctl = controls(node);
    const pick = select(ctl, { label: "Subproblem", options: W.map((_, k) => [k, String(k)]), value: state.i, onChange: (v) => ((state.i = +v), resetOrder(), render()) });
    slider(ctl, { label: "T", min: 2, max: 11, step: 1, value: state.T, format: (v) => String(v), onInput: (v) => ((state.T = v), resetOrder(), render()) });
    slider(ctl, { label: "n_r", min: 1, max: 5, step: 1, value: state.nr, format: (v) => String(v), onInput: (v) => ((state.nr = v), render()) });
    select(ctl, {
      label: "Aggregation",
      options: [
        ["tch", "Tchebycheff"],
        ["pbi", "PBI (θ = 5)"],
        ["ws", "Weighted sum"],
      ],
      value: state.mode,
      onChange: (v) => ((state.mode = v), render()),
    });
    select(ctl, {
      label: "Mating pool",
      options: [
        ["neighbourhood", "Neighbourhood (probability δ = 0.9)"],
        ["population", "Whole population (probability 1 − δ)"],
      ],
      value: state.pool,
      onChange: (v) => ((state.pool = v), resetOrder(), render()),
    });
    button(ctl, "Shuffle order", () => {
      const a = state.order.slice();
      for (let k = a.length - 1; k > 0; k--) {
        const r = Math.floor(Math.random() * (k + 1));
        [a[k], a[r]] = [a[r], a[k]];
      }
      state.order = a;
      render();
    });
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, { x: [0, 1.2], y: [0, 1.2], xLabel: "f₁", yLabel: "f₂", label: "MOEA/D subproblems, the neighbourhood of the selected one and a draggable offspring" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Selected subproblem", shape: "line", cls: "vf-line-2" },
      { label: "Its neighbourhood (candidates)", shape: "line", cls: "vf-line-1" },
      { label: "Offspring y (drag it)", shape: "triangle", cls: "vf-pt vf-pt-2" },
      { label: "Replaced by y", shape: "circle", cls: "vf-ring" },
      { label: "Better, but n_r reached", shape: "circle", cls: "vf-pt-hollow" },
    ]);
    function owner(k) {
      return state.mode === "tch" ? k : N - 1 - k;
    }
    function resetOrder() {
      const B = moeadNeighbours(W, state.T)[state.i];
      state.order = state.pool === "population" ? W.map((_, k) => k) : B.slice();
    }
    const yHandle = handle(plot, {
      cls: "vf-pt vf-pt-2",
      shape: "triangle",
      r: 7,
      label: () => "Offspring y",
      get: () => state.y,
      set: (a, b) => ((state.y = [a, b]), render()),
    });
    function render() {
      pick.value = String(state.i);
      const current = W.map((_, k) => P[owner(k)]);
      const res = moeadReplace(current, W, state.y, state.order, state.nr, state.mode, 5);
      const candidates = new Set(state.order);
      const z = res.z;
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      W.forEach((w, k) => {
        const dir = state.mode === "tch" ? [1 / (w[0] || 1e-4), 1 / (w[1] || 1e-4)] : w;
        const len = Math.hypot(dir[0], dir[1]);
        const end = [z[0] + (3 * dir[0]) / len, z[1] + (3 * dir[1]) / len];
        const cls = k === state.i ? "vf-line-2" : candidates.has(k) ? "vf-line-1" : "vf-line-thin";
        const ray = plot.line(plot.layers.mid, [z, end], cls);
        ray.style.cursor = "pointer";
        ray.addEventListener("click", () => ((state.i = k), resetOrder(), render()));
      });
      marker(plot.layers.marks, "cross", plot.x(z[0]), plot.y(z[1]), 4, "vf-line-ink");
      txt(plot.layers.top, plot.x(z[0]) + 6, plot.y(z[1]) + 16, "z*", { class: "vf-mark-label" });
      const byJ = {};
      res.rows.forEach((r) => (byJ[r.j] = r));
      W.forEach((_, k) => {
        const p = current[k];
        const px = plot.x(p[0]);
        const py = plot.y(p[1]);
        const r = byJ[k];
        const mark = marker(plot.layers.marks, "circle", px, py, 5.5, candidates.has(k) ? "vf-pt vf-pt-1" : "vf-pt vf-pt-muted");
        mark.style.cursor = "pointer";
        mark.addEventListener("click", () => ((state.i = k), resetOrder(), render()));
        if (r && r.result === "replaced") {
          marker(plot.layers.marks, "circle", px, py, 10, "vf-ring");
          s("line", { x1: px, y1: py, x2: plot.x(state.y[0]), y2: plot.y(state.y[1]), class: "vf-line-2 vf-dashed" }, plot.layers.marks);
        } else if (r && r.result !== "kept") marker(plot.layers.marks, "circle", px, py, 10, "vf-pt-hollow");
        txt(plot.layers.top, px + 8, py - 7, String(k), { class: "vf-mark-label" });
      });
      yHandle.place();
      clear(readout);
      table(
        readout,
        ["Order", "Subproblem", "w", "g(current)", "g(offspring)", "Result"],
        res.rows.map((r, n) => [String(n + 1), String(r.j), pair(W[r.j], 1), fmt(r.gx, 4), fmt(r.gy, 4), r.result]),
        [true, true, false, true, true, false]
      );
      const better = res.rows.filter((r) => r.result !== "kept").length;
      h("p", {}, readout, `Ideal point z* = ${pair(z, 3)}. The offspring replaced ${res.replaced} of the ${better} subproblems it improves (n_r = ${state.nr}).`);
      if (state.pool === "population") h("p", {}, readout, "Mating pool: whole population (probability 1 − δ = 0.1): every subproblem is a candidate.");
      if (state.mode !== "tch") h("p", {}, readout, "With PBI and the weighted sum the optimum of a subproblem lies along w itself; with Tchebycheff along (1/w₁, 1/w₂). VAMOS gives subproblem 0 the weight (0, 1), so the labels run in opposite directions.");
    }
    render();
  };

  /* MOEA/D at a glance: one generation, one subproblem per step, with VAMOS' default operators and PBI. */
  widgets["moead-steps"] = function (node) {
    const N = 6;
    const T = 3;
    const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
    const SUB = "₀₁₂₃₄₅₆₇₈₉";
    const xk = (k) => "x" + SUP[k];
    const yk = (k) => "y" + SUB[k];
    const state = { seed: Number(node.dataset.seed || 7548), step: 0 };
    let g = null;
    let titles = [];
    const interesting = (c) =>
      c.steps.some((st) => !st.neighbourhood) && c.steps.some((st) => st.rows.some((r) => !r.visited && r.better)) && c.steps.some((st) => st.replaced.length === 0);
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw populations until a pool is the whole population, n_r stops a visit, and an offspring replaces nothing.
      for (let tries = 0; tries < 300; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(moeadGeneration(seed, N, T))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the subproblems' solutions, the parents and the offspring" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Weight vectors drawn from the ideal point, the subproblems' solutions and the offspring", margin: { t: 22, r: 28 } });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "Each point carries the numbers of the subproblems it solves. The arrow keys step through the figure when it has focus; New population draws another generation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Initial solution", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Offspring of this generation", shape: "triangle", cls: "vf-pt vf-pt-1" },
      { label: "Offspring y of the step", shape: "triangle", cls: "vf-pt vf-pt-2" },
      { label: "Parent, offspring that replaces", shape: "circle", cls: "vf-ring" },
      { label: "Weight vector of the subproblem; DE step 0.5·(x^r₁ − x^r₂) from xᵏ", shape: "line", cls: "vf-line-2" },
      { label: "Weight vectors of the mating pool; x^r₁ − x^r₂", shape: "line", cls: "vf-line-1" },
      { label: "Ideal point z*", shape: "cross", cls: "vf-line-ink" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = moeadGeneration(state.seed, N, T);
      titles = ["the subproblems and their solutions"].concat(g.order.map((k) => `subproblem ${k}`), ["the population after one generation"]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(titles.length - 1, i));
      render();
    }
    const name = (p) => (p.id <= N ? "initial" : yk(p.subproblem));
    const vec = (v, d) => pair(v, d === undefined ? 2 : d);
    const set = (list) => "{" + list.slice().sort((a, b) => a - b).join(", ") + "}";
    const pbi = (f, j, z) => moeadAggregate("pbi", f, g.W[j], z, g.theta);
    function holders(list) {
      const map = new Map();
      list.forEach((p, k) => {
        if (!map.has(p)) map.set(p, []);
        map.get(p).push(k);
      });
      return map;
    }
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(Math.min(v[0], plot.xd[1]));
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      if (label) txt(plot.layers.top, px + 9, py - 7, label, { class: "vf-mark-label" });
    }
    function rays(z, k, pool) {
      g.W.forEach((w, j) => {
        const len = Math.hypot(w[0], w[1]);
        const d = [w[0] / len, w[1] / len];
        const cls = j === k ? "vf-line-2" : pool && pool.includes(j) ? "vf-line-1" : "vf-line-thin";
        fPlot.line(fPlot.layers.mid, [z, [z[0] + 3 * d[0], z[1] + 3 * d[1]]], cls);
        const toRight = d[0] > 1e-9 ? (fPlot.xd[1] - z[0]) / d[0] : Infinity;
        const toTop = d[1] > 1e-9 ? (fPlot.yd[1] - z[1]) / d[1] : Infinity;
        const reach = Math.min(toRight, toTop);
        const px = fPlot.x(z[0] + reach * d[0]);
        const py = fPlot.y(z[1] + reach * d[1]);
        // The labels sit in the widened top and right margins, clear of the points.
        if (toTop < toRight) txt(fPlot.layers.top, px, py - 6, "w" + SUP[j], { class: "vf-mark-label", "text-anchor": "middle" });
        else txt(fPlot.layers.top, px + 4, py + 4, "w" + SUP[j], { class: "vf-mark-label" });
      });
    }
    function ideal(z, faded) {
      marker(fPlot.layers.marks, "cross", fPlot.x(z[0]), fPlot.y(z[1]), 4.5, "vf-line-ink" + (faded ? " vf-faded" : ""));
      if (!faded) txt(fPlot.layers.top, fPlot.x(z[0]) + 7, fPlot.y(z[1]) - 6, "z*", { class: "vf-mark-label" });
    }
    function population(list, opts) {
      for (const [p, ks] of holders(list)) {
        const lost = opts && opts.replaced ? ks.filter((j) => opts.replaced.includes(j)) : [];
        const gone = lost.length === ks.length;
        const label = ks.map((j) => String(j) + (lost.includes(j) ? " ✕" : "")).join(", ");
        const cls = "vf-pt vf-pt-1" + (gone ? " vf-faded" : "");
        const shape = p.id <= N ? "circle" : "triangle";
        const parent = opts && opts.parents ? ks.some((j) => opts.parents.includes(j)) : false;
        draw(xPlot, p.x, shape, cls, label, parent);
        draw(fPlot, p.f, shape, cls, label, false);
        if (lost.length) {
          const y = opts.child.f;
          s("line", { x1: fPlot.x(p.f[0]), y1: fPlot.y(Math.min(p.f[1], fPlot.yd[1])), x2: fPlot.x(y[0]), y2: fPlot.y(Math.min(y[1], fPlot.yd[1])), class: "vf-line-2 vf-dashed" }, fPlot.layers.mid);
        }
      }
    }
    function render() {
      const step = state.step;
      back.disabled = step === 0;
      next.disabled = step === titles.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.back, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.back, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      let headers;
      let rows;
      let numeric;
      if (step === 0 || step === titles.length - 1) {
        const list = step === 0 ? g.pop : g.final;
        const z = step === 0 ? g.z0 : g.steps[N - 1].z;
        rays(z, -1, null);
        if (step > 0) ideal(g.z0, true);
        ideal(z, false);
        population(list, null);
        headers = ["Subproblem", "w", step === 0 ? "B(k)" : "Solution", "x₁", "x₂", "f₁", "f₂", "g(xᵏ)"];
        rows = list.map((p, k) => [String(k), vec(g.W[k], 1), step === 0 ? set(g.B[k]) : name(p), fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), fmt(pbi(p.f, k, z), 3)]);
        numeric = [true, false, false, true, true, true, true, true];
      } else {
        const st = g.steps[step - 1];
        const y = st.child;
        rays(st.z, st.k, st.pool);
        if (st.z[0] !== st.zBefore[0] || st.z[1] !== st.zBefore[1]) ideal(st.zBefore, true);
        ideal(st.z, false);
        const diff = Math.hypot(st.before[st.r1].x[0] - st.before[st.r2].x[0], st.before[st.r1].x[1] - st.before[st.r2].x[1]);
        if (diff > 1e-9) arrow(xPlot, xPlot.layers.mid, st.before[st.r2].x, st.before[st.r1].x, "vf-line-1");
        const base = st.before[st.k].x;
        if (Math.hypot(y.v[0] - base[0], y.v[1] - base[1]) > 1e-9) arrow(xPlot, xPlot.layers.mid, base, y.v, "vf-line-2");
        if (Math.hypot(y.v[0] - y.x[0], y.v[1] - y.x[1]) > 1e-9) {
          xPlot.line(xPlot.layers.mid, [y.v, y.x], "vf-line-thin vf-dashed");
          marker(xPlot.layers.marks, "circle", xPlot.x(y.v[0]), xPlot.y(y.v[1]), 3, "vf-pt-hollow");
        }
        population(st.before, { replaced: st.replaced, parents: [st.r1, st.r2], child: y });
        const cls = "vf-pt vf-pt-2" + (st.replaced.length ? "" : " vf-faded");
        const label = yk(st.k) + (st.replaced.length ? "" : " ✕");
        draw(xPlot, y.x, "triangle", cls, label, false);
        draw(fPlot, y.f, "triangle", cls, label, st.replaced.length > 0);
        const visit = new Map(st.rows.map((r, n) => [r.j, Object.assign({ n }, r)]));
        headers = ["Subproblem", "Solution", "f₁", "f₂", "Visit", "g(current)", "g(y)", "Result"];
        rows = st.before.map((p, j) => {
          const r = visit.get(j);
          let result = "not in the pool";
          if (r) result = !r.visited ? "not visited (n_r reached)" : r.replaced ? "replaced by " + yk(st.k) : "kept";
          const shown = r && r.visited;
          return [String(j), name(p), fmt(p.f[0], 2), fmt(p.f[1], 2), r ? String(r.n + 1) : "–", shown ? fmt(r.gx, 3) : "–", shown ? fmt(r.gy, 3) : "–", result];
        });
        rows.push([yk(st.k), "offspring", fmt(y.f[0], 2), fmt(y.f[1], 2), "–", "–", "–", st.replaced.length ? "holds " + idList(st.replaced) : "discarded"]);
        numeric = [true, false, true, true, true, true, true, false];
      }
      stepText.textContent = `Step ${step + 1} of ${titles.length}: ${titles[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, headers, rows, numeric);
    }
    function describe(step) {
      if (step === 0) {
        return `Six subproblems, one per weight vector wᵏ = (k/5, 1 − k/5), each holding one random solution of the loop figure's problem (f₁ = x₁, f₂ = g·(1 − √(x₁/g)), g = 1 + x₂). The table lists the neighbourhoods B(k) of T = 3 vectors. The ideal point z* = ${vec(g.z0)} is the minimum of the population, and PBI (θ = 5), VAMOS' default aggregation, measures each solution along the ray of its weight vector from z*. The generation visits the subproblems in the random order ${g.order.join(", ")}.`;
      }
      if (step === titles.length - 1) {
        const groups = [...holders(g.final)].filter(([p]) => p.id > N);
        const kept = g.final.filter((p) => p.id <= N).length;
        const shared = groups.filter(([, ks]) => ks.length > 1).map(([p, ks]) => `${name(p)} holds subproblems ${idList(ks)}`);
        const zLast = g.steps[N - 1].z;
        const moved = zLast[0] !== g.z0[0] || zLast[1] !== g.z0[1];
        const children = groups.map(([p]) => name(p));
        const words = ["No", "One", "Two", "Three", "Four", "Five", "Six"];
        return `Every subproblem has been visited once: one generation of ${N} offspring. ${words[N - kept]} subproblem${N - kept === 1 ? "" : "s"} now hold${N - kept === 1 ? "s" : ""} an offspring of this generation${children.length ? ` (${idList(children)})` : ""}, and ${words[kept].toLowerCase()} keep${kept === 1 ? "s" : ""} the initial solution.${shared.length ? ` ${idList(shared)}: nothing stops subproblems from sharing a solution.` : ""} ${moved ? `The ideal point moved from ${vec(g.z0)} to ${vec(zLast)}.` : `The ideal point did not move.`} The next generation visits the subproblems in a new random order.`;
      }
      const st = g.steps[step - 1];
      const k = st.k;
      const y = st.child;
      const pool = st.neighbourhood
        ? `The draw ${fmt(st.u, 3)} < δ = 0.9 makes the neighbourhood B(${k}) = ${set(st.pool)} the mating pool.`
        : `The draw ${fmt(st.u, 3)} ≥ δ = 0.9 makes the whole population the mating pool.`;
      const same = st.before[st.r1] === st.before[st.r2];
      const raw = [0, 1].map((j) => st.before[k].x[j] + 0.5 * (st.before[st.r1].x[j] - st.before[st.r2].x[j]));
      const clipped = raw.some((t) => t < 0 || t > 1) ? ", clipped to the bounds" : "";
      const parents = `Its parents are ${xk(st.r1)} and ${xk(st.r2)}${same ? ", the same solution, so the difference is zero" : ""}: DE gives v = ${xk(k)} + 0.5·(${xk(st.r1)} − ${xk(st.r2)}) = ${vec(y.v)}${clipped}`;
      const changed = [0, 1].filter((j) => y.x[j] !== y.v[j]).map((j) => "x" + SUB[j + 1]);
      const mutation = changed.length ? `polynomial mutation changes ${idList(changed)}` : "polynomial mutation changes no variable";
      const moved = st.z[0] !== st.zBefore[0] || st.z[1] !== st.zBefore[1];
      const zText = moved ? `The ideal point moves from ${vec(st.zBefore)} to ${vec(st.z)} before the comparisons.` : `The ideal point stays at ${vec(st.z)}.`;
      const visited = st.rows.filter((r) => r.visited);
      const verdicts = visited.map((r) => (r.replaced ? `replaces ${xk(r.j)} (g ${fmt(r.gy, 2)} < ${fmt(r.gx, 2)})` : `leaves ${xk(r.j)} (g ${fmt(r.gy, 2)} ≥ ${fmt(r.gx, 2)})`));
      const skipped = st.rows.filter((r) => !r.visited).map((r) => r.j);
      const better = st.rows.filter((r) => !r.visited && r.better).map((r) => r.j);
      const stop = skipped.length ? ` After n_r = 2 replacements the visit stops before ${idList(skipped)}${better.length ? `, although ${yk(k)} would also improve ${idList(better)}` : ""}.` : "";
      const none = st.replaced.length ? "" : ` ${yk(k)} replaces nothing and is discarded.`;
      return `${pool} ${parents}, and ${mutation}: ${yk(k)} = ${vec(y.x)}, with f = ${vec(y.f)}. ${zText} ${yk(k)} visits the pool in the order ${st.candidates.join(", ")}: it ${idList(verdicts)}.${stop}${none}`;
    }
    build();
  };

  function arrow(plot, layer, from, to, cls) {
    const x1 = plot.x(from[0]);
    const y1 = plot.y(from[1]);
    const x2 = plot.x(to[0]);
    const y2 = plot.y(to[1]);
    const len = Math.hypot(x2 - x1, y2 - y1);
    if (len < 1) return;
    const ux = (x2 - x1) / len;
    const uy = (y2 - y1) / len;
    s("line", { x1, y1, x2: x2 - ux * 5, y2: y2 - uy * 5, class: cls }, layer);
    s("path", { d: `M${x2},${y2}L${x2 - ux * 8 - uy * 4},${y2 - uy * 8 + ux * 4}L${x2 - ux * 8 + uy * 4},${y2 - uy * 8 - ux * 4}Z`, class: cls.replace("vf-line", "vf-pt").replace(" vf-dashed", "") }, layer);
  }

  /* I11: one SMPSO velocity update. */
  widgets["pso-velocity"] = function (node) {
    const state = { x: [0.25, 0.3], v: [0.1, 0.05], p: [0.4, 0.2], g: [0.75, 0.7], c1: 2.0, c2: 1.9, r1: 0.6, r2: 0.8, clamp: true };
    const ctl = controls(node);
    const c1s = slider(ctl, { label: "C1", min: 1.5, max: 2.5, step: 0.05, value: state.c1, format: (v) => v.toFixed(2), onInput: (v) => ((state.c1 = v), render()) });
    const c2s = slider(ctl, { label: "C2", min: 1.5, max: 2.5, step: 0.05, value: state.c2, format: (v) => v.toFixed(2), onInput: (v) => ((state.c2 = v), render()) });
    button(ctl, "Resample r1, r2", () => ((state.r1 = Math.random()), (state.r2 = Math.random()), render()));
    button(ctl, "Draw as VAMOS", () => {
      state.c1 = Math.round((1.5 + Math.random()) * 20) / 20;
      state.c2 = Math.round((1.5 + Math.random()) * 20) / 20;
      state.r1 = Math.random();
      state.r2 = Math.random();
      c1s.set(state.c1);
      c2s.set(state.c2);
      render();
    });
    checkbox(ctl, { label: "Speed constraint", checked: true, onChange: (v) => ((state.clamp = v), render()) });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [-0.1, 1.1], y: [-0.1, 1.1], xLabel: "x₁", yLabel: "x₂", label: "One particle of SMPSO: inertia, cognitive and social terms, constriction, clamping and the new position" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Inertia w·v", shape: "line", cls: "vf-line-ink" },
      { label: "Cognitive, towards the personal best p", shape: "line", cls: "vf-line-1" },
      { label: "Social, towards the leader g", shape: "line", cls: "vf-line-3" },
      { label: "Unconstrained velocity", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Velocity after χ and clamping", shape: "line", cls: "vf-line-2" },
      { label: "Speed limit box", shape: "region", cls: "vf-region-1" },
    ]);
    const mk = (key, label, cls, shape) =>
      handle(plot, { cls, shape, r: 6, label: () => label, get: () => state[key], set: (a, b) => ((state[key] = [Math.min(1, Math.max(0, a)), Math.min(1, Math.max(0, b))]), render()), boundsX: [0, 1], boundsY: [0, 1] });
    const handles = [mk("x", "Particle x", "vf-pt vf-pt-ink"), mk("p", "Personal best p", "vf-pt vf-pt-1", "diamond"), mk("g", "Leader g", "vf-pt vf-pt-3", "square")];
    function render() {
      const r = smpsoStep({ ...state, w: 0.1, vmax: 0.5, lower: [0, 0], upper: [1, 1] });
      const x = state.x;
      clear(plot.layers.back);
      clear(plot.layers.mid);
      clear(plot.layers.top);
      plot.rect(plot.layers.back, x[0] - r.delta[0], x[1] - r.delta[1], x[0] + r.delta[0], x[1] + r.delta[1], "vf-region-1");
      plot.line(plot.layers.mid, [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], "vf-line-ink");
      const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
      arrow(plot, plot.layers.mid, [x[0] - state.v[0], x[1] - state.v[1]], x, "vf-line-thin");
      const a1 = add(x, r.inertia);
      const a2 = add(a1, r.cognitive);
      const a3 = add(a2, r.social);
      arrow(plot, plot.layers.mid, x, a1, "vf-line-ink");
      arrow(plot, plot.layers.mid, a1, a2, "vf-line-1");
      arrow(plot, plot.layers.mid, a2, a3, "vf-line-3");
      plot.line(plot.layers.mid, [x, a3], "vf-line-thin vf-dashed");
      arrow(plot, plot.layers.mid, x, add(x, r.clamped), "vf-line-2");
      marker(plot.layers.mid, "circle", plot.x(r.position[0]), plot.y(r.position[1]), 6, "vf-pt vf-pt-2");
      handles.forEach((hd) => hd.place());
      [
        [x, "x"],
        [state.p, "p"],
        [state.g, "g"],
        [r.position, "x′"],
      ].forEach(([q, label]) => txt(plot.layers.top, plot.x(q[0]) + 9, plot.y(q[1]) + 15, label, { class: "vf-mark-label" }));
      clear(readout);
      const v3 = (v) => `(${fmt(v[0])}, ${fmt(v[1])})`;
      h("p", {}, readout, `φ = C1 + C2 = ${fmt(r.phi, 2)} → χ = ${r.phi <= 4 ? "1 (φ ≤ 4)" : fmt(r.chi, 4)}; r1 = ${fmt(state.r1, 2)}, r2 = ${fmt(state.r2, 2)}.`);
      h("p", {}, readout, `w·v = ${v3(r.inertia)}   C1·r1·(p − x) = ${v3(r.cognitive)}   C2·r2·(g − x) = ${v3(r.social)}`);
      h("p", {}, readout, `unconstrained v = ${v3(r.raw)}   χ·v = ${v3(r.constricted)}   ${state.clamp ? "clamped" : "unclamped"} v′ = ${v3(r.clamped)}`);
      h("p", {}, readout, `new position x′ = ${v3(r.position)}`);
      if (r.phi > 4) h("p", {}, readout, `χ < 0: VAMOS evaluates 2 / (2 − φ − √(φ² − 4φ)) without an absolute value, so the step points away from the guides; Clerc and Kennedy's coefficient is |χ| = ${fmt(Math.abs(r.chi), 4)}.`);
      r.hit.forEach(([j, side]) => h("p", {}, readout, `x${j === 0 ? "₁" : "₂"} hit the ${side} bound: x${j === 0 ? "₁" : "₂"} = ${side === "lower" ? 0 : 1} and its velocity reverses.`));
    }
    render();
  };

  /** Labels for the marks of a step-by-step figure: marks at the same place share one label, such as a particle and
      the leader it found, and each label takes the place around its mark that covers the fewest marks and labels. */
  function placeMergedLabels(plot, items, spots) {
    const groups = [];
    for (const item of items) {
      const group = groups.find((other) => Math.hypot(other.px - item.px, other.py - item.py) < 1);
      if (!group) groups.push({ px: item.px, py: item.py, text: [item.text] });
      else if (!group.text.includes(item.text)) group.text.push(item.text);
    }
    const boxes = spots.map((p) => [p.px - p.r, p.py - p.r, p.px + p.r, p.py + p.r]);
    const overlap = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const places = [
      [9, -7, "start"],
      [9, 15, "start"],
      [-9, -7, "end"],
      [-9, 15, "end"],
      [0, -14, "middle"],
      [0, 24, "middle"],
    ];
    for (const group of groups) {
      const text = group.text.join(", ");
      const width = 6.4 * text.length;
      let best = null;
      for (const [dx, dy, anchor] of places) {
        const x = group.px + dx;
        const y = group.py + dy;
        const left = anchor === "start" ? x : anchor === "end" ? x - width : x - width / 2;
        const box = [left, y - 10, left + width, y + 2];
        let cost = boxes.reduce((sum, b) => sum + overlap(box, b), 0);
        if (box[0] < 0 || box[2] > plot.W || box[1] < 0 || box[3] > plot.H) cost += 1e4;
        if (!best || cost < best.cost) best = { cost, x, y, anchor, box };
      }
      boxes.push(best.box);
      txt(plot.layers.top, best.x, best.y, text, { class: "vf-mark-label", "text-anchor": best.anchor });
    }
  }

  /* SMPSO in depth: one iteration in nine steps, from the swarm and its leaders to the updated leaders archive. */
  widgets["smpso-steps"] = function (node) {
    const N = 6;
    const TITLES = ["the swarm and its leaders", "leader selection", "the velocity update", "constriction and the speed limit", "the move and the bounds", "turbulence", "evaluation", "personal bests", "the leaders archive"];
    const SUB = ["₁", "₂"];
    const state = { seed: Number(node.dataset.seed || 60550), step: 0 };
    let g = null;
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    const shows = (c) =>
      c.leaders.length > 2 &&
      c.moves.some((m) => m.chi < 0) &&
      c.moves.some((m) => m.chi === 1) &&
      c.moves.some((m) => m.constricted.some((v, j) => Math.abs(v) > m.delta[j])) &&
      c.moves.some((m) => Math.hypot(m.cognitive[0], m.cognitive[1]) > 0.1) &&
      c.turbulence.some((t) => t.mutated.length > 0) &&
      c.pbest.some((p) => p.replaced) &&
      c.pbest.some((p) => !p.replaced) &&
      c.inserts.some((e) => e.accepted) &&
      c.inserts.some((e) => !e.accepted);
    button(ctl, "New population", () => {
      // Draw swarms until the iteration shows both signs of χ, a clamped step, a visible pull towards a personal best,
      // a turbulence that moves particle 1, kept and replaced personal bests, and particles that enter the archive and
      // particles it rejects.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (shows(smpsoGeneration(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [-0.05, 1.05], y: [-0.05, 1.05], xLabel: "x₁", yLabel: "x₂", label: "Positions, velocities, personal bests and leaders of one SMPSO iteration" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors of the particles, their personal bests and the leaders" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another swarm.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Particle", shape: "circle", cls: "vf-pt vf-pt-ink" },
      { label: "New position", shape: "triangle", cls: "vf-pt vf-pt-2" },
      { label: "Personal best p", shape: "diamond", cls: "vf-pt vf-pt-1" },
      { label: "Leader", shape: "square", cls: "vf-pt vf-pt-3" },
      { label: "Velocity v", shape: "line", cls: "vf-line-thin" },
      { label: "Inertia w·v", shape: "line", cls: "vf-line-ink" },
      { label: "Pull towards p", shape: "line", cls: "vf-line-1" },
      { label: "Pull towards the leader g", shape: "line", cls: "vf-line-3" },
      { label: "Velocity after χ and the speed limit", shape: "line", cls: "vf-line-2" },
      { label: "Chosen leader, personal best, new leader", shape: "circle", cls: "vf-ring" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = smpsoGeneration(state.seed, N);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const same = (a, b) => a[0] === b[0] && a[1] === b[1];
    const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
    // A vector of the tables and texts: two decimals, no negative zero, and no line break inside.
    const num = (v) => fmt(Math.abs(v) < 0.005 ? 0 : v, 2);
    const vec = (v) => `(${num(v[0])}, ${num(v[1])})`;
    // The objective vector of a leader, also of one that entered in this iteration (the new position of its particle).
    const leaderF = (name) => {
      const old = g.leaders.find((l) => l.name === name);
      return old ? old.f : g.after[g.inserts.find((e) => e.leader === name).id - 1].f;
    };
    let labels = [];
    let spots = [];
    function mark(plot, v, shape, cls, label, o) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      const r = (o && o.r) || 5.5;
      marker(plot.layers.marks, shape, px, py, r, cls);
      if (o && o.ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      spots.push({ plot, px, py, r: o && o.ring ? 11 : r + 1 });
      if (label) labels.push({ plot, px, py, text: label });
    }
    function flushLabels() {
      for (const plot of [xPlot, fPlot]) placeMergedLabels(plot, labels.filter((l) => l.plot === plot), spots.filter((p) => p.plot === plot));
      labels = [];
      spots = [];
    }
    // Leaders are larger squares drawn first, so that a particle at a leader's place shows inside it.
    function drawLeaders(list, ringed, out) {
      list.forEach((l) => {
        const gone = !!(out && out.has(l.name));
        const cls = "vf-pt vf-pt-3" + (gone ? " vf-faded" : "");
        const o = { r: 9, ring: !!(ringed && ringed.has(l.name)) };
        mark(xPlot, l.x, "square", cls, l.name + (gone ? " ✕" : ""), o);
        mark(fPlot, l.f, "square", cls, l.name + (gone ? " ✕" : ""), o);
      });
    }
    function drawBests(o) {
      g.before.forEach((q, i) => {
        if (same(q.p, q.x) && !(o && o.all)) return;
        const replaced = o && o.all && g.pbest[i].replaced;
        const cls = "vf-pt vf-pt-1" + (replaced ? " vf-faded" : "");
        const label = "p" + q.id + (replaced ? " ✕" : "");
        const ring = o && o.all && !replaced;
        mark(xPlot, q.p, "diamond", cls, label, { ring });
        mark(fPlot, q.pf, "diamond", cls, label, { ring });
        if (!(o && o.all)) {
          xPlot.line(xPlot.layers.mid, [q.x, q.p], "vf-line-thin vf-dashed");
          fPlot.line(fPlot.layers.mid, [q.f, q.pf], "vf-line-thin vf-dashed");
        }
      });
    }
    function drawParticles(faded) {
      g.before.forEach((q) => {
        const cls = "vf-pt vf-pt-ink" + (faded ? " vf-faded" : "");
        mark(xPlot, q.x, "circle", cls, faded ? null : String(q.id));
        mark(fPlot, q.f, "circle", cls, faded ? null : String(q.id));
      });
    }
    function drawMoved(o) {
      g.after.forEach((q, i) => {
        const t = g.turbulence.find((e) => e.index === i);
        const x = o && o.unmutated ? g.moves[i].position : q.x;
        mark(xPlot, x, "triangle", "vf-pt vf-pt-2", String(q.id), { ring: o && o.ring && o.ring(i, t) });
        if (o && o.f) mark(fPlot, q.f, "triangle", "vf-pt vf-pt-2", String(q.id), { ring: o.ring && o.ring(i, t) });
      });
    }
    function render() {
      const step = state.step;
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.mid, [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], "vf-line-thin");
      xPlot.line(xPlot.layers.mid, [[0, 0], [1, 0]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const chosen = new Set(g.tournaments.map((t) => g.leaders[t.winner].name));
      let headers;
      let rows;
      let extra = null;
      if (step === 0 || step === 1) {
        drawLeaders(g.leaders, step === 1 ? chosen : null);
        drawBests();
        drawParticles(false);
        g.before.forEach((q, i) => {
          if (step === 0) arrow(xPlot, xPlot.layers.mid, q.x, add(q.x, q.v), "vf-line-thin");
          else xPlot.line(xPlot.layers.mid, [q.x, g.moves[i].g], "vf-line-thin vf-dashed");
        });
        if (step === 0) {
          headers = ["Particle", "x₁", "x₂", "v₁", "v₂", "p₁", "p₂", "f₁", "f₂"];
          rows = g.before.map((q) => [String(q.id), fmt(q.x[0], 2), fmt(q.x[1], 2), fmt(q.v[0], 2), fmt(q.v[1], 2), same(q.p, q.x) ? "= x₁" : fmt(q.p[0], 2), same(q.p, q.x) ? "= x₂" : fmt(q.p[1], 2), fmt(q.f[0], 2), fmt(q.f[1], 2)]);
        } else {
          headers = ["Particle", "Leaders drawn", "Their crowding", "Leader g", "Why"];
          const why = { crowding: "larger crowding", coin: "tie, coin", first: "at most two leaders: the first" };
          rows = g.tournaments.map((t) => {
            const drawn = t.b === null ? [t.a] : [t.a, t.b];
            return [String(t.id), drawn.map((k) => g.leaders[k].name).join(" vs "), drawn.map((k) => fmt(g.leaders[k].crowding, 2)).join(" vs "), g.leaders[t.winner].name, why[t.reason]];
          });
        }
        extra = { headers: ["Leader", "Found by particle", "f₁", "f₂", "Crowding"], rows: g.leaders.map((l) => [l.name, String(l.from), fmt(l.f[0], 2), fmt(l.f[1], 2), fmt(l.crowding, 2)]) };
      } else if (step === 2) {
        drawLeaders(g.leaders.filter((l) => chosen.has(l.name)));
        drawBests();
        drawParticles(false);
        g.moves.forEach((m, i) => {
          const x = g.before[i].x;
          const a1 = add(x, m.inertia);
          const a2 = add(a1, m.cognitive);
          arrow(xPlot, xPlot.layers.mid, x, a1, "vf-line-ink");
          arrow(xPlot, xPlot.layers.mid, a1, a2, "vf-line-1");
          arrow(xPlot, xPlot.layers.mid, a2, add(a2, m.social), "vf-line-3");
        });
        headers = ["Particle", "w·v", "C₁", "r₁", "C₁r₁(p − x)", "C₂", "r₂", "C₂r₂(g − x)", "Sum"];
        rows = g.moves.map((m) => [String(m.id), vec(m.inertia), fmt(m.c1, 2), fmt(m.r1, 2), vec(m.cognitive), fmt(m.c2, 2), fmt(m.r2, 2), vec(m.social), vec(m.raw)]);
      } else if (step === 3) {
        drawParticles(false);
        g.moves.forEach((m, i) => {
          const x = g.before[i].x;
          xPlot.line(xPlot.layers.mid, [x, add(x, m.raw)], "vf-line-thin vf-dashed");
          if (!same(m.constricted, m.clamped)) arrow(xPlot, xPlot.layers.mid, x, add(x, m.constricted), "vf-line-thin");
          arrow(xPlot, xPlot.layers.mid, x, add(x, m.clamped), "vf-line-2");
        });
        headers = ["Particle", "φ = C₁ + C₂", "χ", "Sum", "χ · sum", "Clamped to ±0.5"];
        rows = g.moves.map((m) => [String(m.id), fmt(m.phi, 2), fmt(m.chi, 2), vec(m.raw), vec(m.constricted), vec(m.clamped)]);
      } else if (step === 4) {
        drawParticles(true);
        g.before.forEach((q) => mark(fPlot, q.f, "circle", "vf-pt vf-pt-ink", String(q.id)));
        g.moves.forEach((m, i) => {
          const x = g.before[i].x;
          if (m.hit.length) xPlot.line(xPlot.layers.mid, [x, m.tentative], "vf-line-thin vf-dashed");
          arrow(xPlot, xPlot.layers.mid, x, m.position, "vf-line-2");
        });
        drawMoved({ unmutated: true });
        headers = ["Particle", "x", "v", "x + v", "Bound", "New x", "New v"];
        rows = g.moves.map((m, i) => [String(m.id), vec(g.before[i].x), vec(m.clamped), vec(m.tentative), m.hit.length ? m.hit.map(([j, side]) => `x${SUB[j]} ${side}`).join(", ") : "–", vec(m.position), vec(m.velocity)]);
      } else if (step === 5) {
        g.before.forEach((q) => mark(fPlot, q.f, "circle", "vf-pt vf-pt-ink", String(q.id)));
        g.turbulence.forEach((t) => {
          mark(xPlot, t.before, "triangle", "vf-pt vf-pt-2 vf-faded", null);
          if (!same(t.before, t.after)) arrow(xPlot, xPlot.layers.mid, t.before, t.after, "vf-line-thin");
        });
        drawMoved({ ring: (i, t) => !!t });
        headers = ["Particle", "Index", "Turbulence", "Mutation draws", "u", "Position"];
        rows = g.moves.map((m, i) => {
          const t = g.turbulence.find((e) => e.index === i);
          if (!t) return [String(m.id), String(i), "–", "–", "–", vec(m.position)];
          return [String(m.id), String(i), t.mutated.length ? "x" + t.mutated.map((j) => SUB[j]).join(", x") : "none", t.mask.map((v) => fmt(v, 2)).join(", "), t.u.map((v) => (v === null ? "–" : fmt(v, 2))).join(", "), `${vec(t.before)} → ${vec(t.after)}`];
        });
      } else if (step === 6) {
        drawMoved({ f: true });
        g.before.forEach((q, i) => {
          mark(fPlot, q.f, "circle", "vf-pt vf-pt-ink vf-faded", null);
          fPlot.line(fPlot.layers.mid, [q.f, g.after[i].f], "vf-line-thin vf-dashed");
        });
        headers = ["Particle", "x₁", "x₂", "Old f", "New f", "New against old"];
        rows = g.after.map((q, i) => [String(q.id), fmt(q.x[0], 2), fmt(q.x[1], 2), vec(g.before[i].f), vec(q.f), relation(q.f, g.before[i].f)]);
      } else if (step === 7) {
        drawBests({ all: true });
        drawMoved({ f: true, ring: (i) => g.pbest[i].replaced });
        g.before.forEach((q, i) => {
          if (g.pbest[i].replaced) return;
          xPlot.line(xPlot.layers.mid, [g.after[i].x, q.p], "vf-line-thin vf-dashed");
          fPlot.line(fPlot.layers.mid, [g.after[i].f, q.pf], "vf-line-thin vf-dashed");
        });
        headers = ["Particle", "New f", "f(p)", "f(p) ≺ f(x)", "Personal best"];
        rows = g.after.map((q, i) => {
          const p = g.before[i];
          const kept = !g.pbest[i].replaced;
          return [String(q.id), vec(q.f), vec(p.pf), kept ? "yes" : "no", kept ? "kept" : dominates(q.f, p.pf) ? "replaced: better" : same(q.f, p.pf) ? "replaced: equal" : "replaced: incomparable"];
        });
      } else {
        const removed = new Set();
        const fate = new Map();
        g.inserts.forEach((e) => {
          e.removed.forEach((name) => {
            removed.add(name);
            fate.set(name, `removed: dominated by ${e.leader}`);
          });
          if (e.pruned) {
            removed.add(e.pruned);
            fate.set(e.pruned, `removed: most crowded (${fmt(e.crowding.find((c) => c.name === e.pruned).crowding, 2)})`);
          }
        });
        const entered = g.inserts.filter((e) => e.accepted);
        const fresh = entered.map((e) => e.leader);
        // A new leader is the new position of the particle that entered; it can leave again in the same iteration.
        const all = g.leaders.concat(entered.map((e) => ({ name: e.leader, from: e.id, x: g.after[e.id - 1].x, f: g.after[e.id - 1].f })));
        const finalCrowding = new Map(g.leadersAfter.map((l) => [l.name, l.crowding]));
        drawLeaders(all, new Set(fresh.filter((name) => !removed.has(name))), removed);
        drawMoved({ f: true });
        headers = ["Particle", "f₁", "f₂", "Leaders archive"];
        rows = g.after.map((q, i) => [String(q.id), fmt(q.f[0], 2), fmt(q.f[1], 2), insertText(g.inserts[i])]);
        extra = {
          headers: ["Leader", "Found by particle", "f₁", "f₂", "Crowding", "Status"],
          rows: all.map((l) => [l.name, String(l.from), fmt(l.f[0], 2), fmt(l.f[1], 2), removed.has(l.name) ? "–" : fmt(finalCrowding.get(l.name), 2), removed.has(l.name) ? fate.get(l.name) : fresh.includes(l.name) ? "new" : "kept"]),
        };
      }
      flushLabels();
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      // Single numbers align right; ids, names, vectors and words stay left.
      const single = new Set(["x₁", "x₂", "v₁", "v₂", "p₁", "p₂", "f₁", "f₂", "C₁", "C₂", "r₁", "r₂", "φ = C₁ + C₂", "χ", "Crowding"]);
      const numeric = (list) => list.map((label) => single.has(label));
      table(tableBox, headers, rows, numeric(headers));
      if (extra) table(tableBox, extra.headers, extra.rows, numeric(extra.headers));
    }
    function relation(a, b) {
      if (dominates(a, b)) return "better";
      if (dominates(b, a)) return "worse";
      return same(a, b) ? "equal" : "incomparable";
    }
    function insertText(e) {
      if (!e.accepted) {
        return `rejected: ${e.by} ${same(leaderF(e.by), g.after[e.id - 1].f) ? "equals" : "dominates"} it`;
      }
      let text = `enters as ${e.leader}`;
      if (e.removed.length) text += `, removes ${idList(e.removed)}`;
      if (e.pruned) {
        const d = fmt(e.crowding.find((c) => c.name === e.pruned).crowding, 2);
        text += e.pruned === e.leader ? `, then leaves again as the most crowded of ${N + 1} leaders (${d})` : `; the archive then holds ${N + 1} leaders, so ${e.pruned}, the most crowded (${d}), leaves`;
      }
      return text;
    }
    function describe(step) {
      const ids = (list) => idList(list.map((q) => q.id));
      if (step === 0) {
        const differ = g.before.filter((q) => !same(q.p, q.x));
        const pb = differ.length
          ? `Personal best and position differ for ${ids(differ)}: ${differ.length === 1 ? "its" : "their"} last move was dominated by the personal best.`
          : "Every personal best equals its particle's position: no last move was dominated by the personal best.";
        const lead = g.leaders.map((l) => `${l.name} · ${fmt(l.crowding, 2)}`).join(", ");
        return `Six particles of a two-variable problem whose Pareto front is ZDT1's, after one iteration from random positions with zero velocity. Each has a position x, a velocity v (the arrows) and a personal best p. ${pb} The leaders archive holds the ${g.leaders.length} non-dominated positions found so far, with their crowding distances: ${lead}.`;
      }
      if (step === 1) {
        if (g.leaders.length <= 2) return `The archive holds at most two leaders, so every particle takes the first one, ${g.leaders[0].name}, as its guide g.`;
        const games = g.tournaments.map((t) => {
          const a = g.leaders[t.a];
          const b = g.leaders[t.b];
          const w = g.leaders[t.winner];
          return `${t.id}: ${a.name} vs ${b.name} → ${w.name}${t.reason === "coin" ? " (tie, coin)" : ` (${fmt(w.crowding, 2)} against ${fmt((w === a ? b : a).crowding, 2)})`}`;
        });
        return `Each particle draws two distinct leaders and takes the one with the larger crowding distance as its guide g; a tie is decided by a coin. ${games.join("; ")}. The dashed lines join each particle to its guide.`;
      }
      if (step === 2) {
        const pulled = g.moves.filter((m, i) => !same(g.before[i].p, g.before[i].x));
        const cog = pulled.length ? `The pull towards p is zero except for ${idList(pulled.map((m) => m.id))}, whose personal best differs from x.` : "The pull towards p is zero for every particle, because p = x.";
        const top = g.moves.reduce((a, b) => (Math.hypot(...b.social) > Math.hypot(...a.social) ? b : a));
        return `v ← w·v + C₁r₁(p − x) + C₂r₂(g − x), with w = 0.1 and one C₁ and C₂ (uniform in [1.5, 2.5]) and one r₁ and r₂ (uniform in [0, 1]) per particle, shared by both variables. The arrows add the three terms tip to tail; an arrow can continue beyond the plot. ${cog} The longest pull is ${top.id}'s towards its leader: C₂ = ${fmt(top.c2, 2)} and r₂ = ${fmt(top.r2, 2)} give ${vec(top.social)}.`;
      }
      if (step === 3) {
        const one = g.moves.filter((m) => m.chi === 1);
        const neg = g.moves.filter((m) => m.chi < 0);
        const parts = [];
        if (one.length) parts.push(`φ ≤ 4 gives χ = 1 for ${idList(one.map((m) => m.id))}`);
        if (neg.length) parts.push(`φ > 4 gives χ = 2 / (2 − φ − √(φ² − 4φ)) < 0 for ${neg.map((m) => `${m.id} (φ = ${fmt(m.phi, 2)}, χ = ${fmt(m.chi, 2)})`).join(", ")}, which reverses the step, as in VAMOS and jMetal`);
        // Enough digits to tell a clamped component from the limit.
        const exact = (v, limit) => {
          let digits = 2;
          while (digits < 6 && fmt(v, digits) === fmt(limit, digits)) digits += 1;
          return fmt(v, digits);
        };
        const cut = g.moves
          .filter((m) => m.constricted.some((v, j) => Math.abs(v) > m.delta[j]))
          .map((m) => `${m.id}'s from (${m.constricted.map((v, j) => (Math.abs(v) > m.delta[j] ? exact(v, m.clamped[j]) : num(v))).join(", ")}) to ${vec(m.clamped)}`);
        const clamp = cut.length ? `The speed limit δ = 0.5, half the range of each variable, then clamps every component to [−0.5, 0.5]: it cuts ${idList(cut)}.` : "No component exceeds the speed limit δ = 0.5, half the range of each variable.";
        return `φ = C₁ + C₂ decides the constriction χ: ${parts.join("; ")}. ${clamp} The dashed lines are the sums before χ; the orange arrows the velocities.`;
      }
      if (step === 4) {
        const hits = g.moves
          .filter((m) => m.hit.length)
          .map((m) => {
            const one = m.hit.length === 1;
            const bounds = m.hit.map(([j, side]) => `x${SUB[j]} = ${side === "lower" ? 0 : 1}`);
            const changes = m.hit.map(([j]) => `v${SUB[j]} from ${fmt(m.clamped[j], 2)} to ${fmt(m.velocity[j], 2)}`);
            return `Particle ${m.id} would cross ${idList(bounds)}: it stops on ${one ? "that bound" : "those bounds"}, and its velocity changes sign there, ${idList(changes)}.`;
          });
        return `Each particle moves by its velocity, x ← x + v, to the triangles. ${hits.length ? hits.join(" ") : "No particle crosses a bound."} The velocities are kept for the next iteration's inertia.`;
      }
      if (step === 5) {
        const t = g.turbulence[0];
        const who = `Polynomial mutation (probability 1/n = 1/2 per variable, η = 20) perturbs every sixth particle, indices 0, 6, 12, …: here particle ${t.id} only.`;
        if (!t.mutated.length) return `${who} Both of its draws, ${t.mask.map((v) => fmt(v, 2)).join(" and ")}, exceed 1/2, so it does not change this time.`;
        const moved = t.mutated.map((j) => `x${SUB[j]} (draw ${fmt(t.mask[j], 2)} ≤ 1/2) from ${fmt(t.before[j], 2)} to ${fmt(t.after[j], 2)}`);
        return `${who} It mutates ${idList(moved)}. Its velocity does not change; the turbulence moves only the position.`;
      }
      if (step === 6) {
        const worse = g.after.filter((q, i) => dominates(g.before[i].f, q.f));
        const tail = worse.length
          ? `${ids(worse)} moved to ${worse.length === 1 ? "a position that its old one dominates" : "positions that their old ones dominate"}: the swarm always moves, and only the personal bests and the archive keep the good positions.`
          : "No particle moved to a position that its old one dominates, but the swarm always moves: only the personal bests and the archive keep the good positions.";
        return `The new positions are evaluated; the dashed lines join each particle's old and new objective vectors. ${tail}`;
      }
      if (step === 7) {
        const kept = g.after.filter((q, i) => !g.pbest[i].replaced);
        const better = g.after.filter((q, i) => g.pbest[i].replaced && dominates(q.f, g.before[i].pf));
        const other = g.after.filter((q, i) => g.pbest[i].replaced && !dominates(q.f, g.before[i].pf));
        const parts = [];
        if (better.length) parts.push(`${ids(better)} ${better.length === 1 ? "replaces its personal best with a better position" : "replace their personal bests with better positions"}`);
        if (other.length) parts.push(`${ids(other)} ${other.length === 1 ? "replaces its personal best with an incomparable position" : "replace their personal bests with incomparable positions"}`);
        if (kept.length) parts.push(`${ids(kept)} ${kept.length === 1 ? "keeps its personal best, which dominates the new position" : "keep their personal bests, which dominate the new positions"}`);
        return `The new position replaces the personal best unless the personal best dominates it: p ← x unless f(p) ≺ f(x). ${parts.join("; ")}. Rings mark the personal bests after the update.`;
      }
      const events = g.inserts.map((e) => `${e.id} ${insertText(e)}`);
      const final = g.leadersAfter.map((l) => l.name);
      return `The new positions enter the archive one at a time, in particle order: ${events.join("; ")}. The archive, ${idList(final)}, is what VAMOS returns by default, and its crowding distances drive the next iteration's leader tournaments.`;
    }
    build();
  };

  /** SMS-EMOA survival of N + 1 solutions (VAMOS' smsemoa/helpers.py, relative or jMetal reference point). */
  function smsemoaStep(Q, offset, jmetal) {
    const sorted = nonDominatedSort(Q);
    const last = sorted.fronts[sorted.fronts.length - 1];
    const top = Q[0].map((_, j) => Math.max(...Q.map((q) => q[j])));
    const low = Q[0].map((_, j) => Math.min(...Q.map((q) => q[j])));
    const ref = top.map((t, j) => (jmetal ? (t >= 0 ? 1.1 * t : t / 1.1) : t + offset * (t - low[j] > 0 ? t - low[j] : 1)));
    let contributions = null;
    let boxes = null;
    let removed = last[0];
    if (last.length > 1) {
      const res = hvContributions2d(
        last.map((i) => Q[i]),
        ref
      );
      contributions = res.contributions;
      boxes = res.boxes;
      let k = 0;
      contributions.forEach((c, i) => {
        if (c < contributions[k]) k = i;
      });
      removed = last[k];
    }
    return { fronts: sorted.fronts, last, ref, contributions, boxes, removed };
  }

  /** One SMS-EMOA step of the loop figure's problem, recorded for the SMS-EMOA lesson as VAMOS runs it by default:
      two parents drawn uniformly at random, with replacement; the first child of SBX (probability 0.9, η = 20),
      mutated by polynomial mutation (probability 1/2, η = 20); and smsemoaStep on the population and the child,
      with the reference point nadir + 1.0 · (nadir − ideal) of all N + 1 solutions. It is declared among the
      figures, next to smsemoaStep; both are hoisted, so the exported algorithms reach them under Node.js too. */
  function smsemoaGeneration(seed, popSize) {
    const rand = mulberry32(seed);
    const N = popSize;
    let nextId = 1;
    const pop = [];
    for (let i = 0; i < N; i++) {
      const x = [rand(), rand()];
      pop.push({ id: nextId++, x, f: gaEvaluate(x) });
    }
    const p1 = pop[Math.floor(rand() * N)];
    const p2 = pop[Math.floor(rand() * N)];
    const x = rand() <= 0.9 ? sbxPair(p1.x, p2.x, 0, 1, 20, 0.5, rand)[0] : p1.x.slice();
    polynomialMutation(x, 0, 1, 0.5, 20, rand);
    const child = { id: nextId++, x, f: gaEvaluate(x), parents: [p1.id, p2.id] };
    const merged = pop.concat([child]);
    const step = smsemoaStep(merged.map((p) => p.f), 1.0, false);
    return {
      popSize: N,
      pop,
      rank: nonDominatedSort(pop.map((p) => p.f)).rank,
      parents: [p1.id, p2.id],
      child,
      merged,
      fronts: step.fronts,
      last: step.last,
      ref: step.ref,
      contributions: step.contributions,
      boxes: step.boxes,
      removed: step.removed,
      next: merged.filter((_, i) => i !== step.removed),
    };
  }

  /* I9: one SMS-EMOA selection step. */
  widgets["smsemoa-step"] = function (node) {
    const parents = [
      [0.08, 0.9],
      [0.2, 0.62],
      [0.35, 0.45],
      [0.4, 0.4],
      [0.62, 0.22],
      [0.9, 0.06],
    ];
    const children = { A: [0.55, 0.5], B: [0.5, 0.3] };
    const names = ["P1", "P2", "P3", "P4", "P5", "P6", "C"];
    const state = { scenario: "A", Q: parents.concat([children.A]).map((p) => p.slice()), step: 0, offset: 1, jmetal: false };
    const ctl = controls(node);
    select(ctl, {
      label: "Scenario",
      options: [
        ["A", "A. The child is dominated"],
        ["B", "B. The child joins the front"],
      ],
      value: "A",
      onChange: (v) => {
        state.scenario = v;
        state.Q = parents.concat([children[v]]).map((p) => p.slice());
        state.step = 0;
        render();
      },
    });
    const back = button(ctl, "Back", () => ((state.step = Math.max(0, state.step - 1)), render()));
    const next = button(ctl, "Next step", () => ((state.step = Math.min(4, state.step + 1)), render()), true);
    button(ctl, "Reset step", () => ((state.step = 0), render()));
    slider(ctl, { label: "Reference offset", min: 0, max: 1, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => ((state.offset = v), render()) });
    checkbox(ctl, { label: "jMetal rule (1.1 · max)", checked: false, onChange: (v) => ((state.jmetal = v), render()) });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1.9], y: [0, 1.9], xLabel: "f₁", yLabel: "f₂", label: "Six parents and one child; SMS-EMOA removes one solution of the last front" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child C", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Exclusive contribution in the last front", shape: "region", cls: "vf-region-2" },
      { label: "Reference point r (computed)", shape: "diamond", cls: "vf-pt vf-pt-3" },
    ]);
    const handles = state.Q.map((_, i) =>
      handle(plot, {
        cls: "vf-pt vf-pt-muted",
        shape: i === 6 ? "triangle" : "circle",
        label: () => names[i],
        get: () => state.Q[i],
        set: (a, b) => ((state.Q[i] = [Math.min(1, a), Math.min(1, b)]), (state.step = 0), render()),
        boundsX: [0, 1],
        boundsY: [0, 1],
      })
    );
    const STEPS = ["", "Sort into fronts", "Identify the last front", "Compute contributions", "Remove the worst"];
    function render() {
      const r = smsemoaStep(state.Q, state.offset, state.jmetal);
      back.disabled = state.step === 0;
      next.disabled = state.step === 4;
      next.textContent = state.step < 4 ? "Next: " + STEPS[state.step + 1] : "Done";
      const rank = new Array(7).fill(0);
      r.fronts.forEach((f, k) => f.forEach((i) => (rank[i] = k)));
      clear(plot.layers.back);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      if (state.step >= 3 && r.boxes) {
        const pts = r.last.map((i) => state.Q[i]).sort((a, b) => a[0] - b[0]);
        const poly = [[pts[0][0], r.ref[1]]];
        pts.forEach((p, k) => {
          poly.push(p);
          poly.push([k + 1 < pts.length ? pts[k + 1][0] : r.ref[0], p[1]]);
        });
        poly.push(r.ref);
        plot.line(plot.layers.back, poly, "vf-region-1");
        r.boxes.forEach((b) => b && plot.rect(plot.layers.back, b[0], b[1], b[2], b[3], "vf-region-2"));
      }
      marker(plot.layers.marks, "diamond", plot.x(r.ref[0]), plot.y(r.ref[1]), 5, "vf-pt vf-pt-3");
      txt(plot.layers.top, plot.x(r.ref[0]) + 8, plot.y(r.ref[1]) - 6, "r", { class: "vf-mark-label" });
      state.Q.forEach((p, i) => {
        handles[i].place();
        let cls = state.step >= 1 ? "vf-pt " + rankClass(rank[i]) : "vf-pt vf-pt-muted";
        if (state.step >= 4 && i === r.removed) cls += " vf-faded";
        handles[i].setClass(cls);
        if (state.step === 2 && r.last.includes(i)) marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 10.5, "vf-ring");
        txt(plot.layers.top, plot.x(p[0]) + 9, plot.y(p[1]) - 7, names[i] + (state.step >= 4 && i === r.removed ? " ✕" : ""), { class: "vf-mark-label" });
      });
      clear(readout);
      const list = (rows) => rows.map((i) => names[i]).join(", ");
      const lines = [`r = ${pair(r.ref, 3)}, computed from all seven points.`];
      if (state.step >= 1) lines.push(r.fronts.map((f, k) => `Front ${k + 1}: ${list(f)}`).join(" · "));
      if (state.step >= 2) lines.push(r.fronts.length === 1 ? "The last front is front 1: all seven points." : `The last front is front ${r.fronts.length}: ${list(r.last)}.`);
      lines.forEach((t) => h("p", {}, readout, t));
      if (state.step >= 3) {
        if (!r.contributions) h("p", { class: "vf-step-text" }, readout, "One member: removed without computing any hypervolume.");
        else table(readout, ["Point", "Objectives", "Contribution"], r.last.map((i, k) => [names[i], pair(state.Q[i]), fmt(r.contributions[k], 4)]), [false, true, true]);
      }
      if (state.step >= 4) {
        const k = r.last.indexOf(r.removed);
        const what = r.contributions ? `SMS-EMOA removes ${names[r.removed]} (contribution ${fmt(r.contributions[k], 4)}).` : `SMS-EMOA removes ${names[r.removed]}: it is alone in the last front.`;
        h("p", { class: "vf-step-text" }, readout, what + (r.removed === 6 ? " The child loses: the population does not change." : ` C replaces ${names[r.removed]} in the population.`));
      }
      if (state.step === 0) h("p", {}, readout, "Six parents and one child compete for six places. Press Next step.");
    }
    render();
  };

  /* SMS-EMOA, one steady-state step in steps: two random parents, one child, the reference point of the N + 1
     solutions, non-dominated sorting, the last front, its hypervolume contributions and the next population. */
  widgets["smsemoa-steps"] = function (node) {
    const N = 6;
    const TITLES = ["the population P", "parent selection", "variation", "evaluation and the reference point", "non-dominated sorting", "the last front", "hypervolume contributions", "the next population"];
    const state = { seed: Number(node.dataset.seed || 65199), step: 0 };
    let g = null;
    let rankQ = [];
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw steps until the last front has at least three members, so that the contributions decide.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (smsemoaGeneration(seed, N).last.length >= 3) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the child" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1.9], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, fronts, reference point and hypervolume contributions of one SMS-EMOA step" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another step whose last front has at least three members.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Chosen parent, member of the last front, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Reference point r", shape: "diamond", cls: "vf-pt vf-pt-3" },
      { label: "Exclusive contribution in the last front", shape: "region", cls: "vf-region-2" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = smsemoaGeneration(state.seed, N);
      rankQ = new Array(N + 1).fill(0);
      g.fronts.forEach((front, k) => front.forEach((i) => (rankQ[i] = k)));
      fPlot.setDomain([0, Math.ceil((g.ref[0] + 0.05) * 10) / 10], [0, Math.ceil((g.ref[1] + 0.05) * 10) / 10]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    // Rows of Q: the parents (ids 1 to N), then the child (id N + 1).
    const id = (i) => g.merged[i].id;
    const ids = (rows) => rows.map(id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const contribution = (i) => (g.contributions ? g.contributions[g.last.indexOf(i)] : null);
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      plot.labels.push({ x: px, y: py, text: label, r: ring ? 11 : 7 });
    }
    function render() {
      const step = state.step;
      const last = TITLES.length - 1;
      back.disabled = step === 0;
      next.disabled = step === last;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
        plot.labels = [];
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const rows = [];
      const lastSet = new Set(g.last);
      if (step === 3) {
        const ideal = [0, 1].map((m) => Math.min(...g.merged.map((p) => p.f[m])));
        const nadir = [0, 1].map((m) => Math.max(...g.merged.map((p) => p.f[m])));
        fPlot.rect(fPlot.layers.back, ideal[0], ideal[1], nadir[0], nadir[1], "vf-line-thin vf-dashed");
        fPlot.line(fPlot.layers.mid, [nadir, g.ref], "vf-line-thin vf-dashed");
      }
      if (step === 4) {
        g.fronts.forEach((front, k) => {
          const pts = front.map((i) => g.merged[i].f).sort((a, b) => a[0] - b[0]);
          if (pts.length > 1) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(k));
        });
      }
      if (step === 6 && g.boxes) {
        const pts = g.last.map((i) => g.merged[i].f).sort((a, b) => a[0] - b[0]);
        const region = [[pts[0][0], g.ref[1]]];
        pts.forEach((p, k) => {
          region.push(p);
          region.push([k + 1 < pts.length ? pts[k + 1][0] : g.ref[0], p[1]]);
        });
        region.push(g.ref);
        fPlot.line(fPlot.layers.back, region, "vf-region-1");
        g.boxes.forEach((b) => b && fPlot.rect(fPlot.layers.back, b[0], b[1], b[2], b[3], "vf-region-2"));
      }
      if (step >= 3) {
        marker(fPlot.layers.marks, "diamond", fPlot.x(g.ref[0]), fPlot.y(g.ref[1]), 5, "vf-pt vf-pt-3");
        fPlot.labels.push({ x: fPlot.x(g.ref[0]), y: fPlot.y(g.ref[1]), text: "r", r: 7 });
      }
      if (step <= 2) {
        const chosen = new Map();
        g.parents.forEach((pid) => chosen.set(pid, (chosen.get(pid) || 0) + 1));
        g.pop.forEach((p, i) => {
          const k = step >= 1 ? chosen.get(p.id) || 0 : 0;
          const cls = "vf-pt " + rankClass(g.rank[i]);
          const label = String(p.id) + (k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", cls, label, k > 0);
          draw(fPlot, p.f, "circle", cls, label, k > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(g.rank[i] + 1), "–", k ? (k > 1 ? "chosen twice" : "chosen") : "–"]);
        });
        if (step === 2) {
          const c = g.child;
          c.parents.forEach((pid) => xPlot.line(xPlot.layers.mid, [g.merged[pid - 1].x, c.x], "vf-line-thin vf-dashed"));
          draw(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
          rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "?", "–", "not evaluated"]);
        }
      } else {
        g.merged.forEach((p, i) => {
          const isChild = i === N;
          const sorted = step >= 4;
          const removed = step === last && i === g.removed;
          const cls = "vf-pt " + (sorted ? rankClass(rankQ[i]) : "vf-pt-muted") + (removed ? " vf-faded" : "");
          const ring = ((step === 5 || step === 6) && lastSet.has(i)) || (step === last && i !== g.removed);
          let label = String(p.id);
          if (step === 6 && g.contributions && lastSet.has(i)) label += " · " + fmt(contribution(i), 3);
          if (removed) label += " ✕";
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, ring);
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, ring);
          let status = "–";
          if (step === 5 || step === 6) status = lastSet.has(i) ? (g.last.length === 1 ? "alone in the last front" : "last front") : "safe";
          if (step === last) status = i === g.removed ? "removed" : "kept";
          const shown = step >= 6 && g.contributions && lastSet.has(i);
          rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), sorted ? String(rankQ[i] + 1) : "?", shown ? fmt(contribution(i), 4) : "–", status]);
        });
      }
      for (const plot of [xPlot, fPlot]) placeMarkLabels(plot, plot.labels);
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Contribution", "Status"], rows, [false, false, true, true, true, true, true, true, false]);
    }
    function describe(step) {
      const child = g.child;
      const v = g.fronts.length;
      if (step === 0) {
        const fronts = nonDominatedSort(g.pop.map((p) => p.f)).fronts;
        const text = fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ");
        return `${N} solutions of a two-variable problem whose Pareto front is ZDT1's: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂, sorted into fronts: ${text}. Each step of SMS-EMOA creates one child and removes one of the N + 1 = ${N + 1} solutions.`;
      }
      if (step === 1) {
        const [a, b] = g.parents;
        const pick = a === b ? `${a} twice, so crossover has nothing to mix and the child differs from ${a} only by mutation` : `${a} and ${b}`;
        return `VAMOS' default selection("random") draws two parents uniformly at random, with replacement: ${pick}. Fronts and hypervolume play no part in mating; selection("tournament") would use NSGA-II's tournament instead.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 20) creates two children, VAMOS keeps the first, and polynomial mutation (probability 1/2 per variable, η = 20) perturbs it: child ${child.id}, the triangle joined to its parents in the decision space. Its objectives are not known yet.`;
      }
      if (step === 3) {
        return `Child ${child.id} is evaluated, f = ${pair(child.f)}, and joins the ${N} parents: Q holds N + 1 = ${N + 1} solutions. VAMOS first updates the reference point from all ${N + 1}: r = nadir + 1.0 · (nadir − ideal) = ${pair(g.ref)}, one objective range beyond the worst values. The dashed box spans the ideal and the nadir point.`;
      }
      if (step === 4) {
        const text = g.fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ");
        const pushed = g.pop.map((_, i) => i).filter((i) => rankQ[i] > g.rank[i]);
        const push = pushed.length ? ` Its arrival moves ${idList(ids(pushed))} down a front.` : "";
        return `Q falls into ${v} front${v === 1 ? "" : "s"}: ${text}. Child ${child.id} joins front ${rankQ[N] + 1}.${push} VAMOS inserts the child into the fronts of the population instead of sorting Q from scratch; the fronts are the same.`;
      }
      if (step === 5) {
        if (g.last.length === 1) return `Only the last front, front ${v}, can lose a member, and it has one: ${id(g.last[0])}. It is removed without computing any hypervolume; the members of earlier fronts are safe.`;
        return `Only the last front, front ${v} = ${set(g.last)}, can lose a member; the members of earlier fronts are safe, whatever their hypervolume contributions.`;
      }
      if (step === 6) {
        if (!g.contributions) return `Front ${v} has a single member, ${id(g.last[0])}, so no hypervolume is computed: it is removed.`;
        const parts = g.last.map((i) => `${fmt(contribution(i), 4)} for ${id(i)}`);
        const worst = g.removed;
        return `The exclusive contribution of a member of front ${v} is the area that only it dominates, up to r: the rectangles inside the front's shaded region. The contributions are ${idList(parts)}. The two extremes' rectangles reach r, which the margin of the default reference point makes large. The smallest is ${id(worst)}'s.`;
      }
      const kept = g.next.map((p) => p.id).sort((a, b) => a - b);
      if (g.removed === N) return `Child ${child.id} is the one removed: the population does not change, P = {${kept.join(", ")}}. The next step draws two new parents from it.`;
      const why = g.contributions ? `the smallest contribution, ${fmt(contribution(g.removed), 4)}` : "alone in the last front";
      return `${id(g.removed)} leaves (${why}) and child ${child.id} takes its place: P = {${kept.join(", ")}}. The next step draws two new parents from it; VAMOS counts a generation every ${N} children.`;
    }
    build();
  };

  /* A6-A9: one candidate set, the selection rules of CCMO, C-TAEA, ToP and CMOEA-MS. */
  widgets["constrained-selection"] = function (node) {
    const algo = node.getAttribute("data-rule") || "ccmo";
    const labels = CS_CANDIDATES.map((c) => c[0]);
    const F = CS_CANDIDATES.map((c) => [c[1], c[2]]);
    const cv = F.map((f) => ctpViolation(f));
    const N = 8;
    const rules = {
      ccmo: [
        ["pop1", "Population 1: constraint violation first"],
        ["pop2", "Population 2: helper problem, constraints ignored"],
      ],
      ctaea: [
        ["ca", "Convergence archive (CA)"],
        ["da", "Diversity archive (DA)"],
      ],
      top: [
        ["p1", "Phase 1: single-objective, pairwise"],
        ["p2", "Phase 2: NSGA-II with constrained dominance"],
      ],
      cmoeams: [
        ["A", "Stage A: density and violation as two objectives"],
        ["B", "Stage B: violation first"],
      ],
    }[algo];
    const state = { rule: rules[0][0] };
    const ctl = controls(node);
    select(ctl, { label: algo === "ctaea" ? "Archive update" : algo === "top" ? "Phase" : algo === "cmoeams" ? "Stage" : "Rule", options: rules, value: state.rule, onChange: (v) => ((state.rule = v), render()) });
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1.6], xLabel: "f₁", yLabel: "f₂", label: "Sixteen candidates on CTP2 and the eight that the selected rule keeps" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Constrained Pareto front", shape: "line", cls: "vf-line-1" },
      { label: "Kept, feasible", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Kept, infeasible", shape: "square", cls: "vf-pt vf-pt-1" },
      { label: "Not kept", shape: "circle", cls: "vf-pt-hollow" },
    ]);
    drawCtp2(plot);
    const ccmo1 = algo === "ccmo" ? ccmoSelect(F, cv, N) : null;
    const ccmo2 = algo === "ccmo" ? ccmoSelect(F, new Array(F.length).fill(0), N) : null;
    const ctaea = algo === "ctaea" ? ctaeaSelect(F, cv, N) : null;
    const top = algo === "top" ? topSelect(F, cv, N) : null;
    const cms = algo === "cmoeams" ? cmoeamsSelect(F, cv, N) : null;
    const list = (rows) => rows.map((i) => labels[i]).join(", ");
    function strengthTable(res, extra) {
      const rows = F.map((_, i) => i).sort((a, b) => res.fitness[a] - res.fitness[b] || a - b);
      table(
        readout,
        ["Label"].concat(extra.map((e) => e[0]), ["Dominated by", "R", "Fitness", "Kept"]),
        rows.map((i) => [labels[i]].concat(extra.map((e) => e[1](i)), [res.by[i].length > 6 ? `${res.by[i].length} candidates` : list(res.by[i]) || "–", String(res.R[i]), fmt(res.fitness[i], 4), res.kept.includes(i) ? "✓" : ""])),
        [false].concat(extra.map(() => true), [false, true, true, false])
      );
    }
    function render() {
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      clear(readout);
      let kept = [];
      const feasibleKept = () => kept.filter((i) => cv[i] <= 0).length;
      if (algo === "ccmo") {
        const res = state.rule === "pop1" ? ccmo1 : ccmo2;
        kept = res.kept;
        h("p", {}, readout, state.rule === "pop1" ? `Population 1 keeps ${feasibleKept()} of the 9 feasible candidates and ${N - feasibleKept()} infeasible ones: a smaller violation always dominates.` : `Population 2 ignores the constraints and keeps ${N - feasibleKept()} infeasible candidates and ${feasibleKept()} feasible one${feasibleKept() === 1 ? "" : "s"} (${list(kept.filter((i) => cv[i] <= 0))}), the solutions closest to the unconstrained front.`);
        h("p", {}, readout, `Fewer than ${N} candidates have fitness below 1, so the ${N} lowest fitness values are kept. Density uses the distance d₄ to the 4th nearest candidate.`);
        strengthTable(res, [
          ["CV", (i) => fmt(cv[i], 4)],
          ["d₄", (i) => fmt(res.dk[i], 4)],
        ]);
      } else if (algo === "ctaea") {
        if (state.rule === "ca") {
          kept = ctaea.CA;
          const step = ctaea.ca.removals[0];
          h("p", {}, readout, `${ctaea.feasible.length} of 16 candidates are feasible, ${ctaea.feasible.length - N} more than the archive holds. Fronts of the feasible candidates: ${ctaea.ca.fronts.map((f, k) => `front ${k + 1} = ${list(f)}`).join("; ")}.`);
          if (step) h("p", { class: "vf-step-text" }, readout, `Most crowded subregion (after normalizing the selected candidates): ${step.crowded}, with ${step.counts[step.crowded]} members. Its closest pair is ${list(step.closest)}; Tchebycheff values with ideal ${pair(step.ideal, 3)}: ${step.closest.map((i, a) => `${labels[i]} ${fmt(step.tch[a])}`).join(", ")} → ${labels[step.removed]} is removed.`);
          table(
            readout,
            ["Label", "CV", "Front", "Normalized subregion", "Kept"],
            F.map((_, i) => {
              const fr = ctaea.ca.fronts.findIndex((f) => f.includes(i));
              const nrm = ctaea.ca.normalized.find((x) => x.i === i);
              return [labels[i], fmt(cv[i], 4), fr >= 0 ? String(fr + 1) : "infeasible", nrm ? String(nrm.region) : "–", kept.includes(i) ? "✓" : ""];
            }),
            [false, true, false, true, false]
          );
        } else {
          kept = ctaea.DA;
          h("p", {}, readout, `The CA has ${ctaea.caCounts.join(", ")} members in subregions 0–7. The DA fills, round by round, the subregions where the CA has fewest, ignoring the constraints: it keeps ${N - feasibleKept()} infeasible candidates.`);
          table(
            readout,
            ["Step", "Round", "Subregion", "CA members", "Candidates", "Non-dominated", "Chosen"],
            ctaea.trace.map((t, a) => [String(a + 1), String(t.iteration), String(t.region), String(t.ca), list(t.cands), t.nd.map((i, b) => `${labels[i]} ${fmt(t.tch[b])}`).join(", "), labels[t.chosen]]),
            [true, true, true, true, false, false, false]
          );
        }
      } else if (algo === "top") {
        if (state.rule === "p1") {
          kept = top.next;
          top.pairs.forEach((p) => plot.line(plot.layers.mid, [F[p.p], F[p.o]], "vf-line-thin vf-dashed"));
          table(
            readout,
            ["Pair", "Parent CV, mean", "Trial CV, mean", "Winner", "Why"],
            top.pairs.map((p) => {
              const why = cv[p.o] < cv[p.p] ? "smaller violation" : cv[p.o] === cv[p.p] && p.om < p.pm ? "same violation, smaller mean" : cv[p.p] < cv[p.o] ? "parent: smaller violation" : "parent: not worse";
              return [`${labels[p.p]} / ${labels[p.o]}`, `${fmt(cv[p.p], 4)}, ${fmt(p.pm, 4)}`, `${fmt(cv[p.o], 4)}, ${fmt(p.om, 4)}`, labels[p.winner], why];
            }),
            [false, true, true, false, false]
          );
          h("p", {}, readout, `Pf = ${top.next.filter((i) => cv[i] <= 0).length}/${N} = ${fmt(top.Pf)}, δ = ${fmt(top.delta)}. Phase 1 continues while δ ≥ 0.2 or Pf ≤ 1/3 (and only until 90% of the budget): ${top.phaseOneContinues ? "phase 1 continues." : "the next generation is phase 2."}`);
        } else {
          kept = top.kept;
          h("p", {}, readout, `Fronts: ${top.fronts.map((f, k) => `${k + 1} = ${list(f)}`).join("; ")}. Feasible fronts by Pareto dominance, then one front per violation value.`);
          table(
            readout,
            ["Label", "Front", "Crowding", "Kept"],
            F.map((_, i) => [labels[i], String(top.fronts.findIndex((f) => f.includes(i)) + 1), fmt(top.crowd[i]), kept.includes(i) ? "✓" : ""]),
            [false, true, true, false]
          );
        }
      } else {
        const res = state.rule === "A" ? cms.A : cms.B;
        kept = res.kept;
        h("p", {}, readout, `${Math.round(cms.share * 16)} of 16 candidates are feasible: ${fmt(cms.share, 4)} > λ = 0.5. VAMOS uses stage B once 10% of the evaluations are done, stage A before. ${state.rule === "A" ? `Stage A keeps ${N - feasibleKept()} infeasible candidates.` : "Stage B keeps only feasible candidates."}`);
        strengthTable(
          res,
          state.rule === "A"
            ? [
                ["SDE", (i) => fmt(cms.sde[i], 4)],
                ["CVn", (i) => fmt(cms.cvn[i], 4)],
              ]
            : [["CVn", (i) => fmt(cms.cvn[i], 4)]]
        );
      }
      if (algo === "ctaea") {
        ctaea.W.forEach((w, k) => {
          plot.line(plot.layers.mid, [[0, 0], [w[0] * 3, w[1] * 3]], "vf-line-thin");
          const t = Math.min(0.97 / Math.max(w[0], 1e-6), 1.55 / Math.max(w[1], 1e-6));
          txt(plot.layers.top, plot.x(w[0] * t) + (k === 0 ? 5 : -10), plot.y(w[1] * t) + 12, String(k), { class: "vf-mark-label" });
        });
      }
      const strength = algo === "ccmo" ? (state.rule === "pop1" ? ccmo1 : ccmo2) : algo === "cmoeams" ? (state.rule === "A" ? cms.A : cms.B) : null;
      const detail = (i) => {
        if (strength) {
          const dom = strength.by[i].length > 6 ? `${strength.by[i].length} candidates` : list(strength.by[i]) || "nobody";
          const extra = algo === "cmoeams" ? ` · SDE ${fmt(cms.sde[i], 4)} · CVn ${fmt(cms.cvn[i], 4)}` : "";
          return `${extra} · dominated by ${dom} · fitness ${fmt(strength.fitness[i], 4)}`;
        }
        if (algo === "top" && state.rule === "p2") return ` · front ${top.fronts.findIndex((fr) => fr.includes(i)) + 1} · crowding ${fmt(top.crowd[i])}`;
        if (algo === "ctaea") return ` · subregion ${ctaea.regions[i]}`;
        return "";
      };
      F.forEach((f, i) => {
        const px = plot.x(f[0]);
        const py = plot.y(f[1]);
        const hit = s("circle", { cx: px, cy: py, r: 11, class: "vf-hit" }, plot.layers.marks);
        marker(plot.layers.marks, cv[i] > 0 ? "square" : "circle", px, py, 5.5, kept.includes(i) ? "vf-pt vf-pt-1" : "vf-pt-hollow");
        attachTip(hit, tip, () => `${labels[i]} · f = ${pair(f, 3)} · CV ${fmt(cv[i], 4)}${detail(i)}${kept.includes(i) ? " · kept" : ""}`, px, py);
        txt(plot.layers.top, px + 8, py - 6, labels[i], { class: "vf-mark-label" });
      });
    }
    render();
  };

  /** The loop figure's problem with CTP2's constraint, ctpViolation(gaEvaluate(x)), drawn in both spaces: the
      infeasible region (in the objective space only above the unconstrained front f₂ = 1 − √f₁, below which nothing
      is attainable), the unconstrained Pareto set x₂ = 0 and front (dashed), and the constrained ones. drawCtp2
      cannot serve here: it shades from CTP2's own unconstrained front f₁ + f₂ = 1, and this problem also reaches
      the infeasible band between the two fronts. */
  let loopCtp2Front = null;
  function drawLoopCtp2(xPlot, fPlot) {
    const shade = (plot, cols, rows, bad) => {
      const top = plot.yd[1];
      for (let c = 0; c < cols; c++) {
        const u = (c + 0.5) / cols;
        let start = null;
        for (let r = 0; r <= rows; r++) {
          const v = (r / rows) * top;
          const isBad = r < rows && bad(u, v);
          if (isBad && start === null) start = v;
          if (!isBad && start !== null) {
            plot.rect(plot.layers.back, c / cols, start, (c + 1) / cols + 0.0005, v, "vf-region-2");
            start = null;
          }
        }
      }
    };
    shade(xPlot, 100, 200, (x1, x2) => ctpViolation(gaEvaluate([x1, x2])) > 0);
    shade(fPlot, 100, 420, (f1, f2) => f2 >= 1 - Math.sqrt(f1) && ctpViolation([f1, f2]) > 0);
    if (!loopCtp2Front) {
      // Every feasible point lies above the line f₂ = 1 − tan(0.2π)·f₁, so the search can start at f₁ + f₂ = 1.
      const lowest = [];
      for (let i = 0; i <= 1000; i++) {
        const f1 = i / 1000;
        for (let f2 = 1 - f1; f2 <= 1.6; f2 += 0.0005) {
          if (ctpViolation([f1, f2]) <= 0) {
            lowest.push([f1, f2]);
            break;
          }
        }
      }
      loopCtp2Front = nonDominatedIndices(lowest).map((i) => lowest[i]);
    }
    // x₁ = f₁, and g = 1 + x₂ solves g − √(f₁·g) = f₂.
    const set = loopCtp2Front.map(([f1, f2]) => [f1, Math.pow((Math.sqrt(f1) + Math.sqrt(f1 + 4 * f2)) / 2, 2) - 1]);
    xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
    fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
    drawFront(xPlot, xPlot.layers.mid, set, "vf-line-1", 0.0015);
    drawFront(fPlot, fPlot.layers.mid, loopCtp2Front, "vf-line-1", 0.0015);
  }

  /** The class of a solution's marker in the constrained step-throughs: its colour, hollow when infeasible. */
  function solutionClass(sol, colour, faded) {
    return "vf-pt " + colour + (sol.cv > 0 ? " vf-pt-open" : "") + (faded ? " vf-faded" : "");
  }

  /** Write the labels of crowded points beside them, each where it covers the fewest other labels and markers:
      items are { x, y, text } in the plot's coordinates, written in order into plot.layers.top. */
  function placeLabels(plot, items) {
    const offsets = [
      [9, -7, "start"],
      [9, 14, "start"],
      [-9, -7, "end"],
      [-9, 14, "end"],
      [0, -12, "middle"],
      [0, 21, "middle"],
    ];
    const marks = items.map((it) => [it.x - 7, it.y - 7, it.x + 7, it.y + 7]);
    const placed = [];
    const overlap = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    for (const it of items) {
      const w = 6.4 * it.text.length;
      let best = null;
      offsets.forEach(([dx, dy, anchor], k) => {
        const x = it.x + dx;
        const left = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
        const box = [left, it.y + dy - 10, left + w, it.y + dy + 2];
        let cost = k * 0.5;
        for (const other of placed) cost += overlap(box, other);
        for (const m of marks) cost += overlap(box, m);
        if (box[0] < 2 || box[2] > plot.W - 2) cost += 1000;
        if (!best || cost < best.cost) best = { cost, x, y: it.y + dy, anchor, box };
      });
      placed.push(best.box);
      txt(plot.layers.top, best.x, best.y, it.text, { class: "vf-mark-label", "text-anchor": best.anchor });
    }
  }

  /** The objective range of a constrained step-through: f₂ up to 1.6 unless a point lies higher. */
  function constrainedTop(sols) {
    return Math.max(...sols.map((sol) => sol.f[1])) > 1.5 ? 2.1 : 1.6;
  }

  /* A5: one CCMO generation, step by step. */
  widgets["ccmo-steps"] = function (node) {
    const N = 6;
    const TITLES = [
      "the two populations",
      "mating selection in population 1",
      "population 1's offspring",
      "mating selection in population 2",
      "population 2's offspring",
      "evaluation",
      "population 1's fitness, the violation first",
      "population 1's environmental selection",
      "population 2's fitness, without the constraints",
      "population 2's environmental selection",
      "the next generation",
    ];
    const COLOURS = ["vf-pt-1", "vf-pt-2"];
    const state = { seed: Number(node.dataset.seed || 37454), step: 0 };
    let g = null;
    const interesting = (gen) => {
      const [p1, p2] = gen.selections;
      const helped = gen.children.some((c) => c.population === 2 && c.cv === 0 && p1.kept.includes(c.id));
      const explored = gen.children.some((c) => c.cv > 0 && p2.kept.includes(c.id) && !p1.kept.includes(c.id));
      return helped && explored;
    };
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw populations until population 1 keeps a feasible child of population 2, and population 2 an infeasible child that population 1 drops.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(ccmoGeneration(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of both populations and of their children, with the infeasible region shaded" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 1.6], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors of one CCMO generation, with CTP2's infeasible region shaded" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    let lines = null;
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another generation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Population 1", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Population 2", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Child, in the colour of the population that bred it", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Infeasible (hollow)", shape: "circle", cls: "vf-pt vf-pt-open vf-pt-muted" },
      { label: "In both next populations", shape: "circle", cls: "vf-pt vf-pt-ink" },
      { label: "Tournament winner, fitness below 1, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Constrained Pareto set and front", shape: "line", cls: "vf-line-1" },
      { label: "Unconstrained Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = ccmoGeneration(state.seed, N);
      fPlot.setDomain([0, 1], [0, constrainedTop(all())]);
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
      }
      drawLoopCtp2(xPlot, fPlot);
      lines = [xPlot, fPlot].map((plot) => s("g", {}, plot.layers.mid));
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const all = () => g.pops[0].concat(g.pops[1], g.children);
    const byId = (id) => all().find((sol) => sol.id === id);
    const set = (ids) => "{" + ids.slice().sort((a, b) => a - b).join(", ") + "}";
    const childrenOf = (k) => g.children.filter((c) => c.population === k + 1);
    const labels = [[], []];
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      labels[plot === xPlot ? 0 : 1].push({ x: px, y: py, text: label });
    }
    function render() {
      const step = state.step;
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      lines.forEach(clear);
      labels.forEach((list) => (list.length = 0));
      // Which population a step is about: 0, 1, or null for both.
      const focus = step >= 1 && step <= 2 ? 0 : step >= 3 && step <= 4 ? 1 : step >= 6 && step <= 7 ? 0 : step >= 8 && step <= 9 ? 1 : null;
      const mating = step === 1 || step === 3 ? g.mating[focus] : null;
      const wins = new Map();
      if (mating) mating.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
      const sel = step >= 6 && step <= 9 ? g.selections[focus] : null;
      const candidateIndex = (sol) => (sel ? sel.candidates.findIndex((c) => c.id === sol.id) : -1);
      const nextPops = g.selections.map((x) => x.kept);
      if (mating) {
        const pop = g.pops[focus];
        for (const t of mating.tournaments) {
          if (t.a !== t.b) fPlot.line(lines[1], [pop.find((p) => p.id === t.a).f, pop.find((p) => p.id === t.b).f], "vf-line-thin vf-dashed");
        }
      }
      const rows = [];
      for (const sol of all()) {
        const k = sol.population - 1;
        const isChild = !!sol.child;
        const shape = isChild ? "triangle" : "circle";
        const evaluated = !isChild || step >= 5;
        let visible;
        if (step === 0) visible = !isChild;
        else if (step <= 2) visible = !isChild ? k === 0 : step === 2 && k === 0;
        else if (step <= 4) visible = !isChild ? k === 1 : step === 4 && k === 1;
        else if (sel) visible = candidateIndex(sol) >= 0;
        else visible = true;
        const born = !isChild || step >= 2 + 2 * k;
        if (!born) continue;
        let colour = COLOURS[k];
        let ring = false;
        let faded = false;
        let label = String(sol.id);
        let status = "–";
        let R = "–";
        let D = "–";
        let fit = "–";
        if (!isChild && step <= 5) {
          const i = g.pops[k].indexOf(sol);
          R = String(g.R[k][i]);
          D = fmt(g.D[k][i]);
          fit = fmt(g.fitness[k][i]);
        }
        if (mating && !isChild && k === focus) {
          const won = wins.get(sol.id) || 0;
          ring = won > 0;
          if (won > 1) label += " ×" + won;
          status = won ? `won ${won}×` : "–";
        }
        if ((step === 2 || step === 4) && k === focus) {
          if (isChild) status = "not evaluated";
          else {
            const kids = childrenOf(k).filter((c) => c.parents.includes(sol.id));
            ring = kids.length > 0;
            if (kids.length) status = "parent of " + idList(kids.map((c) => c.id));
          }
        }
        if (step === 5 && isChild) status = sol.cv > 0 ? "infeasible, offered to both" : "feasible, offered to both";
        if (sel) {
          const i = candidateIndex(sol);
          if (i < 0) status = "not a candidate";
          else {
            R = String(sel.R[i]);
            D = fmt(sel.D[i]);
            fit = fmt(sel.fitness[i]);
            if (step === 6 || step === 8) {
              ring = sel.fitness[i] < 1;
              status = sel.fitness[i] < 1 ? "fitness below 1" : "fitness 1 or more";
            } else {
              const kept = sel.kept.includes(sol.id);
              ring = kept;
              faded = !kept;
              if (!kept) label += " ✕";
              status = kept ? "kept" : "removed";
            }
          }
        }
        if (step === 10) {
          const inP1 = nextPops[0].includes(sol.id);
          const inP2 = nextPops[1].includes(sol.id);
          colour = inP1 && inP2 ? "vf-pt-ink" : inP1 ? COLOURS[0] : inP2 ? COLOURS[1] : colour;
          faded = !inP1 && !inP2;
          if (faded) label += " ✕";
          status = inP1 && inP2 ? "both populations" : inP1 ? "population 1" : inP2 ? "population 2" : "removed";
          const stored = [0, 1].filter((j) => nextPops[j].includes(sol.id)).map((j) => fmt(g.selections[j].fitness[g.selections[j].candidates.findIndex((c) => c.id === sol.id)]));
          fit = stored.length ? stored.join(" / ") : "–";
        }
        if (visible) {
          const cls = solutionClass(sol, colour, faded);
          draw(xPlot, sol.x, shape, cls, label, ring);
          if (evaluated) draw(fPlot, sol.f, shape, cls, label, ring);
          if (isChild && (step === 2 || step === 4) && k === focus) {
            for (const id of sol.parents) xPlot.line(lines[0], [byId(id).x, sol.x], "vf-line-thin vf-dashed");
          }
        }
        const role = isChild ? `child of ${sol.parents[0]} and ${sol.parents[1]} (population ${k + 1})` : `population ${k + 1}`;
        rows.push([String(sol.id), role, evaluated ? fmt(sol.f[0], 2) : "?", evaluated ? fmt(sol.f[1], 2) : "?", evaluated ? fmt(sol.cv) : "?", R, D, fit, status]);
      }
      placeLabels(xPlot, labels[0]);
      placeLabels(fPlot, labels[1]);
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "f₁", "f₂", "CV", "R", "D", "Fitness", "Status"], rows, [false, false, true, true, true, true, true, true, false]);
    }
    function describe(step) {
      const feasible = (list) => list.filter((sol) => sol.cv === 0).map((sol) => sol.id);
      const infeasible = (list) => list.filter((sol) => sol.cv > 0).map((sol) => sol.id);
      const range = (list) => `${list[0].id} to ${list[list.length - 1].id}`;
      if (step === 0) {
        return `Two populations of ${N} on the loop figure's problem, f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂, under CTP2's constraint, which makes its unconstrained front f₂ = 1 − √f₁ infeasible except at f₁ = 0. Population 1 is ${range(g.pops[0])} and population 2 is ${range(g.pops[1])}, drawn at random, population 1 first. ${idList(feasible(g.pops[0].concat(g.pops[1])))} are feasible (filled); the others are infeasible (hollow). Each population computes its members' fitness R + D, with the violation first in population 1 and without it in population 2; the tournaments use it.`;
      }
      if (step === 1 || step === 3) {
        const k = step === 1 ? 0 : 1;
        const m = g.mating[k];
        const why = (t) => (t.a === t.b ? "the same member twice" : t.tie ? "equal fitness, the one that comes first" : "lower fitness");
        const games = m.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner} (${why(t)})`).join("; ");
        const pairs = m.pairs.map((p) => `(${p[0]}, ${p[1]})`).join(", ");
        return `Population ${k + 1} draws ${N} binary tournaments on its own fitness, each between two members drawn at random: ${games}. The winners of the first ${N / 2} mate with those of the last ${N / 2}: ${pairs}. Ringed members won at least once.`;
      }
      if (step === 2 || step === 4) {
        const k = step === 2 ? 0 : 1;
        const kids = childrenOf(k);
        return `SBX crossover (probability 1, η = 20) recombines each pair and only its first child is kept; polynomial mutation (probability 1/2 per variable, η = 20) perturbs it. Population ${k + 1} creates ${N / 2} children, ${idList(kids.map((c) => c.id))}: the triangles joined to their parents in the decision space. Their objectives are not known yet.`;
      }
      if (step === 5) {
        const ok = feasible(g.children);
        const bad = infeasible(g.children);
        return `The ${N} children are evaluated once, population 1's first. ${ok.length ? `${idList(ok)} ${ok.length === 1 ? "is" : "are"} feasible` : "None is feasible"}${bad.length ? ` and ${idList(bad)} ${bad.length === 1 ? "is" : "are"} infeasible` : ""}. Both populations receive all of them: each will choose ${N} survivors from ${2 * N} candidates, its own members and every child.`;
      }
      if (step === 6 || step === 8) {
        const k = step === 6 ? 0 : 1;
        const sel = g.selections[k];
        const below = sel.candidates.filter((_, i) => sel.fitness[i] < 1).map((c) => c.id);
        const rule =
          k === 0
            ? "Its dominance puts the violation first: a smaller violation dominates a larger one, so every feasible candidate dominates every infeasible one, and candidates with the same violation are compared by Pareto dominance."
            : "It passes a violation of zero for every candidate, so its dominance is Pareto dominance: population 2 solves the problem without its constraints.";
        return `Population ${k + 1}'s candidates are ${range(g.pops[k])} and the children ${range(g.children)}. ${rule} R is the sum of the strengths (the numbers of candidates dominated) of a candidate's dominators, and D = 1/(σ₃ + 2), with σ₃ the distance to the 3rd nearest candidate (k = ⌊√${2 * N}⌋ = 3). Fitness below 1, ringed: ${idList(below)}, the candidates that nobody dominates.`;
      }
      if (step === 7 || step === 9) {
        const k = step === 7 ? 0 : 1;
        const sel = g.selections[k];
        const removed = sel.candidates.map((c) => c.id).filter((id) => !sel.kept.includes(id));
        const fitness = (id) => sel.fitness[sel.candidates.findIndex((c) => c.id === id)];
        const fitOf = (id) => fmt(fitness(id), 2);
        let how;
        if (sel.rule === "fill") how = `Only ${sel.below} candidate${sel.below === 1 ? " has" : "s have"} a fitness below 1, fewer than ${N}, so population ${k + 1} keeps the ${N} lowest fitness values: ${sel.kept.map((id) => `${id} (${fitOf(id)})`).join(", ")}.`;
        else if (sel.rule === "truncate") {
          const pool = sel.candidates.filter((_, i) => sel.fitness[i] < 1);
          const cut = removed.filter((id) => fitness(id) < 1);
          const lists = (sol) =>
            pool
              .filter((o) => o !== sol)
              .map((o) => [euclidean(sol.f, o.f), o.id])
              .sort((a, b) => a[0] - b[0]);
          let why = "each time the candidate whose sorted distances to the remaining ones are lexicographically smallest, the one with the closest neighbour";
          if (cut.length === 1) {
            const out = lists(byId(cut[0]));
            const near = lists(byId(out[0][1]));
            why =
              near[0][1] === cut[0]
                ? `${cut[0]} and ${out[0][1]} are the closest pair (${fmt(out[0][0], 2)} apart in the objective space), and ${cut[0]} goes because its second nearest candidate is closer (${fmt(out[1][0], 2)} against ${fmt(near[1][0], 2)})`
                : `${cut[0]} has the closest neighbour, ${out[0][1]}, ${fmt(out[0][0], 2)} away in the objective space`;
          }
          how = `${sel.below} candidates have a fitness below 1, more than ${N}, so SPEA2's truncation removes ${cut.length === 1 ? "one of them" : cut.length + " of them, one at a time"}: ${why}.`;
        } else how = `Exactly ${N} candidates have a fitness below 1, and population ${k + 1} keeps them.`;
        const ascending = (ids) => ids.slice().sort((a, b) => a - b);
        const keptChildren = sel.kept.filter((id) => byId(id).child);
        const fromOther = ascending(keptChildren.filter((id) => byId(id).population !== k + 1));
        const infeasibleKept = ascending(sel.kept.filter((id) => byId(id).cv > 0));
        const extra =
          k === 0
            ? fromOther.length
              ? ` ${idList(fromOther)}, bred by population 2, ${fromOther.length === 1 ? "joins" : "join"} population 1.`
              : ""
            : infeasibleKept.length
              ? ` ${idList(infeasibleKept)} ${infeasibleKept.length === 1 ? "is" : "are"} infeasible: population 2 never looks at the constraints.`
              : "";
        return `${how} ${idList(removed)} ${removed.length === 1 ? "is" : "are"} removed.${extra} The survivors are stored by increasing fitness.`;
      }
      const [p1, p2] = g.selections.map((x) => x.kept.slice().sort((a, b) => a - b));
      const helped = p1.filter((id) => byId(id).child && byId(id).population === 2);
      const shared = p1.filter((id) => p2.includes(id));
      return `Population 1 = ${set(p1)} and population 2 = ${set(p2)}; the colours now show where each solution belongs.${helped.length ? ` ${idList(helped)}, bred by population 2, ${helped.length === 1 ? "is" : "are"} now in population 1.` : ""}${shared.length ? ` ${idList(shared)} ${shared.length === 1 ? "is" : "are"} in both, with a fitness in each.` : ""} Each population keeps the fitness computed on its candidates for its next tournaments, and population 1 is the result of the run.`;
    }
    build();
  };

  /* A6: one C-TAEA generation, step by step. */
  widgets["ctaea-steps"] = function (node) {
    const N = 6;
    const GENERATION = 2;
    const COLOURS = { CA: "vf-pt-1", DA: "vf-pt-2", both: "vf-pt-ink", child: "vf-pt-muted" };
    const state = { seed: Number(node.dataset.seed || 196524), step: 0 };
    let run = null;
    let g = null;
    let steps = [];
    let lines = null;
    let rays = null;
    const interesting = (r) => {
      const rec = r.records[r.records.length - 1];
      const daInfeasible = rec.daUpdate.kept.some((id) => r.solutions[id - 1].cv > 0);
      return rec.caUpdate.removals.length > 0 && rec.caUpdate.removals.length <= 3 && daInfeasible && rec.fromCA.some(Boolean) && !rec.fromCA.every(Boolean);
    };
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw runs until the CA update truncates its feasible candidates, the DA keeps an infeasible solution, and second parents come from both archives.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(ctaeaGeneration(seed, N, GENERATION))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the archives' members and of the children, with the infeasible region shaded" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 1.6], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, weight vectors and subregions of one C-TAEA generation, with CTP2's infeasible region shaded" });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another run.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Convergence archive (CA)", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Diversity archive (DA)", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "In both archives", shape: "circle", cls: "vf-pt vf-pt-ink" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Infeasible (hollow)", shape: "circle", cls: "vf-pt vf-pt-open vf-pt-muted" },
      { label: "Non-dominated, tournament winner, selected", shape: "circle", cls: "vf-ring" },
      { label: "Weight vector, numbered by subregion", shape: "line", cls: "vf-line-thin" },
      { label: "Normalization box of a removal", shape: "region", cls: "vf-cuboid" },
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Constrained Pareto set and front", shape: "line", cls: "vf-line-1" },
      { label: "Unconstrained Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    const sol = (id) => run.solutions[id - 1];
    const shownIds = () => [...new Set(g.CA.concat(g.DA, g.children))].sort((a, b) => a - b);
    const set = (ids) => "{" + ids.slice().sort((a, b) => a - b).join(", ") + "}";
    const ascending = (ids) => [...new Set(ids)].sort((a, b) => a - b);
    const plural = (list, one, many) => (list.length === 1 ? one : many);
    /** The end of a ray from p in direction d, where it leaves the plot's domain. */
    const rayEnd = (p, d) => {
      const t = Math.min(d[0] > 0 ? (fPlot.xd[1] - p[0]) / d[0] : Infinity, d[1] > 0 ? (fPlot.yd[1] - p[1]) / d[1] : Infinity);
      return [p[0] + d[0] * t, p[1] + d[1] * t];
    };
    function drawRays(layer, origin, dirs, emphasis) {
      dirs.forEach((d, k) => {
        const end = rayEnd(origin, d);
        fPlot.line(layer, [origin, end], k === emphasis ? "vf-line-ink" : "vf-line-thin" + (origin[0] || origin[1] ? " vf-dashed" : ""));
        const px = fPlot.x(end[0]);
        const py = fPlot.y(end[1]);
        const onTop = Math.abs(end[1] - fPlot.yd[1]) < 1e-9;
        txt(layer, onTop ? px + 4 : px - 4, onTop ? py + 12 : py - 5, String(k), { class: "vf-mark-label", "text-anchor": onTop ? "start" : "end" });
      });
    }
    function build() {
      run = ctaeaGeneration(state.seed, N, GENERATION);
      g = run.records[GENERATION - 1];
      steps = [
        { kind: "archives", title: "the two archives" },
        { kind: "shares", title: "the shares of non-dominated solutions" },
        { kind: "first", title: "the first parents" },
        { kind: "second", title: "the second parents" },
        { kind: "vary", title: "variation" },
        { kind: "evaluate", title: "evaluation" },
        { kind: "feasible", title: "CA update, feasible candidates first" },
      ];
      const removals = g.caUpdate.removals;
      removals.forEach((removal, i) => steps.push({ kind: "removal", title: `CA update, removal ${i + 1} of ${removals.length}`, removal }));
      const last = Math.max(...g.daUpdate.trace.map((t) => t.iteration));
      for (let t = 1; t <= last; t++) steps.push({ kind: "da", title: `DA update, iteration ${t}`, iteration: t });
      steps.push({ kind: "next", title: "the next archives" });
      fPlot.setDomain([0, 1], [0, constrainedTop(shownIds().map(sol))]);
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
      }
      drawLoopCtp2(xPlot, fPlot);
      rays = s("g", {}, fPlot.layers.mid);
      drawRays(rays, [0, 0], run.W, -1);
      lines = [xPlot, fPlot].map((plot) => s("g", {}, plot.layers.mid));
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(steps.length - 1, i));
      render();
    }
    const labels = [[], []];
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      labels[plot === xPlot ? 0 : 1].push({ x: px, y: py, text: label });
    }
    const membership = (id, CA, DA) => (CA.includes(id) && DA.includes(id) ? "both" : CA.includes(id) ? "CA" : DA.includes(id) ? "DA" : null);
    const roleOf = (id) => {
      const s2 = sol(id);
      if (g.children.includes(id)) return `child of ${s2.parents[1]} (first) and ${s2.parents[0]} (second)`;
      return { both: "CA and DA", CA: "CA", DA: "DA" }[membership(id, g.CA, g.DA)];
    };
    const game = (t) => {
      const why = {
        feasible: `only ${t.winner} is feasible`,
        first: t.a === t.b ? "the same member twice" : `both feasible, ${t.b} does not dominate ${t.a}`,
        dominates: `both feasible, ${t.b} dominates ${t.a}`,
        third: `both infeasible, the third draw ${t.c} wins`,
      }[t.reason];
      return `${t.a} vs ${t.b} → ${t.winner} (${why})`;
    };
    const tally = (games) => {
      const wins = new Map();
      games.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
      return wins;
    };
    function render() {
      const st = steps[state.step];
      const kind = st.kind;
      back.disabled = state.step === 0;
      next.disabled = state.step === steps.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      lines.forEach(clear);
      labels.forEach((list) => (list.length = 0));
      rays.style.display = kind === "removal" ? "none" : "";
      const before = ["archives", "shares", "first", "second", "vary"].includes(kind);
      const evaluated = (id) => !g.children.includes(id) || !before;
      const ndIds = new Set(g.CA.concat(g.DA).filter((_, i) => g.shares.nd[i]));
      const firstWins = tally(g.first);
      const secondWins = tally(g.second);
      const daTrace = kind === "da" ? g.daUpdate.trace.filter((t) => t.iteration <= st.iteration) : [];
      const picked = new Set(daTrace.map((t) => t.chosen));
      const ca = g.caUpdate;
      const caSelected = new Set(ca.fronts ? ca.fronts.reduce((acc, front) => (acc.length >= N ? acc : acc.concat(front)), []) : ca.kept);
      if (kind === "first" || kind === "second") {
        for (const t of kind === "first" ? g.first : g.second) {
          if (t.a !== t.b) fPlot.line(lines[1], [sol(t.a).f, sol(t.b).f], "vf-line-thin vf-dashed");
        }
      }
      if (kind === "vary") {
        for (const id of g.children) for (const p of sol(id).parents) xPlot.line(lines[0], [sol(p).x, sol(id).x], "vf-line-thin vf-dashed");
      }
      let subregionHeader = "Subregion";
      if (kind === "removal") {
        const r = st.removal;
        subregionHeader = "Subregion (normalized)";
        fPlot.rect(lines[1], r.lo[0], r.lo[1], r.hi[0], r.hi[1], "vf-cuboid");
        drawRays(
          lines[1],
          r.lo,
          run.W.map((w) => [w[0] * (r.hi[0] - r.lo[0]), w[1] * (r.hi[1] - r.lo[1])]),
          r.crowded
        );
        if (r.closest.length === 2) fPlot.line(lines[1], [sol(r.closest[0]).f, sol(r.closest[1]).f], "vf-line-ink");
      }
      const rows = [];
      for (const id of shownIds()) {
        const s2 = sol(id);
        const isChild = g.children.includes(id);
        const shape = isChild ? "triangle" : "circle";
        const was = membership(id, g.CA, g.DA);
        let colour = isChild ? COLOURS.child : COLOURS[was];
        let visible = true;
        let ring = false;
        let faded = false;
        let label = String(id);
        let status = "–";
        let region = evaluated(id) ? String(s2.region) : "?";
        if (kind === "archives" || kind === "shares") {
          visible = !isChild;
          if (kind === "shares" && !isChild) {
            ring = ndIds.has(id);
            status = ring ? "non-dominated in CA ∪ DA" : "dominated in CA ∪ DA";
          }
        } else if (kind === "first") {
          const archive = g.firstFrom === "CA" ? g.CA : g.DA;
          visible = archive.includes(id);
          colour = COLOURS[g.firstFrom];
          const won = firstWins.get(id) || 0;
          ring = won > 0;
          if (won > 1) label += " ×" + won;
          if (won) status = `first parent ${won}×`;
        } else if (kind === "second") {
          visible = !isChild;
          const won = secondWins.get(id) || 0;
          ring = won > 0;
          if (won > 1) label += " ×" + won;
          if (won) status = `second parent ${won}×`;
        } else if (kind === "vary") {
          const first = g.children.filter((c) => sol(c).parents[1] === id);
          const second = g.children.filter((c) => sol(c).parents[0] === id);
          ring = first.length + second.length > 0;
          if (isChild) status = "not evaluated";
          else if (ring) status = [first.length ? `first parent of ${idList(first)}` : "", second.length ? `second parent of ${idList(second)}` : ""].filter(Boolean).join(", ");
        } else if (kind === "evaluate") {
          if (isChild) status = s2.cv > 0 ? "infeasible" : "feasible";
        } else if (kind === "feasible") {
          visible = ca.candidates.includes(id);
          colour = COLOURS.CA;
          if (!visible) status = "not a candidate";
          else {
            const front = ca.fronts ? ca.fronts.findIndex((f) => f.includes(id)) : -1;
            const infFront = ca.infeasible ? ca.infeasible.fronts.findIndex((f) => f.includes(id)) : -1;
            ring = caSelected.has(id);
            faded = !ring;
            if (faded) label += " ✕";
            if (ca.fronts) status = s2.cv > 0 ? "infeasible, out" : `front ${front + 1}, ${ring ? "selected" : "out"}`;
            else if (ca.infeasible && s2.cv > 0) status = `(CV, Tch) front ${infFront + 1}, ${ring ? "kept" : "out"}`;
            else status = ring ? "kept" : "out";
          }
        } else if (kind === "removal") {
          const r = st.removal;
          const at = r.rows.indexOf(id);
          visible = at >= 0;
          colour = COLOURS.CA;
          region = at >= 0 ? String(r.region[at]) : "–";
          if (at < 0) status = ca.candidates.includes(id) ? "out" : "not a candidate";
          else {
            const pair = r.closest.indexOf(id);
            if (pair >= 0) label += " · " + fmt(r.tch[pair]);
            if (id === r.removed) {
              faded = true;
              label += " ✕";
            }
            status = id === r.removed ? `removed, Tch ${fmt(r.tch[pair])}` : pair >= 0 ? `closest pair, Tch ${fmt(r.tch[pair])}` : r.region[at] === r.crowded ? "most crowded subregion" : "selected";
          }
        } else if (kind === "da") {
          visible = g.daUpdate.candidates.includes(id);
          colour = COLOURS.DA;
          if (!visible) status = "not a candidate";
          else {
            const pick = daTrace.find((t) => t.chosen === id);
            ring = picked.has(id);
            status = !pick ? "candidate" : `picked in iteration ${pick.iteration}` + (pick.nd.length > 1 ? `, Tch ${fmt(pick.tch[pick.nd.indexOf(id)])}` : ", the only non-dominated one");
          }
        } else if (kind === "next") {
          const now = membership(id, ca.kept, g.daUpdate.kept);
          if (now) colour = COLOURS[now];
          faded = !now;
          if (faded) label += " ✕";
          status = { both: "CA and DA", CA: "CA", DA: "DA" }[now] || "out";
        }
        if (visible) {
          const cls = solutionClass(s2, colour, faded);
          draw(xPlot, s2.x, shape, cls, label, ring);
          if (evaluated(id)) draw(fPlot, s2.f, shape, cls, label, ring);
        }
        if (isChild && ["archives", "shares", "first", "second"].includes(kind)) continue;
        rows.push([String(id), roleOf(id), evaluated(id) ? fmt(s2.f[0], 2) : "?", evaluated(id) ? fmt(s2.f[1], 2) : "?", evaluated(id) ? fmt(s2.cv) : "?", region, status]);
      }
      placeLabels(xPlot, labels[0]);
      placeLabels(fPlot, labels[1]);
      stepText.textContent = `Step ${state.step + 1} of ${steps.length}: ${st.title}. ${describe(st)}`;
      clear(tableBox);
      table(tableBox, ["Id", "Role", "f₁", "f₂", "CV", subregionHeader, "Status"], rows, [false, false, true, true, true, true, false]);
    }
    function describe(st) {
      const ca = g.caUpdate;
      const da = g.daUpdate;
      const hugeNote = (values) => (values.some((v) => v > 1000) ? " A weight vector at an axis has a component of 10⁻⁶, which makes the Tchebycheff value of a point off that axis huge." : "");
      const feasibleOf = (ids) => ascending(ids.filter((id) => sol(id).cv === 0));
      const infeasibleOf = (ids) => ascending(ids.filter((id) => sol(id).cv > 0));
      const counts = (ids) => run.W.map((_, k) => ids.filter((id) => sol(id).region === k).length);
      if (st.kind === "archives") {
        const both = g.CA.filter((id) => g.DA.includes(id));
        const empty = counts(g.CA)
          .map((c, k) => (c ? -1 : k))
          .filter((k) => k >= 0);
        return `C-TAEA with ${N} weight vectors on the loop figure's problem under CTP2's constraint, the problem of the CCMO lesson's step-through. The rays are the weight vectors, numbered 0 to ${N - 1}: a solution belongs to the subregion of the ray closest to it in angle. The initial population, 1 to ${N}, filled both archives, and generation 1 updated them with children ${N + 1} to ${2 * N}. Now the CA is ${set(g.CA)} and the DA is ${set(g.DA)}${both.length ? `; ${idList(ascending(both))} ${plural(both, "is", "are")} in both` : ""}. Filled points are feasible and hollow ones infeasible: the CA has ${feasibleOf(g.CA).length} feasible members and the DA ${feasibleOf(g.DA).length}.${empty.length ? ` The CA has nobody in subregion${empty.length === 1 ? "" : "s"} ${idList(empty)}.` : ""}`;
      }
      if (st.kind === "shares") {
        const sh = g.shares;
        const ndCA = g.CA.filter((_, i) => sh.ndCA[i]);
        return `Mating starts by comparing the archives. CA ∪ DA has ${sh.size} rows, a solution in both archives counting twice; the ringed ones are non-dominated: ${sh.inFirst} in the CA and ${sh.inSecond} in the DA, so ρ_c = ${sh.inFirst}/${sh.size} and ρ_d = ${sh.inSecond}/${sh.size}. ${sh.rhoC > sh.rhoD ? "ρ_c > ρ_d, so every first parent of this generation comes from the CA." : "ρ_c > ρ_d is false, so every first parent of this generation comes from the DA."} In the CA alone ${ndCA.length} of ${g.CA.length} members are non-dominated (${idList(ascending(ndCA))}): P_C = ${ndCA.length}/${g.CA.length} = ${fmt(sh.PC, 2)} is the probability that a second parent comes from the CA.`;
      }
      if (st.kind === "first") {
        return `${N} binary tournaments in the ${g.firstFrom} choose the first parents. Each draws three members with replacement: if only one of the first two is feasible it wins, if both are the first wins unless the second dominates it, and if neither is the third wins. ${g.first.map(game).join("; ")}. Ringed members won at least once.`;
      }
      if (st.kind === "second") {
        const parts = g.second.map((t, i) => `${g.fromCA[i] ? `u = ${fmt(g.uniforms[i], 2)} < ${fmt(g.shares.PC, 2)}, CA` : `u = ${fmt(g.uniforms[i], 2)} ≥ ${fmt(g.shares.PC, 2)}, DA`}: ${game(t)}`);
        return `For each child a uniform number u chooses the archive of the second parent, the CA when u < P_C, and a tournament in that archive chooses the parent. ${parts.join("; ")}.`;
      }
      if (st.kind === "vary") {
        return `SBX crossover (probability 1, η = 20) recombines each pair, the second parent passed first, and only the first child is kept; polynomial mutation (probability 1/2 per variable, η = 20) perturbs it. Children ${g.children[0]} to ${g.children[N - 1]}, one per weight vector, are the triangles joined to their parents in the decision space. Their objectives are not known yet.`;
      }
      if (st.kind === "evaluate") {
        const ok = feasibleOf(g.children);
        const bad = infeasibleOf(g.children);
        return `The ${N} children are evaluated: ${ok.length ? `${idList(ok)} ${plural(ok, "is", "are")} feasible` : "none is feasible"}${bad.length ? ` and ${idList(bad)} ${plural(bad, "is", "are")} infeasible` : ""}. Both archives are updated with all of them, the CA first.`;
      }
      if (st.kind === "feasible") {
        const feas = ascending(ca.feasible);
        const head = `The CA update takes the CA and the children, ${ca.candidates.length} candidates. ${feas.length} of them are feasible`;
        if (ca.fronts) {
          const used = [];
          let total = 0;
          for (const front of ca.fronts) {
            if (total >= N) break;
            used.push(front);
            total += front.length;
          }
          const fronts = ca.fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ");
          const which = used.length === 1 ? "front 1 holds" : used.length === 2 ? "fronts 1 and 2 hold" : `fronts 1 to ${used.length} hold`;
          const excess = total - N;
          return `${head}, more than ${N}, so the CA will hold only feasible solutions. Non-dominated sorting of the feasible candidates gives ${fronts}. Fronts are added until they hold at least ${N}: ${which} ${total}, ringed, ${excess === 0 ? "exactly the CA's size" : `so ${excess === 1 ? "one" : excess} must still go`}. The infeasible candidates${ca.fronts.length > used.length ? " and the later fronts" : ""} are out.`;
        }
        if (!ca.infeasible) return `${head}, exactly ${N}: they are the new CA, and the infeasible candidates are out.`;
        const inf = ca.infeasible;
        const fronts = inf.fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ");
        return `${head}, fewer than ${N}: all of them stay, and the infeasible candidates are ranked by non-dominated sorting on two criteria, their violation and their Tchebycheff value in their subregion (with the ideal point of the infeasible candidates): ${fronts}. Fronts are added until the CA holds at least ${N}${inf.removed.length ? `, and the largest violations of the last front added, ${idList(inf.removed)}, are dropped` : ""}.`;
      }
      if (st.kind === "removal") {
        const r = st.removal;
        const members = r.rows.filter((_, a) => r.region[a] === r.crowded);
        const [a, b] = r.closest;
        const dominated = (id) => r.rows.some((o) => o !== id && dominates(sol(o).f, sol(id).f));
        const kept = r.closest.filter((id) => id !== r.removed);
        const note = !dominated(r.removed) ? ` ${r.removed} is non-dominated: the removal is decided by crowding, not by rank.` : "";
        const tied = r.counts.map((c, k) => (c === r.counts[r.crowded] ? k : -1)).filter((k) => k >= 0);
        const crowdedText =
          tied.length > 1
            ? `subregions ${idList(tied)} are the most crowded, and the first, ${r.crowded}, holds ${idList(ascending(members))}.`
            : `subregion ${r.crowded} is the most crowded, with ${idList(ascending(members))}.`;
        const pairText =
          r.closest.length === 2
            ? `Its closest pair is ${a} and ${b}, ${fmt(euclidean(sol(a).f, sol(b).f), 2)} apart; with the ideal point of the pair, ${pair(r.ideal, 2)}, and the weight vector of each one's subregion before normalization, their Tchebycheff values are ${a} ${fmt(r.tch[0])} and ${b} ${fmt(r.tch[1])}, so ${r.removed} goes and ${kept[0]} stays.`
            : `Its closest members are ${idList(r.closest)}; the one with the largest Tchebycheff value, ${r.removed}, goes.`;
        return `The ${r.rows.length} selected solutions are normalized with their ideal point ${pair(r.lo, 2)} and nadir point ${pair(r.hi, 2)} (the box), and associated with the weight vectors drawn from the ideal point. Members per subregion: ${r.counts.join(", ")}; ${crowdedText} ${pairText}${hugeNote(r.tch)}${note}`;
      }
      if (st.kind === "da") {
        const t = st.iteration;
        const picks = da.trace.filter((x) => x.iteration === t);
        const total = da.trace.filter((x) => x.iteration <= t).length;
        const earlier = da.trace.filter((x) => x.iteration < t).map((x) => x.chosen);
        const regions = ascending(picks.map((x) => x.region));
        const pickText = regions.map((k) => {
          const mine = picks.filter((x) => x.region === k);
          const c = da.caCounts[k];
          const each = mine.map((x) => `${x.chosen}${x.nd.length > 1 ? ` (non-dominated ${x.nd.map((id, i) => `${id} ${fmt(x.tch[i])}`).join(", ")})` : ""}`);
          return `subregion ${k} (${c === 0 ? "no CA member" : c === 1 ? "1 CA member" : c + " CA members"}) takes ${each.join(", then ")}`;
        });
        // Subregions with a quota left but no candidate: every candidate of theirs is already in the DA, or they never had one.
        const empty = run.W.map((_, k) => k).filter((k) => t - da.caCounts[k] > 0 && !regions.includes(k) && !da.candidates.some((id) => sol(id).region === k && !earlier.includes(id)));
        if (empty.length) pickText.push(`subregion${empty.length === 1 ? "" : "s"} ${idList(empty)} ${plural(empty, "has", "have")} room but no candidate left`);
        if (!pickText.length) pickText.push(`every subregion has at least ${t} CA member${t === 1 ? "" : "s"}, so none receives a candidate`);
        const intro =
          t === 1
            ? `The DA update takes the DA and the children, ${da.candidates.length} candidates, and ignores the constraints. It counts the new CA's members in each subregion, c = (${da.caCounts.join(", ")}), and in iteration itr gives subregion k up to itr − c_k candidates, each time the non-dominated candidate of k with the smallest Tchebycheff value. `
            : "";
        return `${intro}Iteration ${t}: ${pickText.join("; ")}. The DA has ${total} of ${N} members.${hugeNote(picks.flatMap((x) => x.tch))}`;
      }
      const newCA = ca.kept;
      const newDA = da.kept;
      const childCA = g.children.filter((id) => newCA.includes(id));
      const childDA = g.children.filter((id) => newDA.includes(id));
      const entered = childCA.length ? `${plural(childCA, "Child", "Children")} ${idList(ascending(childCA))} entered the CA${childDA.length ? ` and ${idList(ascending(childDA))} the DA` : ""}` : childDA.length ? `No child entered the CA; ${idList(ascending(childDA))} entered the DA` : "No child entered either archive";
      return `The new CA is ${set(newCA)} and the new DA is ${set(newDA)}; the colours now show where each solution belongs. ${entered}. ${infeasibleOf(newDA).length} of the DA's ${newDA.length} members are infeasible, while the CA, the result of the run, is ${infeasibleOf(newCA).length ? "not yet entirely feasible" : "entirely feasible"}.`;
    }
    build();
  };

  /** The background of the loop figure's problem with CTP2's constraint (gaEvaluate, then ctpViolation): the
      infeasible region in the decision space and, with drawCtp2, in the objective space. This problem also reaches
      the points below f₂ = 1 − f₁, which CTP2 does not and which are all infeasible, so that triangle is shaded too
      and CTP2's unconstrained front, which is not this problem's, is left out. */
  function drawConstrainedLoop(xPlot, fPlot) {
    const cols = 120;
    const step = 0.004;
    for (let c = 0; c < cols; c++) {
      const x1 = (c + 0.5) / cols;
      let start = null;
      for (let x2 = 0; x2 <= 1 + step; x2 += step) {
        const bad = x2 <= 1 && ctpViolation(gaEvaluate([x1, x2])) > 0;
        if (bad && start === null) start = x2;
        if (!bad && start !== null) {
          xPlot.rect(xPlot.layers.back, c / cols, start, (c + 1) / cols + 0.0005, Math.min(x2, 1), "vf-region-2");
          start = null;
        }
      }
    }
    drawCtp2(fPlot);
    fPlot.layers.back.querySelectorAll(".vf-front").forEach((line) => line.remove());
    const below = fPlot.line(fPlot.layers.back, [[0, 0], [0, 1], [1, 0], [0, 0]], "vf-region-2");
    fPlot.layers.back.insertBefore(below, fPlot.layers.back.firstChild);
  }

  /** Write point labels where they overlap the other labels and points least: items are { px, py, text }. */
  function spreadPointLabels(plot, items) {
    const offsets = [
      [9, -7, "start"],
      [-9, -7, "end"],
      [9, 14, "start"],
      [-9, 14, "end"],
      [11, 4, "start"],
      [-11, 4, "end"],
      [0, -12, "middle"],
      [0, 21, "middle"],
    ];
    const overlap = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const marks = items.map((it) => [it.px - 7, it.py - 7, it.px + 7, it.py + 7]);
    const placed = [];
    for (const it of items) {
      const w = it.text.length * 6.3 + 2;
      let best = null;
      for (const [dx, dy, anchor] of offsets) {
        const x0 = anchor === "start" ? it.px + dx : anchor === "end" ? it.px + dx - w : it.px - w / 2;
        const box = [x0, it.py + dy - 9, x0 + w, it.py + dy + 2];
        let cost = 0;
        for (const b of placed) cost += 2 * overlap(box, b);
        for (const b of marks) cost += overlap(box, b);
        if (box[0] < 2 || box[2] > plot.W - 2 || box[1] < 0 || box[3] > plot.H - 30) cost += 1000;
        if (!best || cost < best.cost) best = { cost, dx, dy, anchor, box };
      }
      placed.push(best.box);
      txt(plot.layers.top, it.px + best.dx, it.py + best.dy, it.text, { class: "vf-mark-label", "text-anchor": best.anchor });
    }
  }

  /* ToP at a glance: one generation of phase 1, the switch, and one generation of phase 2. */
  widgets["top-steps"] = function (node) {
    const N = 6;
    const STEPS = [
      [1, "the population and the objective of phase 1"],
      [1, "mutant vectors"],
      [1, "trial vectors"],
      [1, "evaluation"],
      [1, "one-to-one replacement"],
      [1, "Pf, δ and the switch"],
      [2, "constrained fronts of the population"],
      [2, "mating selection"],
      [2, "DE/rand/1 offspring"],
      [2, "evaluation and merge"],
      [2, "constrained non-dominated sorting"],
      [2, "whole fronts and crowding distance"],
      [2, "the next population"],
    ];
    const PHASE_2 = STEPS.findIndex((st) => st[0] === 2);
    const SUB = "₀₁₂₃₄₅₆₇₈₉";
    const state = { seed: Number(node.dataset.seed || 789767), step: 0 };
    let g = null;
    const interesting = (c) => {
      if (!c.phase2) return false;
      const m = c.phase1.merged;
      const p2 = c.phase2;
      return (
        c.phase1.pairs.some((p) => m[p.p].cv !== m[p.o].cv) &&
        c.phase1.pairs.some((p) => m[p.p].cv === m[p.o].cv) &&
        p2.children.some((ch) => ch.cv > 0) &&
        p2.split !== null &&
        p2.places > 0 &&
        p2.split.length > p2.places
      );
    };
    const ctl = controls(node);
    const phaseSelect = select(ctl, {
      label: "Phase",
      options: [
        ["1", "Phase 1: generation 1"],
        ["2", "Phase 2: generation 2"],
      ],
      value: "1",
      onChange: (v) => go(v === "1" ? 0 : PHASE_2),
    });
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw runs until phase 1 ends after one generation, the violation and the mean objective both decide pairs,
      // an offspring of phase 2 is infeasible and the crowding distance decides who survives.
      for (let tries = 0; tries < 300; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(topGeneration(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the population, the mutant and trial vectors and the offspring of ToP" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 1.6], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, CTP2's infeasible region and ToP's replacements, fronts and survivors" });
    drawConstrainedLoop(xPlot, fPlot);
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; Phase jumps to the first step of a phase, and New population draws another run.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Trial vector, child", shape: "triangle", cls: "vf-pt vf-pt-1" },
      { label: "Infeasible, CV > 0 (hollow)", shape: "circle", cls: "vf-pt vf-pt-open vf-pt-muted" },
      { label: "Not evaluated yet", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Mutant vector", shape: "diamond", cls: "vf-pt-hollow" },
      { label: "Feasible fronts 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Winner, survivor, best third", shape: "circle", cls: "vf-ring" },
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Constrained Pareto front", shape: "line", cls: "vf-line-1" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = topGeneration(state.seed, N);
      go(0);
    }
    function go(i) {
      // A run whose phase 1 continues has no phase 2 to show.
      const last = g.phase2 ? STEPS.length - 1 : PHASE_2 - 1;
      state.step = Math.max(0, Math.min(last, i));
      render();
    }
    const sub = (id) => String(id).replace(/\d/g, (d) => SUB[Number(d)]);
    const set = (list) => "{" + list.slice().sort((a, b) => a - b).join(", ") + "}";
    const feasibleCls = (p) => (p.cv > 0 ? "vf-pt-1 vf-pt-open" : "vf-pt-1");
    const cvText = (p) => (p.cv > 0 ? fmt(p.cv, 4) : "0");
    function draw(plot, labels, v, shape, cls, label, ring) {
      const px = plot.x(Math.min(Math.max(v[0], plot.xd[0]), plot.xd[1]));
      const py = plot.y(Math.min(Math.max(v[1], plot.yd[0]), plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      labels.push({ px, py, text: label });
    }
    function render() {
      const step = state.step;
      phaseSelect.value = String(STEPS[step][0]);
      back.disabled = step === 0;
      next.disabled = step === (g.phase2 ? STEPS.length - 1 : PHASE_2 - 1);
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      const xl = [];
      const fl = [];
      const out = step < PHASE_2 ? renderPhaseOne(step, xl, fl) : renderPhaseTwo(step - PHASE_2, xl, fl);
      spreadPointLabels(xPlot, xl);
      spreadPointLabels(fPlot, fl);
      stepText.textContent = `Step ${step + 1} of ${STEPS.length}: ${STEPS[step][1]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, out.headers, out.rows, out.numeric);
    }
    function renderPhaseOne(step, xl, fl) {
      const p1 = g.phase1;
      const pop = p1.pop;
      const rows = [];
      const headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "CV", "Mean", step === 4 ? "Status" : "Detail"];
      const numeric = [false, false, true, true, true, true, true, true, false];
      const row = (p, role, known, detail) => [String(p.id), role, fmt(p.x[0], 2), fmt(p.x[1], 2), known ? fmt(p.f[0], 3) : "?", known ? fmt(p.f[1], 3) : "?", known ? cvText(p) : "?", known ? fmt(p.mean, 3) : "?", detail];
      if (step === 5) {
        const P = p1.next.map((i) => p1.merged[i]);
        const sums = new Map(p1.sums.map(([i, v]) => [p1.merged[i].id, v]));
        const order = p1.sums.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]);
        const k = Math.floor(p1.sums.length / 3);
        const third = new Set(order.slice(0, k).map((e) => p1.merged[e[0]].id));
        if (p1.lo) {
          const { lo, hi } = p1;
          fPlot.rect(fPlot.layers.mid, lo[0], lo[1], hi[0], hi[1], "vf-line-thin vf-dashed");
          const level = (sv) => {
            const a = Math.max(0, sv - 1);
            const b = Math.min(1, sv);
            const at = (t) => [lo[0] + t * (hi[0] - lo[0]), lo[1] + (sv - t) * (hi[1] - lo[1])];
            fPlot.line(fPlot.layers.mid, [at(a), at(b)], "vf-line-ink vf-dashed");
          };
          level(order[0][1]);
          if (k > 1) level(order[k - 1][1]);
        }
        P.forEach((p) => {
          const isTrial = p.parent !== undefined;
          const label = sums.has(p.id) ? `${p.id} · ${fmt(sums.get(p.id), 2)}` : String(p.id);
          draw(xPlot, xl, p.x, isTrial ? "triangle" : "circle", "vf-pt " + feasibleCls(p), String(p.id), third.has(p.id));
          draw(fPlot, fl, p.f, isTrial ? "triangle" : "circle", "vf-pt " + feasibleCls(p), label, third.has(p.id));
          const detail = sums.has(p.id) ? `s = ${fmt(sums.get(p.id), 3)}${third.has(p.id) ? ", best third" : ""}` : p.cv > 0 ? "infeasible" : "–";
          rows.push(row(p, isTrial ? `trial vector of ${p.parent}` : "parent", true, detail));
        });
        return { headers, rows, numeric };
      }
      const byId = new Map(pop.map((p) => [p.id, p]));
      if (step === 1) {
        p1.trials.forEach((t, i) => {
          const base = byId.get(t.base);
          xPlot.line(xPlot.layers.mid, [base.x, t.mutant], "vf-line-thin vf-dashed");
          draw(xPlot, xl, t.mutant, "diamond", "vf-pt-hollow", "v" + sub(pop[i].id), false);
        });
      }
      if (step === 2 || step === 3) {
        p1.trials.forEach((t, i) => {
          xPlot.line(xPlot.layers.mid, [pop[i].x, t.x], "vf-line-thin vf-dashed");
          if (step === 3) fPlot.line(fPlot.layers.mid, [pop[i].f, t.f], "vf-line-thin vf-dashed");
        });
      }
      if (step === 4) {
        p1.pairs.forEach((pr) => {
          fPlot.line(fPlot.layers.mid, [p1.merged[pr.p].f, p1.merged[pr.o].f], "vf-line-thin vf-dashed");
          xPlot.line(xPlot.layers.mid, [p1.merged[pr.p].x, p1.merged[pr.o].x], "vf-line-thin vf-dashed");
        });
      }
      const winners = new Set(p1.next.map((i) => p1.merged[i].id));
      pop.forEach((p, i) => {
        const removed = step === 4 && !winners.has(p.id);
        const cls = "vf-pt " + feasibleCls(p) + (removed ? " vf-faded" : "");
        const label = String(p.id) + (step <= 1 && i === p1.best ? " · best" : "") + (removed ? " ✕" : "");
        draw(xPlot, xl, p.x, "circle", cls, label, step === 4 && !removed);
        draw(fPlot, fl, p.f, "circle", cls, label, step === 4 && !removed);
        const t = p1.trials[i];
        let detail = "–";
        if (step === 0) detail = i === p1.best ? "x_best: lowest mean" : "–";
        if (step === 1) {
          const [r1, r2, r3] = t.r;
          detail = `${t.strategy}, r1 = ${r1}, r2 = ${r2}, r3 = ${r3}: v = ${pair(t.mutant, 2)}`;
        }
        if (step >= 2) detail = `competes with ${t.id}`;
        if (step === 4) detail = winners.has(p.id) ? "stays" : `replaced by ${t.id}`;
        rows.push(row(p, "parent", true, detail));
      });
      if (step >= 2) {
        p1.trials.forEach((t, i) => {
          const known = step >= 3;
          const removed = step === 4 && !winners.has(t.id);
          const cls = "vf-pt " + (known ? feasibleCls(t) : "vf-pt-muted") + (removed ? " vf-faded" : "");
          const label = String(t.id) + (removed ? " ✕" : "");
          draw(xPlot, xl, t.x, "triangle", cls, label, step === 4 && !removed);
          if (known) draw(fPlot, fl, t.f, "triangle", cls, label, step === 4 && !removed);
          let detail = "not evaluated";
          if (step === 2) {
            const clamped = t.mutant.some((v) => v < 0 || v > 1);
            const moved = t.x.some((v, j) => Math.abs(v - Math.min(1, Math.max(0, t.mutant[j]))) > 1e-12);
            detail = (clamped ? "v clamped to [0, 1]²" : "v") + (moved ? ", then mutated" : ", not mutated");
          }
          if (step === 3) detail = `competes with ${pop[i].id}`;
          if (step === 4) detail = winners.has(t.id) ? `replaces ${pop[i].id}` : "removed";
          rows.push(row(t, `trial vector of ${pop[i].id}`, known, detail));
        });
      }
      return { headers, rows, numeric };
    }
    function renderPhaseTwo(k, xl, fl) {
      const p2 = g.phase2;
      const rows = [];
      const headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "CV", "Front", "Crowding", "Status"];
      const numeric = [false, false, true, true, true, true, true, true, true, false];
      const cls = (p, front, ranked) => "vf-pt " + (p.cv > 0 ? "vf-pt-1 vf-pt-open" : ranked ? rankClass(front) : "vf-pt-1");
      if (k <= 2) {
        const wins = new Map();
        p2.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
        const byId = new Map(p2.pop.map((p) => [p.id, p]));
        if (k === 0) {
          p2.fronts.forEach((members, f) => {
            const pts = members.map((i) => p2.pop[i].f).sort((a, b) => a[0] - b[0]);
            if (pts.length > 1 && p2.pop[members[0]].cv <= 0) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(f));
          });
        }
        if (k === 1) p2.tournaments.forEach((t) => fPlot.line(fPlot.layers.mid, [byId.get(t.a).f, byId.get(t.b).f], "vf-line-thin vf-dashed"));
        p2.pop.forEach((p, i) => {
          const w = wins.get(p.id) || 0;
          const label = String(p.id) + (k === 1 && w > 1 ? " ×" + w : "");
          draw(xPlot, xl, p.x, "circle", cls(p, p2.front[i], true), label, k === 1 && w > 0);
          draw(fPlot, fl, p.f, "circle", cls(p, p2.front[i], true), label, k === 1 && w > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 3), fmt(p.f[1], 3), cvText(p), String(p2.front[i] + 1), fmt(p2.crowd[i], 2), k === 1 ? (w ? `won ${w}×` : "–") : "–"]);
        });
        if (k === 2) {
          p2.children.forEach((c) => {
            xPlot.line(xPlot.layers.mid, [byId.get(c.parents[0]).x, c.x], "vf-line-thin vf-dashed");
            draw(xPlot, xl, c.x, "triangle", "vf-pt vf-pt-muted", String(c.id), false);
            rows.push([String(c.id), "child", fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "?", "–", "–", `x₁ = ${c.parents[0]}, x₂ = ${c.parents[1]}, x₃ = ${c.parents[2]}`]);
          });
        }
        return { headers, rows, numeric };
      }
      const front = new Array(p2.merged.length).fill(0);
      p2.mergedFronts.forEach((members, f) => members.forEach((i) => (front[i] = f)));
      const keep = new Set(p2.kept);
      const whole = new Set(p2.whole.flat());
      const split = new Set(p2.split || []);
      const sorted = k >= 4;
      if (k === 4) {
        p2.mergedFronts.forEach((members, f) => {
          const pts = members.map((i) => p2.merged[i].f).sort((a, b) => a[0] - b[0]);
          if (pts.length > 1 && p2.merged[members[0]].cv <= 0) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(f));
        });
      }
      if (k === 5 && p2.split) {
        const order = p2.split.slice().sort((a, b) => p2.merged[a].f[0] - p2.merged[b].f[0] || a - b);
        for (let q = 1; q < order.length - 1; q++) {
          const a = p2.merged[order[q - 1]].f;
          const b = p2.merged[order[q + 1]].f;
          fPlot.rect(fPlot.layers.mid, a[0], a[1], b[0], b[1], "vf-cuboid");
        }
      }
      p2.merged.forEach((p, i) => {
        const isChild = i >= N;
        const outOfRace = k === 5 && !whole.has(i) && !split.has(i);
        const removed = k === 6 && !keep.has(i);
        const c = cls(p, front[i], sorted) + (outOfRace || removed ? " vf-faded" : "");
        const ring = (k === 5 && whole.has(i)) || (k === 6 && keep.has(i));
        let label = String(p.id);
        if (k === 5 && split.has(i)) label += " · " + fmt(p2.mergedCrowd[i], 2);
        if (removed) label += " ✕";
        draw(xPlot, xl, p.x, isChild ? "triangle" : "circle", c, label, ring);
        draw(fPlot, fl, p.f, isChild ? "triangle" : "circle", c, label, ring);
        let status = isChild ? "child" : "parent";
        if (k === 5) status = whole.has(i) ? "kept, whole front" : split.has(i) ? `competes for ${p2.places} place${p2.places === 1 ? "" : "s"}` : "discarded";
        if (k === 6) status = keep.has(i) ? "kept" : "removed";
        rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 3), fmt(p.f[1], 3), cvText(p), sorted ? String(front[i] + 1) : "?", k === 5 && split.has(i) ? fmt(p2.mergedCrowd[i], 2) : "–", status]);
      });
      return { headers, rows, numeric };
    }
    function describe(step) {
      const p1 = g.phase1;
      const pop = p1.pop;
      if (step === 0) {
        const best = pop[p1.best];
        const infeasible = pop.filter((p) => p.cv > 0).map((p) => p.id);
        const inf = infeasible.length ? `${idList(infeasible)} ${infeasible.length === 1 ? "is" : "are"} infeasible (hollow)` : "all are feasible";
        return `Six random solutions of the loop figure's problem, with CTP2's constraint on its objectives: ${inf}. Phase 1 solves a single-objective problem: minimize the mean objective (f₁ + f₂)/2 under the constraint. x_best, the solution with the lowest mean, feasible or not, is ${best.id} (mean ${fmt(best.mean, 3)}${best.cv > 0 ? ", infeasible" : ""}). A run starts with δ = 1 ≥ 0.2, so its first generation is in phase 1.`;
      }
      if (step === 1) {
        const by = (name) => p1.trials.filter((t) => t.strategy === name).map((t) => t.parent);
        const ctr = by("current-to-rand");
        const rtb = by("rand-to-best");
        const who = (list) => `${list.length === 1 ? "solution" : "solutions"} ${idList(list)} use${list.length === 1 ? "s" : ""}`;
        const uses = [ctr.length ? `${who(ctr)} DE/current-to-rand/1, v = x + F (x_r1 − x) + F (x_r2 − x_r3)` : "", rtb.length ? `${who(rtb)} DE/rand-to-best/1, v = x_r1 + F (x_best − x_r1) + F (x_r2 − x_r3)` : ""].filter(Boolean).join("; ");
        return `Each solution draws one of two strategies with probability 0.5, and three members r1, r2 and r3 with replacement, so they can repeat or be the solution itself. With F = 0.5, ${uses}. Each dashed line joins a base vector (x, or x_r1) to its mutant; a mutant outside [0, 1]² is drawn at the border.`;
      }
      if (step === 2) {
        const clamped = p1.trials.filter((t) => t.mutant.some((v) => v < 0 || v > 1)).map((t) => t.id);
        const clampText = clamped.length ? ` ${idList(clamped)} ${clamped.length === 1 ? "leaves" : "leave"} the bounds and ${clamped.length === 1 ? "is" : "are"} clamped back.` : " No mutant leaves the bounds.";
        return `With CR = 1, the binomial crossover takes every variable from the mutant, so each trial vector starts as its mutant.${clampText} Polynomial mutation (probability 1/2 per variable, η = 20) then perturbs them: trial vectors ${p1.trials[0].id} to ${p1.trials[N - 1].id}, each joined to the solution it will compete with. Their objectives are not known yet.`;
      }
      if (step === 3) {
        const feasible = p1.trials.filter((t) => t.cv <= 0).length;
        return `The ${N} trial vectors are evaluated: ${feasible} of them ${feasible === 1 ? "is" : "are"} feasible. Each segment joins a solution to its trial vector; a trial vector competes only with the solution at its position.`;
      }
      if (step === 4) {
        const games = p1.pairs.map((pr) => {
          const a = p1.merged[pr.p];
          const b = p1.merged[pr.o];
          const means = `${fmt(b.mean, 3)} against ${fmt(a.mean, 3)}`;
          if (b.cv < a.cv) return `${b.id} replaces ${a.id} (${a.cv > 0 && b.cv <= 0 ? "feasible against infeasible" : "smaller violation"}${b.mean >= a.mean ? `, although its mean is not smaller: ${means}` : ""})`;
          if (a.cv < b.cv) return `${a.id} stays against ${b.id} (${b.id} ${a.cv <= 0 ? "is infeasible" : "is more violated"}${b.mean < a.mean ? `, although its mean is smaller: ${means}` : ""})`;
          const both = a.cv <= 0 ? "both feasible" : "same violation";
          return b.mean < a.mean ? `${b.id} replaces ${a.id} (${both}, smaller mean: ${means})` : `${a.id} stays against ${b.id} (${both}, mean not smaller: ${means})`;
        });
        return `A trial vector replaces its solution if its violation is smaller, or if both are equal and its mean objective is smaller (Deb's feasibility rule on one pair): ${games.join("; ")}.`;
      }
      if (step === 5) {
        const P = p1.next.map((i) => p1.merged[i].id);
        const n = p1.sums.length || p1.next.filter((i) => p1.merged[i].cv <= 0).length;
        const pf = `Pf = ${n}/${N} = ${fmt(p1.Pf, 3)}`;
        const verdict = p1.continues ? "(δ ≥ 0.2 or Pf ≤ 1/3) holds, so the next generation is in phase 1 again." : "(δ ≥ 0.2 or Pf ≤ 1/3) fails, so the next generation is in phase 2, and so are all the later ones; 12 evaluations are far below the limit of 90% of the budget.";
        if (n < 3) return `The population is now ${set(P)}. ${pf}. With fewer than three feasible solutions δ keeps its value, 1. ${verdict}`;
        const order = p1.sums.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]);
        const k = Math.floor(n / 3);
        const s1 = order[0][1];
        const sk = order[k - 1][1];
        const third = order.slice(0, k).map((e) => p1.merged[e[0]].id);
        const how = `The ${n} feasible solutions are normalized with their minimum and maximum in each objective (the box), and each adds its two normalized objectives: ${order.map((e) => `${p1.merged[e[0]].id} · ${fmt(e[1], 3)}`).join(", ")}.`;
        const delta = k === 1 ? `⌊${n}/3⌋ = 1, so the best third is ${third[0]} alone and δ = s₁ − s₁ = 0.` : `The best third is the ⌊${n}/3⌋ = ${k} smallest sums, ${idList(third)}, and δ = s${sub(k)} − s₁ = ${fmt(sk, 3)} − ${fmt(s1, 3)} = ${fmt(p1.delta, 3)}, the gap between the two dashed lines.`;
        return `The population is now ${set(P)}. ${pf}. ${how} ${delta} ${verdict}`;
      }
      const p2 = g.phase2;
      const ids = (rows, from) => rows.map((i) => from[i].id);
      const frontsText = (fronts, from) =>
        fronts
          .map((members, f) => {
            const infeasible = from[members[0]].cv > 0;
            return `front ${f + 1} = ${set(ids(members, from))}${infeasible ? ` (CV ${fmt(from[members[0]].cv, 4)})` : ""}`;
          })
          .join(", ");
      const k = step - PHASE_2;
      if (k === 0) {
        const infeasible = p2.fronts.filter((members) => p2.pop[members[0]].cv > 0).length;
        return `Phase 2 is NSGA-II with ToP's constrained dominance. The population is sorted into fronts: the feasible solutions by Pareto dominance, then one front per violation value${infeasible ? "" : " (none here: all are feasible)"}: ${frontsText(p2.fronts, p2.pop)}. Each solution also has a crowding distance inside its front, listed in the table; the tournaments use both.`;
      }
      if (k === 1) {
        const why = { front: "better front", crowding: "same front, larger crowding distance", tie: "full tie, first in the population", same: "the same solution twice" };
        const games = p2.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner} (${why[t.reason]})`).join("; ");
        return `Six binary tournaments, each between two solutions drawn with replacement, choose the base vector x₁ of each offspring: ${games}. Ringed solutions won at least once.`;
      }
      if (k === 2) {
        return `Each offspring is x₁ + F (x₂ − x₃) with F = 0.5 (DE/rand/1), where x₂ and x₃ are random members of the population; with CR = 1 it is the whole mutant, then it is clamped and mutated as in phase 1: children ${p2.children[0].id} to ${p2.children[N - 1].id}, joined to their x₁. Their objectives are not known yet.`;
      }
      if (k === 3) {
        const infeasible = p2.children.filter((c) => c.cv > 0).map((c) => c.id);
        return `The ${N} children are evaluated (${infeasible.length ? `${idList(infeasible)} ${infeasible.length === 1 ? "is" : "are"} infeasible` : "all are feasible"}) and merged with the ${N} parents: ${2 * N} solutions compete for ${N} places.`;
      }
      if (k === 4) {
        return `The ${2 * N} solutions are sorted with ToP's constrained dominance: the feasible ones by Pareto dominance, then one front per distinct violation, in increasing order: ${frontsText(p2.mergedFronts, p2.merged)}. Every infeasible solution is behind every feasible one, whatever its objectives.`;
      }
      if (k === 5) {
        const used = N - p2.places;
        const fit = p2.whole.length === 0 ? "No front fits entirely" : p2.whole.length === 1 ? `Front 1 fits entirely, ${used} of ${N} places` : `Fronts 1 to ${p2.whole.length} fit entirely, ${used} of ${N} places`;
        if (!p2.split) return `${fit}: the whole fronts fill the places exactly, and the fronts after them are discarded.`;
        const winners = p2.split.filter((i) => p2.kept.includes(i));
        const dist = p2.split
          .slice()
          .sort((a, b) => cmpDesc(p2.mergedCrowd[a], p2.mergedCrowd[b]) || a - b)
          .map((i) => `${p2.merged[i].id} · ${fmt(p2.mergedCrowd[i], 2)}`)
          .join(", ");
        return `${fit}. Front ${p2.whole.length + 1} has ${p2.split.length} members for the ${p2.places} remaining place${p2.places === 1 ? "" : "s"}, and the fronts after it are discarded. Inside it, the extremes get ∞ and the others the sum of the sides of their cuboid divided by the front's ranges: ${dist}. The largest distances win (a tie goes to the solution that comes first): ${idList(ids(winners, p2.merged))}.`;
      }
      const kept = p2.kept.map((i) => p2.merged[i].id);
      const removed = p2.merged.filter((_, i) => !p2.kept.includes(i)).map((p) => p.id);
      const children = p2.kept.filter((i) => i >= N).length;
      return `The next population is ${set(kept)}: ${N - children} parent${N - children === 1 ? "" : "s"} and ${children} child${children === 1 ? "" : "ren"}; ${idList(removed)} leave. Phase 2 repeats this generation until the budget is spent; δ and Pf are no longer computed.`;
    }
    build();
  };

  /* CMOEA-MS at a glance: a generation in stage A, the stage rule, and a generation in stage B. */
  widgets["cmoeams-steps"] = function (node) {
    const N = 6;
    const BUDGET = 150;
    const LAMBDA = 0.5;
    const state = { seed: Number(node.dataset.seed || 27749), step: 0 };
    let g = null;
    let steps = [];
    const interesting = (c) => {
      const [g1, g2] = c.generations;
      const same = (a, b) => a.join() === b.join();
      return !g1.stageB && g2.stageB && g1.A.kept.some((i) => g1.Q[i].cv > 0) && !same(g1.A.kept, g1.B.kept) && !same(g2.A.kept, g2.B.kept);
    };
    const ctl = controls(node);
    const stageSelect = select(ctl, {
      label: "Stage",
      options: [
        ["0", "Generation 1"],
        ["1", "Generation 2"],
      ],
      value: "0",
      onChange: (v) => go(steps.findIndex((st) => st.gen === Number(v))),
    });
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw runs until generation 1 is in stage A and keeps an infeasible solution, generation 2 is in stage B, and
      // in both generations the other stage would have kept other solutions.
      for (let tries = 0; tries < 300; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(cmoeamsGeneration(seed, N, BUDGET, LAMBDA))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the population and of the offspring of CMOEA-MS" });
    const sPlot = createPlot(body, { title: "Stage A's points [SDE, CVn]", x: [0.3, 0.5], y: [0, 1.08], xLabel: "SDE (lower is less crowded)", yLabel: "CVn (normalized violation)", xCount: 4, label: "The points [SDE, CVn] of the union Q, which stage A ranks by Pareto dominance" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 1.6], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, CTP2's infeasible region and the solutions that CMOEA-MS keeps" });
    sPlot.wrap.hidden = true;
    drawConstrainedLoop(xPlot, fPlot);
    for (const plot of [xPlot, sPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; Stage jumps to the first step of a generation, and New population draws another run.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent, feasible", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-1" },
      { label: "Infeasible, CV > 0 (hollow)", shape: "circle", cls: "vf-pt vf-pt-open vf-pt-muted" },
      { label: "Not evaluated yet", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Tournament winner, fitness below 1, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Boundary of what the fitness-below-1 solutions dominate", shape: "line", cls: "vf-line-ink" },
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Constrained Pareto front", shape: "line", cls: "vf-line-1" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = cmoeamsGeneration(state.seed, N, BUDGET, LAMBDA);
      steps = [];
      g.generations.forEach((gen, k) => {
        const stage = gen.stageB ? "B" : "A";
        steps.push({ gen: k, kind: "pop", title: k === 0 ? "the population and its fitness" : "the population and the fitness it got in Q" });
        steps.push({ gen: k, kind: "tournament", title: "mating selection" });
        steps.push({ gen: k, kind: "vary", title: "SBX and polynomial mutation" });
        steps.push({ gen: k, kind: "union", title: "evaluation and the union Q" });
        steps.push({ gen: k, kind: "rule", title: "the stage rule" });
        if (!gen.stageB) steps.push({ gen: k, kind: "sde", title: "shift-based density" });
        steps.push({ gen: k, kind: "fitness", title: gen.stageB ? "fitness in stage B, the violation first" : "fitness in stage A, density and violation as two objectives" });
        steps.push({ gen: k, kind: "select", title: `environmental selection in stage ${stage}` });
      });
      Array.from(stageSelect.options).forEach((opt, k) => (opt.textContent = `Stage ${g.generations[k].stageB ? "B" : "A"}: generation ${k + 1}`));
      const sde = g.generations.filter((gen) => !gen.stageB).flatMap((gen) => gen.sde);
      if (sde.length) sPlot.setDomain([Math.floor(Math.min(...sde) * 40 - 1) / 40, Math.ceil(Math.max(...sde) * 40 + 1) / 40], [0, 1.08]);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(steps.length - 1, i));
      render();
    }
    const set = (list) => "{" + list.slice().sort((a, b) => a - b).join(", ") + "}";
    const feasibleCls = (p) => (p.cv > 0 ? "vf-pt-1 vf-pt-open" : "vf-pt-1");
    function draw(plot, labels, v, shape, cls, label, ring) {
      const px = plot.x(Math.min(Math.max(v[0], plot.xd[0]), plot.xd[1]));
      const py = plot.y(Math.min(Math.max(v[1], plot.yd[0]), plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      labels.push({ px, py, text: label });
    }
    /** The staircase that bounds the region the given points dominate (minimization). */
    function staircase(plot, pts) {
      const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      if (!P.length) return;
      const path = [[P[0][0], plot.yd[1]]];
      P.forEach((p, k) => {
        path.push([p[0], k ? P[k - 1][1] : p[1]], [p[0], p[1]]);
      });
      path.push([plot.xd[1], P[P.length - 1][1]]);
      plot.line(plot.layers.mid, path, "vf-line-ink vf-dashed");
    }
    function render() {
      const st = steps[state.step];
      const gen = g.generations[st.gen];
      stageSelect.value = String(st.gen);
      back.disabled = state.step === 0;
      next.disabled = state.step === steps.length - 1;
      const plane = !gen.stageB && (st.kind === "sde" || st.kind === "fitness" || st.kind === "select");
      xPlot.wrap.hidden = plane;
      sPlot.wrap.hidden = !plane;
      for (const plot of [xPlot, sPlot, fPlot]) {
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      const left = plane ? sPlot : xPlot;
      const ll = [];
      const fl = [];
      const out = st.kind === "pop" || st.kind === "tournament" || st.kind === "vary" ? renderMating(st, gen, ll, fl) : renderUnion(st, gen, left, ll, fl);
      spreadPointLabels(left, ll);
      spreadPointLabels(fPlot, fl);
      stepText.textContent = `Step ${state.step + 1} of ${steps.length}: ${st.title}. ${describe(st, gen)}`;
      clear(tableBox);
      table(tableBox, out.headers, out.rows, out.numeric);
    }
    function renderMating(st, gen, ll, fl) {
      const rows = [];
      const headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "CV", "Fitness", "Detail"];
      const numeric = [false, false, true, true, true, true, true, true, false];
      const wins = new Map();
      gen.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
      const byId = new Map(gen.pop.map((p) => [p.id, p]));
      if (st.kind === "tournament") gen.tournaments.forEach((t) => fPlot.line(fPlot.layers.mid, [byId.get(t.a).f, byId.get(t.b).f], "vf-line-thin vf-dashed"));
      gen.pop.forEach((p, i) => {
        const w = wins.get(p.id) || 0;
        const ring = st.kind === "pop" ? gen.fitness[i] < 1 : st.kind === "tournament" && w > 0;
        const label = String(p.id) + (st.kind === "tournament" && w > 1 ? " ×" + w : "");
        draw(xPlot, ll, p.x, "circle", "vf-pt " + feasibleCls(p), label, ring);
        draw(fPlot, fl, p.f, "circle", "vf-pt " + feasibleCls(p), label, ring);
        rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 3), fmt(p.f[1], 3), p.cv > 0 ? fmt(p.cv, 4) : "0", fmt(gen.fitness[i], 4), st.kind === "tournament" ? (w ? `won ${w}×` : "–") : gen.fitness[i] < 1 ? "fitness below 1" : "–"]);
      });
      if (st.kind === "vary") {
        gen.children.forEach((c) => {
          c.parents.forEach((id) => xPlot.line(xPlot.layers.mid, [byId.get(id).x, c.x], "vf-line-thin vf-dashed"));
          draw(xPlot, ll, c.x, "triangle", "vf-pt vf-pt-muted", String(c.id), false);
          rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "?", "–", "not evaluated"]);
        });
      }
      return { headers, rows, numeric };
    }
    function renderUnion(st, gen, left, ll, fl) {
      const Q = gen.Q;
      const res = gen.stageB ? gen.B : gen.A;
      const keep = new Set(gen.kept);
      const rows = [];
      let headers;
      let numeric;
      const below = Q.map((_, i) => res.fitness[i] < 1);
      if (st.kind === "fitness") {
        const pts = Q.map((q, i) => (gen.stageB ? q.f : [gen.sde[i], gen.cvn[i]])).filter((_, i) => below[i]);
        staircase(gen.stageB ? fPlot : sPlot, pts);
      }
      Q.forEach((q, i) => {
        const isChild = i >= N;
        const removed = st.kind === "select" && !keep.has(i);
        const cls = "vf-pt " + feasibleCls(q) + (removed ? " vf-faded" : "");
        const ring = (st.kind === "fitness" && below[i]) || (st.kind === "select" && keep.has(i));
        // The fitness is written only in the plot where the stage ranks Q: [SDE, CVn] in stage A, the objectives in stage B.
        const label = String(q.id) + (removed ? " ✕" : "");
        const ranked = st.kind === "fitness" && below[i] ? `${q.id} · ${fmt(res.fitness[i], 3)}` : label;
        const lv = left === sPlot ? [gen.sde[i], gen.cvn[i]] : q.x;
        draw(left, ll, lv, isChild ? "triangle" : "circle", cls, left === sPlot ? ranked : label, ring);
        draw(fPlot, fl, q.f, isChild ? "triangle" : "circle", cls, gen.stageB ? ranked : label, ring);
        const role = isChild ? "child" : "parent";
        if (st.kind === "union" || st.kind === "rule") {
          rows.push([String(q.id), role, fmt(q.x[0], 2), fmt(q.x[1], 2), fmt(q.f[0], 3), fmt(q.f[1], 3), q.cv > 0 ? fmt(q.cv, 4) : "0", fmt(gen.cvn[i], 3), q.cv > 0 ? "infeasible" : "feasible"]);
        } else if (st.kind === "sde") {
          rows.push([String(q.id), role, fmt(q.f[0], 3), fmt(q.f[1], 3), fmt(gen.dSde[i], 3), fmt(gen.sde[i], 3), fmt(gen.cvn[i], 3)]);
        } else if (st.kind === "fitness") {
          const dom = res.by[i].map((j) => Q[j].id);
          const cells = [String(q.id), role];
          if (!gen.stageB) cells.push(fmt(gen.sde[i], 3));
          cells.push(fmt(gen.cvn[i], 3), dom.length ? set(dom) : "–", String(res.R[i]), fmt(res.dcos[i], 3), fmt(res.D[i], 4), fmt(res.fitness[i], 4));
          rows.push(cells);
        } else {
          rows.push([String(q.id), role, fmt(gen.cvn[i], 3), fmt(res.fitness[i], 4), keep.has(i) ? "kept" : "removed"]);
        }
      });
      if (st.kind === "union" || st.kind === "rule") {
        headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "CV", "CVn", "Status"];
        numeric = [false, false, true, true, true, true, true, true, false];
      } else if (st.kind === "sde") {
        headers = ["Id", "Role", "f₁", "f₂", "d", "SDE", "CVn"];
        numeric = [false, false, true, true, true, true, true];
      } else if (st.kind === "fitness") {
        headers = ["Id", "Role"].concat(gen.stageB ? [] : ["SDE"], ["CVn", "Dominated by", "R", "d_cos", "D", "Fitness"]);
        numeric = [false, false].concat(gen.stageB ? [] : [true], [true, false, true, true, true, true]);
      } else {
        headers = ["Id", "Role", "CVn", "Fitness", "Status"];
        numeric = [false, false, true, true, false];
      }
      return { headers, rows, numeric };
    }
    const ordinal = (n) => n + (n % 10 === 1 && n % 100 !== 11 ? "st" : n % 10 === 2 && n % 100 !== 12 ? "nd" : n % 10 === 3 && n % 100 !== 13 ? "rd" : "th");
    function describe(st, gen) {
      const Q = gen.Q;
      const idsOf = (rows) => rows.map((i) => Q[i].id);
      const k = Math.floor(Math.sqrt(Q.length));
      if (st.kind === "pop") {
        const below = gen.pop.filter((_, i) => gen.fitness[i] < 1).map((p) => p.id);
        const infeasible = gen.pop.filter((p) => p.cv > 0).map((p) => p.id);
        const inf = infeasible.length ? `${idList(infeasible)} ${infeasible.length === 1 ? "is" : "are"} infeasible (hollow)` : "all are feasible";
        if (st.gen === 0) return `Six random solutions of the loop figure's problem, with CTP2's constraint on its objectives: ${inf}. The initial population gets stage A's fitness, computed on these ${N} alone: it is below 1 for the solutions that no other one dominates, here ${idList(below)} (ringed). The run has a budget of ${BUDGET} evaluations, and λ = ${LAMBDA}.`;
        return `The population that generation 1 selected, in the order of Q₁: ${inf}. Its fitness is the one it got in Q₁, in stage ${g.generations[0].stageB ? "B" : "A"}; it is not computed again for the tournaments. Fitness below 1: ${idList(below)} (ringed).`;
      }
      if (st.kind === "tournament") {
        const games = gen.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner}`).join("; ");
        const pairs = gen.pairs.map((p) => `(${p[0]}, ${p[1]})`).join(", ");
        return `Six binary tournaments, each between two members drawn with replacement; the lower fitness wins, and a tie goes to the member that comes first: ${games}. The i-th winner of the first half mates with the i-th of the second half: ${pairs}.`;
      }
      if (st.kind === "vary") {
        const half = gen.pairs.length;
        const twins = gen.pairs.filter((p) => p[0] === p[1]).map((p) => p[0]);
        const twinText = twins.length ? ` A pair made of one solution twice, here ${idList(twins)}, cannot be crossed: its children start as copies.` : "";
        return `SBX (probability 1, η = 20, each variable crossed with probability 1/2) crosses each pair, with its second parent passed first, and both children are kept: the first children, ${gen.children[0].id} to ${gen.children[half - 1].id}, before the second ones, ${gen.children[half].id} to ${gen.children[2 * half - 1].id}. Polynomial mutation (probability 1/2 per variable, η = 20) then perturbs every child.${twinText} The children are joined to their parents; their objectives are not known yet.`;
      }
      if (st.kind === "union") {
        const top = Math.max(...Q.map((q) => q.cv));
        const worst = Q.filter((q) => q.cv === top).map((q) => q.id);
        return `The ${gen.children.length} children are evaluated, and the union Q of the population and the offspring holds ${Q.length} solutions, in that order. The violation is normalized by its largest value in Q, ${fmt(top, 4)} (solution${worst.length === 1 ? "" : "s"} ${idList(worst)}): CVn = CV / ${fmt(top, 4)}, in [0, 1]. With several constraints, CVn would be the mean of the normalized violations.`;
      }
      if (st.kind === "rule") {
        const feasible = Q.filter((q) => q.cv <= 0).length;
        const tenth = 0.1 * BUDGET;
        const shareOk = gen.share > LAMBDA;
        const evalOk = gen.evaluations >= tenth;
        const why = gen.stageB ? "both hold: Q is in stage B" : !evalOk && shareOk ? "the share is large enough, but too few evaluations are done: Q is in stage A" : !evalOk ? "neither holds: Q is in stage A" : "the share is too small: Q is in stage A";
        return `In Q, ${feasible} of the ${Q.length} solutions are feasible: a share of ${feasible}/${Q.length} = ${fmt(gen.share, 3)}. Stage B needs a share greater than λ = ${LAMBDA} and at least 10% of the budget, ${tenth} of ${BUDGET} evaluations, done, the offspring included: ${gen.evaluations} are done. ${why[0].toUpperCase() + why.slice(1)}. The rule is checked again at every generation.`;
      }
      if (st.kind === "sde") {
        const order = Q.map((_, i) => i).sort((a, b) => gen.sde[a] - gen.sde[b] || a - b);
        return `Stage A replaces the objectives by the shift-based density. With the objectives normalized to [0, 1] over Q, every other solution is shifted to x in the objectives where it is better than x, and d is the ${ordinal(k + 1)} smallest distance from x to the shifted solutions, x's own 0 included (⌊√${Q.length}⌋ + 1 = ${k + 1}); SDE = 1/(d + 2). The least crowded, with the lowest SDE, are ${idList(idsOf(order.slice(0, 2)))}; the most crowded is ${Q[order[order.length - 1]].id}. The left panel now places each solution at [SDE, CVn].`;
      }
      if (st.kind === "fitness") {
        const res = gen.stageB ? gen.B : gen.A;
        const below = Q.map((_, i) => i).filter((i) => res.fitness[i] < 1);
        const common = `S counts the solutions a solution dominates, R adds the S of its dominators, and D = 1/(d_cos + 2), where d_cos is the cosine distance to the ${ordinal(k)} nearest other ${gen.stageB ? "objective vector" : "point"} (⌊√${Q.length}⌋ = ${k}); fitness = R + D, below 1 only for the non-dominated.`;
        if (!gen.stageB) return `Stage A ranks the points [SDE, CVn] of the left panel by Pareto dominance, so a less crowded infeasible solution can beat a crowded feasible one. ${common} Non-dominated: ${idList(idsOf(below))}, whose dashed boundary encloses everything they dominate.`;
        return `Stage B ranks the objectives with the violation first: a smaller CVn dominates, and equal CVn, such as those of the feasible solutions, are compared by Pareto dominance on f₁ and f₂. ${common} Non-dominated: ${idList(idsOf(below))}.`;
      }
      const res = gen.stageB ? gen.B : gen.A;
      const other = gen.stageB ? gen.A : gen.B;
      const below = Q.map((_, i) => i).filter((i) => res.fitness[i] < 1);
      let how;
      if (below.length < N) how = `Only ${below.length} solution${below.length === 1 ? " has" : "s have"} a fitness below 1, so the ${N} lowest fitness values are kept instead.`;
      else if (below.length > N) {
        const cut = below.filter((i) => !gen.kept.includes(i));
        how = `In Q, ${below.length} solutions have a fitness below 1, more than ${N}: the truncation removes, one at a time, the one whose sorted cosine distances between objective vectors are lexicographically smallest, here ${idList(idsOf(cut))}.`;
      } else how = `Exactly ${N} solutions have a fitness below 1, and they are kept.`;
      const infeasible = gen.kept.filter((i) => Q[i].cv > 0).length;
      const after = st.gen === 0 ? "" : " Each later generation applies the stage rule again, and returns to stage A whenever half or more of Q is infeasible.";
      return `${how} The next population, in the order of Q, is ${set(idsOf(gen.kept))}, with ${infeasible === 0 ? "no infeasible solution" : infeasible === 1 ? "one infeasible solution" : `${infeasible} infeasible solutions`}; its fitness goes to the next tournaments. Stage ${gen.stageB ? "A" : "B"} would have kept ${set(idsOf(other.kept))}.${after}`;
    }
    build();
  };

  /* A4: AGE-MOEA curvature estimate and survival score. */
  widgets["agemoea-geometry"] = function (node) {
    const angles = [90, 82, 71, 62, 53, 41, 29, 17, 8, 0];
    const state = { p: 2, N: 6, euclid: false };
    const ctl = controls(node);
    slider(ctl, { label: "True p", min: 0.5, max: 3, step: 0.1, value: state.p, format: (v) => v.toFixed(1), onInput: (v) => ((state.p = v), render()) });
    slider(ctl, { label: "Survivors N", min: 2, max: 10, step: 1, value: state.N, format: (v) => String(v), onInput: (v) => ((state.N = v), render()) });
    checkbox(ctl, { label: "What if: Euclidean distances (p = 2)", checked: false, onChange: (v) => ((state.euclid = v), render()) });
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, { x: [0, 1.1], y: [0, 1.1], xLabel: "f₁", yLabel: "f₂", label: "Ten points on a front of curvature p, AGE-MOEA's estimate and its survival scores" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "True front f₁ᵖ + f₂ᵖ = 1", shape: "line", cls: "vf-front" },
      { label: "Fitted front with the estimate p̂", shape: "line", cls: "vf-line-1" },
      { label: "Kept (selection order shown)", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Dropped", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Central point used for p̂", shape: "circle", cls: "vf-ring" },
    ]);
    function points(p) {
      return angles.map((deg) => {
        const t = (deg * Math.PI) / 180;
        const c = deg === 90 ? 0 : Math.cos(t);
        const sn = deg === 0 ? 0 : Math.sin(t);
        const r = Math.pow(Math.pow(c, p) + Math.pow(sn, p), -1 / p);
        return [r * c, r * sn];
      });
    }
    function curve(p) {
      return frontPoints((t) => Math.pow(Math.max(0, 1 - Math.pow(t, p)), 1 / p), 0, 1, 201);
    }
    function render() {
      const F = points(state.p);
      const res = agemoeaSurvival(F, state.euclid ? 2 : undefined);
      const kept = new Set(agemoeaCut(res.score, state.N));
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      plot.line(plot.layers.mid, [[0, 0], [1.1, 1.1]], "vf-line-thin vf-dashed");
      plot.line(plot.layers.mid, curve(state.p), "vf-front");
      plot.line(plot.layers.mid, curve(res.estimated), "vf-line-1");
      F.forEach((f, i) => {
        const px = plot.x(f[0]);
        const py = plot.y(f[1]);
        marker(plot.layers.marks, "circle", px, py, 5.5, kept.has(i) ? "vf-pt vf-pt-1" : "vf-pt-hollow");
        if (i === res.central) marker(plot.layers.marks, "circle", px, py, 10, "vf-ring");
        const ord = res.order.indexOf(i) + 1;
        txt(plot.layers.top, px + 8, py - 6, `${i + 1}${kept.has(i) ? " ·" + ord : ""}`, { class: "vf-mark-label" });
      });
      clear(readout);
      const c = F[res.central];
      h("p", {}, readout, `Estimated p̂ = ${fmt(res.estimated)} (true p = ${state.p.toFixed(1)}) · central point ${res.central + 1} = ${pair(c, 3)}.`);
      if (state.euclid) h("p", {}, readout, "What if: proximities and distances measured with p = 2 instead of p̂. This is not a VAMOS option; it shows why the estimate matters.");
      table(
        readout,
        ["Point", "f₁", "f₂", "Proximity", "Score", "Order", "Status"],
        F.map((f, i) => [String(i + 1), fmt(f[0], 3), fmt(f[1], 3), fmt(res.proximity[i]), fmt(res.score[i]), String(res.order.indexOf(i) + 1), kept.has(i) ? "kept" : "dropped"]),
        [true, true, true, true, true, true, false]
      );
    }
    render();
  };

  /* AGE-MOEA: one generation in ten steps. The geometry steps replace the decision space by the normalized first
     front, where the corners, the diagonal, the central point and the fitted curve live. */
  widgets["agemoea-steps"] = function (node) {
    const N = 6;
    const TITLES = ["the population Pₜ", "mating selection", "variation", "evaluation and merge", "non-dominated sorting", "normalizing the first front", "the estimated geometry p̂", "the survival score", "whole fronts and the cut", "the next population Pₜ₊₁"];
    const state = { seed: Number(node.dataset.seed || 29299), step: 0 };
    let g = null;
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    const shows = (c) => c.geometry.first.length >= 3 && c.geometry.p === c.geometry.estimated && c.geometry.p > 0.1 && c.geometry.p < 20 && c.fronts[c.last].length > c.places;
    button(ctl, "New population", () => {
      // Draw populations until the first front estimates p̂ from a central point and the last front must be cut.
      for (let tries = 0; tries < 200; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (shows(agemoeaGeneration(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the children" });
    const nPlot = createPlot(body, { title: "Normalized first front", x: [0, 1.15], y: [0, 1.15], xLabel: "f̃₁ = (f₁ − z₁) / a₁", yLabel: "f̃₂ = (f₂ − z₂) / a₂", label: "The first front translated by its ideal point and divided by its intercepts" });
    const fPlot = createPlot(body, { title: "Objective space", x: [0, 1], y: [0, 2.1], xLabel: "f₁ (minimize)", yLabel: "f₂ (minimize)", label: "Objective vectors, fronts and the geometry of one AGE-MOEA generation" });
    for (const plot of [xPlot, nPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "The arrow keys step through the figure when it has focus; New population draws another generation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Tournament winner, corner, central point, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Ideal point z of front 1", shape: "cross", cls: "vf-line-ink" },
      { label: "Line through the corners", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Fitted front f̃₁ᵖ̂ + f̃₂ᵖ̂ = 1", shape: "line", cls: "vf-line-1" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = agemoeaGeneration(state.seed, N);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const ids = (rows) => rows.map((i) => g.merged[i].id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const frontsText = (fronts, members) => fronts.map((front, k) => `front ${k + 1} = {${front.map((i) => members[i].id).sort((a, b) => a - b).join(", ")}}`).join(", ");
    const lp = (t, p) => Math.pow(Math.pow(Math.abs(t[0]), p) + Math.pow(Math.abs(t[1]), p), 1 / p);
    let labels = [];
    let spots = [];
    function mark(plot, v, shape, cls, label, ring) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      spots.push({ plot, px, py, r: ring ? 11 : 7 });
      if (label) labels.push({ plot, px, py, text: label });
    }
    function render() {
      const step = state.step;
      const geometryStep = step >= 5 && step <= 7;
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      xPlot.wrap.hidden = geometryStep;
      nPlot.wrap.hidden = !geometryStep;
      const left = geometryStep ? nPlot : xPlot;
      for (const plot of [xPlot, nPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      const G = g.geometry;
      const first = new Set(G.first);
      const keep = new Set(g.keep);
      const lastFront = g.fronts[g.last];
      const cut = lastFront.length > g.places;
      const whole = new Set(g.fronts.slice(0, g.last).flat());
      const scored = (i) => g.mergedRank[i] <= g.last;
      let headers;
      let rows;
      let numeric;
      if (step <= 2) {
        const wins = new Map();
        g.tournaments.forEach((t) => wins.set(t.winner, (wins.get(t.winner) || 0) + 1));
        if (step === 1) g.tournaments.forEach((t) => fPlot.line(fPlot.layers.mid, [g.pop[t.a].f, g.pop[t.b].f], "vf-line-thin vf-dashed"));
        g.pop.forEach((p, i) => {
          const k = wins.get(i) || 0;
          const cls = "vf-pt " + rankClass(g.rank[i]);
          const label = String(p.id) + (step === 1 && k > 1 ? " ×" + k : "");
          mark(xPlot, p.x, "circle", cls, label, step === 1 && k > 0);
          mark(fPlot, p.f, "circle", cls, label, step === 1 && k > 0);
        });
        rows = g.pop.map((p, i) => [String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(g.rank[i] + 1), fmt(g.score[i], 2), step === 1 ? (wins.get(i) ? `won ${wins.get(i)}×` : "–") : "–"]);
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((id) => xPlot.line(xPlot.layers.mid, [g.pop[id - 1].x, c.x], "vf-line-thin vf-dashed"));
            mark(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "–", "–", "not evaluated"]);
          });
        }
        headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Score", "Status"];
        numeric = [false, false, true, true, true, true, true, true, false];
      } else if (!geometryStep) {
        if (step === 4) {
          g.fronts.forEach((front, k) => {
            const pts = front.map((i) => g.merged[i].f).sort((a, b) => a[0] - b[0]);
            if (pts.length > 1) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(k));
          });
        }
        rows = [];
        g.merged.forEach((p, i) => {
          const child = i >= N;
          const sorted = step >= 4;
          const out = (step === 8 && cut && lastFront.includes(i) && !keep.has(i)) || (step >= 8 && g.mergedRank[i] > g.last) || (step === 9 && !keep.has(i));
          const cls = "vf-pt " + (sorted ? rankClass(g.mergedRank[i]) : "vf-pt-muted") + (out ? " vf-faded" : "");
          const ring = (step === 8 && (whole.has(i) || (lastFront.includes(i) && keep.has(i)))) || (step === 9 && keep.has(i));
          const label = String(p.id) + (step === 9 && !keep.has(i) ? " ✕" : "");
          // The scores of the cut front go only to the objective space, where the cut is measured.
          const score = step === 8 && cut && lastFront.includes(i) ? " · " + fmt(g.mergedScore[i], 2) : "";
          mark(xPlot, p.x, child ? "triangle" : "circle", cls, label, ring);
          mark(fPlot, p.f, child ? "triangle" : "circle", cls, label + score, ring);
          let status = child ? "child" : "parent";
          if (step === 8) status = whole.has(i) ? "kept, whole front" : lastFront.includes(i) ? (cut ? (keep.has(i) ? "kept by score" : "removed by score") : "kept, fills the places") : "discarded";
          if (step === 9) status = keep.has(i) ? "kept" : "removed";
          rows.push([String(p.id), child ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), sorted ? String(g.mergedRank[i] + 1) : "?", step >= 8 && scored(i) ? fmt(g.mergedScore[i], 2) : "–", status]);
        });
        headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Score", "Status"];
        numeric = [false, false, true, true, true, true, true, true, false];
      } else {
        const z = G.ideal;
        const a = G.intercepts;
        const toF = (t) => [z[0] + a[0] * t[0], z[1] + a[1] * t[1]];
        const corners = new Set(G.corners);
        // Objective space: front 1 in colour, every other member faded, the ideal point and the translated axes.
        g.merged.forEach((p, i) => {
          const inFirst = first.has(i);
          const laterScored = step === 7 && !inFirst && scored(i);
          const cls = "vf-pt " + rankClass(g.mergedRank[i]) + (inFirst || laterScored ? "" : " vf-faded");
          const ring = (step === 5 && corners.has(i)) || (step === 6 && i === G.central);
          mark(fPlot, p.f, i >= N ? "triangle" : "circle", cls, inFirst || laterScored ? String(p.id) : null, ring);
        });
        marker(fPlot.layers.marks, "cross", fPlot.x(z[0]), fPlot.y(z[1]), 5, "vf-line-ink");
        labels.push({ plot: fPlot, px: fPlot.x(z[0]), py: fPlot.y(z[1]), text: "z" });
        fPlot.line(fPlot.layers.mid, [z, [z[0] + 1.2 * a[0], z[1]]], "vf-line-thin");
        fPlot.line(fPlot.layers.mid, [z, [z[0], z[1] + 1.2 * a[1]]], "vf-line-thin");
        if (step === 5 && G.corners.length > 1) fPlot.line(fPlot.layers.mid, [toF([1, 0]), toF([0, 1])], "vf-line-thin vf-dashed");
        const curve = frontPoints((t) => Math.pow(Math.max(0, 1 - Math.pow(t, G.p)), 1 / G.p), 0, 1, 121);
        if (step >= 6) {
          fPlot.line(fPlot.layers.mid, curve.map(toF), "vf-line-1");
          nPlot.line(nPlot.layers.mid, curve, "vf-line-1");
        }
        if (step === 6) {
          nPlot.line(nPlot.layers.mid, [[0, 0], [1.15, 1.15]], "vf-line-thin vf-dashed");
          fPlot.line(fPlot.layers.mid, [z, toF([1.15, 1.15])], "vf-line-thin vf-dashed");
        }
        if (step === 5 && G.corners.length > 1) nPlot.line(nPlot.layers.mid, [[1, 0], [0, 1]], "vf-line-thin vf-dashed");
        G.first.forEach((i, k) => {
          const t = G.normalized[k];
          const ring = (step === 5 && corners.has(i)) || (step === 6 && i === G.central);
          const order = G.order.indexOf(i);
          const label = String(g.merged[i].id) + (step === 7 && order >= 0 ? " ·" + (order + 1) : "");
          mark(nPlot, t, i >= N ? "triangle" : "circle", "vf-pt " + rankClass(0), label, ring);
        });
        headers = ["Id", "f₁", "f₂", "f̃₁", "f̃₂", "Role", "P(s)", "Score", "Order"];
        numeric = [false, true, true, true, true, false, true, true, true];
        rows = G.first.map((i, k) => {
          const t = G.normalized[k];
          const order = G.order.indexOf(i);
          return [
            String(g.merged[i].id),
            fmt(g.merged[i].f[0], 2),
            fmt(g.merged[i].f[1], 2),
            fmt(t[0], 2),
            fmt(t[1], 2),
            corners.has(i) ? "corner" : step >= 6 && i === G.central ? "central point" : "–",
            step >= 7 ? fmt(g.mergedProximity[i], 2) : "–",
            step >= 7 ? fmt(g.mergedScore[i], 2) : "–",
            step >= 7 && order >= 0 ? String(order + 1) : "–",
          ];
        });
        if (step === 7) {
          g.merged.forEach((p, i) => {
            if (first.has(i) || !scored(i)) return;
            const t = p.f.map((v, j) => (v - z[j]) / a[j]);
            rows.push([String(p.id), fmt(p.f[0], 2), fmt(p.f[1], 2), fmt(t[0], 2), fmt(t[1], 2), `front ${g.mergedRank[i] + 1}`, fmt(g.mergedProximity[i], 2), fmt(g.mergedScore[i], 2), "–"]);
          });
        }
      }
      for (const plot of [left, fPlot]) {
        placeMergedLabels(
          plot,
          labels.filter((l) => l.plot === plot),
          spots.filter((p) => p.plot === plot)
        );
      }
      labels = [];
      spots = [];
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, headers, rows, numeric);
    }
    function describe(step) {
      const G = g.geometry;
      const first = new Set(G.first);
      const scored = (i) => g.mergedRank[i] <= g.last;
      if (step === 0) {
        const corners = g.geometry0.corners.map((i) => g.pop[i].id);
        const scores = corners.length ? ` In front 1, the corners ${idList(corners.sort((a, b) => a - b))} score ∞ and the other members the greedy score of step 8; later fronts score 1/P, the inverse of their distance to the ideal point.` : "";
        return `Six solutions of a two-variable problem whose Pareto front is ZDT1's: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂. They are sorted into fronts, ${frontsText(g.fronts0, g.pop)}, and VAMOS scores them by running the survival on the population itself.${scores}`;
      }
      if (step === 1) {
        const why = { front: "better front", score: "same front, higher score", coin: "full tie, coin" };
        const games = g.tournaments.map((t) => `${g.pop[t.a].id} vs ${g.pop[t.b].id} → ${g.pop[t.winner].id} (${why[t.reason]})`).join("; ");
        return `Six binary tournaments pick three pairs of parents: ${games}. The better front wins; within a front, the higher survival score, as in the author's code. Ringed members won at least once.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 15, AGE-MOEA's default) recombines each pair and polynomial mutation (probability 1/2 per variable, η = 20) perturbs each child: children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents in the decision space. Their objectives are not known yet.`;
      }
      if (step === 3) return `The ${N} children are evaluated and merged with the ${N} parents: ${2 * N} solutions compete for ${N} places, as in NSGA-II.`;
      if (step === 4) return `The merged population falls into ${g.fronts.length} fronts: ${frontsText(g.fronts, g.merged)}. Lines join the members of each front. Only the measure that cuts the last front differs from NSGA-II.`;
      if (step === 5) {
        const z = G.ideal;
        const intro = `The geometry is estimated on front 1 = ${set(G.first)}. Its ideal point z = ${pair(z, 2)}, the smallest value of each objective in the front, becomes the origin.`;
        if (G.corners.length < 2) return `${intro} With a single member there are no two corners to draw a line through, so VAMOS uses p̂ = 1 and gives front 1 the score 0.`;
        const [c1, c2] = G.corners.map((i) => g.merged[i].id);
        return `${intro} The corners, the members nearest to each translated axis, are ${c1} (f₁ axis) and ${c2} (f₂ axis). The dashed line through them meets the axes at the intercepts a₁ = ${fmt(G.intercepts[0], 2)} and a₂ = ${fmt(G.intercepts[1], 2)}, which divide the objectives: f̃ = (f − z) / a puts the corners at (1, 0) and (0, 1), on the left.`;
      }
      if (step === 6) {
        if (G.first.length < 2) return "With a single member, front 1 has no shape to estimate: p̂ = 1.";
        if (G.central < 0) return "Front 1 has no member besides its two corners, so there is no central point to read and p̂ = 1, a flat front.";
        const k = G.first.indexOf(G.central);
        const t = G.normalized[k];
        const mean = (t[0] + t[1]) / 2;
        const shape = G.p < 1 ? "below 1: a convex front, bulging towards the ideal point, like ZDT1's (p ≈ 0.72)" : G.p > 1 ? "above 1: a concave front" : "1: a flat front";
        const guard = G.estimated !== G.p ? ` VAMOS replaces this estimate by ${fmt(G.p, 2)}.` : "";
        return `The central point is the non-corner member nearest to the diagonal through (1, 1): ${g.merged[G.central].id}, at f̃ = ${pair(t, 2)}. Its mean c̄ = ${fmt(mean, 3)} gives p̂ = ln 2 / ln(1/c̄) = ${fmt(G.estimated, 2)}, ${shape}.${guard} The blue curve is the model f̃₁ᵖ̂ + f̃₂ᵖ̂ = 1, drawn in both spaces.`;
      }
      if (step === 7) {
        if (G.corners.length < 2) return `Front 1 has a single member, which scores 0. Later fronts score 1/P, the inverse of their distance to z.`;
        const corners = idList(G.corners.map((i) => g.merged[i].id));
        const greedy = G.order.filter((i) => g.mergedScore[i] !== Infinity).map((i) => `${g.merged[i].id} (${fmt(g.mergedScore[i], 2)})`);
        let text = `The proximity P(s) is the Lp distance of f̃(s) to the origin with p̂ = ${fmt(G.p, 2)}. The corners ${corners} score ∞ and form the selected set S. Then, ${greedy.length ? `again and again, the member s with the largest D(s) = d̂(s, q₁) + d̂(s, q₂), its Lp distances to the two nearest members of S divided by P(s), joins S with that score: ${greedy.join(", ")}. The labels give the order of selection; the scores never increase along it.` : "no other member is left."}`;
        const later = g.merged.map((_, i) => i).filter((i) => !first.has(i) && scored(i));
        if (later.length) text += ` Later fronts up to front ${g.last + 1}, the one that fills the population, score 1/P, with P measured in front 1's normalized coordinates (table).`;
        return text;
      }
      if (step === 8) {
        const used = N - g.places;
        const lastFront = g.fronts[g.last];
        const fit = g.last === 0 ? "No front fits entirely" : g.last === 1 ? `Front 1 fits entirely, ${used} of ${N} places` : `Fronts 1 ${g.last === 2 ? "and" : "to"} ${g.last} fit entirely, ${used} of ${N} places`;
        if (lastFront.length <= g.places) {
          const lead = g.last === 0 ? `Front 1 fills the ${N} places` : `${fit}, and front ${g.last + 1} fills the remaining ${g.places} place${g.places === 1 ? "" : "s"}`;
          return `${lead} exactly: the scores decide nothing in this survival. The fronts after it are discarded.`;
        }
        const kept = lastFront.filter((i) => g.keep.includes(i)).sort((a, b) => cmpDesc(g.mergedScore[a], g.mergedScore[b]));
        const lost = lastFront.filter((i) => !g.keep.includes(i)).sort((a, b) => cmpDesc(g.mergedScore[a], g.mergedScore[b]));
        const show = (list) => idList(list.map((i) => `${g.merged[i].id} (${fmt(g.mergedScore[i], 2)})`));
        const measure = g.last === 0 ? "its greedy scores" : "1/P, so the members closest to the ideal point survive";
        return `${fit}. Front ${g.last + 1} has ${lastFront.length} members for ${g.places} place${g.places === 1 ? "" : "s"} and is cut by ${measure}: ${show(kept)} ${kept.length === 1 ? "stays" : "stay"}, and ${show(lost)} ${lost.length === 1 ? "is" : "are"} removed.${g.last + 1 < g.fronts.length ? " The fronts after it are discarded." : ""}`;
      }
      const kept = g.keep.slice().sort((a, b) => a - b);
      const parents = kept.filter((i) => i < N).length;
      const children = kept.length - parents;
      const removed = g.merged.map((_, i) => i).filter((i) => !g.keep.includes(i));
      return `Pₜ₊₁ = ${set(kept)}: ${parents} parent${parents === 1 ? "" : "s"} and ${children} child${children === 1 ? "" : "ren"} survive, and ${idList(ids(removed))} leave. The scores of this survival break the ties of the next generation's tournaments.`;
    }
    build();
  };

  /* A2: NSGA-III association and niche preservation. */
  widgets["nsga3-niching"] = function (node) {
    const W = [0, 1, 2, 3, 4, 5].map((k) => [k / 5, 1 - k / 5]);
    const data = [
      ["A", 0, 1, 1],
      ["B", 0.09, 0.99, 1],
      ["C", 0.5, 0.87, 1],
      ["D", 0.87, 0.5, 1],
      ["E", 1, 0, 1],
      ["F", 0.06, 1.18, 2],
      ["G", 0.3, 1.1, 2],
      ["H", 0.41, 1.02, 2],
      ["I", 0.61, 0.975, 2],
      ["J", 0.72, 0.9, 2],
      ["K", 1.105, 0.317, 2],
      ["L", 0.2, 1.3, 3],
      ["M", 0.5, 1.2, 3],
      ["N", 0.68, 1.12, 3],
      ["O", 0.8, 1.05, 3],
      ["P", 1, 0.95, 3],
      ["Q", 1.2, 0.4, 3],
      ["R", 1.3, 0.35, 3],
    ].map(([id, f1, f2, front]) => ({ id, f: [f1, f2], front, ...nsga3Associate([f1, f2], W) }));
    const state = { N: 9, seed: 7, step: 0, focus: null };
    const ctl = controls(node);
    slider(ctl, { label: "Population size N", min: 6, max: 10, step: 1, value: state.N, format: (v) => String(v), onInput: (v) => ((state.N = v), (state.step = 0), render()) });
    button(ctl, "Step", () => ((state.step += 1), render()), true);
    button(ctl, "Fill", () => ((state.step = 99), render()));
    button(ctl, "Reset", () => ((state.step = 0), render()));
    button(ctl, "Reshuffle", () => ((state.seed = Math.floor(Math.random() * 1e6)), (state.step = 0), render()));
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1.4], y: [0, 1.4], xLabel: "f₁ (normalized)", yLabel: "f₂ (normalized)", label: "NSGA-III reference directions, associations and niche preservation" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Front 1, already selected", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Front 2 (last front), candidate", shape: "circle", cls: "vf-pt-hollow" },
      { label: "Picked by niching (round shown)", shape: "circle", cls: "vf-pt vf-pt-2" },
      { label: "Front 3, not examined", shape: "circle", cls: "vf-pt vf-pt-muted vf-faded" },
    ]);
    function render() {
      const rho0 = W.map((_, j) => data.filter((d) => d.front === 1 && d.dir === j).length);
      const cands = data.filter((d) => d.front === 2);
      const K = state.N - 5;
      const all = nsga3Niching(rho0, cands, K, mulberry32(state.seed));
      const shown = all.picks.slice(0, Math.min(state.step, all.picks.length));
      const picked = new Map(shown.map((p) => [p.id, p.round]));
      const rho = rho0.slice();
      shown.forEach((p) => (rho[p.dir] += 1));
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      plot.line(plot.layers.mid, [[0, 1], [1, 0]], "vf-front");
      W.forEach((w, j) => {
        const t = 1.38 / Math.max(w[0], w[1]);
        plot.line(plot.layers.mid, [[0, 0], [w[0] * t, w[1] * t]], "vf-line-thin");
        txt(plot.layers.top, plot.x(w[0] * t * 0.97) + (j === 0 ? 5 : j === 1 ? 5 : -18), plot.y(w[1] * t * 0.97) + (j === 5 ? -8 : j < 2 ? 10 : 4), `w${j + 1}`, { class: "vf-mark-label" });
      });
      data.forEach((d) => {
        const px = plot.x(d.f[0]);
        const py = plot.y(d.f[1]);
        const w = W[d.dir];
        const wn = Math.hypot(w[0], w[1]);
        const dot = (d.f[0] * w[0] + d.f[1] * w[1]) / wn;
        if (d.front === 2 || state.focus === d.id) plot.line(plot.layers.mid, [d.f, [(dot * w[0]) / wn, (dot * w[1]) / wn]], state.focus === d.id ? "vf-line-2" : "vf-line-thin vf-dashed");
        const cls = d.front === 1 ? "vf-pt vf-pt-1" : d.front === 3 ? "vf-pt vf-pt-muted vf-faded" : picked.has(d.id) ? "vf-pt vf-pt-2" : "vf-pt-hollow";
        const hit = s("circle", { cx: px, cy: py, r: 11, class: "vf-hit" }, plot.layers.marks);
        const mark = marker(plot.layers.marks, "circle", px, py, 5.5, cls);
        const show = () => {
          tip.show(`${d.id} · front ${d.front} · w${d.dir + 1} · d⊥ ${fmt(d.dperp, 4)}`, px, py);
          if (state.focus !== d.id) {
            state.focus = d.id;
            render();
          }
        };
        hit.addEventListener("pointerenter", show);
        mark.addEventListener("pointerenter", show);
        hit.addEventListener("pointerleave", () => tip.hide());
        txt(plot.layers.top, px + 8, py - 6, d.id + (picked.has(d.id) ? " ·" + picked.get(d.id) : ""), { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", {}, readout, picked.size >= K ? `Population full: ${data.filter((d) => d.front === 1 || picked.has(d.id)).map((d) => d.id).join(" ")}.` : `Selected ${5 + picked.size} of ${state.N} · places left ${K - picked.size}.`);
      shown.forEach((p) => {
        const d = data.find((x) => x.id === p.id);
        h("p", { class: "vf-step-text" }, readout, `Round ${p.round} · smallest count ${p.low} → ${p.Jmin.map((j) => "w" + (j + 1)).join(", ")} · w${p.dir + 1} has count ${p.count}: ${p.closest ? `closest candidate ${p.id} (d⊥ ${fmt(d.dperp, 3)})` : `random candidate ${p.id}`}`);
      });
      table(
        readout,
        ["Direction", "ρ", "Last-front candidates"],
        W.map((_, j) => ["w" + (j + 1), String(rho[j]), cands.filter((c) => c.dir === j).map((c) => c.id).join(", ") || "none"]),
        [false, true, false]
      );
    }
    render();
  };

  /* NSGA-III at a glance: one generation in eleven steps, normalization and niching included. */
  widgets["nsga3-steps"] = function (node) {
    const N = 6;
    const TITLES = [
      "the population Pₜ",
      "mating selection",
      "variation",
      "evaluation and merge",
      "non-dominated sorting",
      "whole fronts and the last front",
      "normalization",
      "association",
      "niche counts",
      "niching of the last front",
      "the next population Pₜ₊₁",
    ];
    const NORMALIZED = [7, 8, 9];
    const state = { seed: Number(node.dataset.seed || 16892), step: 0 };
    let g = null;
    const interesting = (c) => !!c.niching && c.last.length > c.places && c.places > 1 && c.niching.picks.some((p) => p.closest && c.last.filter((i) => c.association[i].dir === p.dir).length > 1);
    const ctl = controls(node);
    const back = button(ctl, "Previous step", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      // Draw populations until niching cuts the last front and an empty direction chooses between candidates.
      for (let tries = 0; tries < 300; tries++) {
        const seed = Math.floor(Math.random() * 1e6);
        if (interesting(nsga3Generation(seed, N))) {
          state.seed = seed;
          break;
        }
      }
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, { title: "Decision space", x: [0, 1], y: [0, 1], xLabel: "x₁", yLabel: "x₂", label: "Decision vectors of the parents and of the children" });
    const fOptions = {
      title: "Objective space",
      x: [0, 1],
      y: [0, 2.1],
      xLabel: "f₁ (minimize)",
      yLabel: "f₂ (minimize)",
      label: "Objective vectors, fronts, normalization and reference directions of one NSGA-III generation",
      margin: { t: 22, r: 28 },
    };
    const fPlot = createPlot(body, fOptions);
    const fTitle = fPlot.wrap.querySelector(".vf-title");
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    const stepText = h("p", { class: "vf-step-text", "aria-live": "polite" }, readout);
    h("p", {}, readout, "Steps 8 to 10 show the normalized objectives. The arrow keys step through the figure when it has focus; New population draws another generation.");
    const tableBox = h("div", {}, readout);
    legend(node, [
      { label: "Parent", shape: "circle", cls: "vf-pt vf-pt-muted" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-muted" },
      { label: "Front 1, 2, 3 … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Chosen parent, extreme point, survivor", shape: "circle", cls: "vf-ring" },
      { label: "Ideal point z*, nadir estimate", shape: "cross", cls: "vf-line-ink" },
      { label: "Reference direction, line through the extreme points", shape: "line", cls: "vf-line-thin" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT|BUTTON)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      g = nsga3Generation(state.seed, N);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(TITLES.length - 1, i));
      render();
    }
    const ids = (rows) => rows.map((i) => g.merged[i].id);
    const set = (rows) => "{" + ids(rows).sort((a, b) => a - b).join(", ") + "}";
    const dir = (j) => "w" + (j + 1);
    function draw(plot, v, shape, cls, label, ring) {
      const px = plot.x(Math.max(plot.xd[0], Math.min(v[0], plot.xd[1])));
      const py = plot.y(Math.max(plot.yd[0], Math.min(v[1], plot.yd[1])));
      marker(plot.layers.marks, shape, px, py, 6, cls);
      if (ring) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
      if (label) txt(plot.layers.top, px + 9, py - 7, label, { class: "vf-mark-label" });
    }
    function cross(v, label, above) {
      const px = fPlot.x(v[0]);
      const py = fPlot.y(v[1]);
      marker(fPlot.layers.marks, "cross", px, py, 4.5, "vf-line-ink");
      txt(fPlot.layers.top, px + 7, py + (above ? -7 : 15), label, { class: "vf-mark-label" });
    }
    function setSpace(normalized) {
      const inside = g.whole.flat().concat(g.last || []);
      const top = normalized ? Math.max(1.2, Math.ceil(10 * 1.08 * Math.max(...inside.map((i) => Math.max(g.normalized[i][0], g.normalized[i][1])))) / 10) : 0;
      fOptions.xLabel = normalized ? "fⁿ₁ (normalized)" : "f₁ (minimize)";
      fOptions.yLabel = normalized ? "fⁿ₂ (normalized)" : "f₂ (minimize)";
      fTitle.textContent = normalized ? "Normalized objective space" : "Objective space";
      fPlot.svg.setAttribute("aria-label", normalized ? "Normalized objectives, reference directions, associations and niching" : fOptions.label);
      fPlot.setDomain(normalized ? [0, top] : [0, 1], normalized ? [0, top] : [0, 2.1]);
    }
    function render() {
      const step = state.step;
      const normalized = NORMALIZED.includes(step);
      back.disabled = step === 0;
      next.disabled = step === TITLES.length - 1;
      setSpace(normalized);
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.back);
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.back, [[0, 0.004], [1, 0.004]], "vf-front");
      if (!normalized) fPlot.line(fPlot.layers.back, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
      let rows = [];
      let headers = ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Status"];
      let numeric = [false, false, true, true, true, true, true, false];
      const counts = new Map();
      g.draws.forEach((d) => counts.set(d, (counts.get(d) || 0) + 1));
      if (step <= 2) {
        const byId = new Map(g.pop.map((p) => [p.id, p]));
        if (step === 1) {
          for (let k = 0; k < g.draws.length; k += 2) fPlot.line(fPlot.layers.mid, [g.pop[g.draws[k]].f, g.pop[g.draws[k + 1]].f], "vf-line-thin vf-dashed");
        }
        g.pop.forEach((p, i) => {
          const k = counts.get(i) || 0;
          const label = String(p.id) + (step >= 1 && k > 1 ? " ×" + k : "");
          draw(xPlot, p.x, "circle", "vf-pt vf-pt-1", label, step >= 1 && k > 0);
          draw(fPlot, p.f, "circle", "vf-pt vf-pt-1", label, step >= 1 && k > 0);
          rows.push([String(p.id), "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), "–", step >= 1 ? (k ? `drawn ${k}×` : "not drawn") : "–"]);
        });
        if (step === 2) {
          g.children.forEach((c) => {
            c.parents.forEach((id) => xPlot.line(xPlot.layers.mid, [byId.get(id).x, c.x], "vf-line-thin vf-dashed"));
            draw(xPlot, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), false);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), "?", "?", "–", "not evaluated"]);
          });
        }
      } else if (!normalized) {
        const keep = new Set(g.keep);
        const whole = new Set(g.whole.flat());
        const last = new Set(g.last || []);
        const extremes = new Set(g.extremes);
        if (step === 4) {
          g.fronts.forEach((front, k) => {
            const pts = front.map((i) => g.merged[i].f).sort((a, b) => a[0] - b[0]);
            if (pts.length > 1) fPlot.line(fPlot.layers.mid, pts, "vf-front-link " + rankClass(k));
          });
        }
        if (step === 6) {
          const z = g.ideal;
          const nad = g.nadir;
          fPlot.line(fPlot.layers.mid, [[z[0], 3], z, [3, z[1]]], "vf-line-thin");
          fPlot.rect(fPlot.layers.back, z[0], z[1], nad[0], nad[1], "vf-cuboid");
          if (g.hyperplane) fPlot.line(fPlot.layers.mid, [[z[0] + g.intercepts[0], z[1]], [z[0], z[1] + g.intercepts[1]]], "vf-line-ink vf-dashed");
          cross(z, "z*");
          cross(nad, "nadir", true);
        }
        g.merged.forEach((p, i) => {
          const isChild = i >= N;
          const sorted = step >= 4;
          const out = step >= 5 && !whole.has(i) && !last.has(i);
          const removed = step === 10 && !keep.has(i);
          const cls = "vf-pt " + (sorted ? rankClass(g.rank[i]) : "vf-pt-muted") + (out || removed ? " vf-faded" : "");
          const ring = (step === 5 && whole.has(i)) || (step === 6 && extremes.has(i)) || (step === 10 && keep.has(i));
          let label = String(p.id);
          if (removed) label += " ✕";
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls, label, ring && step !== 6);
          draw(fPlot, p.f, isChild ? "triangle" : "circle", cls, label, ring);
          let status = isChild ? "child" : "parent";
          if (step >= 5) status = whole.has(i) ? "kept, whole front" : last.has(i) ? `last front, ${g.places} place${g.places === 1 ? "" : "s"}` : "discarded";
          if (step === 6 && extremes.has(i)) status = "extreme point";
          if (step === 10) status = keep.has(i) ? (whole.has(i) ? "kept, whole front" : "kept by niching") : "removed";
          rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), sorted ? String(g.rank[i] + 1) : "?", status]);
        });
      } else {
        const whole = new Set(g.whole.flat());
        const last = new Set(g.last || []);
        const picks = new Map((g.niching ? g.niching.picks : []).map((p) => [p.id, p]));
        const top = fPlot.xd[1];
        const rho = g.rho.slice();
        if (step === 9) picks.forEach((p) => (rho[p.dir] += 1));
        g.W.forEach((w, j) => {
          const t = top / Math.max(w[0], w[1]);
          fPlot.line(fPlot.layers.mid, [[0, 0], [w[0] * t, w[1] * t]], "vf-line-thin");
          // The labels sit in the widened top and right margins, clear of the points.
          const px = fPlot.x(w[0] * t);
          const py = fPlot.y(w[1] * t);
          if (w[1] > w[0]) txt(fPlot.layers.top, px, py - 6, dir(j), { class: "vf-mark-label", "text-anchor": "middle" });
          else txt(fPlot.layers.top, px + 4, py + 4, dir(j), { class: "vf-mark-label" });
        });
        if (g.hyperplane) fPlot.line(fPlot.layers.mid, [[1, 0], [0, 1]], "vf-line-ink vf-dashed");
        headers = ["Id", "Front", "fⁿ₁", "fⁿ₂", "Direction", "d⊥", "Status"];
        numeric = [false, true, true, true, false, true, false];
        g.merged.forEach((p, i) => {
          const isChild = i >= N;
          const inside = whole.has(i) || last.has(i);
          const a = g.association[i];
          const fn = g.normalized[i];
          const w = g.W[a.dir];
          const wn = Math.hypot(w[0], w[1]);
          const proj = (fn[0] * w[0] + fn[1] * w[1]) / wn;
          if (inside && step === 7) fPlot.line(fPlot.layers.mid, [fn, [(proj * w[0]) / wn, (proj * w[1]) / wn]], "vf-line-2 vf-dashed");
          const pick = picks.get(i);
          const cls = "vf-pt " + rankClass(g.rank[i]) + (inside ? "" : " vf-faded");
          let label = inside ? String(p.id) : "";
          if (step === 9 && pick) label += " ·" + pick.round;
          if (step === 9 && last.has(i) && !pick) label += " ✕";
          const ring = (step === 8 && whole.has(i)) || (step === 9 && (whole.has(i) || !!pick));
          const out = step === 9 && last.has(i) && !pick ? " vf-faded" : "";
          // A discarded solution outside the normalized window is left out rather than drawn at its edge.
          if (inside || (fn[0] <= top && fn[1] <= top)) draw(fPlot, fn, isChild ? "triangle" : "circle", cls + out, label, ring);
          draw(xPlot, p.x, isChild ? "triangle" : "circle", cls + out, label, ring);
          let status = whole.has(i) ? "whole front" : last.has(i) ? "last front" : "discarded";
          if (step === 9 && last.has(i)) status = pick ? `picked in round ${pick.round}${pick.closest ? ", closest" : ", at random"}` : "not picked";
          if (inside) rows.push([String(p.id), String(g.rank[i] + 1), fmt(fn[0], 3), fmt(fn[1], 3), dir(a.dir), fmt(a.dperp, 3), status]);
        });
        if (step === 8) {
          // The niche counts are a property of the directions: one row per direction.
          headers = ["Direction", "ρ", "Whole fronts", "Last front (d⊥)"];
          numeric = [false, true, false, false];
          const on = (list, j) => list.filter((i) => g.association[i].dir === j).sort((a, b) => g.merged[a].id - g.merged[b].id);
          rows = g.W.map((_, j) => [
            dir(j),
            String(g.rho[j]),
            on(g.whole.flat(), j).map((i) => g.merged[i].id).join(", ") || "–",
            on(g.last || [], j).map((i) => `${g.merged[i].id} (${fmt(g.association[i].dperp, 3)})`).join(", ") || "–",
          ]);
        }
      }
      stepText.textContent = `Step ${step + 1} of ${TITLES.length}: ${TITLES[step]}. ${describe(step)}`;
      clear(tableBox);
      table(tableBox, headers, rows, numeric);
    }
    function describe(step) {
      if (step === 0) {
        return `Six solutions of the loop figure's problem (f₁ = x₁, f₂ = g·(1 − √(x₁/g)), g = 1 + x₂). With two objectives and p = 5 divisions there are H = 6 reference directions, and N = H = 6 is the smallest population VAMOS accepts for them. NSGA-III's default selection is random, so the population needs no ranks or crowding distances before mating.`;
      }
      if (step === 1) {
        const pairs = [];
        for (let k = 0; k < g.draws.length; k += 2) pairs.push(`(${g.pop[g.draws[k]].id}, ${g.pop[g.draws[k + 1]].id})`);
        const missing = g.pop.filter((_, i) => !g.draws.includes(i)).map((p) => p.id);
        return `VAMOS' default selection for NSGA-III, selection("random"), draws ${g.draws.length} parents uniformly at random, with replacement, as Deb and Jain do: the pairs ${idList(pairs)}. Ringed members were drawn at least once${missing.length ? `; ${idList(missing)} ${missing.length === 1 ? "was" : "were"} not drawn` : ""}. Neither the fronts nor any density estimate plays a part.`;
      }
      if (step === 2) {
        return `SBX crossover (probability 0.9, η = 30) recombines each pair and polynomial mutation (probability 1/2 per variable, η = 20) perturbs each child, VAMOS' defaults for NSGA-III: children ${g.children[0].id} to ${g.children[N - 1].id}, the triangles joined to their parents. Their objectives are not known yet.`;
      }
      if (step === 3) {
        return `The ${N} children are evaluated and merged with the ${N} parents: Rₜ = Pₜ ∪ Qₜ holds ${2 * N} solutions for ${N} places, as in NSGA-II.`;
      }
      if (step === 4) return `Non-dominated sorting, unchanged from NSGA-II: ${g.fronts.map((front, k) => `front ${k + 1} = ${set(front)}`).join(", ")}.`;
      if (step === 5) {
        const used = N - g.places;
        const fit = g.whole.length === 0 ? "No front fits entirely" : g.whole.length === 1 ? `Front 1 fits entirely, ${used} of ${N} places` : `Fronts 1 to ${g.whole.length} fit entirely, ${used} of ${N} places`;
        if (!g.last) return `${fit}: the whole fronts fill the places exactly, so this generation needs no niching.`;
        return `${fit}. Front ${g.whole.length + 1} = ${set(g.last)} is the last front: ${g.last.length} members for ${g.places} place${g.places === 1 ? "" : "s"}. NSGA-III cuts it with reference directions instead of crowding distances; the later fronts are discarded.`;
      }
      if (step === 6) {
        const e = g.extremes.map((i) => g.merged[i].id);
        const z = pair(g.ideal);
        const plane = g.hyperplane
          ? `The line through them cuts the axes from z* at a₁ = ${fmt(g.intercepts[0], 2)} and a₂ = ${fmt(g.intercepts[1], 2)}, so the nadir estimate is z* + a = ${pair(g.nadir)}.`
          : `The line through them is rejected (${g.extremes[0] === g.extremes[1] ? "the two extreme points are the same solution" : "an intercept is not positive or the estimate exceeds the worst values seen"}), so VAMOS takes the worst point of the first front, ${pair(g.frontWorst)}, as the nadir estimate${g.fromPopulation.some(Boolean) ? ", and the worst value of the merged population where that point is not above z*" : ""}.`;
        return `The ideal point z* = ${z} holds the smallest values seen so far; in this first generation, those of Rₜ. The ASF picks the extreme points of the first front: ${e[0]} on the f₁ axis (z¹, the smallest f₂) and ${e[1]} on the f₂ axis (z², the smallest f₁). ${plane} Each objective is then divided by its range: fⁿ = (f − z*) / (nadir − z*), which puts the extreme points on the unit axes.`;
      }
      const inside = g.whole.flat().concat(g.last || []);
      if (step === 7) {
        const byDir = g.W.map((_, j) => inside.filter((i) => g.association[i].dir === j));
        const text = byDir.map((list, j) => (list.length ? `${dir(j)} ← ${idList(ids(list).sort((a, b) => a - b))}` : null)).filter(Boolean);
        const empty = byDir.map((list, j) => (list.length ? null : dir(j))).filter(Boolean);
        return `In the normalized space, each solution joins the direction at the smallest angle, and d⊥ (dashed) is its distance to that ray: ${text.join("; ")}.${empty.length ? ` ${idList(empty)} ${empty.length === 1 ? "has" : "have"} no solution of these fronts.` : ""} The discarded fronts are faded.`;
      }
      if (step === 8) {
        const levels = [...new Set(g.rho)].sort((a, b) => b - a);
        const count = levels.map((c) => `ρ = ${c} on ${idList(g.W.map((_, j) => j).filter((j) => g.rho[j] === c).map(dir))}`).join("; ");
        const cands = g.W.map((_, j) => {
          const list = (g.last || []).filter((i) => g.association[i].dir === j);
          return list.length ? `${dir(j)} (${idList(ids(list).sort((a, b) => a - b))})` : null;
        }).filter(Boolean);
        if (!g.last) return `The whole fronts give the niche counts ${count}, but they already fill every place.`;
        return `The niche count ρⱼ counts the survivors of the whole fronts on each direction: ${count}. The last front's members sit on ${idList(cands)}.`;
      }
      if (step === 9) {
        if (!g.niching) return "The whole fronts filled the places exactly, so nothing is left to niche.";
        const rounds = [];
        g.niching.picks.forEach((p) => {
          if (!rounds[p.round - 1]) rounds[p.round - 1] = { low: p.low, tied: p.tied, Jmin: p.Jmin, picks: [] };
          rounds[p.round - 1].picks.push(p);
        });
        const text = rounds.map((r, n) => {
          const served = r.picks.map((p) => {
            const d = g.association[p.id].dperp;
            return `${dir(p.dir)} takes ${g.merged[p.id].id}${p.closest ? ` (closest, d⊥ ${fmt(d, 3)})` : " (at random)"}`;
          });
          const places = r.Jmin.length < r.tied.length ? ` Only ${r.Jmin.length === 1 ? "one place is" : r.Jmin.length + " places are"} left, and the random order serves ${idList(r.Jmin.map(dir))}:` : "";
          const tied = r.tied.length === 1 ? `on ${dir(r.tied[0])}` : `on ${idList(r.tied.map(dir))}`;
          return `Round ${n + 1}: the smallest count among the directions with candidates is ${r.low}, ${tied}.${places} ${idList(served)}.`;
        });
        const left = g.last.filter((i) => !g.keep.includes(i));
        return `${text.join(" ")} An empty direction gets its closest candidate, a direction with survivors a random one. ${left.length ? `${idList(ids(left).sort((a, b) => a - b))} ${left.length === 1 ? "is" : "are"} left out.` : ""}`;
      }
      const kept = g.keep.slice().sort((a, b) => a - b);
      const parents = kept.filter((i) => i < N).length;
      const removed = g.merged.map((_, i) => i).filter((i) => !g.keep.includes(i));
      return `Pₜ₊₁ = ${set(kept)}: ${parents} parent${parents === 1 ? "" : "s"} and ${kept.length - parents} child${kept.length - parents === 1 ? "" : "ren"} survive, and ${idList(ids(removed))} leave. VAMOS keeps the ideal point, the worst values and the extreme points ${idList(g.extremes.map((i) => g.merged[i].id))}: the next generation looks for its extreme points among its first front and these.`;
    }
    build();
  };

  /* ---- Tuning: an F-race block by block, and the budgets of the model-based backends ---- */

  /* Racing: eight configurations, four instances and three seeds, raced by fraceRace (RacingTuner's F-race). */
  widgets["frace-race"] = function (node) {
    const N = 8;
    const INSTANCES = 4;
    const B = INSTANCES * 3;
    const SCALES = [1, 4, 0.5, 2];
    const names = Array.from({ length: N }, (_, c) => "C" + (c + 1));
    const blockName = (b) => `instance ${(b % INSTANCES) + 1}, seed ${Math.floor(b / INSTANCES) + 1}`;
    const state = { seed: 4, noise: 0.03, alpha: 0.05, firstTest: 5, k: 0 };
    let scores = [];
    function generate() {
      const rand = mulberry32(state.seed);
      const quality = names.map((_, c) => c);
      for (let c = N - 1; c > 0; c--) {
        const r = Math.floor(rand() * (c + 1));
        [quality[c], quality[r]] = [quality[r], quality[c]];
      }
      const base = SCALES.map(() => 0.7 + 0.1 * rand());
      scores = names.map((_, c) =>
        Array.from({ length: B }, (_, b) => {
          const i = b % INSTANCES;
          return SCALES[i] * (base[i] + 0.012 * quality[c] + state.noise * gaussian(rand));
        })
      );
    }
    const ctl = controls(node);
    const blocks = slider(ctl, { label: "Blocks played", min: 0, max: B, step: 1, value: 0, format: (v) => String(v), onInput: (v) => ((state.k = v), render()) });
    button(ctl, "Next block", () => go(state.k + 1), true);
    button(ctl, "Restart", () => go(0));
    const ctl2 = controls(node);
    select(ctl2, {
      label: "α",
      options: [
        [0.01, "0.01"],
        [0.05, "0.05 (VAMOS default)"],
        [0.1, "0.10"],
      ],
      value: 0.05,
      onChange: (v) => ((state.alpha = +v), render()),
    });
    slider(ctl2, { label: "first_test", min: 1, max: 10, step: 1, value: state.firstTest, format: (v) => String(v), onInput: (v) => ((state.firstTest = v), render()) });
    slider(ctl2, { label: "Noise σ", min: 0.005, max: 0.08, step: 0.005, value: state.noise, format: (v) => v.toFixed(3), onInput: (v) => ((state.noise = v), generate(), render()) });
    button(ctl2, "New scores", () => ((state.seed += 1), generate(), render()));
    const body = h("div", { class: "vf-body" }, node);
    const ticksX = Array.from({ length: B }, (_, b) => b + 1);
    const plot = createPlot(body, {
      width: 400,
      height: 290,
      margin: { l: 40 },
      x: [0.4, B + 0.6],
      y: [0.4, N + 0.6],
      xTicks: ticksX,
      yTicks: names.map((_, c) => c + 1),
      yFormat: (v) => names[N - v],
      xLabel: "Block (instance and seed), in the order played",
      yLabel: "Configuration",
      label: "A race of eight configurations: their rank on every block played, the tests and the eliminations. Focus it and use the arrow keys to play or rewind blocks.",
    });
    plot.svg.setAttribute("tabindex", "0");
    plot.svg.classList.add("vf-keyboard");
    plot.svg.addEventListener("keydown", (e) => {
      const moves = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 };
      if (e.key in moves) go(state.k + moves[e.key]);
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(B);
      else return;
      e.preventDefault();
    });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Rank 1 in its block", shape: "circle", cls: "vf-pt vf-r0" },
      { label: "Rank 2", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Rank 3", shape: "circle", cls: "vf-pt vf-r2" },
      { label: "Rank 4", shape: "circle", cls: "vf-pt vf-r3" },
      { label: "Rank 5 or worse", shape: "circle", cls: "vf-pt vf-r4" },
      { label: "Test that eliminates", shape: "line", cls: "vf-line-2 vf-dashed" },
      { label: "Test without elimination", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Eliminated", shape: "cross", cls: "vf-line-2" },
    ]);
    function go(k) {
      state.k = Math.max(0, Math.min(B, k));
      blocks.set(state.k);
      render();
    }
    const yOf = (c) => N - c;
    const list = (cs) => cs.map((c) => names[c]).join(", ").replace(/, ([^,]*)$/, " and $1");
    function render() {
      const race = fraceRace(scores, { alpha: state.alpha, firstTest: state.firstTest });
      const shown = race.steps.slice(0, state.k);
      const ended = state.k >= race.steps.length && (race.survivors.length <= 1 || state.k >= B);
      for (const layer of ["back", "mid", "marks", "top"]) clear(plot.layers[layer]);
      if (shown.length && !ended) plot.rect(plot.layers.back, shown.length - 0.5, 0.4, shown.length + 0.5, N + 0.6, "vf-region-1");
      shown.forEach((st) => {
        if (st.test) plot.line(plot.layers.mid, [[st.block + 1.5, 0.4], [st.block + 1.5, N + 0.6]], st.eliminated.length ? "vf-line-2 vf-dashed" : "vf-line-thin vf-dashed");
      });
      names.forEach((name, c) => {
        const mine = shown.filter((st) => st.alive.includes(c));
        if (mine.length > 1) plot.line(plot.layers.mid, [[1, yOf(c)], [mine.length, yOf(c)]], "vf-line-thin");
        mine.forEach((st) => {
          const ranks = averageRanks(st.alive.map((a) => -scores[a][st.block]));
          const rank = ranks[st.alive.indexOf(c)];
          const px = plot.x(st.block + 1);
          const py = plot.y(yOf(c));
          const mark = marker(plot.layers.marks, "circle", px, py, 6.5, "vf-pt " + rankClass(Math.ceil(rank) - 1));
          attachTip(mark, tip, `${name}, block ${st.block + 1} (${blockName(st.block)}): score ${fmt(scores[c][st.block], 3)}, rank ${rank} of ${st.alive.length}`, px, py);
        });
        const out = shown.find((st) => st.eliminated.includes(c));
        if (out) marker(plot.layers.top, "cross", plot.x(out.block + 1.5), plot.y(yOf(c)), 4.5, "vf-line-2");
      });
      clear(readout);
      const k = shown.length;
      const experiments = shown.reduce((acc, st) => acc + st.alive.length, 0);
      const waits = Math.max(state.firstTest, 3);
      if (!k) {
        h("p", {}, readout, `No block played yet: ${N} configurations are alive. Each block is one instance and one seed, and every alive configuration runs on it. The first test waits for max(first_test, 3) = ${waits} blocks.`);
      } else {
        const last = shown[k - 1];
        const nowAlive = race.steps[k] ? race.steps[k].alive.length : race.alive.filter(Boolean).length;
        h("p", {}, readout, `Block ${k} of ${B} (${blockName(last.block)}): ${last.alive.length} configurations ran on it. ${experiments} runs so far, against ${N * k} without racing.`);
        let line;
        const t = last.test;
        if (!t) line = `No test after this block: the first test waits for max(first_test, 3) = ${waits} blocks.`;
        else if (t.test === "wilcoxon") {
          line = `Two configurations are alive, so a paired Wilcoxon signed-rank test decides: p = ${fmt(t.p, 4)} ${t.p < state.alpha ? "<" : "≥"} α = ${state.alpha}` + (last.eliminated.length ? `; ${list(last.eliminated)} is eliminated.` : ": no elimination.");
        } else if (t.statistic === null) line = "Every block ranks the configurations alike: the Friedman statistic is undefined and nobody is eliminated.";
        else {
          line = `Friedman test on the ranks of ${last.alive.length} configurations over ${last.block + 1} blocks: T = ${fmt(t.statistic, 2)}, p = ${fmt(t.p, 4)} ${t.p < state.alpha ? "<" : "≥"} α = ${state.alpha}.`;
          if (t.critical !== null) {
            line += ` Conover's critical difference on the rank sums is ${fmt(t.critical, 2)}; the best rank sum is ${fmt(t.rankSums[t.best], 1)} (${names[last.alive[t.best]]}), so ${last.eliminated.length ? list(last.eliminated) + (last.eliminated.length > 1 ? " are" : " is") + " eliminated" : "nobody is eliminated"}.`;
          } else line += " No configuration is eliminated.";
        }
        h("p", {}, readout, line);
        if (ended && race.survivors.length <= 1) h("p", {}, readout, `Only ${names[race.best]} is left after block ${k}, so the race stops and returns it: ${race.experiments} runs instead of ${N * B}.`);
        else if (ended) h("p", {}, readout, `The blocks are exhausted after ${race.experiments} runs instead of ${N * B}. Of the ${race.survivors.length} survivors, ${names[race.best]} has the best mean score over the blocks they all played, and the race returns it.`);
        else h("p", {}, readout, `${nowAlive} configurations go on to the next block.`);
      }
      const lastTest = (c) => {
        for (let i = k - 1; i >= 0; i--) {
          const st = shown[i];
          if (st.test && st.test.rankSums && st.alive.includes(c)) return st.test.rankSums[st.alive.indexOf(c)];
        }
        return null;
      };
      table(
        readout,
        ["Configuration", "Blocks", "Rank sum at its last test", "Mean score", "Status"],
        names.map((name, c) => {
          const played = shown.filter((st) => st.alive.includes(c)).length;
          const mean = played ? scores[c].slice(0, played).reduce((a, v) => a + v, 0) / played : NaN;
          const out = shown.find((st) => st.eliminated.includes(c));
          const done = ended;
          const status = out ? `eliminated after block ${out.block + 1}` : done && race.best === c ? "returned" : "alive";
          const rs = lastTest(c);
          return [name, String(played), rs === null ? "–" : fmt(rs, 1), fmt(mean, 3), status];
        }),
        [false, true, true, true, false]
      );
    }
    generate();
    render();
  };

  /* Model-based backends: the budgets they evaluate and the instance/seed pairs each budget runs. */
  widgets["fidelity-budgets"] = function (node) {
    const INSTANCES = 6;
    const SEEDS = 5;
    const PRESETS = {
      a: [1000, 3000, 9000],
      b: [500, 5000],
      c: [1000, 2000, 4000, 8000],
    };
    const state = { levels: "a", mode: "hyperband", eta: 3, minFrac: 0.5, minSeeds: 1, configs: 27 };
    const ctl = controls(node);
    select(ctl, {
      label: "Backend",
      options: [
        ["levels", "optuna, bohb_optuna: the levels"],
        ["hyperband", "smac3, bohb: Hyperband between the ends"],
      ],
      value: state.mode,
      onChange: (v) => ((state.mode = v), render()),
    });
    select(ctl, {
      label: "--fidelity-levels",
      options: Object.entries(PRESETS).map(([key, lv]) => [key, lv.join(",")]),
      value: state.levels,
      onChange: (v) => ((state.levels = v), render()),
    });
    slider(ctl, { label: "η", min: 2, max: 4, step: 1, value: state.eta, format: (v) => String(v), onInput: (v) => ((state.eta = v), render()) });
    const ctl2 = controls(node);
    const fracSlider = slider(ctl2, { label: "--fidelity-min-instance-frac", min: 0.2, max: 1, step: 0.05, value: state.minFrac, format: (v) => v.toFixed(2), onInput: (v) => ((state.minFrac = v), render()) });
    const seedSlider = slider(ctl2, { label: "--fidelity-min-seed-count", min: 0, max: SEEDS, step: 1, value: state.minSeeds, format: (v) => (v ? String(v) : "0 (all)"), onInput: (v) => ((state.minSeeds = v), render()) });
    const configSlider = slider(ctl2, { label: "BOHB configurations", min: 1, max: 60, step: 1, value: state.configs, format: (v) => String(v), onInput: (v) => ((state.configs = v), render()) });
    button(ctl2, "VAMOS defaults", () => {
      state.minFrac = 1;
      state.minSeeds = 0;
      fracSlider.set(1);
      seedSlider.set(0);
      render();
    });
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, {
      width: 380,
      height: 250,
      x: [0, 1],
      y: [0, INSTANCES * SEEDS],
      yTicks: [0, 5, 10, 15, 20, 25, 30],
      xTicks: [],
      xLabel: "Budget per run (evaluations)",
      yLabel: "Instance/seed pairs",
      label: "Instance/seed pairs that a model-based backend runs at each budget",
    });
    const readout = h("div", { class: "vf-readout" }, body);
    function budgets(levels) {
      if (state.mode === "levels") return levels.slice();
      return hyperbandBudgets(levels[0], levels[levels.length - 1], state.eta);
    }
    function render() {
      const levels = PRESETS[state.levels];
      const opts = { instances: INSTANCES, seeds: SEEDS, minInstanceFrac: state.minFrac, minSeedCount: state.minSeeds };
      const rows = budgets(levels).map((b) => fidelitySlice(b, levels, opts));
      for (const layer of ["back", "mid", "marks", "top"]) clear(plot.layers[layer]);
      plot.setDomain([0, rows.length], [0, INSTANCES * SEEDS]);
      rows.forEach((r, i) => {
        const pairs = r.instances * r.seeds;
        plot.rect(plot.layers.marks, i + 0.2, 0, i + 0.8, pairs, "vf-bar-1");
        txt(plot.layers.top, plot.x(i + 0.5), plot.y(pairs) - 5, `${r.instances}×${r.seeds}`, { "text-anchor": "middle", class: "vf-mark-label" });
        txt(plot.layers.top, plot.x(i + 0.5), plot.y(0) + 15, r.budget.toLocaleString("en-US"), { "text-anchor": "middle" });
      });
      clear(readout);
      const full = rows[rows.length - 1];
      const low = rows[0];
      const unused = levels.filter((v) => !rows.some((r) => r.budget === v));
      h(
        "p",
        {},
        readout,
        state.mode === "levels"
          ? `A trial runs at ${rows.map((r) => r.budget.toLocaleString("en-US")).join(", then ")} evaluations, each from scratch, until the pruner stops it; the median pruner (optuna) or the Hyperband pruner with reduction factor ${state.eta} (bohb_optuna) can stop it after any level but the last.`
          : `Hyperband between ${levels[0].toLocaleString("en-US")} and ${full.budget.toLocaleString("en-US")} evaluations: the largest level divided by powers of η = ${state.eta} gives the budgets ${rows.map((r) => r.budget.toLocaleString("en-US")).join(", ")}. ${unused.length ? `The levels ${unused.map((v) => v.toLocaleString("en-US")).join(" and ")} are not evaluated.` : "Here they coincide with the levels."} (SMAC3 uses η = 3.)`
      );
      h("p", {}, readout, `At the full budget a configuration runs on ${full.instances} instances × ${full.seeds} seeds = ${full.instances * full.seeds} pairs; at ${low.budget.toLocaleString("en-US")} evaluations, on ${low.instances} × ${low.seeds} = ${low.instances * low.seeds}. With VAMOS's defaults (instance fraction 1.0, every seed) every budget runs all ${INSTANCES * SEEDS} pairs.`);
      table(
        readout,
        ["Budget", "Fraction", "Instances", "Seeds", "Evaluations per configuration"],
        rows.map((r) => [r.budget.toLocaleString("en-US"), fmt(r.fraction, 3), String(r.instances), String(r.seeds), (r.budget * r.instances * r.seeds).toLocaleString("en-US")]),
        [true, true, true, true, true]
      );
      if (state.mode === "hyperband") {
        const brackets = hyperbandBrackets(rows.length, state.eta, state.configs);
        h("p", {}, readout, `BOHB samples ${state.configs} configurations (--tune-budget) in ${brackets.length} brackets; each bracket starts its configurations at its own budget and keeps the best 1/η of them at each larger budget:`);
        table(
          readout,
          ["Bracket", ...rows.map((r) => r.budget.toLocaleString("en-US"))],
          brackets.map((rungs, i) => [String(i + 1), ...rows.map((_, j) => (j >= rows.length - rungs.length ? String(rungs[j - (rows.length - rungs.length)]) : "–"))]),
          [false, ...rows.map(() => true)]
        );
      }
      configSlider.input.disabled = state.mode !== "hyperband";
    }
    render();
  };

  /* ---- Advanced extras: reference-direction layers, parallel coordinates, constraint-handling techniques ---- */

  /* Reference directions: one Das and Dennis lattice or Deb and Jain's two layers on the simplex of three objectives. */
  widgets["lattice-layers"] = function (node) {
    const state = { p1: 4, p2: 2 };
    const ctl = controls(node);
    slider(ctl, { label: "Boundary divisions p₁", min: 1, max: 16, step: 1, value: state.p1, onInput: (v) => ((state.p1 = v), render()) });
    slider(ctl, { label: "Inside divisions p₂ (0: none)", min: 0, max: 8, step: 1, value: state.p2, onInput: (v) => ((state.p2 = v), render()) });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, {
      x: [-0.08, 1.08],
      y: [-0.1, 0.95],
      axes: false,
      margin: { l: 10, r: 10, t: 10, b: 10 },
      width: 340,
      height: 300,
      label: "Reference directions of one or two layers on the unit simplex of three objectives",
    });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Boundary layer (p₁)", shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Inside layer (p₂), shrunk by half", shape: "triangle", cls: "vf-pt vf-pt-2" },
      { label: "Inside direction that repeats a boundary one", shape: "circle", cls: "vf-ring" },
    ]);
    const V = [
      [0, 0],
      [1, 0],
      [0.5, 0.866],
    ];
    const xy = (w) => [w[0] * V[0][0] + w[1] * V[1][0] + w[2] * V[2][0], w[0] * V[0][1] + w[1] * V[1][1] + w[2] * V[2][1]];
    const count = (m, p) => (p > 0 ? binomial(p + m - 1, m - 1) : 0);
    const interior = (m, p) => (p >= m ? binomial(p - 1, m - 1) : 0);
    const num = (v) => v.toLocaleString("en-US");
    function render() {
      const { p1, p2 } = state;
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      plot.line(plot.layers.mid, [V[0], V[1], V[2], V[0]], "vf-line-ink");
      ["w₁ = 1", "w₂ = 1", "w₃ = 1"].forEach((label, i) => txt(plot.layers.top, plot.x(V[i][0]) + (i === 1 ? -36 : i === 0 ? 0 : 8), plot.y(V[i][1]) + (i === 2 ? 4 : 16), label, { class: "vf-mark-label" }));
      const layers = p2 > 0 ? twoLayerLattice(3, p1, p2) : { boundary: dasDennis(3, p1), inside: [] };
      if (p2 > 0) {
        const corners = [0, 1, 2].map((i) => xy([0, 1, 2].map((j) => (i === j ? 0.5 : 0) + 1 / 6)));
        plot.line(plot.layers.mid, corners.concat([corners[0]]), "vf-line-thin vf-dashed");
      }
      const key = (w) => w.map((v) => v.toFixed(9)).join(",");
      const boundaryKeys = new Set(layers.boundary.map(key));
      const r = p1 > 12 ? 2.6 : 3.6;
      const mark = (w, shape, cls, layerName) => {
        const [x, y] = xy(w);
        const px = plot.x(x);
        const py = plot.y(y);
        marker(plot.layers.marks, shape, px, py, r, cls);
        const hit = s("circle", { cx: px, cy: py, r: 8, class: "vf-hit" }, plot.layers.marks);
        attachTip(hit, tip, `${layerName}: w = (${w.map((v) => fmt(v, 3)).join(", ")})`, px, py);
      };
      layers.boundary.forEach((w) => mark(w, "circle", "vf-pt vf-pt-1", "Boundary layer"));
      let shared = 0;
      layers.inside.forEach((w) => {
        mark(w, "triangle", "vf-pt vf-pt-2", "Inside layer");
        if (boundaryKeys.has(key(w))) {
          shared += 1;
          const [x, y] = xy(w);
          s("circle", { cx: plot.x(x), cy: plot.y(y), r: r + 4, class: "vf-ring" }, plot.layers.top);
        }
      });
      const n1 = count(3, p1);
      const n2 = count(3, p2);
      clear(readout);
      h("p", {}, readout, `Boundary layer: C(p₁ + 2, 2) = C(${p1 + 2}, 2) = ${n1} directions, ${interior(3, p1)} of them with every coordinate above zero; the others lie on the edges of the triangle.`);
      h("p", {}, readout, p2 > 0 ? `Inside layer: C(${p2 + 2}, 2) = ${n2} directions, each w/2 + 1/6, so every coordinate is at least 1/6.` : "No inside layer: a single Das and Dennis lattice.");
      h("p", {}, readout, `H = ${n1 + n2} directions. NSGA-III needs pop_size ≥ ${n1 + n2}; MOEA/D and RVEA need exactly ${n1 + n2}.`);
      if (shared) h("p", { class: "vf-step-text" }, readout, `${shared} inside direction${shared === 1 ? " repeats a boundary direction" : "s repeat boundary directions"}: VAMOS rejects these layers with a ConfigurationError.`);
      const rows = [3, 5, 8, 10, 15].map((m) => [String(m), num(count(m, p1)), p2 > 0 ? num(count(m, p1) + count(m, p2)) : "–", num(interior(m, p1))]);
      table(readout, ["m", `H, p₁ = ${p1}`, p2 > 0 ? `H, p₁ = ${p1} and p₂ = ${p2}` : "H, two layers", "Interior directions, p₁ only"], rows, [false, true, true, true]);
    }
    render();
  };

  /** A DTLZ2-like front: m objectives on the unit sphere from uniform position variables. */
  function dtlz2Front(n, m, seed) {
    const rand = mulberry32(seed);
    return Array.from({ length: n }, () => {
      const x = Array.from({ length: m - 1 }, rand);
      return Array.from({ length: m }, (_, i) => {
        let v = i > 0 ? Math.sin((x[m - 1 - i] * Math.PI) / 2) : 1;
        for (let j = 0; j < m - 1 - i; j++) v *= Math.cos((x[j] * Math.PI) / 2);
        return v;
      });
    });
  }

  /* Visualizing many objectives: parallel coordinates with axis reordering and brushing, linked to a scatter-plot matrix. */
  widgets["parallel-coordinates"] = function (node) {
    const n = 60;
    const m = 5;
    const base = dtlz2Front(n, m, 7);
    const names = Array.from({ length: m }, (_, i) => "f" + (i + 1));
    const state = { order: names.map((_, i) => i), brushes: {}, scaled: false, shared: false, refocus: -1, drag: null };
    const ctl = controls(node);
    select(ctl, {
      label: "Objectives",
      options: [
        ["equal", "DTLZ2: every fᵢ in [0, 1]"],
        ["scaled", "Scaled: fᵢ × 2i, as in WFG"],
      ],
      value: "equal",
      onChange: (v) => ((state.scaled = v === "scaled"), draw()),
    });
    select(ctl, {
      label: "Axes",
      options: [
        ["own", "Each axis on its own range"],
        ["shared", "One scale for every axis"],
      ],
      value: "own",
      onChange: (v) => ((state.shared = v === "shared"), draw()),
    });
    button(ctl, "Clear brushes", () => ((state.brushes = {}), draw()));
    button(ctl, "Reset order", () => ((state.order = names.map((_, i) => i)), draw()));
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const pc = createPlot(body, {
      x: [-0.4, m - 0.6],
      y: [-0.2, 1.12],
      axes: false,
      margin: { l: 8, r: 8, t: 8, b: 8 },
      width: 420,
      height: 300,
      title: "Parallel coordinates",
      label: "Parallel coordinates of sixty solutions of a five-objective front. Focus an axis: Left and Right move it, Up and Down move its brush, Shift with Up and Down resizes it, Escape clears it.",
    });
    const sm = createPlot(body, { x: [0, m], y: [0, m], axes: false, margin: { l: 6, r: 6, t: 6, b: 6 }, width: 300, height: 300, title: "Scatter-plot matrix", label: "Scatter-plot matrix of the same solutions, every pair of objectives, each on its own range" });
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Selected by every brush", shape: "line", cls: "vf-pc-line" },
      { label: "Filtered out", shape: "line", cls: "vf-pc-line vf-pc-off" },
      { label: "Brush", shape: "region", cls: "vf-pc-brush" },
      { label: "Pair of neighbouring axes", shape: "region", cls: "vf-pc-adjacent" },
    ]);
    let F = [];
    let lo = [];
    let span = [];
    let gLo = 0;
    let gSpan = 1;
    const lines = [];
    const dots = [];
    // u: a value as a fraction of its objective's own range; a: its height on the axis.
    const uOf = (row, i) => (span[i] > 0 ? (F[row][i] - lo[i]) / span[i] : 0.5);
    const aOfU = (u, i) => (state.shared ? (lo[i] + u * span[i] - gLo) / gSpan : u);
    const uOfA = (a, i) => (state.shared ? (gLo + a * gSpan - lo[i]) / span[i] : a);
    const clamp01 = (v) => Math.min(1, Math.max(0, v));
    const selected = (row) => Object.keys(state.brushes).every((i) => {
      const u = uOf(row, +i);
      return u >= state.brushes[i][0] - 1e-12 && u <= state.brushes[i][1] + 1e-12;
    });
    function crossings(i, j) {
      let c = 0;
      for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if ((F[a][i] - F[b][i]) * (F[a][j] - F[b][j]) < 0) c += 1;
      return c;
    }
    function draw() {
      F = base.map((f) => f.map((v, i) => (state.scaled ? v * 2 * (i + 1) : v)));
      lo = names.map((_, i) => Math.min(...F.map((f) => f[i])));
      span = names.map((_, i) => Math.max(...F.map((f) => f[i])) - lo[i]);
      gLo = Math.min(...lo);
      gSpan = Math.max(...names.map((_, i) => lo[i] + span[i])) - gLo;
      drawParallel();
      drawMatrix();
      update();
    }
    function drawParallel() {
      [pc.layers.back, pc.layers.mid, pc.layers.marks, pc.layers.top].forEach(clear);
      lines.length = 0;
      for (let row = 0; row < n; row++) {
        const pts = state.order.map((i, k) => [k, aOfU(uOf(row, i), i)]);
        lines.push(pc.line(pc.layers.mid, pts, "vf-pc-line"));
      }
      state.order.forEach((i, k) => {
        const x = pc.x(k);
        const g = s("g", { class: "vf-pc-axis", tabindex: "0", role: "group" }, pc.layers.top);
        g.setAttribute("aria-label", `Axis ${names[i]}, position ${k + 1} of ${m}. Left and Right arrows move it; Up and Down move its brush; Shift with Up and Down resizes the brush; Escape clears it.`);
        s("line", { x1: x, x2: x, y1: pc.y(0), y2: pc.y(1), class: "vf-axis-line" }, g);
        const brush = state.brushes[i];
        if (brush) {
          const y0 = pc.y(aOfU(brush[0], i));
          const y1 = pc.y(aOfU(brush[1], i));
          s("rect", { x: x - 6, y: y1, width: 12, height: Math.max(1.5, y0 - y1), class: "vf-pc-brush" }, g);
        }
        // Each axis is labelled with the values at its ends: its own range, or the shared one.
        const bottom = state.shared ? gLo : lo[i];
        const top = state.shared ? gLo + gSpan : lo[i] + span[i];
        txt(g, x, pc.y(1) - 5, tickFormat(+top.toFixed(2)), { "text-anchor": "middle" });
        txt(g, x, pc.y(0) + 13, tickFormat(+bottom.toFixed(2)), { "text-anchor": "middle" });
        s("rect", { x: x - 16, y: pc.y(0) + 19, width: 32, height: 18, rx: 4, class: "vf-pc-ring" }, g);
        txt(g, x, pc.y(0) + 32, names[i], { "text-anchor": "middle", class: "vf-axis-label" });
        const grip = s("rect", { x: x - 18, y: pc.y(0) + 17, width: 36, height: 22, class: "vf-hit vf-pc-grip" }, g);
        const bodyHit = s("rect", { x: x - 12, y: pc.y(1), width: 24, height: pc.y(0) - pc.y(1), class: "vf-hit vf-pc-body" }, g);
        grip.addEventListener("pointerdown", (e) => start(e, { type: "move", obj: i }, g));
        bodyHit.addEventListener("pointerdown", (e) => {
          const p = pc.pointer(e);
          if (!p) return;
          const u = clamp01(uOfA(p.y, i));
          start(e, { type: "brush", obj: i, from: u }, g);
          state.brushes[i] = [u, u];
        });
        g.addEventListener("keydown", (e) => keys(e, i));
        if (state.refocus === i) {
          state.refocus = -1;
          g.focus({ preventScroll: true });
        }
      });
    }
    function start(e, drag, g) {
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      state.drag = drag;
      if (pc.svg.setPointerCapture) pc.svg.setPointerCapture(e.pointerId);
      if (g.focus) g.focus({ preventScroll: true });
    }
    pc.svg.addEventListener("pointermove", (e) => {
      const drag = state.drag;
      if (!drag) return;
      const p = pc.pointer(e);
      if (!p) return;
      if (drag.type === "move") {
        const target = Math.min(m - 1, Math.max(0, Math.round(p.x)));
        const from = state.order.indexOf(drag.obj);
        if (target !== from) {
          state.order.splice(from, 1);
          state.order.splice(target, 0, drag.obj);
          state.refocus = drag.obj;
          drawParallel();
          update();
        }
      } else {
        const u = clamp01(uOfA(p.y, drag.obj));
        state.brushes[drag.obj] = [Math.min(drag.from, u), Math.max(drag.from, u)];
        state.refocus = drag.obj;
        drawParallel();
        update();
      }
    });
    const stop = () => {
      const drag = state.drag;
      state.drag = null;
      if (drag && drag.type === "brush") {
        const b = state.brushes[drag.obj];
        if (b && b[1] - b[0] < 0.01) {
          delete state.brushes[drag.obj]; // a click without a drag clears the brush
          state.refocus = drag.obj;
          drawParallel();
          update();
        }
      }
    };
    pc.svg.addEventListener("pointerup", stop);
    pc.svg.addEventListener("pointercancel", stop);
    function keys(e, i) {
      const k = state.order.indexOf(i);
      let b = state.brushes[i];
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const target = k + (e.key === "ArrowLeft" ? -1 : 1);
        if (target < 0 || target >= m) return;
        state.order.splice(k, 1);
        state.order.splice(target, 0, i);
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const step = e.key === "ArrowUp" ? 0.05 : -0.05;
        if (!b) b = [0, 0.25];
        else if (e.shiftKey) b = [b[0], Math.min(1, Math.max(b[0] + 0.05, b[1] + step))];
        else {
          const shift = Math.min(1 - b[1], Math.max(-b[0], step));
          b = [b[0] + shift, b[1] + shift];
        }
        state.brushes[i] = [+b[0].toFixed(4), +b[1].toFixed(4)];
      } else if (e.key === "Escape" || e.key === "Delete" || e.key === "Backspace") {
        if (!b) return;
        delete state.brushes[i];
      } else return;
      e.preventDefault();
      state.refocus = i;
      drawParallel();
      update();
    }
    function drawMatrix() {
      [sm.layers.back, sm.layers.mid, sm.layers.marks, sm.layers.top].forEach(clear);
      dots.length = 0;
      for (let row = 0; row < n; row++) dots.push([]);
      const pad = 0.08;
      for (let r = 0; r < m; r++) {
        for (let c = 0; c < m; c++) {
          const x0 = c;
          const y0 = m - 1 - r;
          sm.rect(sm.layers.back, x0 + 0.02, y0 + 0.02, x0 + 0.98, y0 + 0.98, "vf-pc-cell");
          if (r === c) {
            txt(sm.layers.top, sm.x(x0 + 0.5), sm.y(y0 + 0.5) + 4, names[r], { "text-anchor": "middle", class: "vf-axis-label" });
            continue;
          }
          for (let row = 0; row < n; row++) {
            const px = sm.x(x0 + pad + (1 - 2 * pad) * uOf(row, c));
            const py = sm.y(y0 + pad + (1 - 2 * pad) * uOf(row, r));
            dots[row].push(s("circle", { cx: px, cy: py, r: 1.7, class: "vf-pc-dot" }, sm.layers.marks));
          }
        }
      }
    }
    function update() {
      const keep = Array.from({ length: n }, (_, row) => selected(row));
      lines.forEach((line, row) => line.setAttribute("class", keep[row] ? "vf-pc-line" : "vf-pc-line vf-pc-off"));
      dots.forEach((list, row) => list.forEach((dot) => dot.setAttribute("class", keep[row] ? "vf-pc-dot" : "vf-pc-dot vf-pc-off")));
      clear(sm.layers.top);
      const pairs = [];
      for (let k = 0; k < m - 1; k++) pairs.push([state.order[k], state.order[k + 1]]);
      for (const [a, b] of pairs) {
        for (const [r, c] of [
          [a, b],
          [b, a],
        ]) {
          sm.rect(sm.layers.top, c + 0.02, m - 1 - r + 0.02, c + 0.98, m - 1 - r + 0.98, "vf-pc-adjacent");
        }
      }
      names.forEach((name, r) => txt(sm.layers.top, sm.x(r + 0.5), sm.y(m - 1 - r + 0.5) + 4, name, { "text-anchor": "middle", class: "vf-axis-label" }));
      clear(readout);
      const counts = pairs.map(([a, b]) => crossings(a, b));
      const total = counts.reduce((x, y) => x + y, 0);
      h("p", {}, readout, `Axis order: ${state.order.map((i) => names[i]).join(", ")}. Line crossings between neighbouring axes, out of ${(n * (n - 1)) / 2} pairs of solutions: ${pairs.map(([a, b], k) => `${names[a]}–${names[b]} ${counts[k]}`).join(", ")}; ${total} in all.`);
      h("p", {}, readout, `Only ${m - 1} of the ${(m * (m - 1)) / 2} pairs of objectives are neighbours in parallel coordinates; the scatter-plot matrix outlines them and shows the other ${((m - 1) * (m - 2)) / 2} as well.`);
      const brushed = Object.keys(state.brushes).map(Number);
      const passing = keep.filter(Boolean).length;
      if (!brushed.length) h("p", {}, readout, `No brush: all ${n} solutions are selected. Drag along an axis, or focus it and press ↑ or ↓, to select an interval.`);
      else {
        const parts = brushed
          .sort((a, b) => a - b)
          .map((i) => `${names[i]} in [${fmt(lo[i] + state.brushes[i][0] * span[i], 2)}, ${fmt(lo[i] + state.brushes[i][1] * span[i], 2)}]`);
        h("p", {}, readout, `Brushes: ${parts.join(" and ")}. ${passing} of ${n} solutions pass every brush and stay highlighted in both views.`);
      }
    }
    draw();
  };

  /* Constraint-handling techniques: ten solutions with two objectives and two constraints, ranked by each technique. */
  const CH_SOLUTIONS = [
    ["A", [0.1, 0.95], [-0.2, -0.1]],
    ["B", [0.3, 0.7], [-0.1, -0.3]],
    ["C", [0.55, 0.5], [-0.3, -0.2]],
    ["D", [0.85, 0.3], [-0.2, -0.4]],
    ["E", [0.95, 0.75], [-0.1, -0.1]],
    ["F", [0.2, 0.4], [0.05, -0.2]],
    ["G", [0.45, 0.2], [0.12, 0.08]],
    ["H", [0.7, 0.1], [-0.1, 0.35]],
    ["I", [0.15, 0.15], [0.6, 0.5]],
    ["J", [0.6, 0.85], [0.3, -0.1]],
  ];

  widgets["constraint-techniques"] = function (node) {
    const labels = CH_SOLUTIONS.map((c) => c[0]);
    const F = CH_SOLUTIONS.map((c) => c[1]);
    const G = CH_SOLUTIONS.map((c) => c[2]);
    const cv = overallViolation(G);
    const count = violatedCount(G);
    const n = F.length;
    const ALL = "NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA, RVEA and SMPSO";
    const FRONT = "NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA";
    const techniques = [
      ["cdp", "cdp: constrained domination (default)", ALL],
      ["violated_constraints", "violated_constraints", ALL],
      ["epsilon_constrained", "epsilon_constrained", ALL],
      ["push_pull", "push_pull", ALL],
      ["stochastic_ranking", "stochastic_ranking", FRONT],
      ["self_adaptive_penalty", "self_adaptive_penalty", FRONT],
    ];
    const state = { name: "cdp", epsilon: 0.25, pushing: true, probability: 0.45, seed: 1 };
    const ctl = controls(node);
    select(ctl, { label: "Technique", options: techniques.map((t) => [t[0], t[1]]), value: state.name, onChange: (v) => ((state.name = v), params(), render()) });
    const extra = h("div", { class: "vf-controls" }, node);
    const body = h("div", { class: "vf-body" }, node);
    const plot = createPlot(body, { x: [0, 1.5], y: [0, 1.5], xLabel: "f₁", yLabel: "f₂", label: "Ten solutions in objective space, coloured by the front that the chosen technique gives them" });
    const tip = tooltip(plot);
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Feasible", shape: "circle", cls: "vf-pt vf-pt-ink" },
      { label: "Infeasible", shape: "square", cls: "vf-pt vf-pt-ink" },
      { label: "Front 1", shape: "circle", cls: "vf-pt vf-r0" },
      { label: "Front 2", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Front 3", shape: "circle", cls: "vf-pt vf-r2" },
      { label: "Front 4", shape: "circle", cls: "vf-pt vf-r3" },
      { label: "Front 5 or later", shape: "circle", cls: "vf-pt vf-r4" },
      { label: "Objectives the technique ranks", shape: "circle", cls: "vf-pt-hollow" },
    ]);
    function params() {
      clear(extra);
      const nm = state.name;
      if (nm === "push_pull") {
        select(extra, {
          label: "Phase",
          options: [
            ["push", "Push: constraints ignored"],
            ["pull", "Pull: ε-constrained"],
          ],
          value: state.pushing ? "push" : "pull",
          // params() rebuilds this select, so the new one takes the focus back.
          onChange: (v) => ((state.pushing = v === "push"), params(), render(), extra.querySelector("select").focus()),
        });
      }
      if (nm === "epsilon_constrained" || (nm === "push_pull" && !state.pushing)) {
        slider(extra, { label: "Current ε", min: 0, max: 1.2, step: 0.01, value: state.epsilon, format: (v) => v.toFixed(2), onInput: (v) => ((state.epsilon = v), render()) });
      }
      if (nm === "stochastic_ranking") {
        slider(extra, { label: "probability p_f", min: 0, max: 1, step: 0.05, value: state.probability, format: (v) => v.toFixed(2), onInput: (v) => ((state.probability = v), render()) });
        button(extra, "Draw again", () => ((state.seed += 1), render()));
      }
    }
    function options() {
      const rand = mulberry32(1000 + state.seed);
      return {
        epsilon: state.epsilon,
        pushing: state.pushing,
        probability: state.probability,
        uniforms: Array.from({ length: n * (n - 1) }, rand),
      };
    }
    const list = (rows) => rows.map((i) => labels[i]).join(", ") || "none";
    function describe(res, o) {
      const nm = state.name;
      const feasibleCount = cv.filter((v) => v <= 0).length;
      if (nm === "cdp") return "Feasible solutions first, ranked by Pareto dominance; then the infeasible ones by overall violation CV, one front per value.";
      if (nm === "violated_constraints") return `The same rule over the number of violated constraints: ${list([5, 7, 9])} violate one and compare by Pareto dominance, then ${list([6, 8])} violate two.`;
      if (nm === "push_pull" && o.pushing) return "Push phase: Pareto dominance on the objectives alone, as if there were no constraints.";
      if (nm === "epsilon_constrained" || nm === "push_pull") {
        const within = cv.map((v, i) => (v > 0 && v <= o.epsilon ? i : -1)).filter((i) => i >= 0);
        return `CV ≤ ε = ${fmt(o.epsilon, 2)} counts as feasible: ${list(within)} join the feasible solutions. The others follow by CV, and equal CV shares a front.`;
      }
      if (nm === "stochastic_ranking") return `A bubble sort compares neighbours by Pareto rank when both are feasible or with probability ${fmt(o.probability, 2)}, by CV otherwise. Sorted order: ${list(res.order)}. Each infeasible solution is a front of its own.`;
      return `${feasibleCount} of ${n} solutions are feasible (r_f = ${fmt(feasibleCount / n, 2)}). Pareto dominance ranks d(x) + (1 − r_f) X(x) + r_f Y(x) per objective, from normalized objectives and violations.`;
    }
    function render() {
      const o = options();
      const res = constraintRanks(state.name, F, G, o);
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      const moved = res.objectives || null;
      F.forEach((f, i) => {
        const px = plot.x(f[0]);
        const py = plot.y(f[1]);
        if (moved && (moved[i][0] !== f[0] || moved[i][1] !== f[1])) {
          plot.line(plot.layers.mid, [f, moved[i]], "vf-line-thin vf-dashed");
          marker(plot.layers.mid, cv[i] > 0 ? "square" : "circle", plot.x(moved[i][0]), plot.y(moved[i][1]), 4, "vf-pt-hollow");
        }
        marker(plot.layers.marks, cv[i] > 0 ? "square" : "circle", px, py, 6, "vf-pt " + rankClass(res.ranks[i]));
        const hit = s("circle", { cx: px, cy: py, r: 11, class: "vf-hit" }, plot.layers.marks);
        const where = moved ? ` · ranked at ${pair(moved[i], 3)}` : "";
        attachTip(hit, tip, `${labels[i]} · f = ${pair(f)} · CV ${fmt(cv[i], 2)}${where} · front ${res.ranks[i] + 1}`, px, py);
        txt(plot.layers.top, px + 8, py - 7, `${labels[i]} ${res.ranks[i] + 1}`, { class: "vf-mark-label" });
      });
      clear(readout);
      h("p", {}, readout, describe(res, o));
      h("p", {}, readout, `Taken by: ${techniques.find((t) => t[0] === state.name)[2]}.`);
      const head = ["Solution", "f₁, f₂", "CV", "Violated"];
      const numeric = [false, false, true, true];
      if (moved) {
        head.push("Ranked as");
        numeric.push(false);
      }
      head.push("Front");
      numeric.push(true);
      const rows = F.map((_, i) => i).sort((a, b) => res.ranks[a] - res.ranks[b] || a - b);
      table(
        readout,
        head,
        rows.map((i) => {
          const row = [labels[i], pair(F[i]), fmt(cv[i], 2), String(count[i])];
          if (moved) row.push(pair(moved[i]));
          row.push(String(res.ranks[i] + 1));
          return row;
        }),
        numeric
      );
    }
    params();
    render();
  };

  /* ---- Benchmarks ---- */

  function paddedDomain(lo, hi) {
    const pad = 0.06 * (hi - lo || 1);
    return [lo - pad, hi + pad];
  }

  /* Benchmarks: a gallery of two-objective fronts and the properties of their problems. */
  widgets["benchmark-gallery"] = function (node) {
    const keys = Object.keys(BENCHMARK_FRONTS);
    const state = { key: node.dataset.problem || "zdt1" };
    const ctl = controls(node);
    select(ctl, {
      label: "Problem",
      options: keys.map((k) => [k, BENCHMARK_FRONTS[k].label]),
      value: state.key,
      onChange: (v) => {
        state.key = v;
        render();
      },
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "f₁", yLabel: "f₂", label: "The Pareto front of the selected problem with two objectives" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Pareto front (two objectives)", shape: "line", cls: "vf-line-1" },
      { label: "Points at evenly spaced values of the position variable", shape: "circle", cls: "vf-pt vf-pt-1" },
    ]);
    const details = h("div", {}, node);
    function render() {
      const front = BENCHMARK_FRONTS[state.key];
      const pts = benchmarkFront(state.key, 2001);
      const pieces = frontPieces(pts, 0.01);
      const lo = [0, 1].map((j) => Math.min(...pts.map((p) => p[j])));
      const hi = [0, 1].map((j) => Math.max(...pts.map((p) => p[j])));
      plot.setDomain(paddedDomain(Math.min(0, lo[0]), hi[0]), paddedDomain(Math.min(0, lo[1]), hi[1]));
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      for (const piece of pieces) if (piece.length > 1) plot.line(plot.layers.mid, piece, "vf-line-1");
      const t0 = front.lo || 0;
      for (let i = 0; i <= 20; i++) {
        const p = front.point(t0 + ((1 - t0) * i) / 20);
        if (isFrontPoint(pts, p)) {
          marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 3.2, "vf-pt vf-pt-1");
        }
      }
      clear(readout);
      h(
        "p",
        {},
        readout,
        `${front.label} with two objectives: the front has ${pieces.length} piece${pieces.length === 1 ? "" : "s"}, ` +
          `f₁ from ${fmt(lo[0])} to ${fmt(hi[0])} and f₂ from ${fmt(lo[1])} to ${fmt(hi[1])}.`
      );
      clear(details);
      table(
        details,
        ["Property", front.label],
        [
          ["Front shape", front.shape],
          ["What it tests", front.tests],
          ["Separable", front.separable],
          ["Objectives", front.objectives],
          ["Default n", front.n],
          ["Packaged reference fronts", front.files],
        ]
      );
    }
    function isFrontPoint(pts, p) {
      // A point of the curve is on the front unless a sampled front point dominates it.
      return !pts.some((q) => q[0] <= p[0] + 1e-12 && q[1] <= p[1] + 1e-12 && (q[0] < p[0] - 1e-9 || q[1] < p[1] - 1e-9));
    }
    render();
  };

  /* Benchmarks: where uniformly drawn position variables land on a biased front. */
  widgets["position-bias"] = function (node) {
    const count = 200;
    const tenths = 10;
    const state = { key: node.dataset.problem || "dtlz4", alpha: 100, seed: 3 };
    const ctl = controls(node);
    select(ctl, {
      label: "Problem",
      options: Object.keys(BIAS_PROBLEMS).map((k) => [k, BIAS_PROBLEMS[k].label]),
      value: state.key,
      onChange: (v) => {
        state.key = v;
        alphaLabel.style.display = v === "dtlz4" ? "" : "none";
        render();
      },
    });
    const alphaControl = slider(ctl, { label: "DTLZ4's α", min: 1, max: 100, step: 1, value: state.alpha, format: (v) => String(v), onInput: (v) => ((state.alpha = v), render()) });
    const alphaLabel = alphaControl.input.parentNode;
    alphaLabel.style.display = state.key === "dtlz4" ? "" : "none";
    button(ctl, "New sample", () => {
      state.seed += 1;
      render();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "f₁", yLabel: "f₂", label: "Pareto-optimal solutions with uniformly drawn position variables" });
    const bars = createPlot(body, {
      x: [0, tenths],
      y: [0, 1],
      xLabel: "Tenth of the front's length, from its f₁-minimal end",
      yLabel: "Share of the points",
      xTicks: [0, 2, 4, 6, 8, 10],
      yFormat: (v) => Math.round(v * 100) + "%",
      label: "Share of the points in each tenth of the front",
    });
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Pareto front", shape: "line", cls: "vf-front" },
      { label: `${count} solutions with uniform position variables`, shape: "circle", cls: "vf-pt vf-pt-1" },
      { label: "Share per tenth of the front", shape: "region", cls: "vf-bar-1" },
      { label: "Uniform share, 10%", shape: "line", cls: "vf-line-2" },
    ]);
    function render() {
      const problem = BIAS_PROBLEMS[state.key];
      const front = BENCHMARK_FRONTS[problem.front];
      const t0 = front.lo || 0;
      // The front's arc length as a function of its parameter t, on a fine grid.
      const grid = 4000;
      const curve = [];
      const length = [0];
      for (let i = 0; i <= grid; i++) curve.push(front.point(t0 + ((1 - t0) * i) / grid));
      for (let i = 1; i <= grid; i++) length.push(length[i - 1] + Math.hypot(curve[i][0] - curve[i - 1][0], curve[i][1] - curve[i - 1][1]));
      const total = length[grid];
      const flip = curve[0][0] > curve[grid][0]; // measure the length from the end where f₁ is smallest
      const arcAt = (t) => {
        const x = Math.min(grid, Math.max(0, ((t - t0) / (1 - t0)) * grid));
        const i = Math.min(grid - 1, Math.floor(x));
        const share = (length[i] + (x - i) * (length[i + 1] - length[i])) / total;
        return flip ? 1 - share : share;
      };
      const rand = mulberry32(state.seed);
      const counts = new Array(tenths).fill(0);
      const points = [];
      for (let k = 0; k < count; k++) {
        const u = Array.from({ length: problem.positions }, () => rand());
        const t = problem.position(u, state.alpha);
        points.push(front.point(t));
        counts[Math.min(tenths - 1, Math.floor(arcAt(t) * tenths))] += 1;
      }
      const lo = [0, 1].map((j) => Math.min(...curve.map((p) => p[j])));
      const hi = [0, 1].map((j) => Math.max(...curve.map((p) => p[j])));
      plot.setDomain(paddedDomain(Math.min(0, lo[0]), hi[0]), paddedDomain(Math.min(0, lo[1]), hi[1]));
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      plot.line(plot.layers.mid, curve, "vf-front");
      for (const p of points) marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 3, "vf-pt vf-pt-1");
      const shares = counts.map((c) => c / count);
      const top = Math.max(0.2, Math.ceil(Math.max(...shares) * 10) / 10);
      bars.setDomain([0, tenths], [0, top]);
      clear(bars.layers.mid);
      clear(bars.layers.top);
      shares.forEach((v, i) => {
        if (v <= 0) return;
        s("rect", { x: bars.x(i) + 1, y: bars.y(v), width: Math.max(0.5, bars.x(i + 1) - bars.x(i) - 2), height: bars.y(0) - bars.y(v), class: "vf-bar-1" }, bars.layers.mid);
      });
      bars.line(bars.layers.top, [[0, 0.1], [tenths, 0.1]], "vf-line-2");
      const most = Math.max(...counts);
      const least = Math.min(...counts);
      const name = state.key === "dtlz4" ? `DTLZ4 with α = ${state.alpha}` : problem.label;
      clear(readout);
      h(
        "p",
        {},
        readout,
        `${name}: of ${count} Pareto-optimal solutions with uniformly drawn position variables, the fullest tenth of the front holds ` +
          `${most} (${fmt((100 * most) / count, 1)}%) and the emptiest ${least} (${fmt((100 * least) / count, 1)}%). ` +
          "Without bias, each tenth would hold about 10%."
      );
    }
    render();
  };

  /* Benchmarks: the feasible regions and constrained fronts of CTP1-CTP7. */
  widgets["ctp-landscape"] = function (node) {
    const notes = {
      ctp1: "Two constraints cut off the lower part of the unconstrained front f₂ = exp(−f₁): the constrained front runs along their boundaries, away from the unconstrained one.",
      ctp2: "Narrow feasible tunnels reach the unconstrained front. The constrained front is 13 short segments, one at the end of each tunnel.",
      ctp3: "The tunnels narrow to single points: the constrained front is 13 isolated points on the unconstrained front.",
      ctp4: "The same 13 points as CTP3 at the end of longer tunnels (a = 0.75), which are harder to follow.",
      ctp5: "Tunnels spaced unevenly (c = 2): the front is a first arch, up to f₁ ≈ 0.256, and isolated points.",
      ctp6: "Infeasible bands cross the objective space between the region that random designs reach and the front: a population has to pass through them.",
      ctp7: "Only disconnected pieces of the unconstrained front are feasible, separated by infeasible regions that reach down to it.",
    };
    const state = { key: node.dataset.problem || "ctp2" };
    const ctl = controls(node);
    select(ctl, {
      label: "Problem",
      options: Object.keys(CTP_PROBLEMS).map((k) => [k, k.toUpperCase()]),
      value: state.key,
      onChange: (v) => {
        state.key = v;
        render();
      },
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1.6], xLabel: "f₁", yLabel: "f₂", label: "Objective space of the selected CTP problem with its infeasible regions" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Infeasible region", shape: "region", cls: "vf-region-2" },
      { label: "Unconstrained front", shape: "line", cls: "vf-front" },
      { label: "Constrained Pareto front", shape: "circle", cls: "vf-pt vf-pt-1" },
    ]);
    function render() {
      const key = state.key;
      const top = CTP_PROBLEMS[key].ymax;
      plot.setDomain([0, 1], [0, top]);
      clear(plot.layers.back);
      clear(plot.layers.marks);
      const cols = 160;
      const rows = 320;
      let cells = 0;
      let feasibleCells = 0;
      for (let c = 0; c < cols; c++) {
        const f1 = (c + 0.5) / cols;
        const base = ctpBase(key, f1);
        let start = null;
        for (let r = 0; r <= rows; r++) {
          const f2 = (r * top) / rows;
          const above = f2 >= base && r < rows;
          const bad = above && ctpTotalViolation(key, [f1, f2]) > 0;
          if (above) {
            cells += 1;
            if (!bad) feasibleCells += 1;
          }
          if (bad && start === null) start = f2;
          if (!bad && start !== null) {
            plot.rect(plot.layers.back, c / cols, start, (c + 1) / cols + 0.0005, f2, "vf-region-2");
            start = null;
          }
        }
      }
      plot.line(plot.layers.back, frontPoints((t) => ctpBase(key, t), 0, 1, 101), "vf-front");
      const front = ctpFront(key, 1001);
      const pieces = frontPieces(front, 0.004);
      for (const piece of pieces) {
        if (piece.length > 1) plot.line(plot.layers.marks, piece, "vf-line-1");
        else marker(plot.layers.marks, "circle", plot.x(piece[0][0]), plot.y(piece[0][1]), 3.5, "vf-pt vf-pt-1");
      }
      clear(readout);
      h(
        "p",
        {},
        readout,
        `${key.toUpperCase()}: ${fmt((100 * feasibleCells) / cells, 1)}% of the plotted region above the unconstrained front is feasible. ` +
          `The constrained front has ${pieces.length} piece${pieces.length === 1 ? "" : "s"} or isolated point${pieces.length === 1 ? "" : "s"}, ` +
          `f₁ from ${fmt(front[0][0])} to ${fmt(front[front.length - 1][0])}.`
      );
      h("p", {}, readout, notes[key]);
    }
    render();
  };

  /* ---- Operators: offspring clouds, binary and permutation step-throughs, integer SBX, tour moves ---- */

  const OP = OPERATORS;

  /* The real-coded operators of the offspring-cloud figure: parameters with VAMOS's defaults. */
  const CLOUD_OPERATORS = {
    sbx: {
      label: "SBX",
      parents: 2,
      params: [{ key: "eta", label: "η_c", min: 0, max: 50, step: 1, value: 20 }],
      run: (P, v, r) => sbxPair(P[0], P[1], 0, 1, v.eta, 0.5, r),
      text: "SBX recombines each variable with probability 0.5; its children stay inside the bounds and spread in proportion to the parents' distance.",
    },
    blx_alpha: {
      label: "BLX-α",
      parents: 2,
      params: [{ key: "alpha", label: "α", min: 0, max: 1, step: 0.05, value: 0.5 }],
      run: (P, v, r) => OP.blxAlphaBeta(P[0], P[1], v.alpha, v.alpha, r),
      text: "BLX-α draws each variable uniformly from the parents' interval, widened by α times its length on both sides: a rectangle around the parents.",
    },
    blx_alpha_beta: {
      label: "BLX-αβ",
      parents: 2,
      params: [
        { key: "alpha", label: "α (below)", min: 0, max: 1, step: 0.05, value: 0.5 },
        { key: "beta", label: "β (above)", min: 0, max: 1, step: 0.05, value: 0.5 },
      ],
      run: (P, v, r) => OP.blxAlphaBeta(P[0], P[1], v.alpha, v.beta, r),
      text: "VAMOS's BLX-αβ widens the interval by α below the smaller parent value and by β above the larger one; α ≠ β shifts the rectangle towards one bound.",
    },
    arithmetic: {
      label: "Arithmetic",
      parents: 2,
      params: [],
      run: (P, v, r) => OP.arithmetic(P[0], P[1], r),
      text: "Arithmetic crossover draws one weight λ per pair: both children lie on the segment between the parents, symmetric about its midpoint.",
    },
    whole_arithmetic: {
      label: "Whole arithmetic",
      parents: 2,
      params: [{ key: "alpha", label: "α", min: 0, max: 1, step: 0.05, value: 0.5 }],
      run: (P, v) => OP.wholeArithmetic(P[0], P[1], v.alpha),
      text: "Whole arithmetic crossover uses the fixed weight α: every pair gives the same two children, and α = 0.5 gives the midpoint twice.",
    },
    laplace: {
      label: "Laplace",
      parents: 2,
      params: [
        { key: "a", label: "location a", min: -0.5, max: 0.5, step: 0.05, value: 0 },
        { key: "b", label: "scale b", min: 0.05, max: 1, step: 0.05, value: 0.5 },
      ],
      run: (P, v, r) => OP.laplace(P[0], P[1], v.a, v.b, r),
      text: "Laplace crossover moves each parent by β |p₁ − p₂|, with β from a Laplace distribution: a cross of heavy-tailed clouds centred on the parents.",
    },
    fuzzy: {
      label: "Fuzzy",
      parents: 2,
      params: [{ key: "d", label: "d", min: 0, max: 1, step: 0.05, value: 0.5 }],
      run: (P, v, r) => OP.fuzzy(P[0], P[1], v.d, 0, 1, r),
      text: "VAMOS's fuzzy crossover draws each variable from a triangular distribution over the widened interval, with its mode at the parents' midpoint.",
    },
    de: {
      label: "Differential evolution",
      parents: 3,
      names: ["x", "a", "b"],
      params: [
        { key: "F", label: "F", min: 0, max: 1, step: 0.05, value: 0.5 },
        { key: "CR", label: "CR", min: 0, max: 1, step: 0.05, value: 0.9 },
      ],
      run: (P, v, r) => OP.deMating(P[0], P[1], P[2], v.F, v.CR, r),
      text: "DE adds the scaled difference F (a − b) to x and takes each variable from that mutant with probability CR (at least one): with fixed parents only three children are possible.",
    },
    pcx: {
      label: "PCX",
      parents: 3,
      params: [
        { key: "se", label: "σ_η", min: 0, max: 0.5, step: 0.01, value: 0.1 },
        { key: "sz", label: "σ_ζ", min: 0, max: 0.5, step: 0.01, value: 0.1 },
      ],
      run: (P, v, r) => OP.pcx(P, v.se, v.sz, r),
      text: "PCX centres one child on each parent: it moves along the direction to the centroid (σ_ζ) and orthogonally to it (σ_η times the other parents' distance).",
    },
    undx: {
      label: "UNDX",
      parents: 3,
      params: [
        { key: "zeta", label: "ζ", min: 0, max: 1, step: 0.05, value: 0.5 },
        { key: "eta", label: "η", min: 0, max: 1, step: 0.05, value: 0.35 },
      ],
      run: (P, v, r) => OP.undx(P, v.zeta, v.eta, r),
      text: "UNDX draws an ellipse around the midpoint of P1 and P2: ζ scales it along P1–P2, and η times P3's distance to that line scales it across.",
    },
    simplex: {
      label: "SPX",
      parents: 3,
      params: [{ key: "eps", label: "ε", min: 0.5, max: 3, step: 0.05, value: 2 }],
      run: (P, v, r) => OP.spx(P, v.eps, r),
      text: "SPX expands the triangle of the three parents about their centroid by ε and draws children uniformly inside it; ε = √(m + 1) = 2 is the default.",
    },
    pm: {
      label: "Polynomial",
      parents: 1,
      params: [{ key: "eta", label: "η_m", min: 0, max: 100, step: 1, value: 20 }],
      run: (P, v, r) => [polynomialMutation(P[0].slice(), 0, 1, v.prob, v.eta, r)],
      text: "Polynomial mutation perturbs each mutated variable with a spread set by η_m, squeezed near the bounds so that the value stays inside them.",
    },
    gaussian: {
      label: "Gaussian",
      parents: 1,
      params: [{ key: "sigma", label: "σ (share of range)", min: 0.01, max: 0.5, step: 0.01, value: 0.1 }],
      run: (P, v, r) => [OP.gaussianMutation(P[0], 0, 1, v.prob, v.sigma, r)],
      text: "Gaussian mutation adds N(0, (σ·range)²) noise to each mutated variable; σ is a share of the variable's range.",
    },
    uniform: {
      label: "Uniform",
      parents: 1,
      params: [{ key: "perturb", label: "perturb", min: 0, max: 1, step: 0.05, value: 0.1 }],
      run: (P, v, r) => [OP.uniformMutation(P[0], 0, 1, v.prob, v.perturb, r)],
      text: "Uniform mutation moves each mutated variable by U(−perturb, perturb) times its range: a box around the parent.",
    },
    uniform_reset: {
      label: "Uniform reset",
      parents: 1,
      params: [],
      run: (P, v, r) => [OP.uniformReset(P[0], 0, 1, v.prob, r)],
      text: "Uniform reset replaces each mutated variable with a uniform value between its bounds, forgetting the parent's value.",
    },
    non_uniform: {
      label: "Non-uniform",
      parents: 1,
      params: [],
      run: (P, v, r) => [OP.nonUniform(P[0], 0, 1, v.prob, r)],
      text: "Non-uniform mutation moves each mutated variable towards a random bound by a uniform share of the distance, because VAMOS uses the progress t/T = 0.",
    },
    cauchy: {
      label: "Cauchy",
      parents: 1,
      params: [{ key: "gamma", label: "γ (share of range)", min: 0.01, max: 0.5, step: 0.01, value: 0.1 }],
      run: (P, v, r) => [OP.cauchyMutation(P[0], 0, 1, v.prob, v.gamma, r)],
      text: "Cauchy mutation adds noise of scale γ·range with heavy tails: most steps are small, a few are very long.",
    },
    levy_flight: {
      label: "Lévy flight",
      parents: 1,
      params: [
        { key: "beta", label: "β", min: 0.3, max: 2, step: 0.05, value: 1.5 },
        { key: "scale", label: "scale", min: 0.005, max: 0.1, step: 0.005, value: 0.01 },
      ],
      run: (P, v, r) => [OP.levyFlight(P[0], 0, 1, v.prob, v.beta, v.scale, r)],
      text: "Lévy flight mutation draws Mantegna's heavy-tailed steps, scaled by scale·range; a smaller β gives heavier tails.",
    },
    power_law: {
      label: "Power law",
      parents: 1,
      params: [{ key: "index", label: "index", min: 0.1, max: 10, step: 0.1, value: 1 }],
      run: (P, v, r) => [OP.powerLaw(P[0], 0, 1, v.prob, v.index, r)],
      text: "Power mutation moves a mutated variable by a share s = u^index of its distance to one bound, so it never leaves the bounds; a larger index gives smaller steps.",
    },
    linked_polynomial: {
      label: "Linked polynomial",
      parents: 1,
      params: [{ key: "eta", label: "η_m", min: 0, max: 100, step: 1, value: 20 }],
      run: (P, v, r) => [OP.linkedPolynomial(P[0], 0, 1, v.prob, v.eta, r)],
      text: "Linked polynomial mutation uses one random number per individual for all its mutated variables, so they move together along a curve.",
    },
  };

  const CLOUD_CROSSOVERS = ["sbx", "blx_alpha", "blx_alpha_beta", "arithmetic", "whole_arithmetic", "laplace", "fuzzy", "de", "pcx", "undx", "simplex"];
  const CLOUD_MUTATIONS = ["pm", "gaussian", "uniform", "uniform_reset", "non_uniform", "cauchy", "levy_flight", "power_law", "linked_polynomial"];

  widgets["operator-cloud"] = function (node) {
    const mutation = node.dataset.group === "mutation";
    const keys = mutation ? CLOUD_MUTATIONS : CLOUD_CROSSOVERS;
    const state = {
      op: node.dataset.operator && keys.includes(node.dataset.operator) ? node.dataset.operator : keys[1],
      parents: [
        [0.35, 0.3],
        [0.6, 0.55],
        [0.3, 0.7],
      ],
      x: [[0.3, 0.6]],
      values: {},
      prob: 0.5,
      seed: 7,
    };
    const ctl = controls(node);
    select(ctl, {
      label: mutation ? "Mutation" : "Crossover",
      value: state.op,
      options: keys.map((k) => [k, CLOUD_OPERATORS[k].label]),
      onChange: (v) => ((state.op = v), buildParams(), render()),
    });
    button(ctl, "New sample", () => ((state.seed += 1), render()));
    const params = controls(node);
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, {
      x: [-0.3, 1.3],
      y: [-0.3, 1.3],
      xLabel: "Variable x₁ (bounds [0, 1])",
      yLabel: "Variable x₂ (bounds [0, 1])",
      label: mutation ? "Mutated copies of one parent in two variables" : "Children of fixed parents in two variables",
      height: 340,
      width: 340,
      margin: { l: 44, b: 40 },
    });
    plot.rect(plot.layers.back, 0, 0, 1, 1, "vf-region-1");
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Bounds [0, 1]²", shape: "region", cls: "vf-region-1" },
      { label: "Child inside the bounds", shape: "circle", cls: "vf-cloud" },
      { label: "Child outside (the repair moves it back)", shape: "circle", cls: "vf-pt-2" },
      { label: mutation ? "Parent (drag it)" : "Parents (drag them)", shape: "circle", cls: "vf-pt-1" },
    ]);

    const shapes = ["circle", "square", "triangle"];
    const handles = [0, 1, 2].map((k) =>
      handle(plot, {
        shape: shapes[k],
        cls: "vf-pt vf-pt-1",
        boundsX: [0, 1],
        boundsY: [0, 1],
        get: () => (mutation ? state.x[0] : state.parents[k]),
        set: (x, y) => {
          if (mutation) state.x[0] = [x, y];
          else state.parents[k] = [x, y];
          render();
        },
        label: () => parentName(k),
      })
    );
    function parentName(k) {
      const spec = CLOUD_OPERATORS[state.op];
      if (mutation) return "Parent";
      return spec.names ? spec.names[k] : "P" + (k + 1);
    }

    function buildParams() {
      clear(params);
      const spec = CLOUD_OPERATORS[state.op];
      state.values = {};
      for (const p of spec.params) {
        state.values[p.key] = p.value;
        slider(params, { label: p.label, min: p.min, max: p.max, step: p.step, value: p.value, format: (v) => String(+v.toFixed(3)), onInput: (v) => ((state.values[p.key] = v), render()) });
      }
      if (mutation) slider(params, { label: "Probability per variable", min: 0.05, max: 1, step: 0.05, value: state.prob, format: (v) => v.toFixed(2), onInput: (v) => ((state.prob = v), render()) });
    }

    function render() {
      const spec = CLOUD_OPERATORS[state.op];
      const parents = mutation ? state.x : state.parents.slice(0, spec.parents);
      const rand = mulberry32(state.seed);
      const values = Object.assign({ prob: state.prob }, state.values);
      const children = [];
      const target = 600;
      while (children.length < target) for (const c of spec.run(parents, values, rand)) children.push(c);
      clear(plot.layers.mid);
      let outside = 0;
      let dist = 0;
      let moved = 0;
      for (const c of children) {
        const out = c[0] < 0 || c[0] > 1 || c[1] < 0 || c[1] > 1;
        if (out) outside += 1;
        const d = Math.min(...parents.map((p) => Math.hypot(c[0] - p[0], c[1] - p[1])));
        dist += d;
        if (d > 1e-12) moved += 1;
        s("circle", { cx: plot.x(c[0]).toFixed(1), cy: plot.y(c[1]).toFixed(1), r: 2, class: out ? "vf-pt-2" : "vf-cloud" }, plot.layers.mid);
      }
      handles.forEach((hd, k) => {
        const used = mutation ? k === 0 : k < spec.parents;
        hd.node.style.display = used ? "" : "none";
        if (used) hd.node.setAttribute("tabindex", "0");
        else hd.node.setAttribute("tabindex", "-1");
        hd.place();
      });
      clear(plot.layers.top);
      parents.forEach((p, k) => txt(plot.layers.top, plot.x(p[0]) + 9, plot.y(p[1]) - 8, parentName(k), { class: "vf-mark-label" }));
      clear(readout);
      h("p", {}, readout, spec.text);
      h(
        "p",
        {},
        readout,
        `${children.length} ${mutation ? "mutated copies" : "children"}: ${outside} (${fmt((100 * outside) / children.length, 1)}%) lie outside the bounds; ${moved} differ from ${mutation ? "the parent" : "every parent"}; mean distance to the nearest parent: ${fmt(dist / children.length, 3)}.`
      );
      const where = parents.map((p, k) => `${parentName(k)} = ${pair(p)}`).join(", ");
      h("p", {}, readout, `${where}. Drag a parent or focus it with Tab and move it with the arrow keys.`);
    }
    buildParams();
    render();
  };

  /* A row of cells for the binary and permutation step-throughs. */
  function cellRow(svg, y, label, values, o) {
    const g = s("g", {}, svg);
    txt(g, 4, y + 17, label, { class: "vf-mark-label" });
    values.forEach((v, j) => {
      const x = o.x0 + j * o.w;
      const cls = o.cls ? o.cls(j) : "vf-pt-hollow";
      const rect = s("rect", { x, y, width: o.w - 3, height: 24, rx: 4, class: cls }, g);
      if (o.focus) o.focus(rect, j);
      txt(g, x + (o.w - 3) / 2, y + 17, v === null || v === undefined || v < 0 ? "·" : String(v), { "text-anchor": "middle", class: o.textCls ? o.textCls(j) : "" });
    });
    return g;
  }

  widgets["binary-steps"] = function (node) {
    const n = 12;
    const state = {
      op: node.dataset.operator || "one_point",
      p1: [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
      p2: [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
      step: 0,
      seed: 3,
      prob: 1 / 12,
      cursor: [0, 0],
    };
    const names = {
      one_point: "One-point crossover",
      two_point: "Two-point crossover",
      uniform: "Uniform crossover",
      hux: "HUX",
      bitflip: "Bit-flip mutation",
      segment_inversion: "Segment inversion",
    };
    const ctl = controls(node);
    select(ctl, { label: "Operator", value: state.op, options: Object.keys(names).map((k) => [k, names[k]]), onChange: (v) => ((state.op = v), (state.step = 0), draw(), probabilityControl(), render()) });
    const prev = button(ctl, "Previous step", () => ((state.step = Math.max(0, state.step - 1)), render()));
    const next = button(ctl, "Next step", () => ((state.step = Math.min(2, state.step + 1)), render()), true);
    button(ctl, "New draw", () => ((state.seed += 1), draw(), render()));
    const probCtl = controls(node);
    const W = 40 + n * 28 + 8;
    const svg = s("svg", { viewBox: `0 0 ${W} 200`, class: "vf-plot", role: "group", "aria-label": "Two parent bit strings and the children an operator builds from them" }, h("div", { class: "vf-plot-wrap" }, node));
    const text = h("p", { class: "vf-step-text", "aria-live": "polite" }, h("div", { class: "vf-readout" }, node));
    legend(node, [
      { label: "Bit from P1", shape: "region", cls: "vf-cell-a" },
      { label: "Bit from P2", shape: "region", cls: "vf-cell-b" },
      { label: "Cut, mask or flipped bit", shape: "region", cls: "vf-cell-mark" },
    ]);
    let info = {};
    const mutationOp = () => state.op === "bitflip" || state.op === "segment_inversion";

    function draw() {
      const rand = mulberry32(state.seed * 7919 + 1);
      if (state.op === "one_point") info = { cut: OP.drawOnePoint(n, rand) };
      else if (state.op === "two_point") {
        const [lo, hi] = OP.drawTwoPoint(n, rand);
        info = { lo, hi };
      } else if (state.op === "uniform") info = { keep: state.p1.map(() => rand() < 0.5) };
      else if (state.op === "hux") info = { chosen: OP.drawHux(state.p1, state.p2, rand) };
      else if (state.op === "bitflip") info = { mask: state.p1.map(() => rand() <= state.prob) };
      else {
        const [a, b] = OP.drawSegment(n, rand);
        info = { a, b };
      }
    }

    // Built when the operator changes, not on each draw, so that moving the slider keeps it and its focus.
    function probabilityControl() {
      clear(probCtl);
      if (state.op === "bitflip") slider(probCtl, { label: "Flip probability per bit", min: 0, max: 0.5, step: 1 / 24, value: state.prob, format: (v) => v.toFixed(3) + (Math.abs(v - 1 / 12) < 1e-9 ? " = 1/n" : ""), onInput: (v) => ((state.prob = v), draw(), render()) });
    }

    function children() {
      const { p1, p2 } = state;
      if (state.op === "one_point") return OP.onePoint(p1, p2, info.cut);
      if (state.op === "two_point") return OP.twoPoint(p1, p2, info.lo, info.hi);
      if (state.op === "uniform") return OP.uniformBits(p1, p2, info.keep);
      if (state.op === "hux") return OP.hux(p1, p2, info.chosen);
      if (state.op === "bitflip") return [OP.bitFlip(p1, info.mask)];
      return [OP.segmentInversion(p1, info.a, info.b)];
    }

    // Source of each child bit: 0 = P1, 1 = P2 (for crossovers), flipped for mutations.
    function fromSecond(child, j) {
      if (state.op === "one_point") return child === 0 ? j >= info.cut : j < info.cut;
      if (state.op === "two_point") return (j >= info.lo && j < info.hi) === (child === 0);
      if (state.op === "uniform") return info.keep[j] === (child === 1);
      if (state.op === "hux") return info.chosen.includes(j) === (child === 0);
      return false;
    }
    function marked(j) {
      if (state.op === "uniform") return !info.keep[j];
      if (state.op === "hux") return info.chosen.includes(j);
      if (state.op === "two_point") return j >= info.lo && j < info.hi;
      if (state.op === "one_point") return j >= info.cut;
      if (state.op === "bitflip") return info.mask[j];
      return j >= info.a && j <= info.b;
    }
    const hamming = (a, b) => a.reduce((s2, v, j) => s2 + (v !== b[j] ? 1 : 0), 0);
    const bits = (a) => a.join("");

    let focusTarget = null;
    function render() {
      clear(svg);
      const x0 = 40;
      const w = 28;
      for (let j = 0; j < n; j++) txt(svg, x0 + j * w + 12.5, 12, String(j), { "text-anchor": "middle" });
      const rows = mutationOp() ? [["x", state.p1, 0]] : [["P1", state.p1, 0], ["P2", state.p2, 1]];
      rows.forEach(([label, row, r], ri) => {
        cellRow(svg, 20 + ri * 30, label, row, {
          x0,
          w,
          cls: () => (r === 0 ? "vf-cell-a" : "vf-cell-b"),
          focus: (rect, j) => {
            rect.setAttribute("tabindex", state.cursor[0] === ri && state.cursor[1] === j ? "0" : "-1");
            rect.setAttribute("class", rect.getAttribute("class") + " vf-focusable");
            rect.setAttribute("role", "button");
            rect.setAttribute("aria-label", `${label} bit ${j} is ${row[j]}. Press Enter to flip it; arrow keys move between bits.`);
            const flip = () => {
              row[j] = 1 - row[j];
              state.cursor = [ri, j];
              focusTarget = [ri, j];
              draw();
              render();
            };
            rect.addEventListener("click", flip);
            rect.addEventListener("keydown", (e) => {
              const moves = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] };
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                flip();
              } else if (moves[e.key]) {
                e.preventDefault();
                const rr = Math.min(rows.length - 1, Math.max(0, ri + moves[e.key][0]));
                const jj = Math.min(n - 1, Math.max(0, j + moves[e.key][1]));
                state.cursor = [rr, jj];
                focusTarget = [rr, jj];
                render();
              }
            });
            if (focusTarget && focusTarget[0] === ri && focusTarget[1] === j) {
              focusTarget = null;
              setTimeout(() => rect.focus({ preventScroll: true }), 0);
            }
          },
        });
      });
      const yMask = 20 + rows.length * 30 + 8;
      if (state.step >= 1) {
        cellRow(svg, yMask, mutationOp() ? "flip" : state.op === "uniform" ? "swap" : "cut", new Array(n).fill(null), {
          x0,
          w,
          cls: (j) => (marked(j) ? "vf-cell-mark" : "vf-cell-empty"),
          textCls: () => "",
        });
        if (state.op === "one_point") s("line", { x1: x0 + info.cut * w - 1.5, x2: x0 + info.cut * w - 1.5, y1: 16, y2: 196, class: "vf-line-2" }, svg);
        if (state.op === "two_point") for (const c of [info.lo, info.hi]) s("line", { x1: x0 + c * w - 1.5, x2: x0 + c * w - 1.5, y1: 16, y2: 196, class: "vf-line-2" }, svg);
      }
      const kids = children();
      if (state.step >= 2) {
        kids.forEach((kid, k) => {
          cellRow(svg, yMask + 38 + k * 30, mutationOp() ? "x′" : "C" + (k + 1), kid, {
            x0,
            w,
            cls: (j) => (mutationOp() ? (marked(j) ? "vf-cell-mark" : "vf-cell-a") : fromSecond(k, j) ? "vf-cell-b" : "vf-cell-a"),
          });
        });
      }
      prev.disabled = state.step === 0;
      next.disabled = state.step === 2;
      text.textContent = describe(kids);
    }

    function describe(kids) {
      const { p1, p2 } = state;
      const head = `Step ${state.step + 1} of 3. `;
      if (state.step === 0) {
        if (mutationOp()) return head + `The child x = ${bits(p1)} enters mutation. Click a bit, or focus it and press Enter, to change it.`;
        return head + `Parents P1 = ${bits(p1)} and P2 = ${bits(p2)} differ in ${hamming(p1, p2)} of ${n} positions. Click a bit, or focus it and press Enter, to change it.`;
      }
      if (state.step === 1) {
        if (state.op === "one_point") return head + `The cut point ${info.cut} is drawn uniformly from 1 to ${n - 1}: the children will exchange positions ${info.cut} to ${n - 1}.`;
        if (state.op === "two_point") return head + `Two positions drawn from 0 to ${n - 1} give the segment [${info.lo}, ${info.hi}): the children will exchange positions ${info.lo} to ${info.hi - 1}.`;
        if (state.op === "uniform") {
          const swapped = info.keep.filter((k) => !k).length;
          return head + `Each position is exchanged with probability 0.5: this draw exchanges ${swapped} of ${n} positions, marked in the swap row.`;
        }
        if (state.op === "hux") {
          const diff = hamming(p1, p2);
          return head + `HUX exchanges exactly half of the ${diff} differing positions (at least one): ${info.chosen.length} chosen at random, ${info.chosen.join(", ") || "none"}.`;
        }
        if (state.op === "bitflip") {
          const flips = info.mask.filter(Boolean).length;
          return head + `Each bit flips with probability ${state.prob.toFixed(3)}, ${fmt(n * state.prob, 2)} bits on average: this draw flips ${flips} (${info.mask.map((m, j) => (m ? j : -1)).filter((j) => j >= 0).join(", ") || "none"}).`;
        }
        return head + `Two distinct positions give the segment from ${info.a} to ${info.b}; every bit inside it will be complemented.`;
      }
      if (mutationOp()) return head + `x′ = ${bits(kids[0])}, at Hamming distance ${hamming(p1, kids[0])} from x.`;
      return head + `C1 = ${bits(kids[0])} (distance ${hamming(kids[0], p1)} to P1, ${hamming(kids[0], p2)} to P2) and C2 = ${bits(kids[1])} (distance ${hamming(kids[1], p1)} to P1, ${hamming(kids[1], p2)} to P2). Every bit comes from a parent, so a position where both parents agree keeps its value.`;
    }
    draw();
    probabilityControl();
    render();
  };

  widgets["integer-sbx"] = function (node) {
    const state = { p1: 3, p2: 4, eta: 20, truncate: false };
    const lower = 0;
    const upper = 10;
    const ctl = controls(node);
    slider(ctl, { label: "Parent 1", min: 0, max: 10, step: 1, value: state.p1, onInput: (v) => ((state.p1 = v), render()) });
    slider(ctl, { label: "Parent 2", min: 0, max: 10, step: 1, value: state.p2, onInput: (v) => ((state.p2 = v), render()) });
    slider(ctl, { label: "η_c", min: 0, max: 50, step: 1, value: state.eta, onInput: (v) => ((state.eta = v), render()) });
    select(ctl, {
      label: "Rounding",
      value: "round",
      options: [
        ["round", "Nearest integer (VAMOS)"],
        ["trunc", "Truncation int() (jMetal)"],
      ],
      onChange: (v) => ((state.truncate = v === "trunc"), render()),
    });
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [-0.5, 10.5], y: [0, 1], xLabel: "Child value (bounds [0, 10])", yLabel: "Share of children", xTicks: [0, 2, 4, 6, 8, 10], yFormat: (v) => Math.round(v * 100) + "%", label: "Distribution of integer SBX children" });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Children (20,000 crossovers)", shape: "region", cls: "vf-bar-1" },
      { label: "Parents", shape: "line", cls: "vf-line-2" },
    ]);
    function render() {
      const rand = mulberry32(5);
      const counts = new Array(11).fill(0);
      let sum = 0;
      const total = 40000;
      for (let i = 0; i < total / 2; i++) {
        for (const c of OP.integerSbx([state.p1], [state.p2], lower, upper, state.eta, 1, rand, state.truncate)) {
          counts[c[0]] += 1;
          sum += c[0];
        }
      }
      const shares = counts.map((c) => c / total);
      const top = Math.max(0.1, Math.ceil(Math.max(...shares) * 10) / 10);
      plot.setDomain([-0.5, 10.5], [0, top]);
      clear(plot.layers.mid);
      clear(plot.layers.top);
      shares.forEach((v, k) => {
        if (v <= 0) return;
        plot.rect(plot.layers.mid, k - 0.38, 0, k + 0.38, v, "vf-bar-1");
      });
      [state.p1, state.p2].forEach((p, k) => {
        s("line", { x1: plot.x(p), x2: plot.x(p), y1: plot.y(0), y2: plot.y(top), class: "vf-line-2 vf-dashed" }, plot.layers.top);
        txt(plot.layers.top, plot.x(p) + 4, plot.m.t + 12 + 14 * k, "p" + (k + 1), { class: "vf-mark-label" });
      });
      const mean = sum / total;
      const same = (counts[state.p1] + (state.p2 !== state.p1 ? counts[state.p2] : 0)) / total;
      clear(readout);
      h("p", {}, readout, `Mean child ${fmt(mean, 3)} for a parents' mean of ${fmt((state.p1 + state.p2) / 2, 1)}; ${fmt(100 * same, 1)}% of the children equal a parent.`);
      h(
        "p",
        {},
        readout,
        state.truncate
          ? "Truncation sends every fraction down, towards zero, so the children's mean falls below the parents' mean."
          : "Rounding to the nearest integer keeps SBX's symmetry: the children's mean stays at the parents' mean."
      );
    }
    render();
  };

  widgets["permutation-steps"] = function (node) {
    const n = 8;
    const P1 = [0, 1, 2, 3, 4, 5, 6, 7];
    const P2 = [2, 6, 4, 0, 5, 7, 1, 3];
    const state = { op: node.dataset.operator || "pmx", a: 2, b: 5, start: 0, step: 0, seed: 1, keep: [false, true, false, true, true, false, false, true] };
    const names = { pmx: "PMX", ox: "Order crossover (OX)", cx: "Cycle crossover (CX)", position: "Position-based" };
    const ctl = controls(node);
    select(ctl, { label: "Operator", value: state.op, options: Object.keys(names).map((k) => [k, names[k]]), onChange: (v) => ((state.op = v), (state.step = 0), buildParams(), render()) });
    const prev = button(ctl, "Previous step", () => ((state.step = Math.max(0, state.step - 1)), render()));
    const next = button(ctl, "Next step", () => ((state.step = Math.min(2, state.step + 1)), render()), true);
    const params = controls(node);
    const W = 40 + n * 34 + 8;
    const svg = s("svg", { viewBox: `0 0 ${W} 160`, class: "vf-plot", role: "group", "aria-label": "Two parent permutations and the children a permutation crossover builds" }, h("div", { class: "vf-plot-wrap" }, node));
    const text = h("p", { class: "vf-step-text", "aria-live": "polite" }, h("div", { class: "vf-readout" }, node));
    legend(node, [
      { label: "From P1", shape: "region", cls: "vf-cell-a" },
      { label: "From P2", shape: "region", cls: "vf-cell-b" },
      { label: "Segment, cycle or kept position", shape: "region", cls: "vf-cell-mark" },
    ]);

    function buildParams() {
      clear(params);
      if (state.op === "pmx" || state.op === "ox") {
        const maxEnd = state.op === "pmx" ? n - 1 : n - 2;
        state.b = Math.min(state.b, maxEnd);
        state.a = Math.min(state.a, state.b - (state.op === "pmx" ? 1 : 0));
        let endCtl = null;
        const startCtl = slider(params, {
          label: "Segment start",
          min: 0,
          max: maxEnd - (state.op === "pmx" ? 1 : 0),
          step: 1,
          value: state.a,
          onInput: (v) => {
            state.a = v;
            const minLen = state.op === "pmx" ? 1 : 0;
            if (state.b < v + minLen) {
              state.b = v + minLen;
              endCtl.set(state.b);
            }
            render();
          },
        });
        endCtl = slider(params, {
          label: "Segment end (included)",
          min: state.op === "pmx" ? 1 : 0,
          max: maxEnd,
          step: 1,
          value: state.b,
          onInput: (v) => {
            state.b = v;
            const minLen = state.op === "pmx" ? 1 : 0;
            if (state.a > v - minLen) {
              state.a = v - minLen;
              startCtl.set(state.a);
            }
            render();
          },
        });
      } else if (state.op === "cx") {
        slider(params, { label: "Start position", min: 0, max: n - 1, step: 1, value: state.start, onInput: (v) => ((state.start = v), render()) });
      } else {
        button(params, "New positions", () => {
          state.seed += 1;
          const rand = mulberry32(state.seed * 104729);
          const k = 1 + Math.floor(rand() * n);
          const order = P1.map((_, i) => i);
          for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
          }
          state.keep = P1.map((_, i) => order.slice(0, k).includes(i));
          render();
        });
      }
    }

    function compute() {
      if (state.op === "pmx") {
        const c1 = OP.pmxChild(P1, P2, state.a, state.b);
        const c2 = OP.pmxChild(P2, P1, state.a, state.b);
        const inSeg = (i) => i >= state.a && i <= state.b;
        return { kids: [c1, c2], mark: inSeg, src: (k, i) => (inSeg(i) ? 1 - k : k) };
      }
      if (state.op === "ox") {
        const lo = state.a;
        const hi = state.b + 1;
        const inSeg = (i) => i >= lo && i < hi;
        return { kids: [OP.oxChild(P1, P2, lo, hi), OP.oxChild(P2, P1, lo, hi)], mark: inSeg, src: (k, i) => (inSeg(i) ? k : 1 - k) };
      }
      if (state.op === "cx") {
        const res = OP.cycle(P1, P2, state.start);
        const inCyc = (i) => res.cycle.includes(i);
        return { kids: res.children, mark: inCyc, src: (k, i) => (inCyc(i) ? k : 1 - k), cycle: res.cycle };
      }
      const keep = (i) => state.keep[i];
      return { kids: [OP.positionBased(P1, P2, state.keep), OP.positionBased(P2, P1, state.keep)], mark: keep, src: (k, i) => (keep(i) ? k : 1 - k) };
    }

    function render() {
      clear(svg);
      const x0 = 40;
      const w = 34;
      for (let j = 0; j < n; j++) txt(svg, x0 + j * w + 15.5, 12, String(j), { "text-anchor": "middle" });
      const res = compute();
      const markCls = (base) => (j) => (state.step >= 1 && res.mark(j) ? "vf-cell-mark" : base);
      cellRow(svg, 20, "P1", P1, { x0, w, cls: markCls("vf-cell-a") });
      cellRow(svg, 50, "P2", P2, { x0, w, cls: markCls("vf-cell-b") });
      if (state.step >= 1) {
        res.kids.forEach((kid, k) => {
          const partial = kid.map((g, i) => (state.step >= 2 || res.mark(i) ? g : null));
          cellRow(svg, 92 + k * 30, "C" + (k + 1), partial, { x0, w, cls: (i) => (partial[i] === null ? "vf-cell-empty" : res.src(k, i) === 0 ? "vf-cell-a" : "vf-cell-b") });
        });
      }
      prev.disabled = state.step === 0;
      next.disabled = state.step === 2;
      text.textContent = describe(res);
    }

    const list = (a) => "[" + a.join(" ") + "]";
    function describe(res) {
      const head = `Step ${state.step + 1} of 3. `;
      const [c1, c2] = res.kids;
      if (state.op === "pmx") {
        if (state.step === 0) return head + `Positions ${state.a} to ${state.b} form the matching section of P1 = ${list(P1)} and P2 = ${list(P2)}.`;
        const pairs = [];
        for (let i = state.a; i <= state.b; i++) pairs.push(`${P2[i]}↔${P1[i]}`);
        if (state.step === 1) return head + `C1 receives P2's section and C2 receives P1's. The section defines the mapping ${pairs.join(", ")}.`;
        return head + `Every other position of C1 copies P1; a gene already in the section is replaced through the mapping, repeatedly, until it is new. C1 = ${list(c1)}, C2 = ${list(c2)}.`;
      }
      if (state.op === "ox") {
        const lo = state.a;
        const hi = state.b + 1;
        if (state.step === 0) return head + `The segment covers positions ${lo} to ${hi - 1} of P1 = ${list(P1)} and P2 = ${list(P2)}.`;
        if (state.step === 1) return head + `C1 copies P1's segment and C2 copies P2's, in place.`;
        const used = new Set(P1.slice(lo, hi));
        return head + `C1 is completed with P2's remaining genes in P2's order, ${list(P2.filter((g) => !used.has(g)))}, written from position ${hi % n} onwards and wrapping around. C1 = ${list(c1)}, C2 = ${list(c2)}.`;
      }
      if (state.op === "cx") {
        if (state.step === 0) return head + `Start at position ${state.start}. A cycle follows P2's gene at the current position to its position in P1.`;
        const path = res.cycle.map((i) => `${i} (P1 ${P1[i]}, P2 ${P2[i]})`).join(" → ");
        if (state.step === 1) return head + `The cycle visits positions ${path} and closes. Its ${res.cycle.length} positions hold the same set of genes in both parents.`;
        return head + `C1 takes P1's genes on the cycle and P2's elsewhere; C2 the reverse. C1 = ${list(c1)}, C2 = ${list(c2)}. Every gene keeps a position it had in one parent.`;
      }
      const kept = state.keep.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
      if (state.step === 0) return head + `A random number of positions, here ${kept.length} (${kept.join(", ")}), is drawn; press New positions for another draw.`;
      if (state.step === 1) return head + `C1 keeps P1's genes at those positions and C2 keeps P2's.`;
      return head + `The free positions of C1 are filled, left to right, with P2's other genes in P2's order. C1 = ${list(c1)}, C2 = ${list(c2)}.`;
    }
    buildParams();
    render();
  };

  widgets["tour-moves"] = function (node) {
    const cities = [
      [0.1, 0.2],
      [0.3, 0.85],
      [0.55, 0.15],
      [0.75, 0.8],
      [0.92, 0.35],
      [0.6, 0.5],
      [0.35, 0.45],
      [0.12, 0.62],
    ];
    const start = [0, 2, 1, 3, 4, 5, 6, 7];
    const n = cities.length;
    const state = { tour: start.slice(), move: "inversion", i: 1, j: 2 };
    const ctl = controls(node);
    const moveSelect = select(ctl, {
      label: "Mutation",
      value: state.move,
      options: [
        ["inversion", "Inversion / 2-opt"],
        ["swap", "Swap"],
        ["insert", "Insert"],
      ],
      onChange: (v) => ((state.move = v), render()),
    });
    const si = slider(ctl, { label: "Position i", min: 0, max: n - 1, step: 1, value: state.i, onInput: (v) => ((state.i = v), render()) });
    const sj = slider(ctl, { label: "Position j", min: 0, max: n - 1, step: 1, value: state.j, onInput: (v) => ((state.j = v), render()) });
    const ctl2 = controls(node);
    button(ctl2, "Apply the move", () => ((state.tour = proposed()), render()), true);
    button(ctl2, "Best inversion", () => {
      let best = null;
      for (let a = 0; a < n; a++)
        for (let b = a + 1; b < n; b++) {
          const len = OP.tourLength(cities, OP.invert(state.tour, a, b));
          if (!best || len < best[0] - 1e-12) best = [len, a, b];
        }
      state.move = "inversion";
      moveSelect.value = "inversion";
      state.i = best[1];
      state.j = best[2];
      si.set(state.i);
      sj.set(state.j);
      render();
    });
    button(ctl2, "Reset tour", () => ((state.tour = start.slice()), render()));
    const body = h("div", { class: "vf-body vf-body--side" }, node);
    const plot = createPlot(body, { x: [0, 1], y: [0, 1], xLabel: "x", yLabel: "y", label: "A tour of eight cities before and after a mutation", width: 320, height: 300, margin: { l: 36, b: 36 } });
    const readout = h("div", { class: "vf-readout" }, body);
    legend(node, [
      { label: "Current tour", shape: "line", cls: "vf-line-thin vf-dashed" },
      { label: "Tour after the move", shape: "line", cls: "vf-line-1" },
      { label: "New edges", shape: "line", cls: "vf-line-2" },
    ]);
    function proposed() {
      const lo = Math.min(state.i, state.j);
      const hi = Math.max(state.i, state.j);
      if (state.move === "inversion") return OP.invert(state.tour, lo, hi);
      if (state.move === "swap") return OP.swap(state.tour, state.i, state.j);
      return OP.insert(state.tour, state.i, state.j);
    }
    function render() {
      const next = proposed();
      const pts = (perm) => perm.concat([perm[0]]).map((c) => cities[c]);
      clear(plot.layers.mid);
      clear(plot.layers.points);
      clear(plot.layers.top);
      plot.line(plot.layers.mid, pts(state.tour), "vf-line-thin vf-dashed");
      plot.line(plot.layers.mid, pts(next), "vf-line-1");
      const before = new Set(OP.tourEdges(state.tour));
      const added = OP.tourEdges(next).filter((e) => !before.has(e));
      for (const e of added) {
        const [a, b] = e.split("-").map(Number);
        plot.line(plot.layers.mid, [cities[a], cities[b]], "vf-line-2");
      }
      cities.forEach((c, k) => {
        marker(plot.layers.points, "circle", plot.x(c[0]), plot.y(c[1]), 5, "vf-pt vf-pt-ink");
        txt(plot.layers.top, plot.x(c[0]) + 7, plot.y(c[1]) - 6, String(k), { class: "vf-mark-label" });
      });
      const l0 = OP.tourLength(cities, state.tour);
      const l1 = OP.tourLength(cities, next);
      clear(readout);
      const verb = { inversion: `reverses positions ${Math.min(state.i, state.j)} to ${Math.max(state.i, state.j)}`, swap: `exchanges the cities at positions ${state.i} and ${state.j}`, insert: `moves the city at position ${state.i} to position ${state.j}` }[state.move];
      h("p", {}, readout, `Current tour [${state.tour.join(" ")}]: length ${fmt(l0, 3)}, ${OP.tourCrossings(cities, state.tour)} crossing(s).`);
      h("p", {}, readout, `The move ${verb}: [${next.join(" ")}], length ${fmt(l1, 3)} (${l1 - l0 <= 0 ? "" : "+"}${fmt(l1 - l0, 3)}), ${OP.tourCrossings(cities, next)} crossing(s); ${added.length} of ${n} edges change.`);
      h("p", {}, readout, "On a symmetric tour an inversion replaces exactly two edges, which is a 2-opt move; a swap replaces up to four and an insertion up to three.");
    }
    render();
  };

  /* ---- Basic: genetic algorithms and families of algorithms ---- */

  function idList(list) {
    const names = list.map(String);
    if (names.length <= 1) return names.join("");
    return names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
  }

  /* The evolutionary loop, one phase at a time, generational or steady-state. */
  widgets["ga-loop"] = function (node) {
    const N = 6;
    const BUDGET = 30;
    const MODES = {
      plus: { label: "Generational, elitist (μ + λ)", short: "(μ + λ)" },
      comma: { label: "Generational, non-elitist (μ, λ)", short: "(μ, λ)" },
      steady: { label: "Steady-state (μ + 1)", short: "(μ + 1)" },
    };
    const PHASES = { select: "select", vary: "vary", evaluate: "evaluate", survive: "survive" };
    const state = { mode: node.dataset.mode || "plus", seed: 44, step: 0 };
    let steps = [];
    const ctl = controls(node);
    select(ctl, {
      label: "Replacement",
      options: Object.keys(MODES).map((k) => [k, MODES[k].label]),
      value: state.mode,
      onChange: (v) => {
        state.mode = v;
        build();
      },
    });
    const back = button(ctl, "Back", () => go(state.step - 1));
    const next = button(ctl, "Next step", () => go(state.step + 1), true);
    const cycleButton = button(ctl, "Next cycle", () => {
      let i = state.step + 1;
      while (i < steps.length - 1 && steps[i].phase !== "survive") i += 1;
      go(i);
    });
    button(ctl, "Restart", () => go(0));
    button(ctl, "New population", () => {
      state.seed = Math.floor(Math.random() * 1e6);
      build();
    });
    const body = h("div", { class: "vf-body vf-body--two" }, node);
    const xPlot = createPlot(body, {
      title: "Decision space",
      x: [0, 1],
      y: [0, 1],
      xLabel: "x₁",
      yLabel: "x₂",
      label: "Decision vectors of the population and of the children",
    });
    const fPlot = createPlot(body, {
      title: "Objective space",
      x: [0, 1],
      y: [0, 2.1],
      xLabel: "f₁ (minimize)",
      yLabel: "f₂ (minimize)",
      label: "Objective vectors of the population and of the evaluated children",
    });
    for (const plot of [xPlot, fPlot]) {
      plot.svg.setAttribute("tabindex", "0");
      plot.svg.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight");
    }
    const readout = h("div", { class: "vf-readout" }, node);
    legend(node, [
      { label: "Population member (front 1, 2 … darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
      { label: "Child", shape: "triangle", cls: "vf-pt vf-pt-2" },
      { label: "Chosen parent / survivor", shape: "circle", cls: "vf-ring" },
      { label: "Pareto set and front", shape: "line", cls: "vf-front" },
    ]);
    node.addEventListener("keydown", (e) => {
      if (e.target && /^(SELECT|INPUT)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight") go(state.step + 1);
      else if (e.key === "ArrowLeft") go(state.step - 1);
      else return;
      e.preventDefault();
    });
    function build() {
      steps = gaLoop(state.mode, state.seed, N, BUDGET);
      go(0);
    }
    function go(i) {
      state.step = Math.max(0, Math.min(steps.length - 1, i));
      render();
    }
    function point(plot, layer, v, shape, cls, label, labelLayer) {
      const px = plot.x(v[0]);
      const py = plot.y(Math.min(v[1], plot.yd[1]));
      marker(layer, shape, px, py, 6, cls);
      if (label !== undefined) txt(labelLayer, px + 9, py - 7, label, { class: "vf-mark-label" });
      return [px, py];
    }
    function render() {
      const st = steps[state.step];
      const mode = MODES[state.mode];
      back.disabled = state.step === 0;
      next.disabled = state.step === steps.length - 1;
      cycleButton.disabled = state.step === steps.length - 1;
      for (const plot of [xPlot, fPlot]) {
        clear(plot.layers.mid);
        clear(plot.layers.marks);
        clear(plot.layers.top);
      }
      xPlot.line(xPlot.layers.mid, [[0, 0.004], [1, 0.004]], "vf-front");
      plot2Front();
      const rank = st.rank || rankAndCrowding(st.pop.map((p) => p.f)).rank;
      const chosen = new Map();
      if (st.pairs) st.pairs.forEach((pair) => pair.forEach((p) => chosen.set(p.id, (chosen.get(p.id) || 0) + 1)));
      const rows = [];
      if (st.phase === "survive") {
        const keptIds = new Set(st.kept.map((p) => p.id));
        st.merged.forEach((p, i) => {
          const isChild = i >= st.pop.length;
          const kept = keptIds.has(p.id);
          const cls = "vf-pt " + rankClass(st.mergedRank[i]) + (kept ? "" : " vf-faded");
          const label = String(p.id) + (kept ? "" : " ✕");
          for (const [plot, v] of [
            [xPlot, p.x],
            [fPlot, p.f],
          ]) {
            const [px, py] = point(plot, plot.layers.marks, v, isChild ? "triangle" : "circle", cls, label, plot.layers.top);
            if (kept) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
          }
          rows.push([String(p.id), isChild ? "child" : "parent", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(st.mergedRank[i] + 1), kept ? "kept" : "removed"]);
        });
      } else {
        st.pop.forEach((p, i) => {
          const k = chosen.get(p.id) || 0;
          const cls = "vf-pt " + rankClass(rank[i]);
          const label = String(p.id) + (st.phase === "select" && k > 1 ? " ×" + k : "");
          for (const [plot, v] of [
            [xPlot, p.x],
            [fPlot, p.f],
          ]) {
            const [px, py] = point(plot, plot.layers.marks, v, "circle", cls, label, plot.layers.top);
            if (st.phase === "select" && k) marker(plot.layers.marks, "circle", px, py, 10.5, "vf-ring");
          }
          rows.push([String(p.id), "member", fmt(p.x[0], 2), fmt(p.x[1], 2), fmt(p.f[0], 2), fmt(p.f[1], 2), String(rank[i] + 1), st.phase === "select" ? (k ? `parent ×${k}` : "–") : st.phase === "done" ? (rank[i] === 0 ? "returned" : "–") : "–"]);
        });
        if (st.children) {
          st.children.forEach((c) => {
            const parents = c.parents.map((id) => st.pop.find((p) => p.id === id));
            parents.forEach((p) => xPlot.line(xPlot.layers.mid, [p.x, c.x], "vf-line-thin vf-dashed"));
            point(xPlot, xPlot.layers.marks, c.x, "triangle", "vf-pt vf-pt-2", String(c.id), xPlot.layers.top);
            if (c.f) point(fPlot, fPlot.layers.marks, c.f, "triangle", "vf-pt vf-pt-2", String(c.id), fPlot.layers.top);
            rows.push([String(c.id), `child of ${c.parents[0]} and ${c.parents[1]}`, fmt(c.x[0], 2), fmt(c.x[1], 2), c.f ? fmt(c.f[0], 2) : "?", c.f ? fmt(c.f[1], 2) : "?", "–", c.f ? "evaluated" : "not evaluated"]);
          });
        }
      }
      clear(readout);
      const cycles = steps.filter((s) => s.phase === "survive").length;
      const perCycle = state.mode === "steady" ? 1 : N;
      const phaseNo = { select: 1, vary: 2, evaluate: 3, survive: 4 }[st.phase];
      const where = st.phase === "init" ? "Start" : st.phase === "done" ? "End" : `Cycle ${st.cycle} of ${cycles}, phase ${phaseNo} of 4: ${PHASES[st.phase]}`;
      h("p", { class: "vf-step-text" }, readout, `${where}. ${describe(st, mode)}`);
      h(
        "p",
        {},
        readout,
        `${mode.label}: ${perCycle} offspring per cycle, ${cycles} survival steps for a budget of ${BUDGET} evaluations. Evaluations used: ${st.evaluations} of ${BUDGET}. Step ${state.step + 1} of ${steps.length}; the arrow keys step through the figure when it has focus.`
      );
      table(readout, ["Id", "Role", "x₁", "x₂", "f₁", "f₂", "Front", "Status"], rows, [false, false, true, true, true, true, true, false]);
    }
    function plot2Front() {
      fPlot.line(fPlot.layers.mid, frontPoints((t) => 1 - Math.sqrt(t), 0, 1, 60), "vf-front");
    }
    function describe(st, mode) {
      if (st.phase === "init") return `${N} random solutions (decision vectors drawn uniformly in [0, 1]²), evaluated: f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂. Their front, 1 is best, colours each point.`;
      if (st.phase === "select") {
        const why = { front: "better front", crowding: "same front, more isolated", coin: "full tie, coin" };
        const games = st.tournaments.map((t) => `${t.a} vs ${t.b} → ${t.winner} (${why[t.reason]})`).join("; ");
        return `${st.tournaments.length} binary tournaments pick ${st.pairs.length === 1 ? "one pair" : st.pairs.length + " pairs"} of parents: ${games}. A member can win several times; ringed members are parents.`;
      }
      if (st.phase === "vary") {
        const ids = st.children.map((c) => c.id);
        if (state.mode === "steady") return `SBX crossover recombines parents ${st.pairs[0][0].id} and ${st.pairs[0][1].id} into two children; the first, ${ids[0]}, is kept and mutated by polynomial mutation. It exists in the decision space only: its objectives are unknown.`;
        return `SBX crossover recombines each pair and polynomial mutation perturbs each child: children ${ids[0]} to ${ids[ids.length - 1]} (triangles, dashed lines to their parents). They exist in the decision space only: their objectives are unknown.`;
      }
      if (st.phase === "evaluate") return `The problem computes f₁ and f₂ of ${st.children.length === 1 ? "the child" : "the " + st.children.length + " children"}: ${st.evaluations} of ${BUDGET} evaluations are now used.`;
      if (st.phase === "survive") {
        const removedParents = st.removed.filter((p) => st.pop.some((q) => q.id === p.id));
        const removedChildren = st.removed.filter((p) => !removedParents.includes(p));
        if (state.mode === "comma") {
          const lost = removedParents.filter((p) => st.mergedRank[st.merged.indexOf(p)] === 0).map((p) => p.id);
          return `${mode.short}: the ${st.children.length} children replace the ${st.pop.length} parents, whatever their quality.` + (lost.length ? ` Lost: ${idList(lost)}, which no child dominates.` : " This time no parent of front 1 is lost.");
        }
        if (state.mode === "steady") {
          const out = st.removed[0];
          if (removedChildren.length) return `${mode.short}: the ${st.pop.length} members and child ${out.id} compete for ${N} places by front, then crowding distance. The child is the worst of the ${st.merged.length}: it is discarded and the population does not change.`;
          return `${mode.short}: the ${st.pop.length} members and child ${st.children[0].id} compete for ${N} places by front, then crowding distance. The worst of the ${st.merged.length}, ${out.id}, is removed and the child takes its place at once.`;
        }
        const parts = [];
        if (removedParents.length) parts.push(`parent${removedParents.length > 1 ? "s" : ""} ${idList(removedParents.map((p) => p.id))}`);
        if (removedChildren.length) parts.push(`child${removedChildren.length > 1 ? "ren" : ""} ${idList(removedChildren.map((p) => p.id))}`);
        return `${mode.short}: the ${st.pop.length} parents and ${st.children.length} children compete for ${N} places, by front and then by crowding distance (NSGA-II's rule). Removed: ${parts.join(" and ")}.`;
      }
      const nd = st.pop.filter((_, i) => st.rank[i] === 0).map((p) => p.id);
      return `The budget is spent and the run stops. With a configuration from builder() or default(), VAMOS returns the non-dominated members of the final population: ${idList(nd)}.`;
    }
    build();
  };

  /* One population, the survivors of each family of algorithms. */
  widgets["family-survivors"] = function (node) {
    const defaults = [
      [0.05, 0.95],
      [0.12, 0.62],
      [0.18, 0.56],
      [0.35, 0.4],
      [0.42, 0.37],
      [0.62, 0.2],
      [0.95, 0.04],
      [0.3, 0.72],
      [0.55, 0.45],
      [0.75, 0.3],
    ];
    const names = defaults.map((_, i) => "P" + (i + 1));
    const FAMILIES = [
      ["dominance", "Dominance-based"],
      ["decomposition", "Decomposition-based"],
      ["indicator", "Indicator-based"],
      ["swarm", "Swarm-based"],
    ];
    const fig = pointSetFigure(node, {
      points: defaults,
      names,
      layout: "",
      x: [0, 1.05],
      y: [0, 1.05],
      label: "Ten draggable solutions and the ones that the selected family of algorithms keeps",
    });
    const state = fig.state;
    state.family = node.dataset.family || "dominance";
    state.N = 5;
    const pressed = {};
    for (const [key, label] of FAMILIES) {
      pressed[key] = button(fig.ctl, label, () => {
        state.family = key;
        fig.render();
      });
    }
    slider(fig.ctl, { label: "Places", min: 2, max: 9, step: 1, value: state.N, format: (v) => String(v), onInput: (v) => ((state.N = v), fig.render()) });
    button(fig.ctl, "Reset points", () => {
      defaults.forEach((p, i) => (state.P[i] = p.slice()));
      fig.render();
    });
    const legendBox = h("div", {}, node);
    const plot = fig.plot;
    fig.render = function () {
      const P = state.P;
      const N = state.N;
      const res = familySurvivors(P, N);
      const r = rankAndCrowding(P);
      for (const [key] of FAMILIES) {
        pressed[key].setAttribute("aria-pressed", String(key === state.family));
        pressed[key].classList.toggle("vf-button--primary", key === state.family);
      }
      clear(plot.layers.back);
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      fig.labels();
      const kept = new Set(state.family === "dominance" ? res.dominance : state.family === "decomposition" ? res.decomposition.kept : state.family === "indicator" ? res.indicator.kept : res.swarm.kept);
      const ring = state.family === "swarm" ? new Set(res.swarm.leaders) : kept;
      let items;
      if (state.family === "decomposition") {
        const z = res.decomposition.z;
        res.decomposition.W.forEach((w, k) => {
          const d = [w[1], w[0]];
          const len = 2 / Math.max(1e-9, Math.hypot(d[0], d[1]));
          plot.line(plot.layers.mid, [z, [z[0] + d[0] * len, z[1] + d[1] * len]], "vf-line-thin vf-dashed");
          // Label each direction just outside the edge where it leaves the plot.
          const tx = (plot.xd[1] - z[0]) / Math.max(d[0], 1e-9);
          const ty = (plot.yd[1] - z[1]) / Math.max(d[1], 1e-9);
          const ex = plot.x(z[0] + d[0] * Math.min(tx, ty));
          const ey = plot.y(z[1] + d[1] * Math.min(tx, ty));
          if (ty <= tx) txt(plot.layers.top, ex, ey - 3, "w" + (k + 1), { class: "vf-mark-label", "text-anchor": "middle" });
          else txt(plot.layers.top, ex + 3, ey + 4, "w" + (k + 1), { class: "vf-mark-label" });
        });
        marker(plot.layers.marks, "diamond", plot.x(z[0]), plot.y(z[1]), 5, "vf-pt vf-pt-3");
        txt(plot.layers.top, plot.x(z[0]) + 8, plot.y(z[1]) + 14, "z*", { class: "vf-mark-label" });
        items = [
          { label: "Kept (best for a weight vector)", shape: "circle", cls: "vf-ring" },
          { label: "Direction of each weight vector", shape: "line", cls: "vf-line-thin vf-dashed" },
          { label: "Ideal point z*", shape: "diamond", cls: "vf-pt vf-pt-3" },
        ];
      } else if (state.family === "indicator") {
        const keptPts = res.indicator.kept.map((i) => P[i]);
        const ref = res.indicator.ref || [1.05, 1.05];
        const nd = nonDominatedIndices(keptPts)
          .map((i) => keptPts[i])
          .sort((a, b) => a[0] - b[0]);
        if (nd.length) {
          const poly = [[nd[0][0], ref[1]]];
          nd.forEach((p, k) => {
            poly.push(p);
            poly.push([k + 1 < nd.length ? nd[k + 1][0] : ref[0], p[1]]);
          });
          poly.push(ref);
          plot.line(plot.layers.back, poly, "vf-region-1");
        }
        res.indicator.removed.forEach((i, k) => txt(plot.layers.top, plot.x(P[i][0]) - 6, plot.y(P[i][1]) + 20, "×" + (k + 1), { class: "vf-mark-label" }));
        items = [
          { label: "Kept", shape: "circle", cls: "vf-ring" },
          { label: "Removed (order shown)", shape: "circle", cls: "vf-pt vf-pt-muted vf-faded" },
          { label: "Hypervolume of the kept set", shape: "region", cls: "vf-region-1" },
        ];
      } else if (state.family === "swarm") {
        items = [
          { label: "Particle (every one moves on)", shape: "circle", cls: "vf-pt vf-r1" },
          { label: "Leader (non-dominated, in the archive)", shape: "circle", cls: "vf-ring" },
        ];
      } else {
        const split = r.fronts.find((front) => front.some((i) => kept.has(i)) && front.some((i) => !kept.has(i)));
        if (split) split.forEach((i) => txt(plot.layers.top, plot.x(P[i][0]) - 6, plot.y(P[i][1]) + 20, fmt(r.crowding[i], 2), { class: "vf-mark-label" }));
        items = [
          { label: "Front 1, 2, … (darker is better)", shape: "circle", cls: "vf-pt vf-r1" },
          { label: "Kept", shape: "circle", cls: "vf-ring" },
          { label: "Crowding distance in the front that is cut", shape: "circle", cls: "vf-pt-hollow" },
        ];
      }
      clear(legendBox);
      legend(legendBox, items);
      P.forEach((p, i) => {
        const hd = fig.handles[i];
        hd.place();
        hd.node.classList.toggle("vf-selected", i === state.active);
        hd.setClass("vf-pt " + rankClass(r.rank[i]) + (kept.has(i) ? "" : " vf-faded"));
        if (ring.has(i)) marker(plot.layers.marks, "circle", plot.x(p[0]), plot.y(p[1]), 10.5, "vf-ring");
      });
      clear(fig.readout);
      const list = (rows) => idList(rows.map((i) => names[i]));
      const removed = (set) => P.map((_, i) => i).filter((i) => !set.has(i));
      let text;
      if (state.family === "dominance") {
        const split = r.fronts.find((front) => front.some((i) => kept.has(i)) && front.some((i) => !kept.has(i)));
        text = `NSGA-II's rule: whole fronts while they fit, then the largest crowding distances. Front 1 is ${list(r.fronts[0])}.`;
        if (split) text += ` Front ${r.rank[split[0]] + 1} does not fit, so crowding keeps ${list(split.filter((i) => kept.has(i)))} and removes ${list(split.filter((i) => !kept.has(i)))}.`;
        text += ` Kept: ${list([...kept])}.`;
      } else if (state.family === "decomposition") {
        const d = res.decomposition;
        const wins = d.W.map((w, k) => `w${k + 1} = (${fmt(w[0], 2)}, ${fmt(w[1], 2)}) → ${names[d.best[k]]}`).join("; ");
        text = `MOEA/D's rule: ${N} weight vectors, and each keeps the member with the smallest Tchebycheff value max_i w_i·|f_i − z*_i|, with z* = ${pair(d.z)}: ${wins}.`;
        text += d.kept.length < N ? ` Only ${d.kept.length} distinct members are kept: a member that wins several subproblems is stored once for each.` : ` Kept: ${list(d.kept)}.`;
      } else if (state.family === "indicator") {
        const d = res.indicator;
        text = d.removed.length
          ? `SMS-EMOA's rule, once per removal: sort into fronts and remove the member of the last front whose loss costs the least hypervolume, with r = nadir + (nadir − ideal) of the members left. Removed in order: ${d.removed.map((i) => names[i]).join(", ")}; the last r is ${pair(d.ref)}. Kept: ${list(d.kept)}.`
          : `Every member fits in the ${N} places: nothing is removed.`;
      } else {
        text = `SMPSO's swarm keeps every particle: each one moves to a new position at every iteration, whatever its quality. Elitism lives in the leaders archive of non-dominated positions, here ${list(res.swarm.leaders)}, with room for as many leaders as there are particles, the VAMOS default. The places setting does not apply.`;
      }
      h("p", { class: "vf-step-text" }, fig.readout, text);
      if (state.family !== "swarm" && removed(kept).length) h("p", {}, fig.readout, `Not kept: ${list(removed(kept))}.`);
      const mark = (set, i) => (set.has(i) ? "✓" : "–");
      const sets = [new Set(res.dominance), new Set(res.decomposition.kept), new Set(res.indicator.kept)];
      const leaders = new Set(res.swarm.leaders);
      table(
        fig.readout,
        ["Point", "f₁", "f₂", "Front", "Dominance", "Decomp.", "Indicator", "Swarm"],
        P.map((p, i) => [names[i], fmt(p[0], 2), fmt(p[1], 2), String(r.rank[i] + 1), mark(sets[0], i), mark(sets[1], i), mark(sets[2], i), leaders.has(i) ? "leader" : "particle"]),
        [false, true, true, true, false, false, false, false]
      );
    };
    fig.render();
  };

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  function init() {
    document.querySelectorAll("[data-vf-widget]").forEach((node) => {
      if (node.getAttribute("data-vf-ready")) return;
      const build = widgets[node.getAttribute("data-vf-widget")];
      if (!build) return;
      node.setAttribute("data-vf-ready", "true");
      clear(node);
      build(node);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
  if (typeof document$ !== "undefined") document$.subscribe(init);
})(typeof window !== "undefined" ? window : globalThis);
