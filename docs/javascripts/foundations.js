/* Interactive figures for the Foundations learning path.

   Dependency-free DOM and SVG. Every figure is a progressive enhancement: the
   lesson text explains the idea without JavaScript, and each container keeps a
   short fallback paragraph until its figure is drawn. The algorithms below
   follow the NumPy reference implementation of VAMOS
   (src/vamos/foundation/kernel/numpy_backend.py and operator_primitives.py,
   and src/vamos/foundation/quality_indicators/hypervolume.py).
   tests/docs/test_foundations.py checks that every figure a lesson uses is
   implemented here and, when Node.js is installed, runs the non-dominated
   sorting, crowding distance, hypervolume and hypervolume contributions
   against VAMOS. The distance indicators and the Das-Dennis lattices were
   checked by hand against MooCore and VAMOS when the path was added. */
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

  /** SPEA2 strength S, raw fitness R, density D = 1 / (σᵏ + 2) and fitness F = R + D. */
  function spea2Fitness(F, k) {
    const n = F.length;
    const S = F.map((a) => F.filter((b) => dominates(a, b)).length);
    const R = F.map((b, i) => F.reduce((sum, a, j) => (j !== i && dominates(a, b) ? sum + S[j] : sum), 0));
    const kk = Math.max(1, Math.min(n - 1, k || Math.floor(Math.sqrt(n))));
    const D = F.map((a, i) => {
      const d = F.map((b, j) => (j === i ? Infinity : euclidean(a, b))).sort((x, y) => x - y);
      return 1 / (d[kk - 1] + 2);
    });
    return { S, R, D, fitness: R.map((r, i) => r + D[i]), k: kk };
  }

  /** SPEA2 environmental selection: non-dominated first, then truncation by nearest neighbours. */
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
    while (archive.length > size) {
      const lists = archive.map((i) =>
        archive
          .filter((j) => j !== i)
          .map((j) => euclidean(F[i], F[j]))
          .sort((a, b) => a - b)
      );
      let worst = 0;
      for (let a = 1; a < archive.length; a++) if (lexLess(lists[a], lists[worst])) worst = a;
      removed.push(archive[worst]);
      archive.splice(worst, 1);
    }
    return { kept: archive, removed, filled, nd, ...fit };
  }

  /** Additive epsilon indicator I(a, b): smallest shift of a that weakly dominates b. */
  function epsilonPair(a, b) {
    let m = -Infinity;
    for (let i = 0; i < a.length; i++) m = Math.max(m, a[i] - b[i]);
    return m;
  }

  /** IBEA environmental selection (VAMOS' ibea/helpers.py): additive epsilon or hypervolume difference. */
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
    while (alive.length > size) {
      let worst = alive[0];
      for (const i of alive) if (fitness[i] < fitness[worst]) worst = i;
      alive.splice(alive.indexOf(worst), 1);
      removed.push(worst);
      for (const x of alive) fitness[x] += T[x][worst];
    }
    return { fitness: initial, final: fitness, kept: alive, removed, c: c || 1, terms: T };
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

  /** One MOEA/D replacement step: strict improvement, at most nr replacements, in the given order. */
  function moeadReplace(current, W, y, order, nr, mode, theta) {
    const z = current[0].map((_, m) => Math.min(y[m], ...current.map((f) => f[m])));
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

  /** NSGA-III niche preservation, returning every pick with its round and reason. */
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
      const Jmin = shuffle(J.filter((j) => rho[j] === low)).slice(0, K - picks.length);
      for (const j of Jmin) {
        const C = shuffle(left.filter((c) => c.dir === j));
        let chosen = C[0];
        if (rho[j] === 0) for (const c of C) if (c.dperp < chosen.dperp) chosen = c;
        picks.push({ id: chosen.id, dir: j, round, count: rho[j], low, Jmin: Jmin.slice(), closest: rho[j] === 0 });
        left.splice(left.indexOf(chosen), 1);
        rho[j] += 1;
        if (picks.length >= K) break;
      }
    }
    return { picks, rho };
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
        p = !Number.isFinite(est) || est <= 0.1 ? 1 : Math.min(est, 20);
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
    return { p, estimated, central, corners, normalized: Nf, intercepts: a, proximity, score, order };
  }

  /** The N survivors of a last front by survival score (descending; later index first on ties). */
  function agemoeaCut(score, N) {
    return score
      .map((v, i) => i)
      .sort((x, y) => (score[y] === score[x] ? y - x : score[y] > score[x] ? 1 : -1))
      .slice(0, N);
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

  const CTAEA_W = (count) =>
    Array.from({ length: count }, (_, k) => {
      const a = k / (count - 1);
      return [Math.max(a, 1e-6), Math.max(1 - a, 1e-6)];
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

  /** C-TAEA convergence archive and diversity archive updates (ctaea/archives.py). */
  function ctaeaSelect(F, cv, n) {
    const W = CTAEA_W(n);
    const idx = F.map((_, i) => i);
    const feasible = idx.filter((i) => cv[i] <= 0);
    const ca = { removals: [], fronts: null, normalized: null };
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
        const closest = members.filter((i) => members.some((j) => j !== i && Math.abs(euclidean(F[i], F[j]) - dmin) < 1e-15));
        const tch = ctaeaTchebycheff(
          closest.map((i) => F[i]),
          W
        );
        let worst = 0;
        tch.forEach((v, a) => {
          if (v > tch[worst]) worst = a;
        });
        ca.normalized = ca.normalized || CA.map((i, a) => ({ i, f: norm[a], region: region[a] }));
        ca.removals.push({ counts, crowded, closest, tch, removed: closest[worst], ideal: [0, 1].map((j) => Math.min(...closest.map((i) => F[i][j]))) });
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
      for (const front of fronts) {
        if (CA.length >= n) break;
        if (CA.length + front.length <= n) CA.push(...front);
        else
          CA.push(
            ...front
              .slice()
              .sort((a, b) => cv[a] - cv[b] || a - b)
              .slice(0, n - CA.length)
          );
      }
    }
    const regions = F.map((f) => ctaeaAssociate(f, W));
    const caCounts = W.map((_, k) => CA.filter((i) => regions[i] === k).length);
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
    return { W, CA, DA, ca, caCounts, regions, trace, feasible };
  }

  /** ToP phase 1 (pairwise replacement on the mean objective) and phase 2 (constrained NSGA-II survival). */
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
    if (feas.length >= 3) {
      const lo = [0, 1].map((j) => Math.min(...feas.map((i) => F[i][j])));
      const hi = [0, 1].map((j) => Math.max(...feas.map((i) => F[i][j])));
      const sums = feas.map((i) => F[i].reduce((s2, v, j) => s2 + (v - lo[j]) / (hi[j] - lo[j]), 0)).sort((a, b) => a - b);
      delta = sums[Math.floor(feas.length / 3) - 1] - sums[0];
    }
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
    return { pairs, next, Pf, delta, phaseOneContinues: delta >= 0.2 || Pf <= 1 / 3, fronts, crowd, kept: kept.sort((a, b) => a - b) };
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

  /** Ma and Wang's ShiP objectives, as VAMOS's shift_penalty_objectives. */
  function shiftPenaltyObjectives(F, G, constant) {
    const measure = overallViolation(G);
    const feasible = measure.map((v) => v <= 0);
    const shifted = F.map((f) => f.slice());
    const k = feasible.filter(Boolean).length;
    if (!F.length || k === 0 || k === F.length) return shifted;
    const share = k / F.length;
    for (let i = 0; i < F[0].length; i++) {
      const values = F.filter((_, r) => feasible[r]).map((f) => f[i]);
      const largest = Math.max(...F.map((f) => f[i]));
      F.forEach((f, r) => {
        if (feasible[r]) return;
        const own = f[i];
        const above = values.filter((v) => v > own);
        const nadir = above.length ? Math.min(...above) : largest; // the local feasible nadir
        shifted[r][i] = own + constant * share * (nadir - own) + measure[r];
      });
    }
    return shifted;
  }

  /** Ma, Wang and Song's ToR: positions by constrained domination and by Pareto dominance, their fitness and VAMOS's ranks. */
  function twoRankings(F, G) {
    const n = F.length;
    const measure = overallViolation(G);
    const rows = Array.from({ length: n }, (_, i) => i);
    const positions = (levels) => {
      const distance = new Array(n).fill(0);
      for (const level of new Set(levels)) {
        const members = rows.filter((i) => levels[i] === level);
        crowdingDistance(F, members).forEach((d, a) => (distance[members[a]] = d));
      }
      const desc = (a, b) => (distance[a] === distance[b] ? 0 : distance[a] > distance[b] ? -1 : 1);
      const order = rows.slice().sort((a, b) => levels[a] - levels[b] || desc(a, b) || a - b);
      const pos = new Array(n);
      order.forEach((row, a) => (pos[row] = a + 1));
      return pos;
    };
    const Rc = positions(ranksByViolation(F, measure, true));
    const Rp = positions(paretoRanks(F));
    const k = measure.filter((v) => v <= 0).length;
    // Twice n times the fitness, an integer: n (R^c + R^p) + k (R^c - R^p).
    const doubled = Rc.map((rc, i) => n * (rc + Rp[i]) + k * (rc - Rp[i]));
    const ranks = new Array(n);
    rows
      .slice()
      .sort((a, b) => doubled[a] - doubled[b] || Rc[a] - Rc[b])
      .forEach((row, a) => (ranks[row] = a));
    return { ranks, Rc, Rp, alpha: 0.5 + (0.5 * k) / n, fitness: doubled.map((d) => d / (2 * n)) };
  }

  /** Rows sorted by their fronts among themselves, then by crowding distance in descending order, then by index. */
  function frontOrder(values, rows) {
    if (!rows.length) return [];
    const { rank, crowding } = rankAndCrowding(rows.map((i) => values[i]));
    const desc = (a, b) => (crowding[a] === crowding[b] ? 0 : crowding[a] > crowding[b] ? -1 : 1);
    return rows
      .map((_, a) => a)
      .sort((a, b) => rank[a] - rank[b] || desc(a, b) || a - b)
      .map((a) => rows[a]);
  }

  function positionsOf(order) {
    const ranks = new Array(order.length);
    order.forEach((row, a) => (ranks[row] = a));
    return ranks;
  }

  /** IDEA's violation measure: per constraint, the dense rank of each violation from 1 (0 if satisfied), summed. */
  function violationMeasure(G) {
    const measure = new Array(G.length).fill(0);
    for (let j = 0; j < (G[0] || []).length; j++) {
      const levels = Array.from(new Set(G.map((g) => g[j]).filter((v) => v > 0))).sort((a, b) => a - b);
      G.forEach((g, r) => {
        if (g[j] > 0) measure[r] += levels.indexOf(g[j]) + 1;
      });
    }
    return measure;
  }

  /** Ray, Singh, Isaacs and Smith's IDEA, as VAMOS orders it: the best round(proportion N) infeasible rows first. */
  function infeasibilityDriven(F, G, proportion, populationSize) {
    const measure = violationMeasure(G);
    const extended = F.map((f, i) => f.concat([measure[i]]));
    const rows = F.map((_, i) => i);
    const infeasible = frontOrder(extended, rows.filter((i) => measure[i] > 0));
    const feasible = frontOrder(extended, rows.filter((i) => measure[i] === 0));
    const quota = Math.floor(proportion * populationSize + 0.5);
    const order = infeasible.slice(0, quota).concat(feasible, infeasible.slice(quota));
    return { ranks: positionsOf(order), measure, quota };
  }

  /** Li et al.'s normalized violation C: the mean over the constraints of each violation over the largest one. */
  function normalizedViolation(G) {
    const c = (G[0] || []).length;
    const excess = G.map((g) => g.map((v) => Math.max(0, v)));
    const largest = Array.from({ length: c }, (_, j) => Math.max(...excess.map((e) => e[j])));
    return excess.map((e) => (c ? e.reduce((acc, v, j) => acc + (largest[j] > 0 ? v / largest[j] : 0), 0) / c : 0));
  }

  /** ATM's fitness with feasible and infeasible rows (Li et al., eqs. 10-15), as VAMOS's adaptive_tradeoff_fitness. */
  function adaptiveTradeoffFitness(F, C, phi) {
    const m = F[0].length;
    const feasible = C.map((v) => v <= 0);
    const fea = F.filter((_, i) => feasible[i]);
    const floor = Array.from({ length: m }, (_, k) => phi * Math.min(...fea.map((f) => f[k])) + (1 - phi) * Math.max(...fea.map((f) => f[k])));
    const converted = F.map((f, i) => (feasible[i] ? f.slice() : f.map((v, k) => Math.max(floor[k], v))));
    const low = Array.from({ length: m }, (_, k) => Math.min(...converted.map((f) => f[k])));
    const span = Array.from({ length: m }, (_, k) => Math.max(...converted.map((f) => f[k])) - low[k]);
    const infC = C.filter((_, i) => !feasible[i]);
    const lowest = Math.min(...infC);
    const spread = Math.max(...infC) - lowest;
    return converted.map((f, i) => {
      const penalty = !feasible[i] && spread > 0 ? (C[i] - lowest) / spread : 0;
      return f.map((v, k) => (span[k] > 0 ? (v - low[k]) / span[k] : 0) + penalty);
    });
  }

  /** Wang, Cai, Zhou and Zeng's ATM in Li, Wang, Yang and Cai's version: three phases by the feasible share. */
  function adaptiveTradeoff(F, G, phi) {
    const C = normalizedViolation(G);
    const rows = F.map((_, i) => i);
    const feasibleCount = C.filter((v) => v <= 0).length;
    if (feasibleCount === F.length) return { ranks: positionsOf(frontOrder(F, rows)), C, phase: 3 };
    if (feasibleCount > 0) {
      const fitness = adaptiveTradeoffFitness(F, C, phi);
      return { ranks: positionsOf(frontOrder(fitness, rows)), C, phase: 2, fitness };
    }
    const values = F.map((f, i) => f.concat([C[i]]));
    let remaining = rows.slice();
    const order = [];
    while (remaining.length) {
      const r = paretoRanks(remaining.map((i) => values[i]));
      const leading = remaining.filter((_, a) => r[a] === 0).sort((a, b) => C[a] - C[b] || remaining.indexOf(a) - remaining.indexOf(b));
      const taken = leading.slice(0, Math.ceil(leading.length / 2));
      order.push(...taken);
      remaining = remaining.filter((i) => !taken.includes(i));
    }
    return { ranks: positionsOf(order), C, phase: 1 };
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
    if (name === "two_rankings") return twoRankings(F, G);
    if (name === "infeasibility_driven") return infeasibilityDriven(F, G, o.proportion, o.populationSize);
    if (name === "adaptive_tradeoff") return adaptiveTradeoff(F, G, o.phi);
    let objectives;
    if (name === "self_adaptive_penalty") objectives = selfAdaptivePenaltyObjectives(F, G);
    else if (name === "static_penalty") objectives = F.map((f, i) => f.map((v) => v + o.coefficient * cv[i]));
    else if (name === "dynamic_penalty") {
      const weight = Math.pow(o.constant * o.generation, o.alpha);
      objectives = F.map((f, i) => {
        const severity = G[i].reduce((acc, g) => acc + Math.pow(Math.max(0, g), o.beta), 0);
        return f.map((v) => v + weight * severity);
      });
    } else if (name === "violation_objective") objectives = F.map((f, i) => f.concat([cv[i]]));
    else if (name === "shift_penalty") objectives = shiftPenaltyObjectives(F, G, o.constant);
    else throw new Error("Unknown constraint-handling technique " + name);
    return { ranks: paretoRanks(objectives), objectives };
  }

  const algorithms = {
    ccmoSelect,
    ctaeaSelect,
    topSelect,
    cmoeamsSelect,
    CS_CANDIDATES,
    agemoeaSurvival,
    agemoeaCut,
    smpsoStep,
    nsga3Associate,
    nsga3Niching,
    moeadWeights,
    moeadNeighbours,
    moeadAggregate,
    moeadReplace,
    spea2Fitness,
    spea2Select,
    epsilonPair,
    ibeaSelect,
    rveaSelect,
    mulberry32,
    dominates,
    nonDominatedSort,
    crowdingDistance,
    rankAndCrowding,
    selectSurvivors,
    nonDominatedIndices,
    hypervolume2d,
    hvContributions2d,
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
        txt(g, x, pc.y(1) - 5, tickFormat(+(lo[i] + span[i]).toFixed(2)), { "text-anchor": "middle" });
        txt(g, x, pc.y(0) + 13, tickFormat(+lo[i].toFixed(2)), { "text-anchor": "middle" });
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
    const COEFFICIENTS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 100];
    const ALL = "NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA, RVEA and SMPSO";
    const FRONT = "NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA";
    const techniques = [
      ["cdp", "cdp: constrained domination (default)", ALL],
      ["violated_constraints", "violated_constraints", ALL],
      ["epsilon_constrained", "epsilon_constrained", ALL],
      ["push_pull", "push_pull", ALL],
      ["stochastic_ranking", "stochastic_ranking", FRONT],
      ["self_adaptive_penalty", "self_adaptive_penalty", FRONT],
      ["static_penalty", "static_penalty", FRONT],
      ["dynamic_penalty", "dynamic_penalty", FRONT],
      ["violation_objective", "violation_objective", FRONT],
      ["two_rankings", "two_rankings (ToR)", "NSGA-II only"],
      ["shift_penalty", "shift_penalty (ShiP)", FRONT],
      ["infeasibility_driven", "infeasibility_driven (IDEA)", "NSGA-II only"],
      ["adaptive_tradeoff", "adaptive_tradeoff (ATM)", "NSGA-II only"],
    ];
    const state = { name: "cdp", epsilon: 0.25, pushing: true, probability: 0.45, seed: 1, coefficient: 3, generation: 1, constant: 2, proportion: 0.2, phi: 0.5 };
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
          onChange: (v) => ((state.pushing = v === "push"), params(), render()),
        });
      }
      if (nm === "epsilon_constrained" || (nm === "push_pull" && !state.pushing)) {
        slider(extra, { label: "Current ε", min: 0, max: 1.2, step: 0.01, value: state.epsilon, format: (v) => v.toFixed(2), onInput: (v) => ((state.epsilon = v), render()) });
      }
      if (nm === "stochastic_ranking") {
        slider(extra, { label: "probability p_f", min: 0, max: 1, step: 0.05, value: state.probability, format: (v) => v.toFixed(2), onInput: (v) => ((state.probability = v), render()) });
        button(extra, "Draw again", () => ((state.seed += 1), render()));
      }
      if (nm === "static_penalty") {
        slider(extra, { label: "coefficient", min: 0, max: COEFFICIENTS.length - 1, step: 1, value: state.coefficient, format: (v) => String(COEFFICIENTS[v]), onInput: (v) => ((state.coefficient = v), render()) });
      }
      if (nm === "dynamic_penalty") {
        slider(extra, { label: "Generation k", min: 1, max: 30, step: 1, value: state.generation, onInput: (v) => ((state.generation = v), render()) });
      }
      if (nm === "shift_penalty") {
        slider(extra, { label: "constant c", min: 0.5, max: 5, step: 0.5, value: state.constant, format: (v) => v.toFixed(1), onInput: (v) => ((state.constant = v), render()) });
      }
      if (nm === "infeasibility_driven") {
        slider(extra, { label: "proportion", min: 0.1, max: 0.5, step: 0.1, value: state.proportion, format: (v) => v.toFixed(1), onInput: (v) => ((state.proportion = v), render()) });
      }
      if (nm === "adaptive_tradeoff") {
        slider(extra, { label: "φ, feasible share of the last survivors", min: 0, max: 1, step: 0.1, value: state.phi, format: (v) => v.toFixed(1), onInput: (v) => ((state.phi = v), render()) });
      }
    }
    function options() {
      const rand = mulberry32(1000 + state.seed);
      return {
        epsilon: state.epsilon,
        pushing: state.pushing,
        probability: state.probability,
        uniforms: Array.from({ length: n * (n - 1) }, rand),
        coefficient: COEFFICIENTS[state.coefficient],
        constant: state.name === "dynamic_penalty" ? 0.5 : state.constant,
        alpha: 2,
        beta: 2,
        generation: state.generation,
        proportion: state.proportion,
        populationSize: n,
        phi: state.phi,
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
      if (nm === "self_adaptive_penalty") return `${feasibleCount} of ${n} solutions are feasible (r_f = ${fmt(feasibleCount / n, 2)}). Pareto dominance ranks d(x) + (1 − r_f) X(x) + r_f Y(x) per objective, from normalized objectives and violations.`;
      if (nm === "static_penalty") return `Pareto dominance on fᵢ + ${o.coefficient} · CV: the coefficient decides whether a violation outweighs better objectives.`;
      if (nm === "dynamic_penalty") return `Weight (C · k)^α = (0.5 · ${o.generation})² = ${fmt(Math.pow(0.5 * o.generation, 2), 2)}; Pareto dominance on fᵢ + weight · Σ max(0, g_j)². The weight grows with the generation.`;
      if (nm === "violation_objective") return "Pareto dominance on (f₁, f₂, CV): an infeasible solution with good objectives is non-dominated.";
      if (nm === "two_rankings") return `P_fea = ${fmt(feasibleCount / n, 2)}, α = 0.5 + 0.5 P_fea = ${fmt(res.alpha, 2)}, fitness = α R^c + (1 − α) R^p; VAMOS gives each solution a front of its own, in order of fitness.`;
      if (nm === "infeasibility_driven") return `With a population of N = ${n}, the best ${res.quota} infeasible solutions (proportion · N, rounded), sorted by fronts and crowding over (f₁, f₂, violation measure), come first; then the feasible ones; then the other infeasible ones. One solution per front.`;
      if (nm === "adaptive_tradeoff") return `${feasibleCount} of ${n} solutions are feasible: phase two. Each objective of an infeasible solution becomes max(φ f_best + (1 − φ) f_worst, f), from the feasible ones, with φ = ${fmt(o.phi, 1)}; objectives are normalized, the normalized violation of the infeasible ones is added, and fronts and crowding order the sums. One solution per front.`;
      return `P_fea = ${fmt(feasibleCount / n, 2)}: each infeasible objective moves c · P_fea = ${fmt(o.constant * (feasibleCount / n), 2)} of the way to the smallest feasible value above it, and CV is added.`;
    }
    function render() {
      const o = options();
      const res = constraintRanks(state.name, F, G, o);
      clear(plot.layers.mid);
      clear(plot.layers.marks);
      clear(plot.layers.top);
      const moved = res.objectives && res.objectives[0].length === 2 ? res.objectives : null;
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
      const taken = techniques.find((t) => t[0] === state.name)[2];
      h("p", {}, readout, describe(res, o));
      h("p", {}, readout, `Taken by: ${taken}.`);
      const head = ["Solution", "f₁, f₂", "CV", "Violated"];
      const numeric = [false, false, true, true];
      if (moved) {
        head.push("Ranked as");
        numeric.push(false);
      } else if (state.name === "two_rankings") {
        head.push("R^c", "R^p", "Fitness");
        numeric.push(true, true, true);
      } else if (state.name === "violation_objective" || state.name === "adaptive_tradeoff") {
        head.push("Ranked as");
        numeric.push(false);
      } else if (state.name === "infeasibility_driven") {
        head.push("Violation measure");
        numeric.push(true);
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
          else if (state.name === "two_rankings") row.push(String(res.Rc[i]), String(res.Rp[i]), fmt(res.fitness[i], 2));
          else if (state.name === "violation_objective") row.push(`(${fmt(F[i][0], 2)}, ${fmt(F[i][1], 2)}, ${fmt(cv[i], 2)})`);
          else if (state.name === "adaptive_tradeoff") row.push(pair(res.fitness[i]));
          else if (state.name === "infeasibility_driven") row.push(String(res.measure[i]));
          row.push(String(res.ranks[i] + 1));
          return row;
        }),
        numeric
      );
    }
    params();
    render();
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
