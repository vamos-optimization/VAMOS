import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;
import org.uma.jmetal.util.constrainthandling.ConstrainedDominance;
import org.uma.jmetal.util.constrainthandling.ConstraintHandlingTechnique;
import org.uma.jmetal.util.constrainthandling.EpsilonConstrained;
import org.uma.jmetal.util.constrainthandling.PushPullSearch;
import org.uma.jmetal.util.constrainthandling.SelfAdaptivePenalty;
import org.uma.jmetal.util.constrainthandling.StochasticRanking;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.pseudorandom.PseudoRandomGenerator;
import org.uma.jmetal.util.ranking.Ranking;

/**
 * Ranks of fixed populations by the constraint-handling techniques of jMetal (package
 * util.constrainthandling), for tools/constraint_handling_reference_ranks.py, which documents the
 * input read from standard input. Constraint values arrive in the VAMOS convention and are negated.
 */
public class ChtRanks {
  static final class Replay implements PseudoRandomGenerator {
    final double[] values;
    int next;
    Replay(double[] values) { this.values = values; }
    public double nextDouble() { return values[next++]; }
    public int nextInt(int lowerBound, int upperBound) { throw new UnsupportedOperationException(); }
    public double nextDouble(double lowerBound, double upperBound) { throw new UnsupportedOperationException(); }
    public void setSeed(long seed) {}
    public long getSeed() { return 0; }
    public String getName() { return "replay"; }
  }

  static Scanner in;

  static double next() { return Double.parseDouble(in.next()); }

  static List<DoubleSolution> read(int n, int m, int k) {
    List<DoubleSolution> list = new ArrayList<>();
    List<Bounds<Double>> bounds = List.of(Bounds.create(0.0, 1.0));
    for (int i = 0; i < n; i++) {
      DoubleSolution solution = new DefaultDoubleSolution(bounds, m, k);
      for (int j = 0; j < m; j++) solution.objectives()[j] = next();
      for (int j = 0; j < k; j++) solution.constraints()[j] = -next();
      list.add(solution);
    }
    return list;
  }

  public static void main(String[] args) throws Exception {
    in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    int cases = in.nextInt();
    // Solutions draw their variables when they are created: the replayed uniforms serve the ranking only.
    PseudoRandomGenerator original = JMetalRandom.getInstance().getRandomGenerator();
    StringBuilder out = new StringBuilder();
    for (int c = 0; c < cases; c++) {
      String name = in.next();
      double[] p = new double[in.nextInt()];
      for (int i = 0; i < p.length; i++) p[i] = next();
      int m = in.nextInt();
      int k = in.nextInt();
      ConstraintHandlingTechnique<DoubleSolution> technique = switch (name) {
        case "cdp" -> new ConstrainedDominance<>();
        case "violated_constraints" -> new ConstrainedDominance<>(ConstrainedDominance.ViolationMeasure.NUMBER_OF_VIOLATED_CONSTRAINTS);
        case "epsilon_constrained" -> new EpsilonConstrained<>(p[0], p[1], p[2]);
        case "push_pull" -> new PushPullSearch<>(p[0], p[1], p[2], p[3], p[4], p[5]);
        case "stochastic_ranking" -> new StochasticRanking<>(p[0]);
        case "self_adaptive_penalty" -> new SelfAdaptivePenalty<>();
        default -> throw new IllegalArgumentException(name);
      };
      int updates = in.nextInt();
      for (int u = 0; u < updates; u++) {
        double progress = next();
        int n = in.nextInt();
        technique.update(read(n, m, k), progress);
      }
      int n = in.nextInt();
      List<DoubleSolution> population = read(n, m, k);
      double[] uniforms = new double[in.nextInt()];
      for (int i = 0; i < uniforms.length; i++) uniforms[i] = next();
      Replay replay = new Replay(uniforms);
      JMetalRandom.getInstance().setRandomGenerator(replay);
      Ranking<DoubleSolution> ranking = technique.ranking().compute(population);
      JMetalRandom.getInstance().setRandomGenerator(original);
      out.append("ranks");
      for (DoubleSolution solution : population) out.append(' ').append(ranking.getRank(solution));
      out.append('\n');
      double epsilon = Double.NaN;
      String pushing = "na";
      if (technique instanceof EpsilonConstrained<DoubleSolution> e) epsilon = e.epsilon();
      if (technique instanceof PushPullSearch<DoubleSolution> pp) {
        epsilon = pp.epsilon();
        pushing = String.valueOf(pp.pushing());
      }
      out.append("state ").append(Double.toString(epsilon)).append(' ').append(pushing).append(' ').append(replay.next).append('\n');
    }
    System.out.print(out);
  }
}
