package org.uma.jmetal.algorithm.multiobjective.top;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.mutation.impl.PolynomialMutation;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;
import org.uma.jmetal.util.pseudorandom.RandomGenerator;

/**
 * Values of jMetal's ToP (pull request jMetal/jMetal#525, whose classes ToPSelection and
 * ToPDifferentialEvolution and the static methods of ToP are package-private, hence this package)
 * on fixed inputs, for tools/top_reference_values.py, which documents the input read from standard
 * input. The random numbers of the operators are replayed from the input.
 */
public class TopReference {
  /** A generator that returns the given uniform numbers in order and counts them. */
  @SuppressWarnings("serial")
  private static final class Replay implements RandomGenerator<Double> {
    private final double[] values;
    private int next;

    Replay(double[] values) {
      this.values = values;
    }

    @Override
    public Double getRandomValue() {
      return values[next++];
    }
  }

  /** The mutation that leaves the trial vector unchanged. */
  @SuppressWarnings("serial")
  private static final class Identity implements MutationOperator<DoubleSolution> {
    @Override
    public DoubleSolution execute(DoubleSolution solution) {
      return solution;
    }

    @Override
    public double mutationProbability() {
      return 0.0;
    }
  }

  private static final StringBuilder OUT = new StringBuilder();

  private static void line(double[] values) {
    StringBuilder text = new StringBuilder();
    for (double value : values) {
      text.append(value).append(' ');
    }
    OUT.append(text.toString().trim()).append('\n');
  }

  private static void line(int[] values) {
    StringBuilder text = new StringBuilder();
    for (int value : values) {
      text.append(value).append(' ');
    }
    OUT.append(text.toString().trim()).append('\n');
  }

  private static double[] doubles(Scanner in, int n) {
    double[] values = new double[n];
    for (int i = 0; i < n; i++) {
      values[i] = Double.parseDouble(in.next());
    }
    return values;
  }

  /** n solutions with m objectives and one constraint, the negated violation. */
  private static List<DoubleSolution> violated(Scanner in, int n, int m) {
    List<Bounds<Double>> bounds = List.of(Bounds.create(0.0, 1.0));
    List<DoubleSolution> solutions = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      DoubleSolution solution = new DefaultDoubleSolution(bounds, m, 1);
      double[] objectives = doubles(in, m);
      System.arraycopy(objectives, 0, solution.objectives(), 0, m);
      double violation = Double.parseDouble(in.next());
      solution.constraints()[0] = violation == 0.0 ? 0.0 : -violation;
      solutions.add(solution);
    }
    return solutions;
  }

  /** n solutions with m objectives and k constraints in jMetal's convention (negative is violated). */
  private static List<DoubleSolution> constrained(Scanner in, int n, int m, int k) {
    List<Bounds<Double>> bounds = List.of(Bounds.create(0.0, 1.0));
    List<DoubleSolution> solutions = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      DoubleSolution solution = new DefaultDoubleSolution(bounds, m, k);
      System.arraycopy(doubles(in, m), 0, solution.objectives(), 0, m);
      System.arraycopy(doubles(in, k), 0, solution.constraints(), 0, k);
      solutions.add(solution);
    }
    return solutions;
  }

  private static List<Bounds<Double>> bounds(double[] lower, double[] upper) {
    List<Bounds<Double>> bounds = new ArrayList<>();
    for (int j = 0; j < lower.length; j++) {
      bounds.add(Bounds.create(lower[j], upper[j]));
    }
    return bounds;
  }

  private static DoubleSolution vector(List<Bounds<Double>> bounds, double[] values) {
    DoubleSolution solution = new DefaultDoubleSolution(bounds, 1, 0);
    for (int j = 0; j < values.length; j++) {
      solution.variables().set(j, values[j]);
    }
    return solution;
  }

  private static double[] variables(DoubleSolution solution) {
    return solution.variables().stream().mapToDouble(Double::doubleValue).toArray();
  }

  private static void selection(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int n = in.nextInt();
      int m = in.nextInt();
      List<DoubleSolution> solutions = violated(in, n, m);
      Map<DoubleSolution, Integer> rows = new IdentityHashMap<>();
      for (int i = 0; i < n; i++) {
        rows.put(solutions.get(i), i);
      }
      int[] fronts = ToPSelection.frontNumbers(solutions);
      double[] distances = ToPSelection.crowdingDistances(solutions, fronts);
      line(fronts);
      line(distances);
      int sizes = in.nextInt();
      for (int s = 0; s < sizes; s++) {
        int size = in.nextInt();
        List<DoubleSolution> selected = ToPSelection.select(solutions, size);
        line(selected.stream().mapToInt(rows::get).toArray());
      }
      int draws = in.nextInt();
      int[] winners = new int[draws];
      for (int d = 0; d < draws; d++) {
        winners[d] = ToPSelection.binaryTournament(in.nextInt(), in.nextInt(), fronts, distances);
      }
      line(winners);
    }
  }

  private static void differentialEvolution(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int operator = in.nextInt();
      int trials = in.nextInt();
      int n = in.nextInt();
      double cr = Double.parseDouble(in.next());
      double f = Double.parseDouble(in.next());
      List<Bounds<Double>> bounds = bounds(doubles(in, n), doubles(in, n));
      int vectors = operator == 2 ? 3 : 4;
      List<DoubleSolution[]> groups = new ArrayList<>();
      for (int t = 0; t < trials; t++) {
        DoubleSolution[] group = new DoubleSolution[vectors];
        for (int v = 0; v < vectors; v++) {
          group[v] = vector(bounds, doubles(in, n));
        }
        groups.add(group);
      }
      Replay draws = new Replay(doubles(in, trials * n));
      ToPDifferentialEvolution de = new ToPDifferentialEvolution(cr, f, new Identity(), draws);
      for (DoubleSolution[] g : groups) {
        DoubleSolution trial =
            switch (operator) {
              case 0 -> de.currentToRand1(g[0], g[1], g[2], g[3]);
              case 1 -> de.randToBest1(g[0], g[1], g[2], g[3]);
              default -> de.rand1(g[0], g[1], g[2]);
            };
        line(variables(trial));
      }
      OUT.append(draws.next).append('\n');
    }
  }

  private static void polynomialMutation(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int rows = in.nextInt();
      int n = in.nextInt();
      double probability = Double.parseDouble(in.next());
      double eta = Double.parseDouble(in.next());
      List<Bounds<Double>> bounds = bounds(doubles(in, n), doubles(in, n));
      List<DoubleSolution> solutions = new ArrayList<>();
      for (int r = 0; r < rows; r++) {
        solutions.add(vector(bounds, doubles(in, n)));
      }
      Replay draws = new Replay(doubles(in, in.nextInt()));
      PolynomialMutation mutation = new PolynomialMutation(probability, eta, draws);
      for (DoubleSolution solution : solutions) {
        line(variables(mutation.execute(solution)));
      }
      OUT.append(draws.next).append('\n');
    }
  }

  private static void phaseOne(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int n = in.nextInt();
      int m = in.nextInt();
      int k = in.nextInt();
      double delta = Double.parseDouble(in.next());
      List<DoubleSolution> population = constrained(in, n, m, k);
      List<DoubleSolution> offspring = constrained(in, n, m, k);
      line(population.stream().mapToDouble(ToP::constraintViolation).toArray());
      line(offspring.stream().mapToDouble(ToP::constraintViolation).toArray());
      line(population.stream().mapToDouble(ToP::meanObjective).toArray());
      OUT.append(ToP.indexOfLowestMeanObjective(population)).append('\n');
      OUT.append(ToP.feasibleRatio(population)).append('\n');
      OUT.append(ToP.updatedDelta(population, delta)).append('\n');
      int[] replaces = new int[n];
      for (int i = 0; i < n; i++) {
        replaces[i] = ToP.offspringReplacesParent(offspring.get(i), population.get(i)) ? 1 : 0;
      }
      line(replaces);
      List<DoubleSolution> next = ToP.oneToOneReplacement(population, offspring);
      int[] kept = new int[n];
      for (int i = 0; i < n; i++) {
        kept[i] = next.get(i) == offspring.get(i) ? 1 : 0;
      }
      line(kept);
      OUT.append(ToP.feasibleRatio(next)).append('\n');
      OUT.append(ToP.updatedDelta(next, delta)).append('\n');
    }
  }

  private static void phaseRule(Scanner in) {
    int cases = in.nextInt();
    int[] decisions = new int[cases];
    for (int c = 0; c < cases; c++) {
      double delta = Double.parseDouble(in.next());
      double feasibleRatio = Double.parseDouble(in.next());
      int evaluations = in.nextInt();
      int maxEvaluations = in.nextInt();
      decisions[c] = ToP.phaseOneContinues(delta, feasibleRatio, evaluations, maxEvaluations) ? 1 : 0;
    }
    line(decisions);
  }

  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    selection(in);
    differentialEvolution(in);
    polynomialMutation(in);
    phaseOne(in);
    phaseRule(in);
    System.out.print(OUT);
  }
}
