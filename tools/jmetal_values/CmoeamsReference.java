package org.uma.jmetal.algorithm.multiobjective.cmoeams;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import org.uma.jmetal.operator.crossover.impl.DifferentialEvolutionCrossover;
import org.uma.jmetal.operator.crossover.impl.DifferentialEvolutionCrossover.DE_VARIANT;
import org.uma.jmetal.operator.crossover.impl.SBXCrossover;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;
import org.uma.jmetal.util.pseudorandom.BoundedRandomGenerator;
import org.uma.jmetal.util.pseudorandom.RandomGenerator;

/**
 * Values of jMetal's CMOEA-MS (pull request jMetal/jMetal#526, whose classes CMOEAMSFitness and
 * CMOEAMSSelection and the static method CMOEAMS.isStageB are package-private, hence this package)
 * and of jMetal's DifferentialEvolutionCrossover and SBXCrossover on fixed inputs, for
 * tools/cmoeams_reference_values.py, which documents the input read from standard input. The random
 * numbers of the tournaments and of the operators are replayed from the input.
 */
public class CmoeamsReference {
  /** A generator that returns the given uniform numbers in order and counts them. */
  @SuppressWarnings("serial")
  private static final class Replay
      implements RandomGenerator<Double>, BoundedRandomGenerator<Double> {
    private final double[] values;
    private int next;

    Replay(double[] values) {
      this.values = values;
    }

    @Override
    public Double getRandomValue() {
      return values[next++];
    }

    @Override
    public Double getRandomValue(Double lowerBound, Double upperBound) {
      return values[next++];
    }
  }

  /** A generator that returns the given integers in order and counts them. */
  @SuppressWarnings("serial")
  private static final class IntegerReplay implements BoundedRandomGenerator<Integer> {
    private final int[] values;
    private int next;

    IntegerReplay(int[] values) {
      this.values = values;
    }

    @Override
    public Integer getRandomValue(Integer lowerBound, Integer upperBound) {
      return values[next++];
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

  private static void line(boolean[] values) {
    int[] flags = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      flags[i] = values[i] ? 1 : 0;
    }
    line(flags);
  }

  private static double[] doubles(Scanner in, int n) {
    double[] values = new double[n];
    for (int i = 0; i < n; i++) {
      values[i] = Double.parseDouble(in.next());
    }
    return values;
  }

  private static int[] integers(Scanner in, int n) {
    int[] values = new int[n];
    for (int i = 0; i < n; i++) {
      values[i] = in.nextInt();
    }
    return values;
  }

  private static double[][] matrix(Scanner in, int rows, int columns) {
    double[][] values = new double[rows][];
    for (int r = 0; r < rows; r++) {
      values[r] = doubles(in, columns);
    }
    return values;
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

  /**
   * Populations: the violation, the SDE, the fitness of both stages, the cosine distances, the
   * environmental selection of both stages at several sizes, the truncation at several sizes and
   * the tournaments of both stages on fixed draws.
   */
  private static void populations(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int n = in.nextInt();
      int m = in.nextInt();
      int k = in.nextInt();
      double[][] objectives = matrix(in, n, m);
      double[][] constraints = matrix(in, n, k);
      double[] violation = CMOEAMSFitness.constraintViolation(constraints);
      double[] stageA = CMOEAMSFitness.stageAFitness(objectives, violation);
      double[] stageB = CMOEAMSFitness.stageBFitness(objectives, violation);
      line(violation);
      line(CMOEAMSFitness.shiftBasedDensity(objectives));
      line(stageA);
      line(stageB);
      for (double[] row : CMOEAMSFitness.cosineDistances(objectives)) {
        line(row);
      }
      for (int size : integers(in, in.nextInt())) {
        line(CMOEAMSSelection.environmentalSelection(stageA, objectives, size));
        line(CMOEAMSSelection.environmentalSelection(stageB, objectives, size));
      }
      for (int removed : integers(in, in.nextInt())) {
        line(CMOEAMSSelection.truncation(objectives, removed));
      }
      int[] draws = integers(in, 2 * in.nextInt());
      line(CMOEAMSSelection.tournamentSelection(stageA, draws.length / 2, new IntegerReplay(draws)));
      line(CMOEAMSSelection.tournamentSelection(stageB, draws.length / 2, new IntegerReplay(draws)));
    }
  }

  /** Points with a violation: the fitness and the cosine distances. */
  private static void points(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int n = in.nextInt();
      int d = in.nextInt();
      double[][] points = matrix(in, n, d);
      double[] violation = doubles(in, n);
      line(CMOEAMSFitness.fitness(points, violation));
      for (double[] row : CMOEAMSFitness.cosineDistances(points)) {
        line(row);
      }
    }
  }

  /** The stage decisions: violations, lambda, evaluations and maximum evaluations. */
  private static void stageRule(Scanner in) {
    int cases = in.nextInt();
    int[] decisions = new int[cases];
    for (int c = 0; c < cases; c++) {
      double[] violation = doubles(in, in.nextInt());
      double lambda = Double.parseDouble(in.next());
      int evaluations = in.nextInt();
      int maxEvaluations = in.nextInt();
      decisions[c] = CMOEAMS.isStageB(violation, lambda, evaluations, maxEvaluations) ? 1 : 0;
    }
    line(decisions);
  }

  /** DifferentialEvolutionCrossover RAND_1_BIN (0) or RAND_1_EXP (1) on fixed draws. */
  private static void differentialEvolution(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int variant = in.nextInt();
      int rows = in.nextInt();
      int n = in.nextInt();
      double cr = Double.parseDouble(in.next());
      double f = Double.parseDouble(in.next());
      List<Bounds<Double>> bounds = bounds(doubles(in, n), doubles(in, n));
      int[] used = new int[rows];
      for (int r = 0; r < rows; r++) {
        DoubleSolution first = vector(bounds, doubles(in, n));
        DoubleSolution second = vector(bounds, doubles(in, n));
        DoubleSolution current = vector(bounds, doubles(in, n));
        IntegerReplay positions = new IntegerReplay(integers(in, 2));
        Replay draws = new Replay(doubles(in, n));
        DifferentialEvolutionCrossover crossover =
            new DifferentialEvolutionCrossover(
                cr, f, variant == 0 ? DE_VARIANT.RAND_1_BIN : DE_VARIANT.RAND_1_EXP, positions, draws);
        crossover.setCurrentSolution(current);
        line(variables(crossover.execute(List.of(first, second, current)).get(0)));
        used[r] = draws.next;
      }
      line(used);
    }
  }

  /** SBXCrossover on fixed pairs and draws, replayed in jMetal's order. */
  private static void sbx(Scanner in) {
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int pairs = in.nextInt();
      int n = in.nextInt();
      double probability = Double.parseDouble(in.next());
      double eta = Double.parseDouble(in.next());
      List<Bounds<Double>> bounds = bounds(doubles(in, n), doubles(in, n));
      List<DoubleSolution[]> parents = new ArrayList<>();
      for (int p = 0; p < pairs; p++) {
        parents.add(
            new DoubleSolution[] {vector(bounds, doubles(in, n)), vector(bounds, doubles(in, n))});
      }
      Replay draws = new Replay(doubles(in, in.nextInt()));
      SBXCrossover crossover = new SBXCrossover(probability, eta, draws);
      for (DoubleSolution[] pair : parents) {
        List<DoubleSolution> children = crossover.execute(List.of(pair[0], pair[1]));
        line(variables(children.get(0)));
        line(variables(children.get(1)));
      }
      OUT.append(draws.next).append('\n');
    }
  }

  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    populations(in);
    points(in);
    stageRule(in);
    differentialEvolution(in);
    sbx(in);
    System.out.print(OUT);
  }
}
