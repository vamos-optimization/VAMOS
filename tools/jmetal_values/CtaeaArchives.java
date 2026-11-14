package org.uma.jmetal.algorithm.multiobjective.ctaea.util;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.pseudorandom.PseudoRandomGenerator;

/**
 * Weight vectors, subregions, archive updates, mating proportions and tournament decisions of
 * jMetal's C-TAEA (pull request jMetal/jMetal#524) on fixed inputs, for
 * tools/ctaea_reference_archives.py, which documents the input read from standard input. The
 * worst solution of the truncation is package-private, hence this package. Each solution has one
 * constraint, the negated violation, so that its jMetal violation is the given one; the tournament
 * replays fixed draws through a generator that returns them in order.
 */
public class CtaeaArchives {
  /** A generator that returns the queued integers, so that a tournament replays fixed draws. */
  @SuppressWarnings("serial")
  private static final class Replay implements PseudoRandomGenerator {
    private final Deque<Integer> draws = new ArrayDeque<>();

    @Override
    public int nextInt(int lowerBound, int upperBound) {
      int value = draws.removeFirst();
      if (value < lowerBound || value > upperBound) {
        throw new IllegalStateException("draw " + value + " outside [" + lowerBound + ", " + upperBound + "]");
      }
      return value;
    }

    // The variables of a new solution, which no class of C-TAEA reads.
    @Override
    public double nextDouble(double lowerBound, double upperBound) {
      return lowerBound;
    }

    @Override
    public double nextDouble() {
      return 0.0;
    }

    @Override
    public void setSeed(long seed) {}

    @Override
    public long getSeed() {
      return 0;
    }

    @Override
    public String getName() {
      return "replay";
    }
  }

  private static String line(double[] values) {
    StringBuilder text = new StringBuilder();
    for (double value : values) {
      text.append(value).append(' ');
    }
    return text.toString().trim();
  }

  private static String line(int[] values) {
    StringBuilder text = new StringBuilder();
    for (int value : values) {
      text.append(value).append(' ');
    }
    return text.toString().trim();
  }

  private static String rows(List<DoubleSolution> selected, Map<DoubleSolution, Integer> index) {
    StringBuilder text = new StringBuilder();
    for (DoubleSolution solution : selected) {
      text.append(index.get(solution)).append(' ');
    }
    return text.toString().trim();
  }

  private static double[][] objectives(List<DoubleSolution> solutions) {
    double[][] matrix = new double[solutions.size()][];
    for (int i = 0; i < solutions.size(); i++) {
      matrix[i] = solutions.get(i).objectives().clone();
    }
    return matrix;
  }

  private static List<DoubleSolution> readSolutions(Scanner in, int m) {
    List<Bounds<Double>> bounds = List.of(Bounds.create(0.0, 1.0));
    int n = in.nextInt();
    List<DoubleSolution> solutions = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      DoubleSolution solution = new DefaultDoubleSolution(bounds, m, 1);
      for (int j = 0; j < m; j++) {
        solution.objectives()[j] = Double.parseDouble(in.next());
      }
      double violation = Double.parseDouble(in.next());
      solution.constraints()[0] = violation == 0.0 ? 0.0 : -violation;
      solutions.add(solution);
    }
    return solutions;
  }

  private static Map<DoubleSolution, Integer> indexOf(List<DoubleSolution> solutions) {
    Map<DoubleSolution, Integer> index = new IdentityHashMap<>();
    for (int i = 0; i < solutions.size(); i++) {
      index.put(solutions.get(i), i);
    }
    return index;
  }

  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    StringBuilder out = new StringBuilder();

    int weightCases = in.nextInt();
    for (int c = 0; c < weightCases; c++) {
      double[][] weights = UniformWeightVectors.generate(in.nextInt(), in.nextInt());
      out.append(weights.length).append('\n');
      for (double[] weight : weights) {
        out.append(line(weight)).append('\n');
      }
    }

    Replay replay = new Replay();
    JMetalRandom.getInstance().setRandomGenerator(replay);
    CTAEATournamentSelection<DoubleSolution> tournament = new CTAEATournamentSelection<>();
    int tournamentCases = in.nextInt();
    for (int c = 0; c < tournamentCases; c++) {
      int m = in.nextInt();
      List<DoubleSolution> archive = readSolutions(in, m);
      Map<DoubleSolution, Integer> index = indexOf(archive);
      int draws = in.nextInt();
      int[] winners = new int[draws];
      for (int k = 0; k < draws; k++) {
        replay.draws.clear();
        for (int d = 0; d < 3; d++) {
          replay.draws.addLast(in.nextInt());
        }
        winners[k] = index.get(tournament.execute(archive));
      }
      out.append(line(winners)).append('\n');
    }

    int archiveCases = in.nextInt();
    for (int c = 0; c < archiveCases; c++) {
      int requested = in.nextInt();
      int m = in.nextInt();
      double[][] weights = UniformWeightVectors.generate(requested, m);
      out.append(weights.length).append('\n');
      for (double[] weight : weights) {
        out.append(line(weight)).append('\n');
      }
      List<DoubleSolution> convergence = readSolutions(in, m);
      List<DoubleSolution> diversity = readSolutions(in, m);
      List<DoubleSolution> offspring = readSolutions(in, m);

      List<DoubleSolution> candidates = new ArrayList<>(convergence);
      candidates.addAll(offspring);
      List<DoubleSolution> diversityCandidates = new ArrayList<>(diversity);
      diversityCandidates.addAll(offspring);
      double[][] candidateObjectives = objectives(candidates);

      ConvergenceArchiveUpdate<DoubleSolution> convergenceUpdate = new ConvergenceArchiveUpdate<>(weights);
      out.append(line(Subregions.associate(candidateObjectives, weights))).append('\n');
      out.append(line(Subregions.tchebycheffInOwnSubregion(candidateObjectives, weights))).append('\n');
      out.append(indexOf(candidates).get(convergenceUpdate.worstOfTheMostCrowdedSubregion(candidates))).append('\n');

      List<DoubleSolution> newConvergence = convergenceUpdate.update(convergence, offspring);
      out.append(rows(newConvergence, indexOf(candidates))).append('\n');
      List<DoubleSolution> newDiversity =
          new DiversityArchiveUpdate<DoubleSolution>(weights).update(newConvergence, diversity, offspring);
      out.append(rows(newDiversity, indexOf(diversityCandidates))).append('\n');
      MatingProportions proportions = MatingProportions.of(newConvergence, newDiversity);
      out.append(
              line(
                  new double[] {
                    proportions.convergenceShare(),
                    proportions.diversityShare(),
                    proportions.nonDominatedShareOfConvergenceArchive()
                  }))
          .append('\n');
    }
    System.out.print(out);
  }
}
