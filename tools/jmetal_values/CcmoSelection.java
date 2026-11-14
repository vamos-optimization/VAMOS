package org.uma.jmetal.algorithm.multiobjective.ccmo;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;

/**
 * Fitness and environmental selection of jMetal's CCMO (pull request jMetal/jMetal#521, whose
 * classes CCMOFitness and CCMOEnvironmentalSelection are package-private, hence this package) on
 * fixed populations, for tools/ccmo_reference_selection.py, which documents the input read from
 * standard input. Each solution has one constraint, the negated violation, so that its jMetal
 * violation is the given one.
 */
public class CcmoSelection {
  private static String line(double[] values) {
    StringBuilder text = new StringBuilder();
    for (double value : values) {
      text.append(value).append(' ');
    }
    return text.toString().trim();
  }

  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    List<Bounds<Double>> bounds = List.of(Bounds.create(0.0, 1.0));
    StringBuilder out = new StringBuilder();
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      int n = in.nextInt();
      int m = in.nextInt();
      double[][] objectives = new double[n][m];
      double[] violations = new double[n];
      List<DoubleSolution> solutions = new ArrayList<>();
      Map<DoubleSolution, Integer> rows = new IdentityHashMap<>();
      for (int i = 0; i < n; i++) {
        DoubleSolution solution = new DefaultDoubleSolution(bounds, m, 1);
        for (int j = 0; j < m; j++) {
          objectives[i][j] = Double.parseDouble(in.next());
          solution.objectives()[j] = objectives[i][j];
        }
        violations[i] = Double.parseDouble(in.next());
        solution.constraints()[0] = violations[i] == 0.0 ? 0.0 : -violations[i];
        solutions.add(solution);
        rows.put(solution, i);
      }
      out.append(line(CCMOFitness.compute(solutions, true))).append('\n');
      out.append(line(CCMOFitness.compute(solutions, false))).append('\n');
      int selections = in.nextInt();
      for (int s = 0; s < selections; s++) {
        int size = in.nextInt();
        boolean withConstraints = in.nextInt() == 1;
        var result = CCMOEnvironmentalSelection.execute(solutions, size, withConstraints);
        StringBuilder selected = new StringBuilder();
        for (DoubleSolution solution : result.population()) {
          selected.append(rows.get(solution)).append(' ');
        }
        out.append(selected.toString().trim()).append('\n');
        out.append(line(result.fitness())).append('\n');
      }
    }
    System.out.print(out);
  }
}
