import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import org.uma.jmetal.component.catalogue.ea.replacement.impl.agemoea.AGEMOEA2EnvironmentalSelection;
import org.uma.jmetal.component.catalogue.ea.replacement.impl.agemoea.AGEMOEAEnvironmentalSelection;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;

/**
 * Values of jMetal's AGE-MOEA-II environmental selection (package
 * component.catalogue.ea.replacement.impl.agemoea), for tools/agemoea2_reference_values.py, which
 * documents the input read from standard input. As a subclass, it reaches the two steps that
 * AGE-MOEA-II overrides: the exponent of a normalized front and its pairwise distances.
 */
public class AgeMoea2Values extends AGEMOEA2EnvironmentalSelection<DoubleSolution> {
  static Scanner in;

  AgeMoea2Values(int numberOfObjectives) {
    super(numberOfObjectives);
  }

  static List<double[]> rows(int m, int n) {
    List<double[]> rows = new ArrayList<>();
    for (int i = 0; i < m; i++) {
      double[] row = new double[n];
      for (int j = 0; j < n; j++) row[j] = Double.parseDouble(in.next());
      rows.add(row);
    }
    return rows;
  }

  static void print(double value) {
    System.out.print(String.format(Locale.ROOT, " %.17g", value));
  }

  static double[] scores(AGEMOEAEnvironmentalSelection<DoubleSolution> selection, List<double[]> objectives, int n) {
    List<DoubleSolution> front = new ArrayList<>();
    for (double[] row : objectives) {
      DoubleSolution solution = new DefaultDoubleSolution(List.of(Bounds.create(0.0, 1.0)), n, 0);
      System.arraycopy(row, 0, solution.objectives(), 0, n);
      front.add(solution);
    }
    selection.computeSurvivalScore(front);
    double[] scores = new double[front.size()];
    for (int i = 0; i < scores.length; i++) {
      scores[i] = (Double) front.get(i).attributes().get(AGEMOEAEnvironmentalSelection.getAttributeId());
    }
    return scores;
  }

  public static void main(String[] args) {
    in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    int cases = in.nextInt();
    for (int c = 0; c < cases; c++) {
      String kind = in.next();
      int m = in.nextInt();
      int n = in.nextInt();
      AgeMoea2Values selection = new AgeMoea2Values(n);
      System.out.print(kind);
      switch (kind) {
        case "geometry" -> {
          List<double[]> front = rows(m, n);
          int k = in.nextInt();
          List<Integer> extremes = new ArrayList<>();
          for (int i = 0; i < k; i++) extremes.add(in.nextInt());
          print(selection.computeGeometry(front, extremes));
        }
        case "distances" -> {
          double p = Double.parseDouble(in.next());
          List<double[]> front = rows(m, n);
          selection.P = p; // pairwiseDistances projects the midpoints with the field P
          for (double[] row : selection.pairwiseDistances(front, p)) {
            for (double value : row) print(value);
          }
        }
        case "survival" -> {
          List<double[]> front = rows(m, n);
          for (double value : scores(new AGEMOEAEnvironmentalSelection<>(n), front, n)) print(value);
          for (double value : scores(selection, front, n)) print(value);
        }
        default -> throw new IllegalArgumentException("unknown case kind " + kind);
      }
      System.out.println();
    }
  }
}
