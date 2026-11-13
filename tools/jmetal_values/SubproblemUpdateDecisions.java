import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import org.uma.jmetal.component.catalogue.ea.replacement.subproblemupdate.SubproblemUpdateCriterion;
import org.uma.jmetal.component.catalogue.ea.replacement.subproblemupdate.impl.FeasibilityRulesCriterion;
import org.uma.jmetal.component.catalogue.ea.replacement.subproblemupdate.impl.ImprovedEpsilonCriterion;
import org.uma.jmetal.component.catalogue.ea.replacement.subproblemupdate.impl.ViolationThresholdCriterion;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.doublesolution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.bounds.Bounds;

/**
 * Decisions of the subproblem update criteria of jMetal (package subproblemupdate), for
 * tools/subproblem_update_reference_decisions.py, which documents the input read from standard
 * input. Constraint values arrive in the VAMOS convention and are negated.
 */
public class SubproblemUpdateDecisions {
  static Scanner in;

  static double next() {
    return Double.parseDouble(in.next());
  }

  static DoubleSolution solution(int k) {
    DoubleSolution solution = new DefaultDoubleSolution(List.of(Bounds.create(0.0, 1.0)), 1, k);
    for (int j = 0; j < k; j++) solution.constraints()[j] = -next();
    return solution;
  }

  static double field(Object criterion, String name) throws Exception {
    Field field = criterion.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field.getDouble(criterion);
  }

  public static void main(String[] args) throws Exception {
    in = new Scanner(System.in);
    in.useLocale(Locale.ROOT);
    int cases = in.nextInt();
    StringBuilder out = new StringBuilder();
    for (int c = 0; c < cases; c++) {
      String name = in.next();
      double tao = next();
      double threshold = next();
      int tc = in.nextInt();
      int n = in.nextInt();
      int k = in.nextInt();
      SubproblemUpdateCriterion<DoubleSolution> criterion = switch (name) {
        case "feasibility_rules" -> new FeasibilityRulesCriterion<>();
        case "improved_epsilon" -> new ImprovedEpsilonCriterion<>(tao, threshold, tc);
        case "violation_threshold" -> new ViolationThresholdCriterion<>();
        default -> throw new IllegalArgumentException(name);
      };
      int steps = in.nextInt();
      for (int s = 0; s < steps; s++) {
        List<DoubleSolution> population = new ArrayList<>();
        for (int i = 0; i < n; i++) population.add(solution(k));
        DoubleSolution child = solution(k);
        criterion.update(population, child);
        int candidates = in.nextInt();
        out.append("replaces");
        for (int q = 0; q < candidates; q++) {
          DoubleSolution current = solution(k);
          double childValue = next();
          double currentValue = next();
          out.append(' ').append(criterion.replaces(child, childValue, current, currentValue) ? 1 : 0);
        }
        out.append('\n');
        double state = Double.NaN;
        if (criterion instanceof ImprovedEpsilonCriterion) state = field(criterion, "epsilonK");
        if (criterion instanceof ViolationThresholdCriterion) state = field(criterion, "threshold");
        out.append("state ").append(Double.toString(state)).append('\n');
      }
    }
    System.out.print(out);
  }
}
