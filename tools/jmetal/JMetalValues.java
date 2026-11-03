import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import org.uma.jmetal.problem.doubleproblem.DoubleProblem;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;

/**
 * Evaluates jMetal problems at given points, for tools/problem_reference_values.py.
 *
 * <p>Each input line is {@code class<TAB>arguments<TAB>x}: a problem class, its constructor
 * arguments separated by commas (empty for the default constructor), and the variable values. The
 * constructor is the public one with as many parameters; each argument is converted to its
 * parameter's type (int, double or boolean, or their boxed types). Each output line is
 * {@code objectives<TAB>constraints}, in jMetal's own convention, with 17 significant digits.
 */
public final class JMetalValues {
  private JMetalValues() {}

  public static void main(String[] args) throws Exception {
    BufferedReader input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    String line;
    while ((line = input.readLine()) != null) {
      if (line.isBlank()) {
        continue;
      }
      String[] fields = line.split("\t", -1);
      DoubleProblem problem = instantiate(fields[0], fields[1]);
      DoubleSolution solution = problem.createSolution();
      String[] values = fields[2].split(",");
      if (values.length != problem.numberOfVariables()) {
        throw new IllegalArgumentException(fields[0] + " has " + problem.numberOfVariables() + " variables, got " + values.length);
      }
      for (int i = 0; i < values.length; i++) {
        solution.variables().set(i, Double.parseDouble(values[i]));
      }
      problem.evaluate(solution);
      System.out.println(join(solution.objectives()) + "\t" + join(solution.constraints()));
    }
  }

  private static DoubleProblem instantiate(String className, String arguments) throws Exception {
    String[] tokens = arguments.isEmpty() ? new String[0] : arguments.split(",");
    for (Constructor<?> constructor : Class.forName(className).getConstructors()) {
      Class<?>[] types = constructor.getParameterTypes();
      if (types.length != tokens.length) {
        continue;
      }
      Object[] values = new Object[types.length];
      for (int i = 0; i < types.length; i++) {
        values[i] = convert(tokens[i], types[i]);
      }
      return (DoubleProblem) constructor.newInstance(values);
    }
    throw new IllegalArgumentException(className + " has no public constructor with " + tokens.length + " parameters");
  }

  private static Object convert(String token, Class<?> type) {
    if (type == int.class || type == Integer.class) {
      return Integer.valueOf(token);
    }
    if (type == double.class || type == Double.class) {
      return Double.valueOf(token);
    }
    if (type == boolean.class || type == Boolean.class) {
      return Boolean.valueOf(token);
    }
    if (type == String.class) {
      return token;
    }
    throw new IllegalArgumentException("Unsupported constructor parameter type " + type);
  }

  private static String join(double[] values) {
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        text.append(',');
      }
      text.append(String.format(Locale.ROOT, "%.17g", values[i]));
    }
    return text.toString();
  }
}
