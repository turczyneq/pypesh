import numpy as np
import matplotlib.pyplot as plt


def _evaluate_simpsons_rule(f_values, f, a, b):
    """
    Evaluate Simpson's Rule using cached function values.
    """
    m = (a + b) / 2
    if str(m) not in f_values.keys():
        f_values[str(m)] = f(m)
    fa, fb, fm = f_values[str(a)], f_values[str(b)], f_values[str(m)]
    simp = (b - a) / 6 * (fa + 4 * fm + fb)
    return simp


def _asr(f_values, f, a, b, absolute_tolerance):
    """
    Efficient recursive implementation of adaptive Simpson's rule using cached function values.
    This version ensures adaptive refinement of point density.
    """
    # Midpoint
    m = (a + b) / 2
    
    # Calculate Simpson's rule for the whole interval [a, b]
    full_simp = _evaluate_simpsons_rule(f_values,f, a, b)
    
    # Calculate Simpson's rule for the left and right subintervals [a, m] and [m, b]
    left_simp = _evaluate_simpsons_rule(f_values, f, a, m)
    right_simp = _evaluate_simpsons_rule(f_values, f, m, b)
    
    # Estimate error (delta) between the full interval and subdivided intervals
    delta = left_simp + right_simp - full_simp
    
    # If the error is small enough, return the sum of the left and right intervals
    if abs(delta) <= 15 * absolute_tolerance:
        # Add the error correction term delta / 15 for better accuracy
        return left_simp + right_simp + delta / 15
    else:

        # Continue refining in both subintervals
        left_result = _asr(f_values, f, a, m, absolute_tolerance / 2)
        right_result = _asr(f_values, f, m, b, absolute_tolerance / 2)
        
        return left_result + right_result


def integrate_with_adaptive_simpson(f, a, b, absolute_tolerance):
    """
    Integrate f from a to b using Adaptive Simpson's Rule with a maximum error of eps.
    This version tracks points where f is evaluated, ensuring denser points in regions requiring more refinement.
    """
    # Initial function evaluations
    f_values = {str(x): f(x) for x in [a, b, (a + b) / (2)]}
    
    # Track the points where the function is evaluated for plotting
    xargs = [a, b, (a + b) / 2]
    yargs = [f(a), f(b), f((a + b) / 2)]
    
    # Call the recursive function with the initial parameters
    result = _asr(f_values, f, a, b, absolute_tolerance)

    xargs = [float(x) for x in f_values.keys()]
    yargs = [float(x) for x in f_values.values()]

    # plt.figure(figsize=(8, 8))
    # plt.scatter(xargs,yargs)
    # plt.show()

    return result



def _example_use_simpson():
    import math

    a, b = 0.0, 1.0

    def fun(x):
        print(f"Function call {x:.2f}")
        return math.sin(x)

    sin_integral = integrate_with_adaptive_simpson(fun, a, b, 1e-4)
    print(
        "Simpson's integration of sine from {} to {} = {}\n".format(a, b, sin_integral)
    )