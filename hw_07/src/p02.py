#!/usr/bin/env python

# files: p02.py
#
# revision history:
# 20250220 (YQ): initial version
#
# This file contains a Python implementation of significance testing for error rate
# reductions in machine learning experiments. The code computes whether a new system’s
# error rate is statistically significant compared to a baseline and determines the
# minimum decrease in error rate required for significance.
#
# ------------------------------------------------------------------------------

# import required system modules
#
import os
import sys
import math
from scipy.stats import norm

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define global constants
#
BASELINE_ERROR = 0.20  # baseline error rate (20%)
NEW_ERROR_RATE = 0.19  # new system error rate (19%)
CONFIDENCE_LEVEL_BASE = 0.80  # confidence level for N = 1000 in part (a)
CONFIDENCE_LEVELS = [0.85, 0.90, 0.95]  # confidence levels for part (c)
SAMPLE_SIZES = [100, 500, 2000, 5000, 10000]  # sample sizes for part (c)


# ------------------------------------------------------------------------------
#
# classes are listed here
#
# ------------------------------------------------------------------------------

class ErrorSignificanceAnalyzer:
    """
    Class: ErrorSignificanceAnalyzer

    arguments:
     none

    description:
     This class implements significance testing for error rate improvements.
     It calculates the z-statistic for the observed error rate difference,
     compares it against the critical value for a given confidence level, and
     computes the minimum error rate decrease required for statistical significance.
    """

    def __init__(self, baseline_error=BASELINE_ERROR):
        """
        method: __init__

        arguments:
         baseline_error: baseline error rate.

        return:
         none

        description:
         Initializes an instance of ErrorSignificanceAnalyzer.
        """
        self.baseline_error = baseline_error
        ErrorSignificanceAnalyzer.__CLASS_NAME__ = self.__class__.__name__
        print("%s (line: %s) %s::__init__: Instance created with baseline_error = %.2f" % (__FILE__, sys._getframe().f_lineno, ErrorSignificanceAnalyzer.__CLASS_NAME__, self.baseline_error))
        #
        # end of method

    def compute_standard_error(self, N):
        """
        method: compute_standard_error

        arguments:
         N: number of files in the experiment.

        return:
         se: standard error computed from the baseline error rate.

        description:
         Computes the standard error using the formula:
         se = sqrt(p * (1 - p) / N)
         where p is the baseline error rate.
        """
        se = math.sqrt(self.baseline_error * (1 - self.baseline_error) / N)
        return se
        #
        # end of method

    def compute_z_score(self, new_error, N):
        """
        method: compute_z_score

        arguments:
         new_error: error rate of the new system.
         N: number of files in the experiment.

        return:
         z: computed z-score for the error rate difference.

        description:
         Computes the z-score for the difference between the baseline error rate
         and the new error rate using the standard error.
        """
        se = self.compute_standard_error(N)
        diff = self.baseline_error - new_error
        z = diff / se
        return z
        #
        # end of method

    def is_significant(self, new_error, N, confidence):
        """
        method: is_significant

        arguments:
         new_error: error rate of the new system.
         N: number of files in the experiment.
         confidence: desired confidence level (one-tailed).

        return:
         significant: boolean indicating whether the improvement is statistically significant.

        description:
         Determines if the error rate improvement is statistically significant by comparing
         the computed z-score to the critical z-value for the given confidence level.
        """
        z = self.compute_z_score(new_error, N)
        z_crit = norm.ppf(confidence)
        return z >= z_crit
        #
        # end of method

    def min_decrease_required(self, N, confidence):
        """
        method: min_decrease_required

        arguments:
         N: number of files in the experiment.
         confidence: desired confidence level (one-tailed).

        return:
         min_decrease: minimum decrease in error rate required for significance.

        description:
         Computes the minimum required decrease in error rate that would be statistically
         significant. This is given by: delta = z_crit * standard_error.
        """
        se = self.compute_standard_error(N)
        z_crit = norm.ppf(confidence)
        min_decrease = z_crit * se
        return min_decrease
        #
        # end of method

    def format_results_table(self, results):
        """
        method: format_results_table

        arguments:
         results: a list of dictionaries containing results for each experiment.

        return:
         table: a formatted string representing the results table.

        description:
         Formats the results into a table with columns for sample size, confidence level,
         significance result for the observed new error rate, and the minimum decrease required.
        """
        header = "{:<10} {:<12} {:<15} {:<25}".format("N", "Confidence", "19.0% Significant?", "Min. Decrease Required (%)")
        lines = [header, "-" * len(header)]
        for res in results:
            line = "{:<10} {:<12} {:<15} {:<25.4f}".format(res["N"], f"{res['confidence'] * 100:.0f}%", res["significant"], res["min_decrease"] * 100)
            lines.append(line)
        table = "\n".join(lines)
        return table
        #
        # end of method


# ------------------------------------------------------------------------------
#
# main routine
#
# ------------------------------------------------------------------------------
def main():
    """
    method: main

    arguments:
     none

    return:
     none

    description:
     Main routine to execute the error significance analysis.
     For N = 1000 with a baseline of 20.0% and a new system error rate of 19.0%,
     the analysis determines if the improvement is statistically significant at 80%
     confidence. It then computes the minimum decrease required for significance.
     The routine repeats the analysis for various sample sizes and confidence levels,
     and prints the results in a formatted table.
    """
    analyzer = ErrorSignificanceAnalyzer(baseline_error=BASELINE_ERROR)

    results = []

    # part (a) and (b) for N = 1000 at 80% confidence
    #
    N = 1000
    confidence = CONFIDENCE_LEVEL_BASE
    significant = "Yes" if analyzer.is_significant(NEW_ERROR_RATE, N, confidence) else "No"
    min_decrease = analyzer.min_decrease_required(N, confidence)
    results.append({"N": N, "confidence": confidence, "significant": significant, "min_decrease": min_decrease})

    # part (c) for various N and confidence levels 85%, 90% and 95%
    #
    for N in SAMPLE_SIZES:
        for confidence in CONFIDENCE_LEVELS:
            significant = "Yes" if analyzer.is_significant(NEW_ERROR_RATE, N, confidence) else "No"
            min_decrease = analyzer.min_decrease_required(N, confidence)
            results.append({"N": N, "confidence": confidence, "significant": significant, "min_decrease": min_decrease})

    table = analyzer.format_results_table(results)
    print("%s (line: %s) main::__init__: Output:\n%s" % (__FILE__, sys._getframe().f_lineno, table))
    #
    # end of method

if __name__ == "__main__":
    main()

#
# end of file
