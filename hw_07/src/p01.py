#!/usr/bin/env python

# files: p01.py
#
# revision history:
# 20250220 (YQ): initial version
#
# This file contains a Python implementation of an entropy and mutual information
# analysis for dataset set #8 and statistically independent uniform random data.
# The analysis includes data processing, quantization of two-dimensional feature
# vectors, computation of marginal, joint, and conditional entropies, as well as
# mutual information.
#
#------------------------------------------------------------------------------

# import required system modules
#
import os
import sys
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define global constants
#
QUANT_LEVELS = 128  # number of quantization levels
EPSILON = 1e-12  # small constant to avoid log(0)


#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class EntropyAnalyzer:
    """
    Class: EntropyAnalyzer

    arguments:
     none

    description:
     This class implements an entropy and mutual information analysis for
     dataset set #8 and for statistically independent uniform random data.
     It processes input CSV data, performs quantization of two-dimensional feature
     vectors, computes marginal entropies H(x1), H(x2), joint entropy H(x1, x2),
     conditional entropies H(x1|x2) and H(x2|x1), and mutual information I(x1;x2).
    """

    def __init__(self, quant_levels=QUANT_LEVELS):
        """
        method: __init__

        arguments:
         quant_levels: number of quantization levels for discretization.

        return:
         none

        description:
         Initializes an instance of EntropyAnalyzer.
        """
        self.quant_levels = quant_levels
        EntropyAnalyzer.__CLASS_NAME__ = self.__class__.__name__
        print("%s (line: %s) %s::__init__: Instance created with quant_levels = %d" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__, self.quant_levels))
        #
        # end of method

    def quantize_data(self, data):
        """
        method: quantize_data

        arguments:
         data: a numpy array of input data.

        return:
         quantized_data: a numpy array of quantized data.

        description:
         Quantizes each element of the input data using the formula:
         y = round((x / (x_max - x_min)) * quant_levels).
         (Adjusts for the case when the range is zero)
        """
        print("%s (line: %s) %s::quantize_data: Quantizing data" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        x_min = np.min(data)
        x_max = np.max(data)
        range_val = x_max - x_min
        if range_val == 0:
            quantized_data = np.zeros_like(data, dtype=int)
        else:
            quantized_data = np.round((data / range_val) * self.quant_levels).astype(int)
        return quantized_data
        #
        # end of method

    def compute_entropy(self, values):
        """
        method: compute_entropy

        arguments:
         values: a one-dimensional numpy array of quantized values.

        return:
         entropy: computed entropy in bits.

        description:
         Computes the entropy H(x) for the given values using base-2 logarithm.
        """
        print("%s (line: %s) %s::compute_entropy: Computing entropy" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        unique_vals, counts = np.unique(values, return_counts=True)
        probabilities = counts / float(np.sum(counts))
        entropy = -np.sum(probabilities * np.log2(probabilities + EPSILON))
        return entropy
        #
        # end of method

    def compute_joint_entropy(self, values1, values2):
        """
        method: compute_joint_entropy

        arguments:
         values1: a one-dimensional numpy array of quantized values for x1.
         values2: a one-dimensional numpy array of quantized values for x2.

        return:
         joint_entropy: computed joint entropy H(x1, x2) in bits.

        description:
         Computes the joint entropy H(x1, x2) by considering the frequency of pairs
         of quantized values.
        """
        print("%s (line: %s) %s::compute_joint_entropy: Computing joint entropy" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        combined = values1 * (self.quant_levels + 1) + values2
        unique_vals, counts = np.unique(combined, return_counts=True)
        probabilities = counts[counts > 0] / float(np.sum(counts))
        joint_entropy = -np.sum(probabilities * np.log2(probabilities + EPSILON))
        return joint_entropy
        #
        # end of method

    def compute_conditional_entropy(self, joint_entropy, marginal_entropy):
        """
        method: compute_conditional_entropy

        arguments:
         joint_entropy: the joint entropy H(x1, x2) in bits.
         marginal_entropy: the marginal entropy H(x1) or H(x2) in bits.

        return:
         conditional_entropy: computed conditional entropy in bits.

        description:
         Computes the conditional entropy H(x|y) using:
         H(x|y) = H(x, y) - H(y).
        """
        print("%s (line: %s) %s::compute_conditional_entropy: Computing conditional entropy" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        conditional_entropy = joint_entropy - marginal_entropy
        return conditional_entropy
        #
        # end of method

    def compute_mutual_information(self, entropy1, entropy2, joint_entropy):
        """
        method: compute_mutual_information

        arguments:
         entropy1: entropy H(x1) in bits.
         entropy2: entropy H(x2) in bits.
         joint_entropy: joint entropy H(x1, x2) in bits.

        return:
         mutual_information: computed mutual information I(x1;x2) in bits.

        description:
         Computes the mutual information I(x1;x2) using:
         I(x1;x2) = H(x1) + H(x2) - H(x1, x2).
        """
        print("%s (line: %s) %s::compute_mutual_information: Computing mutual information" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        mutual_information = entropy1 + entropy2 - joint_entropy
        return mutual_information
        #
        # end of method

    def process_file(self, filename):
        """
        method: process_file

        arguments:
         filename: string path to the CSV file containing two-dimensional feature vectors.

        return:
         results: a dictionary containing entropy and mutual information metrics.

        description:
         Reads the CSV file, quantizes each column of the data, and computes:
         H(x1), H(x2), H(x1, x2), H(x1|x2), H(x2|x1), and I(x1;x2).
        """
        print("%s (line: %s) %s::process_file: Processing file %s" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__, filename))
        data_df = pd.read_csv(filename)
        x1 = data_df.iloc[:, 0].values
        x2 = data_df.iloc[:, 1].values
        qx1 = self.quantize_data(x1)
        qx2 = self.quantize_data(x2)
        H_x1 = self.compute_entropy(qx1)
        H_x2 = self.compute_entropy(qx2)
        H_joint = self.compute_joint_entropy(qx1, qx2)
        H_x1_given_x2 = self.compute_conditional_entropy(H_joint, H_x2)
        H_x2_given_x1 = self.compute_conditional_entropy(H_joint, H_x1)
        I_x1_x2 = self.compute_mutual_information(H_x1, H_x2, H_joint)
        results = {"H(x1)": H_x1, "H(x2)": H_x2, "H(x1, x2)": H_joint, "H(x1|x2)": H_x1_given_x2, "H(x2|x1)": H_x2_given_x1, "I(x1;x2)": I_x1_x2}
        return results
        #
        # end of method

    def process_uniform(self, num_samples):
        """
        method: process_uniform

        arguments:
         num_samples: number of samples to generate for each variable.

        return:
         results: a dictionary containing entropy and mutual information metrics
                  for uniform random data.

        description:
         Generates statistically independent uniform random data for x1 and x2,
         quantizes the data, and computes:
         H(x1), H(x2), H(x1, x2), H(x1|x2), H(x2|x1), and I(x1;x2).
         Also prints the theoretical entropy for comparison.
        """
        print("%s (line: %s) %s::process_uniform: Processing uniform random data" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        x1 = np.random.uniform(0, 1, num_samples)
        x2 = np.random.uniform(0, 1, num_samples)
        qx1 = self.quantize_data(x1)
        qx2 = self.quantize_data(x2)
        H_x1 = self.compute_entropy(qx1)
        H_x2 = self.compute_entropy(qx2)
        H_joint = self.compute_joint_entropy(qx1, qx2)
        H_x1_given_x2 = self.compute_conditional_entropy(H_joint, H_x2)
        H_x2_given_x1 = self.compute_conditional_entropy(H_joint, H_x1)
        I_x1_x2 = self.compute_mutual_information(H_x1, H_x2, H_joint)
        results = {"H(x1)": H_x1, "H(x2)": H_x2, "H(x1, x2)": H_joint, "H(x1|x2)": H_x1_given_x2, "H(x2|x1)": H_x2_given_x1, "I(x1;x2)": I_x1_x2}
        theoretical_entropy = np.log2(self.quant_levels + 1)
        print("%s (line: %s) %s::process_uniform: Theoretical entropy for uniform variable: %.4f bits" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__, theoretical_entropy))
        return results
        #
        # end of method

    def format_results(self, results_dict):
        """
        method: format_results

        arguments:
         results_dict: a dictionary where keys are dataset identifiers and values are
         dictionaries of computed metrics.

        return:
         output: a formatted string representation of the results table.

        description:
         Formats the computed results into a table with columns representing
         the train set, dev set, and uniform random data.
        """
        print("%s (line: %s) %s::format_results: Formatting results" % (__FILE__, sys._getframe().f_lineno, EntropyAnalyzer.__CLASS_NAME__))
        header = "{:<12} {:>12} {:>12} {:>12}".format("Metric", "Train", "Dev", "Uniform")
        lines = [header, "-" * len(header)]
        metrics = ["H(x1)", "H(x2)", "H(x1, x2)", "H(x1|x2)", "H(x2|x1)", "I(x1;x2)"]
        for metric in metrics:
            train_val = results_dict["train"].get(metric, 0)
            dev_val = results_dict["dev"].get(metric, 0)
            uniform_val = results_dict["uniform"].get(metric, 0)
            line = "{:<12} {:>12.4f} {:>12.4f} {:>12.4f}".format(metric, train_val, dev_val, uniform_val)
            lines.append(line)
        output = "\n".join(lines)
        return output
        #
        # end of method


#------------------------------------------------------------------------------
#
# main routine
#
#------------------------------------------------------------------------------
def main():
    """
    method: main

    arguments:
     none (input is expected via command line arguments)

    return:
     none

    description:
     Main routine to execute the entropy analysis.
     Reads train.csv and dev.csv files, processes the data to compute entropy and
     mutual information metrics, processes uniform random data for comparison, and
     outputs a formatted table of results.
    """
    if len(sys.argv) < 3:
        print("Usage: {} <train_csv> <dev_csv>".format(sys.argv[0]))
        sys.exit(1)
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    analyzer = EntropyAnalyzer(quant_levels=QUANT_LEVELS)
    train_results = analyzer.process_file(train_file)
    dev_results = analyzer.process_file(dev_file)
    num_samples = pd.read_csv(train_file).shape[0]
    uniform_results = analyzer.process_uniform(num_samples)
    results_dict = {"train": train_results, "dev": dev_results, "uniform": uniform_results}
    output = analyzer.format_results(results_dict)
    print("%s (line: %s) main::__init__: Output:\n%s" % (__FILE__, sys._getframe().f_lineno, output))
    #
    # end of method


if __name__ == "__main__":
    main()

#
# end of file
