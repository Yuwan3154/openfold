"""parse_usalign on a stored USalign output (cluster probe 10-16, chain 5zo3_A): with -u the TM normalised by the user-specified length is the THIRD TM-score line, not the mean of the two own-length ones."""
import unittest

import t2_diversity as td

OUT_U = """Length of Structure_1: 211 residues
Length of Structure_2: 205 residues
Aligned length= 177, RMSD=   4.11, Seq_ID=n_identical/n_aligned= 1.000
TM-score= 0.58894 (normalized by length of Structure_1: L=211, d0=5.40)
TM-score= 0.60203 (normalized by length of Structure_2: L=205, d0=5.33)
TM-score= 0.59762 (normalized by user-specified L=207.00 and d0=5.35)
(You should use TM-score normalized by length of the reference structure)
"""
OUT_PLAIN = "\n".join(OUT_U.splitlines()[:5]) + "\n"


class ParseUsalign(unittest.TestCase):
    def test_user_length_line(self):
        self.assertAlmostEqual(td.parse_usalign(OUT_U, 207), 0.59762)

    def test_own_length_mean(self):
        self.assertAlmostEqual(td.parse_usalign(OUT_U), (0.58894 + 0.60203) / 2)
        self.assertAlmostEqual(td.parse_usalign(OUT_PLAIN), (0.58894 + 0.60203) / 2)

    def test_mismatch_fails_loudly(self):
        with self.assertRaises(AssertionError):
            td.parse_usalign(OUT_PLAIN, 207)   # -u line missing from the output
        with self.assertRaises(AssertionError):
            td.parse_usalign(OUT_U, 208)   # another length than the one printed


if __name__ == "__main__":
    unittest.main(verbosity=0)
