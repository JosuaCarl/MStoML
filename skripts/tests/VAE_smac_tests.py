import sys
import io
import unittest

sys.path.append("..")
from VAE.VAE_smac import *

last_timestamp = time.time()
step = 0
runtimes = {}

def capture_output(method, args):
    capturedOutput = io.StringIO()               # Make StringIO.
    sys.stdout = capturedOutput                  # Redirect stdout.
    method(*args)                                   # Call function.
    sys.stdout = sys.__stdout__                  # Reset redirect.
    return capturedOutput.getvalue() 

class Test_time_step(unittest.TestCase):
    
    def test_verbose_equal(self):
        self.assertTrue(capture_output(time_step,
                                        {"message": "test",
                                         "verbosity": 1,
                                         "min_verbosity": 1}).startswith("test ("))
        self.assertTrue(capture_output(time_step,
                                        {"message": "test",
                                         "verbosity": 1,
                                         "min_verbosity": 1}).endswith("s)"))
        
    def test_non_verbose(self):
        self.assertEqual(capture_output(time_step,
                                        {"message": "test",
                                         "verbosity": 1,
                                         "min_verbosity": 2}), "")


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)



if __name__ == '__main__':
    unittest.main()