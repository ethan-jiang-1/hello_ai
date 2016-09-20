import unittest
# from app.main import Main
#from app.svm_size import SvmSize
from app_sklearn.svm_size.svm_size import SvmSize


class SvmSizeTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_svm(self):

        svm_machine = SvmSize()

        svm_machine.training()
        svm_machine.predict()


        import pdb; pdb.set_trace()
        pass
