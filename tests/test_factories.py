import unittest
import rivapy.instruments as instruments
from rivapy.instruments.factory import _factory, create

class FactoryTestsMeta(type):
    
    
    def __new__(cls, name, bases, attrs):
        for k,v in _factory().items():
            attrs['test_%s' % k] = cls.gen(v)
        return super(FactoryTestsMeta, cls).__new__(cls, name, bases, attrs)
    
    @classmethod
    def gen(cls, target_class):
        examples = target_class._create_sample(2)
        # Return a testcase that tests ``x``.
        def fn(self):
            ins = target_class._create_sample(4, seed=42)
            for i in range(len(ins)):
                b = create(ins[i].to_dict())
                self.assertEqual(b.hash(), ins[i].hash())
            #errors = NotebookTestsMeta._notebook_run(notebook_file)
            #self.assertTrue(errors == [])
            
        return fn

class InstrumentTests(unittest.TestCase, metaclass = FactoryTestsMeta):
    pass
#    def test_depp(self):
#        self.assertEqual(0,0)

if __name__ == '__main__':
    unittest.main()
