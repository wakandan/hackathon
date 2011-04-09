import unittest
import math
import funcs 

class TestFuncs(unittest.TestCase):
    def test_transform(self):
        f = math.sin  
        input_data = range(12*12)
        input_data = map(lambda x: 5*math.pi*x/12, input_data)
        input_data = map(f, input_data)
        output_data = funcs.transform(input_data) 
        return output_data

    def test_draw_1(self):
        #funcs.draw(range(10))
        input_data = self.test_transform() 
        x_axis =  map(lambda x: x/12.0, range(12*12))
        print x_axis 
        funcs.draw(input_data, x_axis)

    def test_draw_2(self):
        self.assertRaises(Exception, funcs.draw(range(10), range(20)))

if __name__ == '__main__':
    unittest.main()

         
