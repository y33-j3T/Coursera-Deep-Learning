import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import EagerTensor
from numpy import int64

def test_loop(test_cases):
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            assert test_case["result"] == test_case["expected"]
            success += 1
    
        except:
            fails += 1
            print(f'{test_case["name"]}: {test_case["error_message"]}\nExpected: {test_case["expected"]}\nResult: {test_case["result"]}\n')

    if fails == 0:
        print("\033[92m All public tests passed")

    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        raise Exception(test_case["error_message"])

        
def test_my_rmse(my_rmse):

    test_y_true = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    test_y_pred = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    
    expected = 1.7795130420052185
    
    result = my_rmse(test_y_true, test_y_pred)
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(result),
            "expected": EagerTensor,
            "error_message": f'output has an incorrect type.'
        },
        {
            "name": "output_check",
            "result": result,
            "expected": expected,
            "error_message": "Output is incorrect. Please check the equation."
        }
    ]
    
    test_loop(test_cases)
    
def test_model_loss(model_loss):
    
    test_y_true = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    test_y_pred = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    
    expected = 1.7795130420052185
    
    result = model_loss(test_y_true, test_y_pred)
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(result),
            "expected": EagerTensor,
            "error_message": f'output has an incorrect type.'
        },
        {
            "name": "output_check",
            "result": result,
            "expected": expected,
            "error_message": "Output is incorrect. Please check the equation."
        }
    ]
    
    test_loop(test_cases)