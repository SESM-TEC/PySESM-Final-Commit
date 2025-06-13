import os
import inspect
import sys

def print_pytest_instructions():
    # Get the caller's filename
    caller_frame = inspect.stack()[1]
    caller_file = os.path.basename(caller_frame.filename)
    
    print(f"This is a test file meant to be run with pytest, not directly with Python.")
    print("To run the tests, use one of these commands:")
    print(f"\npytest {caller_file}  # Run all tests in this file")
    
    # Get all test functions from the caller's module
    caller_module = sys.modules[caller_frame.frame.f_globals['__name__']]
    test_functions = [name for name, obj in inspect.getmembers(caller_module) 
                     if inspect.isfunction(obj) and name.startswith('test_')]
    
    if test_functions:
        print("\nOr run specific tests:")
        for func in test_functions:
            print(f"pytest {caller_file}::{func}  # Run just the {func} test")
            
    print("\npytest  # Run all tests in the current directory")
    exit(1)
    
