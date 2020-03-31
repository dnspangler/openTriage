import os
import json
import requests
from argparse import ArgumentParser

#TODO: Implement automated testing for multiple frameworks and ui, maybe as a shell script though

def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('--addr', type=str, default='127.0.0.1', help='address of prediction server (defaults to localhost)')
    parser.add_argument('--file', type=str, default='test.json', help='specify file in testData to send (defaults to test)')
    parser.add_argument('--fw', type=str, default='news_adhoc', help='specify framework (defaults to NEWS)')

    return parser.parse_args()

# Lots of ugly dependence on global variabes here... Might get rid of these later.

def test_post(addr,data, type_json=True):
    # Set headers indicating JSON payload
    if type_json:
        headers = {'Accept' : 'application/json',
               'Content-Type' : 'application/json'}
    
    #Send via https if using https port
    
    target = f'https://{addr}/predict/'

    # Return results of post
    results = requests.post(target, data=data, headers=headers, verify=False)
    return results

def test_ui(addr,link):

    target = f'https://{addr}{link}'
        
    print("UI URL:",target)
    #Return results of get
    results = requests.get(target, verify=False)
    return results

def run_tests(test_file,addr):

    print(f"Request using: '{test_file}'")

    # Try predict endpoint
    try:
        with open(test_file, 'rb') as f:
            results = test_post(addr,data=f)
    except Exception as e:
        return "Post failed: " + str(e)
    
    print(results)

    # Try to parse minimum expected items in returned payload
    try:
        r = results.json()
        # For each returned item...
        for key, value in r.items():
            # Try to parse score
            try:
                print(key, value['score'])
            except Exception as e:
                return "No score found: " + str(e)
            # Try to get link

            #TODO: Add test for redis cache assignment correctness

            if 'link' in value:
                try:
                    ui_results = test_ui(addr,value['link'])
                    if ui_results:
                        print("UI sucessfully rendered")
                    else:
                        return "UI render failed: " + ui_results.text
                except Exception as e:
                    return "UI get failed: "  + str(e)
            else:
                print("No link in returned data")
            
    except Exception as e:
        return "Invalid json returned: " + str(e)
    
    return 1


if __name__ == "__main__":

    args = parseArgs()
    
    test_file = f"frameworks/{args.fw}/data/api/{args.file}"

    t = run_tests(test_file,args.addr)

    if t == 1:
        print("Test passed")
    else:
        print(t)
