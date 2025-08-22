#!/usr/bin/env python3
"""
Test script for integrated search tool functionality
"""
import sys
import os
import json
from dotenv import load_dotenv

# Add the current directory to the path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from utils import tavily_search, TOOLS

def test_search_function():
    """Test the search function directly"""
    print("🧪 Testing tavily_search function")
    print("=" * 50)
    
    test_query = "Python pandas dataframe tutorial"
    print(f"Query: '{test_query}'")
    
    result = tavily_search(test_query)
    print(f"Result length: {len(result)} characters")
    print("\nFirst 500 characters:")
    print(result[:500] + "..." if len(result) > 500 else result)
    
    return "❌ Search failed" not in result

def test_tools_configuration():
    """Test that the tools are properly configured"""
    print("\n🔧 Testing TOOLS configuration")
    print("=" * 50)
    
    print(f"Number of tools: {len(TOOLS)}")
    
    tool_names = []
    for tool in TOOLS:
        if 'function' in tool and 'name' in tool['function']:
            tool_names.append(tool['function']['name'])
    
    print(f"Tool names: {tool_names}")
    
    # Check if search tool is present
    has_search = "tavily_search" in tool_names
    has_code = "add_and_execute_jupyter_code_cell" in tool_names
    
    print(f"✅ Has search tool: {has_search}")
    print(f"✅ Has code execution tool: {has_code}")
    
    if has_search:
        # Find search tool and check its configuration
        search_tool = None
        for tool in TOOLS:
            if tool['function']['name'] == 'tavily_search':
                search_tool = tool
                break
        
        if search_tool:
            print("\n📋 Search tool configuration:")
            print(f"   Description: {search_tool['function']['description']}")
            print(f"   Parameters: {list(search_tool['function']['parameters']['properties'].keys())}")
            print(f"   Required: {search_tool['function']['parameters']['required']}")
    
    return has_search and has_code

def test_api_key():
    """Test API key availability"""
    print("\n🔑 Testing API key configuration")
    print("=" * 50)
    
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        print(f"✅ TAVILY_API_KEY found: {api_key[:8]}...")
        return True
    else:
        print("❌ TAVILY_API_KEY not found in environment")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Integrated Search Tool")
    print("=" * 50)
    
    tests = [
        ("API Key", test_api_key),
        ("Tools Configuration", test_tools_configuration), 
        ("Search Function", test_search_function)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🚀 Running {test_name} test...")
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Search tool is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)