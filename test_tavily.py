#!/usr/bin/env python3
"""
Test script for Tavily search functionality
"""
import os
import datetime
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_tavily_search():
    """Test basic Tavily search functionality"""
    
    
    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found in environment")
        return False
    
    print(f"✅ Found Tavily API key: {api_key[:8]}...")
    
    try:
        # Initialize client
        client = TavilyClient(api_key=api_key)
        print("✅ Tavily client initialized successfully")
        
        # Test query with current year
        current_year = datetime.datetime.now().year
        test_query = f"Python matplotlib tutorial {current_year}"
        
        print(f"🔍 Testing search with query: '{test_query}'")
        print(f"📏 Query length: {len(test_query)} characters")
        
        # Test search with parameters
        response = client.search(
            query=test_query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
            include_raw_content=False
        )
        
        print("✅ Search completed successfully!")
        print(f"📊 Response keys: {list(response.keys())}")
        print(f"📝 Found {len(response.get('results', []))} results")
        
        # Display first result
        if response.get('results'):
            first_result = response['results'][0]
            print(f"\n📋 First Result:")
            print(f"   Title: {first_result.get('title', 'N/A')}")
            print(f"   URL: {first_result.get('url', 'N/A')}")
            print(f"   Content preview: {first_result.get('content', 'N/A')[:200]}...")
            print(f"   Score: {first_result.get('score', 'N/A')}")
        
        # Display answer if available
        if response.get('answer'):
            print(f"\n💡 AI Answer: {response['answer'][:300]}...")
        
        # Test query length limit (400 characters)
        long_query = "a" * 401
        print(f"\n🧪 Testing query length limit with {len(long_query)} character query...")
        
        try:
            long_response = client.search(query=long_query, max_results=1)
            print("⚠️  Long query succeeded - no length limit enforced by API")
        except Exception as e:
            print(f"✅ Long query failed as expected: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Tavily search: {str(e)}")
        return False

def test_search_formatting():
    """Test search result formatting for LLM consumption"""
    
    # Mock response for testing formatting
    mock_response = {
        "query": "Python pandas tutorial 2025",
        "results": [
            {
                "title": "Pandas Tutorial - Complete Guide",
                "url": "https://example.com/pandas-tutorial",
                "content": "This is a comprehensive guide to pandas library...",
                "score": 0.95
            },
            {
                "title": "Advanced Pandas Techniques",
                "url": "https://example.com/advanced-pandas", 
                "content": "Learn advanced pandas techniques for data analysis...",
                "score": 0.87
            }
        ],
        "answer": "Pandas is a powerful data manipulation library for Python..."
    }
    
    print("\n🎨 Testing search result formatting:")
    
    # Format results for LLM
    formatted_results = format_search_results_for_llm(mock_response)
    print(formatted_results)
    
    return True

def format_search_results_for_llm(response):
    """Format search results for LLM consumption"""
    
    query = response.get('query', 'Unknown query')
    results = response.get('results', [])
    answer = response.get('answer', '')
    
    formatted = f"🔍 **Search Results for:** {query}\n\n"
    
    if answer:
        formatted += f"**Quick Answer:** {answer}\n\n"
    
    if results:
        formatted += f"**Found {len(results)} relevant sources:**\n\n"
        
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            content = result.get('content', '')
            score = result.get('score', 0)
            
            # Truncate content to reasonable length
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted += f"**{i}. {title}** (Score: {score:.2f})\n"
            formatted += f"   🔗 {url}\n"
            formatted += f"   📄 {content}\n\n"
    else:
        formatted += "No results found.\n"
    
    return formatted

if __name__ == "__main__":
    print("🧪 Testing Tavily Search Integration")
    print("=" * 50)
    
    # Test basic search
    success = test_tavily_search()
    
    if success:
        print("\n" + "=" * 50)
        test_search_formatting()
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")