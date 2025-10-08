#!/usr/bin/env python3
"""
Test script to verify LLM reasoning in MHQA agent
"""
import os
import json

def test_llm_mhqa():
    """Test the MHQA agent with LLM reasoning"""
    
    # Test question
    question = "What is the capital of France?"
    
    print("🧠 Testing MHQA Agent with LLM Reasoning")
    print("=" * 50)
    print(f"Question: {question}")
    print()
    
    # Test with LLM (if API key is available)
    print("1. Testing with LLM reasoning...")
    try:
        os.system(f'python -m agent_systems.MHQA_agent.main --question "{question}" --use_llm -o test_llm_output.json')
        
        # Check output
        with open("test_llm_output.json", "r") as f:
            data = json.load(f)
            
        print("✅ LLM reasoning test completed!")
        print(f"Model: {data.get('model_name', 'unknown')}")
        print(f"Success: {data.get('success', False)}")
        
        # Show the READ step result
        for step in data.get('steps', []):
            if step.get('phase') == 'READ':
                obs = step.get('content', '')
                if 'output' in obs:
                    try:
                        result = json.loads(obs.split('<output>')[1].split('</output>')[0])
                        print(f"Answer: {result.get('answer', 'N/A')}")
                        if 'reasoning' in result:
                            print(f"Reasoning: {result['reasoning'][:100]}...")
                    except:
                        print("Could not parse LLM output")
                break
                
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        print("This is expected if OPENAI_API_KEY is not set")
    
    print()
    
    # Test with heuristic fallback
    print("2. Testing with heuristic reader...")
    try:
        os.system(f'python -m agent_systems.MHQA_agent.main --question "{question}" --no_llm -o test_heuristic_output.json')
        
        # Check output
        with open("test_heuristic_output.json", "r") as f:
            data = json.load(f)
            
        print("✅ Heuristic reader test completed!")
        print(f"Model: {data.get('model_name', 'unknown')}")
        print(f"Success: {data.get('success', False)}")
        
        # Show the READ step result
        for step in data.get('steps', []):
            if step.get('phase') == 'READ':
                obs = step.get('content', '')
                if 'output' in obs:
                    try:
                        result = json.loads(obs.split('<output>')[1].split('</output>')[0])
                        print(f"Answer: {result.get('answer', 'N/A')}")
                    except:
                        print("Could not parse heuristic output")
                break
                
    except Exception as e:
        print(f"❌ Heuristic test failed: {e}")
    
    print()
    print("🎯 Test completed! Check the output files for details.")

if __name__ == "__main__":
    test_llm_mhqa()
