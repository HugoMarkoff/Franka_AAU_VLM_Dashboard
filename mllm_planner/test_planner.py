#!/usr/bin/env python3

from online_mllm_planner import robot_action_planner

def main():
    print("Testing Robot Action Planner")
    print("="*40)
    
    # Test with a simple task
    result = robot_action_planner(
        task_description="pick up the pen",
        image_path="/home/robolab/llmplanner/images/old.jpeg",
        capture_output=True,
        quiet=False  # Show output on terminal
    )
    
    print("\n" + "="*40)
    print("📋 RESULTS:")
    print("="*40)
    
    if result["success"]:
        print(f"✅ Success: {result['success']}")
        print(f"📝 Task: {result['task_description']}")
        print(f"🔍 Objects: {result['detected_objects']}")
        print(f"📋 Plan: {result['action_plan']}")
        print(f"📄 Captured Output Length: {len(result['terminal_output'])} characters")
    else:
        print(f"❌ Failed: {result}")
    
    return result

if __name__ == "__main__":
    result = main()
    print(result)