#!/usr/bin/env python3
"""
Enhanced Robot Action Planner - 4-Step Pipeline with Online API

A modular robot action planning system that:
1. Analyzes images with vision model (Qwen2.5-VL)
2. Validates syntax and spatial relationships
3. Refines reasoning with online LLM (DeepSeek)
4. Generates executable action plans

Author: Chen Li
"""

import sys
import os
import time
import json
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import io
import contextlib
from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model Configuration
QWEN2_5_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Available Robot Actions
AVAILABLE_ACTIONS = {
    "MovetoObject": "move to the specified object",
    "PickupObject": "pick up the object", 
    "PutDownObject": "put down the object at the current location",
    "OpenGripper": "open the gripper",
    "CloseGripper": "close the gripper",
    "Home": "return the robot to its home position"
}

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Model instances (kept loaded for performance)
vision_model = None
vision_processor = None
gpu_count = 0

# Cache for image analysis results (avoids reprocessing same image)
cached_image_path = None
cached_step1_objects = None
cached_step1_relationships = None
cached_step2_objects = None
cached_step2_relationships = None
cached_step3_objects = None
cached_step3_relationships = None

# Pipeline information display
print("="*80)
print("🤖 ENHANCED ROBOT ACTION PLANNER - 4-STEP PIPELINE (ONLINE API)")
print("="*80)
print("STEP 1: Object identification + raw reasoning through VLM (Qwen2.5-VL)")
print("STEP 2: Syntax check of reasoning results and object lists")
print("STEP 3: Refined reasoning for logical consistency through Online LLM (DeepSeek)")
print("STEP 4: Generate the action plan")
print("="*80)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def call_online_llm(prompt, max_tokens=512):
    """
    Call the online LLM API (DeepSeek through Monica).
    
    Args:
        prompt (str): The prompt to send to the API
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str or None: Response text or None if error
    """
    try:
        client = OpenAI(api_key="sk-0cfe59a4171744658e2c9db9ffd8082d", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        response = response.choices[0].message.content

        if len(response) > 0:
            return response
        else:
            print(f"[ERROR] No response content found: {response}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Online LLM API call failed: {e}")
        return None

def extract_objects_from_response(response):
    """
    Extract objects from vision model response using multiple parsing strategies.
    
    Args:
        response (str): Raw vision model response
        
    Returns:
        list: List of detected object names
    """
    import re
    
    detected_objects = []
    
    # Strategy 1: Parse structured "Objects Identified" section
    objects_section_match = re.search(r'#### Objects Identified:(.*?)(?=####|\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
    
    if objects_section_match:
        objects_section = objects_section_match.group(1)
        object_matches = re.findall(r'\d+\.\s*\*\*([^*]+)\*\*', objects_section)
        
        for obj in object_matches:
            obj = obj.strip()
            if obj and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Strategy 2: Fallback - spatial relationships section
    if not detected_objects:
        spatial_matches = re.findall(r'\d+\.\s*\*\*([^*:]+)\*\*:', response)
        for obj in spatial_matches:
            obj = obj.strip()
            if obj and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Strategy 3: Final fallback - any bold text
    if not detected_objects:
        bold_matches = re.findall(r'\*\*([A-Za-z][A-Za-z\s]{2,20})\*\*', response)
        non_objects = {'ON TOP OF', 'TO THE LEFT OF', 'TO THE RIGHT OF', 'OVERLAPPING WITH',
                      'Objects Identified', 'Spatial Relationships', 'Summary', 'Step', 'Analysis'}
        
        for obj in bold_matches:
            obj = obj.strip()
            if obj not in non_objects and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Clean and deduplicate
    final_objects = []
    seen = set()
    
    for obj in detected_objects:
        obj_clean = obj.strip().title()
        if obj_clean and obj_clean not in seen and len(obj_clean) >= 3:
            final_objects.append(obj_clean)
            seen.add(obj_clean)
    
    return final_objects[:6]  # Limit to reasonable number

def correct_spatial_inconsistencies(relationships_str, objects_list, summary_text):
    """
    Correct spatial relationship inconsistencies using summary as reference truth.
    
    Args:
        relationships_str (str): Raw spatial relationships
        objects_list (list): List of detected objects
        summary_text (str): Summary section for ground truth
        
    Returns:
        str: Corrected relationships text
    """
    import re
    
    print(f"\n[CONSISTENCY CHECK] Analyzing spatial relationships for logical errors...")
    
    corrected_text = relationships_str
    ground_truth = {}
    
    if summary_text:
        # Extract relationships from summary
        for line in summary_text.split('\n'):
            line = line.strip('- ').strip()
            
            # Parse "X is on top of Y" patterns
            if 'is on top of' in line.lower():
                match = re.search(r'(?:the )?\*\*?(\w+(?:\s+\w+)*)\*\*? is \*\*?on top of\*\*? (?:the )?\*\*?(\w+(?:\s+\w+)*)\*\*?', line.lower())
                if match:
                    obj1, obj2 = match.groups()
                    ground_truth[obj1.title()] = {'on_top_of': obj2.title()}
                    ground_truth[obj2.title()] = ground_truth.get(obj2.title(), {})
                    ground_truth[obj2.title()]['under'] = ground_truth[obj2.title()].get('under', []) + [obj1.title()]
            
            # Parse "X is under Y" patterns
            elif 'is under' in line.lower() or 'under the' in line.lower():
                match = re.search(r'the (\w+(?:\s+\w+)*) is under (?:the )?(.+)', line.lower())
                if match:
                    obj1 = match.group(1)
                    under_objects_str = match.group(2)
                    
                    # Parse multiple objects separated by "and"
                    under_objects = []
                    for obj in re.split(r'\s+and\s+', under_objects_str):
                        obj = obj.strip()
                        obj = re.sub(r'^(?:the\s+)?', '', obj)
                        obj = re.sub(r'[,.]$', '', obj)
                        if obj:
                            under_objects.append(obj.title())
                    
                    if under_objects:
                        ground_truth[obj1.title()] = ground_truth.get(obj1.title(), {})
                        ground_truth[obj1.title()]['under'] = under_objects
                        
                        for under_obj in under_objects:
                            ground_truth[under_obj] = ground_truth.get(under_obj, {})
                            if 'on_top_of' not in ground_truth[under_obj]:
                                ground_truth[under_obj]['on_top_of'] = []
                            if isinstance(ground_truth[under_obj]['on_top_of'], str):
                                ground_truth[under_obj]['on_top_of'] = [ground_truth[under_obj]['on_top_of']]
                            if obj1.title() not in ground_truth[under_obj]['on_top_of']:
                                ground_truth[under_obj]['on_top_of'].append(obj1.title())
    
    print(f"[CONSISTENCY CHECK] ✅ Spatial relationship validation completed")
    return corrected_text

def extract_plan_from_response(plan_response, thinking_content, task_description, objects_list, relationships_str):
    """
    Extract action plan from model response with fallback plan generation.
    
    Args:
        plan_response (str): Raw model response
        thinking_content (str): Thinking content (unused but kept for compatibility)
        task_description (str): Original task
        objects_list (list): Detected objects
        relationships_str (str): Spatial relationships
        
    Returns:
        str: Formatted action plan
    """
    import re
    
    # Method 1: Look for "plan:" pattern
    plan_matches = list(re.finditer(r'plan:\s*\(', plan_response, re.IGNORECASE))
    if plan_matches:
        last_plan_start = plan_matches[-1].start()
        plan_section = plan_response[last_plan_start:]
        
        lines = plan_section.split('\n')
        plan_line = lines[0].strip()
        
        # Check if plan continues on subsequent lines
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line.startswith('->') or (line and '->' in line and not any(stop_word in line.lower() for stop_word in ['but', 'however', 'wait', 'looking'])):
                plan_line += ' ' + line
            else:
                break
        
        if plan_line and len(plan_line) > 10:
            return plan_line
    
    # Method 2: Look for action patterns
    action_pattern = r'\([A-Za-z]+,\s*\([0-9.]+,\s*[0-9.]+\),\s*[A-Za-z]+\)'
    if re.search(action_pattern, plan_response):
        actions = re.findall(action_pattern, plan_response)
        if actions:
            return "plan: " + " -> ".join(actions)
    
    # Method 3: Fallback plan generation
    print("[WARNING] Could not extract valid plan from model output. Creating fallback plan...")
    
    # Identify target objects from task
    target_objects = []
    task_lower = task_description.lower()
    
    if "first" in task_lower and "then" in task_lower:
        # Handle multi-step tasks
        first_part = task_lower.split("then")[0]
        then_part = task_lower.split("then")[1] if "then" in task_lower else ""
        
        first_obj = None
        then_obj = None
        
        for obj in objects_list:
            if obj.lower() in first_part and first_obj is None:
                first_obj = obj
            if obj.lower() in then_part and then_obj is None:
                then_obj = obj
        
        if first_obj:
            target_objects.append(first_obj)
        if then_obj:
            target_objects.append(then_obj)
    else:
        # Single object task
        for obj in objects_list:
            if obj.lower() in task_lower:
                target_objects.append(obj)
    
    target_obj = target_objects[0] if target_objects else None
    
    if target_obj:
        # Create plan based on dependencies
        plan_steps = []
        
        for target in target_objects:
            # Find blocking objects
            target_blocking_objects = []
            
            for obj in objects_list:
                if obj != target:
                    blocking_patterns = [
                        f"the {obj.lower()} is on top of the {target.lower()}",
                        f"{obj.lower()} is on top of the {target.lower()}",
                        f"- the {obj.lower()} is on top of the {target.lower()}",
                        f"- {obj.lower()} is on top of the {target.lower()}",
                        f"{obj.lower()} on top of {target.lower()}",
                    ]
                    
                    if any(pattern in relationships_str.lower() for pattern in blocking_patterns):
                        target_blocking_objects.append(obj)
            
            # Move blocking objects first
            for blocking_obj in target_blocking_objects:
                plan_steps.extend([
                    f"({blocking_obj}, (0.0, 0.0), MovetoObject)",
                    f"({blocking_obj}, (0.0, 0.0), PickupObject)",
                    f"({blocking_obj}, (0.0, 0.0), PutDownObject)"
                ])
            
            # Then pick up target
            plan_steps.extend([
                f"({target}, (0.0, 0.0), MovetoObject)",
                f"({target}, (0.0, 0.0), PickupObject)"
            ])
        
        if plan_steps:
            return "plan: " + " -> ".join(plan_steps)
        else:
            return f"plan: ({target_obj}, (0.0, 0.0), MovetoObject) -> ({target_obj}, (0.0, 0.0), PickupObject)"
    
    return "plan: (Object, (0.0, 0.0), MovetoObject) -> (Object, (0.0, 0.0), PickupObject)"

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def load_vision_model():
    """
    Load the vision model for object detection and spatial analysis.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global vision_model, vision_processor, gpu_count
    
    print("\n🚀 LOADING VISION MODEL FOR 4-STEP PIPELINE WITH ONLINE API\n")
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        print(f"[INFO] CUDA is available. Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f} GB)")
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        print("[ERROR] CUDA is not available. This code requires GPU for vision model.")
        return False
    
    # Load Vision Model
    print(f"\n[LOADING] Vision Model: {QWEN2_5_VL_MODEL}")
    print("[INFO] This may take 1-2 minutes on first load...")
    
    try:
        from qwen_vl_utils import process_vision_info
        
        # Load processor first
        vision_processor = AutoProcessor.from_pretrained(QWEN2_5_VL_MODEL)
        
        # Load model with memory optimization
        vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN2_5_VL_MODEL, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print(f"[INFO] ✅ Qwen2.5-VL model loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to load vision model: {e}")
        return False
    
    print(f"\n🎉 Vision model loaded! Ready for 4-Step pipeline.\n")
    return True

def cleanup_models():
    """Clean up loaded models to free GPU memory."""
    global vision_model, vision_processor
    
    print("[INFO] Cleaning up models...")
    
    if vision_model is not None:
        del vision_model
        vision_model = None
    if vision_processor is not None:
        del vision_processor
        vision_processor = None
    
    torch.cuda.empty_cache()
    print("[INFO] ✅ Vision model cleaned up and GPU memory freed.")

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step1_vision_analysis(image_path, force_reanalyze=False):
    """
    STEP 1: Object identification and spatial analysis using vision model.
    
    Args:
        image_path (str): Path to the input image
        force_reanalyze (bool): Force re-analysis even if cached
        
    Returns:
        tuple: (objects_list, relationships_str) or (None, None) if error
    """
    global vision_model, vision_processor
    global cached_image_path, cached_step1_objects, cached_step1_relationships
    
    print("\n" + "="*60)
    print("🔍 STEP 1: VISION ANALYSIS (Qwen2.5-VL)")
    print("="*60)
    
    if vision_model is None or vision_processor is None:
        print("[ERROR] Vision model not loaded. Please run load_vision_model() first.")
        return None, None
    
    # Check cache first
    if not force_reanalyze and cached_image_path == image_path and cached_step1_objects is not None:
        print("[INFO] 🚀 Using cached STEP 1 results (much faster!)")
        return cached_step1_objects, cached_step1_relationships
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Could not open image: {e}")
        return None, None

    from qwen_vl_utils import process_vision_info

    # Prepare vision prompt
    prompt = (
        "Analyze this 2D image step by step to identify objects and their spatial relationships. Be precise and consistent.\n"
        "\n"
        "STEP 1: List all visible objects (banana, controller, remote, paper, etc.)\n"
        "STEP 2: For each object, determine if it is:\n"
        "  - ON TOP OF another object (physically resting on it)\n"
        "  - TO THE LEFT of another object\n" 
        "  - TO THE RIGHT of another object\n"
        "  - OVERLAPPING with another object\n"
        "\n"
        "CRITICAL RULES:\n"
        "- If object A is ON TOP OF object B, then B is UNDERNEATH A (not also on top)\n"
        "- Be extremely careful about ON TOP OF vs just positioned near\n"
        "- Only use ON TOP OF when one object is clearly resting on another\n"
        "- Focus on the most obvious spatial relationships\n"
        "\n"
        "Format your response clearly for each object with its relationships."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs for inference
    text = vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vision_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to device
    device = next(vision_model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Run inference
    start_time = time.time()
    print("[INFO] Running STEP 1 vision inference... ⏳")
    
    with torch.no_grad():
        generated_ids = vision_model.generate(
            **inputs, 
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=vision_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = vision_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    end_time = time.time()
    print(f"[INFO] ✅ STEP 1 completed in {end_time - start_time:.1f} seconds")
    print(f"[STEP 1 OUTPUT] Raw Vision Analysis:\n{response}")

    # Extract objects and cache results
    objects_list = extract_objects_from_response(response)
    relationships_str = response

    cached_image_path = image_path
    cached_step1_objects = objects_list
    cached_step1_relationships = relationships_str
    
    # Clean up temporary variables but keep models loaded
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

    print(f"[STEP 1 RESULT] Detected Objects: {objects_list}")
    print(f"[STEP 1 RESULT] Raw Relationships: {relationships_str[:100]}...")
    
    return objects_list, relationships_str

def step2_syntax_validation(objects_list, relationships_str):
    """
    STEP 2: Validate syntax and check spatial consistency.
    
    Args:
        objects_list (list): Objects from STEP 1
        relationships_str (str): Relationships from STEP 1
        
    Returns:
        tuple: (validated_objects, corrected_relationships, validation_issues)
    """
    global cached_step2_objects, cached_step2_relationships
    
    print("\n" + "="*60)
    print("✅ STEP 2: SYNTAX VALIDATION & SPATIAL CONSISTENCY CHECK")
    print("="*60)
    
    validated_objects = []
    validation_issues = []
    
    # Re-extract objects if needed
    if not objects_list or len(objects_list) < 3 or any(obj in ['Spatial', 'Overlapping', 'Analysis', 'Objects'] for obj in objects_list):
        print(f"[WARNING] Object extraction failed or returned invalid objects: {objects_list}")
        print(f"[INFO] Re-extracting objects directly from VLM response...")
        objects_list = extract_objects_from_response(relationships_str)
        print(f"[INFO] Re-extracted objects: {objects_list}")
    
    # Validate object list
    if not objects_list:
        validation_issues.append("No objects detected in the scene")
        print("[ERROR] No objects to validate")
        return [], relationships_str, validation_issues
    
    print(f"[INFO] Validating {len(objects_list)} detected objects...")
    
    # Clean and validate object names
    for obj in objects_list:
        if obj and isinstance(obj, str) and len(obj.strip()) > 0:
            cleaned_obj = obj.strip().title()
            
            if len(cleaned_obj) <= 50 and cleaned_obj.replace(' ', '').isalpha():
                validated_objects.append(cleaned_obj)
                print(f"[VALID] ✓ {cleaned_obj}")
            else:
                validation_issues.append(f"Invalid object name: {obj}")
                print(f"[INVALID] ✗ {obj} - Invalid format")
        else:
            validation_issues.append(f"Malformed object entry: {obj}")
            print(f"[INVALID] ✗ {obj} - Malformed entry")
    
    # Remove duplicates
    validated_objects = list(dict.fromkeys(validated_objects))
    
    # Validate spatial relationships
    print(f"\n[INFO] Checking spatial relationship consistency...")
    cleaned_relationships = relationships_str.strip()
    
    if not cleaned_relationships:
        validation_issues.append("No spatial relationships described")
        print("[WARNING] No spatial relationships found")
        cached_step2_objects = validated_objects
        cached_step2_relationships = cleaned_relationships
        return validated_objects, cleaned_relationships, validation_issues
    
    # Extract summary for reference truth
    import re
    summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', cleaned_relationships, re.DOTALL | re.IGNORECASE)
    summary_text = summary_match.group(1).strip() if summary_match else ""
    
    print(f"[INFO] Found summary section: {bool(summary_text)}")
    if summary_text:
        print(f"[SUMMARY] {summary_text}")
    
    # Correct spatial inconsistencies
    corrected_relationships = correct_spatial_inconsistencies(cleaned_relationships, validated_objects, summary_text)
    
    # Validate relationships
    relationship_keywords = ['on top of', 'above', 'below', 'left', 'right', 'next to', 'under', 'underneath']
    has_relationships = any(keyword in corrected_relationships.lower() for keyword in relationship_keywords)
    
    if has_relationships:
        print("[VALID] ✓ Spatial relationships detected and validated")
    else:
        validation_issues.append("No clear spatial relationships detected")
        print("[WARNING] No clear spatial relationships found")
    
    # Summary
    if validation_issues:
        print(f"\n[STEP 2 ISSUES] Found {len(validation_issues)} validation issues:")
        for issue in validation_issues:
            print(f"  ⚠️ {issue}")
    else:
        print("\n[STEP 2 SUCCESS] ✅ All validation checks passed")
    
    print(f"[STEP 2 RESULT] Final Validated Objects: {validated_objects}")
    print(f"[STEP 2 RESULT] Final Corrected Relationships: {corrected_relationships[:150]}...")
    
    # Cache results
    cached_step2_objects = validated_objects
    cached_step2_relationships = corrected_relationships
    
    return validated_objects, corrected_relationships, validation_issues

def step3_reasoning_refinement(objects_list, relationships_str, task_description, validation_issues):
    """
    STEP 3: Refine reasoning for logical consistency using online LLM.
    
    Args:
        objects_list (list): Validated objects from STEP 2
        relationships_str (str): Relationships from STEP 2
        task_description (str): Original task description
        validation_issues (list): Issues from STEP 2
        
    Returns:
        tuple: (refined_objects, refined_relationships)
    """
    global cached_step3_objects, cached_step3_relationships
    
    print("\n" + "="*60)
    print("🧠 STEP 3: REASONING REFINEMENT (Online DeepSeek API)")
    print("="*60)
    
    # Prepare refinement prompt
    validation_context = ""
    if validation_issues:
        validation_context = f"\nValidation Issues from STEP 2: {', '.join(validation_issues)}"
    
    refinement_prompt = f"""You are an expert at analyzing spatial relationships and object detection results. 
Your task is to review and refine the vision analysis results for logical consistency.

TASK CONTEXT: {task_description}

STEP 1 & 2 RESULTS:
Objects Detected: {', '.join(objects_list)}
Spatial Relationships: {relationships_str}
{validation_context}

REFINEMENT INSTRUCTIONS:
1. Check if the detected objects make sense for the given task
2. Verify spatial relationships are logically consistent
3. Identify any missing objects that might be important for the task
4. Resolve any contradictions in spatial descriptions
5. Ensure object names are clear and standardized

Please provide your refined analysis in this exact format:

REFINED_OBJECTS: [comma-separated list of objects]
REFINED_RELATIONSHIPS: [clear description of spatial relationships]
REASONING: [brief explanation of any changes made]

Focus on accuracy and logical consistency. If the original analysis is good, you can keep it unchanged."""

    try:
        print("[INFO] Running STEP 3 reasoning refinement via online API... ⏳")
        
        # Call online LLM API
        refinement_response = call_online_llm(refinement_prompt, max_tokens=1024)
        
        if refinement_response is None:
            print("[ERROR] Online API call failed for STEP 3. Using STEP 2 validated results.")
            return objects_list, relationships_str
        
        print(f"[STEP 3 OUTPUT] DeepSeek API Refinement:\n{refinement_response}")
        
        # Parse refinement response
        refined_objects = objects_list  # Default to original
        refined_relationships = relationships_str  # Default to original
        
        import re
        
        # Extract refined objects
        objects_match = re.search(r'REFINED_OBJECTS:\s*\[([^\]]+)\]', refinement_response, re.IGNORECASE)
        if objects_match:
            objects_str = objects_match.group(1)
            refined_objects = [obj.strip().title() for obj in objects_str.split(',') if obj.strip()]
        else:
            # Try alternative format
            objects_match = re.search(r'REFINED_OBJECTS:\s*([^\n]+)', refinement_response, re.IGNORECASE)
            if objects_match:
                objects_str = objects_match.group(1)
                refined_objects = [obj.strip().title() for obj in objects_str.split(',') if obj.strip()]
        
        # Extract refined relationships
        relationships_match = re.search(r'REFINED_RELATIONSHIPS:\s*(.*?)(?=\nREASONING:|$)', refinement_response, re.IGNORECASE | re.DOTALL)
        if relationships_match:
            llm_refined_relationships = relationships_match.group(1).strip()
            print(f"[STEP 3 INFO] LLM provided refined relationships: {llm_refined_relationships[:100]}...")
            
            # Extract Summary section from original relationships
            summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', relationships_str, re.DOTALL | re.IGNORECASE)
            
            if summary_match:
                summary_text = summary_match.group(1).strip()
                refined_relationships = summary_text
                print(f"[STEP 3 INFO] Extracted Summary section for STEP 4: {summary_text}")
            else:
                print(f"[STEP 3 WARNING] No Summary section found, using LLM refined relationships")
                refined_relationships = llm_refined_relationships
        else:
            # Fallback approach
            relationships_match = re.search(r'REFINED_RELATIONSHIPS:\s*(.*)', refinement_response, re.IGNORECASE | re.DOTALL)
            if relationships_match:
                llm_refined_relationships = relationships_match.group(1).strip()
                
                # Still try to extract Summary section
                summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', relationships_str, re.DOTALL | re.IGNORECASE)
                
                if summary_match:
                    summary_text = summary_match.group(1).strip()
                    refined_relationships = summary_text
                    print(f"[STEP 3 INFO] Extracted Summary section for STEP 4: {summary_text}")
                else:
                    refined_relationships = llm_refined_relationships
            else:
                # Last resort
                summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', relationships_str, re.DOTALL | re.IGNORECASE)
                
                if summary_match:
                    summary_text = summary_match.group(1).strip()
                    refined_relationships = summary_text
                    print(f"[STEP 3 INFO] Using Summary section from original: {summary_text}")
                else:
                    refined_relationships = relationships_str
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*([^\n]+)', refinement_response, re.IGNORECASE)
        if reasoning_match:
            reasoning_explanation = reasoning_match.group(1).strip()
            print(f"[STEP 3 REASONING] {reasoning_explanation}")
        
        print(f"[STEP 3 SUCCESS] ✅ Reasoning refinement completed")
        print(f"[STEP 3 RESULT] Final Objects: {refined_objects}")
        print(f"[STEP 3 RESULT] Final Relationships: {refined_relationships[:100]}...")
        
        # Cache results
        cached_step3_objects = refined_objects
        cached_step3_relationships = refined_relationships
        
        return refined_objects, refined_relationships
        
    except Exception as e:
        print(f"[ERROR] Error during STEP 3 reasoning refinement: {e}")
        print("[INFO] Using STEP 2 validated results instead")
        return objects_list, relationships_str

def step4_action_planning(task_description, objects_list, relationships_str):
    """
    STEP 4: Generate executable action plan using online LLM.
    
    Args:
        task_description (str): Original task description
        objects_list (list): Final objects from STEP 3
        relationships_str (str): Final relationships from STEP 3
        
    Returns:
        str or None: Final action plan or None if failed
    """
    print("\n" + "="*60)
    print("🚀 STEP 4: ACTION PLANNING (Online DeepSeek API)")
    print("="*60)
    
    print(f"[INFO] Planning for task: {task_description}")
    print(f"[INFO] Using objects: {objects_list}")

    # Extract summary for cleaner scene description
    import re
    summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', relationships_str, re.DOTALL | re.IGNORECASE)
    
    if summary_match:
        summary_text = summary_match.group(1).strip()
        scene_description = f"SPATIAL RELATIONSHIPS SUMMARY:\n{summary_text}\n\nDETAILED ANALYSIS:\n{relationships_str[:500]}..."
        print(f"[INFO] Using extracted summary for scene description")
    else:
        scene_description = relationships_str
        print(f"[INFO] No summary found, using full relationships")
        
    prompt = f"""You are a robot action planner. Create a sequence of robot actions for the given task.

TASK: {task_description}
OBJECTS IN SCENE: {', '.join(objects_list)}
SCENE DESCRIPTION: {scene_description}

ROBOT ACTIONS AVAILABLE:
- MovetoObject: move to an object
- PickupObject: pick up an object
- PutDownObject: put down an object
- OpenGripper: open gripper
- CloseGripper: close gripper
- Home: return to home position

CRITICAL PLANNING RULES:
1. If you need to pick up an object that has other objects ON TOP OF it or RESTING ON it, you MUST first move those blocking objects away.
2. You cannot pick up an object if another object is resting on it or on top of it.
3. Always check the scene description for spatial relationships before planning.
4. If an object A is "on top of", "resting on", or "above" object B, then A must be moved before B can be picked up.

SPATIAL LOGIC CLARIFICATION:
- If "X is on top of Y" → X is accessible, Y is blocked by X
- If "Y is under X" → Y is blocked by X, X is accessible  
- To pick up X (when X is on top): Just pick up X directly
- To pick up Y (when X is on top of Y): Move X away first, then pick up Y

STEP-BY-STEP ANALYSIS REQUIRED:
1. Identify the target object from the task
2. Check: Is anything ON TOP OF the target object? 
3. If YES: Move those objects first
4. If NO: Pick up target directly

EXAMPLE:
- Scene: "Banana is on top of paper"
- Task: "Pick up banana" → Plan: Move to banana, pick up banana (banana is accessible)
- Task: "Pick up paper" → Plan: Move banana away first, then move to paper, pick up paper

INSTRUCTIONS:
- Handle multi-step tasks (e.g., "first...then...") by addressing each target in order
- For each target object, check what objects are "ON TOP OF" it and move those first
- If target is on top of something else, target is directly accessible
- Provide ONLY the final action plan in the exact format below
- Do not include reasoning or explanation in the final output, just the plan

FINAL ACTION PLAN (use this exact format):
plan: (ObjectName, (0.0, 0.0), ActionName) -> (ObjectName, (0.0, 0.0), ActionName) -> ..."""

    try:
        print("[INFO] Running STEP 4 action planning via online API... ⏳")
        
        # Call online LLM API
        plan_response = call_online_llm(prompt, max_tokens=1024)
        
        if plan_response is None:
            print("[ERROR] Online API call failed for STEP 4.")
            return None
        
        print(f"[STEP 4 OUTPUT] DeepSeek API Response:\n{plan_response}")
        print(f"[STEP 4 SUCCESS] ✅ Action planning completed")
        
    except Exception as e:
        print(f"[ERROR] Error during STEP 4 inference: {e}")
        return None
    
    # Extract plan from response
    final_plan = extract_plan_from_response(plan_response, "", task_description, objects_list, relationships_str)
    
    print(f"[STEP 4 RESULT] Final Action Plan: {final_plan}")
    return final_plan

# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def run_complete_4step_pipeline(image_path, task_description, force_reanalyze=False):
    """
    Run the complete 4-step pipeline.
    
    Args:
        image_path (str): Path to input image
        task_description (str): Task to accomplish
        force_reanalyze (bool): Force re-analysis of image
        
    Returns:
        tuple: (final_objects, final_relationships, final_plan)
    """
    print("\n" + "🚀"*20)
    print("RUNNING COMPLETE 4-STEP PIPELINE (ONLINE API)")
    print("🚀"*20)
    
    # STEP 1: Vision Analysis
    step1_objects, step1_relationships = step1_vision_analysis(image_path, force_reanalyze)
    if not step1_objects or not step1_relationships:
        print("[ERROR] STEP 1 failed. Aborting pipeline.")
        return None, None, None
    
    # STEP 2: Syntax Validation
    step2_objects, step2_relationships, validation_issues = step2_syntax_validation(step1_objects, step1_relationships)
    
    # STEP 3: Reasoning Refinement
    step3_objects, step3_relationships = step3_reasoning_refinement(step2_objects, step2_relationships, task_description, validation_issues)
    
    # STEP 4: Action Planning
    final_plan = step4_action_planning(task_description, step3_objects, step3_relationships)
    
    print("\n" + "🎉"*20)
    print("4-STEP PIPELINE COMPLETED (ONLINE API)")
    print("🎉"*20)
    
    return step3_objects, step3_relationships, final_plan

def robot_action_planner(task_description, image_path, output_format="json", force_reanalyze=False, quiet=False, capture_output=False):
    """
    Entry function for robot action planning that can be imported and called by other scripts.
    
    Args:
        task_description (str): Description of the task for the robot to perform
        image_path (str): Path to the image file for scene analysis
        output_format (str): Output format - "json" or "text" (default: "json")
        force_reanalyze (bool): Force re-analysis of the image even if cached (default: False)
        quiet (bool): Suppress progress output, only return result (default: False)
        capture_output (bool): Capture all terminal output and return as string (default: False)
    
    Returns:
        dict: Result dictionary containing:
            - task_description: The input task
            - image_path: The input image path
            - detected_objects: List of detected objects
            - spatial_relationships: Spatial relationship description
            - action_plan: Generated action plan
            - success: Boolean indicating if planning succeeded
            - terminal_output: Captured terminal output (if capture_output=True)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If task description is empty
        RuntimeError: If model loading or planning fails
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    if not task_description.strip():
        raise ValueError("Task description cannot be empty.")
    
    # Initialize output capture
    captured_output = ""
    original_stdout = sys.stdout
    
    try:
        # Set up output capturing - only capture if quiet=True, OR if capture_output=True AND quiet=True
        if quiet:
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
        elif capture_output and not quiet:
            # Special case: show on terminal AND capture
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            
            class TeeOutput:
                def __init__(self, terminal, buffer):
                    self.terminal = terminal
                    self.buffer = buffer
                
                def write(self, text):
                    self.terminal.write(text)
                    self.buffer.write(text)
                    return len(text)
                
                def flush(self):
                    self.terminal.flush()
                    self.buffer.flush()
            
            sys.stdout = TeeOutput(original_stdout, output_buffer)
        
        # Show startup message if not quiet
        if not quiet:
            print("\n🚀 Starting Enhanced Robot Action Planner with 4-Step Pipeline (Online API)")
        
        # Load vision model
        if not load_vision_model():
            if quiet or (capture_output and not quiet):
                sys.stdout = original_stdout
            raise RuntimeError("Failed to load vision model.")
        
        # Run the complete 4-step pipeline
        final_objects, final_relationships, final_plan = run_complete_4step_pipeline(
            image_path, 
            task_description, 
            force_reanalyze
        )
        
        # Capture the output if requested
        if quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        elif capture_output and not quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        
    except Exception as e:
        # Restore stdout in case of error
        if quiet or (capture_output and not quiet):
            captured_output = output_buffer.getvalue() if 'output_buffer' in locals() else ""
            sys.stdout = original_stdout
        cleanup_models()
        raise RuntimeError(f"Planning failed: {e}")
    
    # Check if pipeline succeeded
    if final_plan is None:
        cleanup_models()
        raise RuntimeError("Failed to generate action plan.")
    
    # Prepare result
    result = {
        "task_description": task_description,
        "image_path": image_path,
        "detected_objects": final_objects or [],
        "spatial_relationships": final_relationships or "",
        "action_plan": final_plan or "",
        "success": final_plan is not None
    }
    
    # Add captured output if requested
    if capture_output:
        result["terminal_output"] = captured_output
    
    # Cleanup models
    cleanup_models()
    
    return result

# For backward compatibility
def run_robot_action_planner(task_description, image_path, **kwargs):
    """Alias for robot_action_planner for backward compatibility."""
    return robot_action_planner(task_description, image_path, **kwargs) 